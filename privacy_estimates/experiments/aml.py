import hydra
import os
import logging
import json
import re
import shlex
import fnmatch
import pandas as pd

from urllib.parse import urlparse, parse_qs
from azure.ai.ml import MLClient, load_component
from azure.ai.ml.entities import Component, PipelineJob, Job, JobResourceConfiguration, QueueSettings
from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential
from dataclasses import dataclass, field
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import TypeVar, Type, Any, get_type_hints, Dict, Union, List, OrderedDict
from dataclasses import is_dataclass
from yaml import safe_load
from parmap import map as pmap
from typing import Optional, Callable, Optional, Iterable, get_args, get_origin
from collections.abc import Mapping
from subprocess import check_output, CalledProcessError
from tqdm import tqdm
from functools import lru_cache
from importlib.metadata import version

from privacy_estimates.experiments.utils import SingletonMeta


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_credential() -> ChainedTokenCredential:
    return ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(),
    )


@dataclass
class RegistryConfig:
    registry_name: str
    location: str = None
    credential: Optional[DefaultAzureCredential] = None

    def __post_init__(self) -> None:
        self.ml_client = self._get_ml_client()

    def _get_ml_client(self) -> MLClient:
        credential = self.credential
        if credential is None:
            credential = get_credential()
        return MLClient(credential=credential, registry_name=self.registry_name, registry_location=self.location)


class ComputeConfig:
    def apply(self, job: Job) -> Job:
        raise NotImplementedError("Must be implemented in subclass")


@dataclass
class ServerlessComputeConfig(ComputeConfig):
    instance_type: str
    instance_count: int = 1
    job_tier: str = "Standard"
    process_count_per_instance: int = 1
    locations: Optional[List[str]] = None

    def apply(self, job: Job) -> Job:
        job.compute = "serverless"
        job.resources = JobResourceConfiguration(instance_count=self.instance_count, instance_type=self.instance_type,
                                                 locations=self.locations)
        job.queue_settings = QueueSettings(job_tier=self.job_tier)
        if job.distribution is not None:
            job.distribution.process_count_per_instance = self.process_count_per_instance
        elif self.process_count_per_instance > 1:
            raise ValueError("Cannot set process_count_per_instance without setting distribution")
        return job


@dataclass
class ClusterComputeConfig(ComputeConfig):
    cluster_name: str
    process_count_per_instance: int = 1

    def apply(self, job: Job) -> Job:
        job.compute = self.cluster_name
        if job.distribution is not None:
            job.distribution.process_count_per_instance = self.process_count_per_instance
        elif self.process_count_per_instance > 1:
            raise ValueError("Cannot set process_count_per_instance without setting distribution")
        return job


def from_dict(cls, d: Dict) -> ClusterComputeConfig:
    d = d.copy()
    type = d.pop("type", None)
    if type is None:
        raise ValueError("Creating ClusterComputeConfig from dict requires a 'type' key")
    if type == "serverless":
        return ServerlessComputeConfig(**d)
    elif type == "cluster":
        return ClusterComputeConfig(**d)
    else:
        raise ValueError(f"Unknown compute type {type}")

setattr(ComputeConfig, "from_dict", classmethod(from_dict))


@dataclass
class WorkspaceConfig:
    workspace_name: str
    resource_group: str
    subscription_id: str
    default_compute: Optional[str] = None
    compute: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.ml_client = self._get_ml_client()

        if self.default_compute is not None:
            if "default" in self.compute:
                raise ValueError("Cannot specify both default_compute and 'default' in compute")
            self.compute["default"] = self.default_compute

        for n, c in self.compute.items():
            if isinstance(c, str):
                self.compute[n] = ClusterComputeConfig(cluster_name=c)
            elif isinstance(c, Mapping):
                self.compute[n] = ComputeConfig.from_dict(c)
            else:
                raise ValueError(f"Invalid compute configuration for {n}")

        self.default_compute = self.compute.get("default", None)

    def _get_ml_client(self) -> MLClient:
        credential = get_credential()
        return MLClient(credential=credential, subscription_id=self.subscription_id, resource_group_name=self.resource_group,
                        workspace_name=self.workspace_name)

    @property
    def workspace(self):
        from azureml.core import Workspace
        return Workspace(subscription_id=self.subscription_id, resource_group=self.resource_group,
                         workspace_name=self.workspace_name)

    def get_job(self, name: str):
        return self.ml_client.jobs.get(name=name)

    @classmethod
    def from_yaml(cls, path: Union[str,Path]):
        path = Path(path)
        with path.open() as f:
            cfg = safe_load(f)
        return cls(
            subscription_id=cfg['subscription_id'],
            workspace_name=cfg['workspace_name'],
            resource_group=cfg['resource_group'],
            cpu_compute=cfg.get('cpu_compute', None),
            gpu_compute=cfg.get('gpu_compute', None),
            large_memory_cpu_compute=cfg.get('large_memory_cpu_compute', None),
            compute=cfg.get('compute', {}),
        )

    @classmethod
    def from_env_vars(cls, **kwargs):
        subscription = os.environ.get('AZUREML_ARM_SUBSCRIPTION', None)
        workspace = os.environ.get('AZUREML_ARM_WORKSPACE_NAME', None)
        group = os.environ.get('AZUREML_ARM_RESOURCE_GROUP', None)
        if not all([subscription, workspace, group]):
            return None
        return cls(
            subscription_id=os.environ['AZUREML_ARM_SUBSCRIPTION'],
            workspace_name=os.environ['AZUREML_ARM_WORKSPACE_NAME'],
            resource_group=os.environ['AZUREML_ARM_RESOURCE_GROUP'],
            **kwargs
        )

    @classmethod
    def from_az_cli(cls, **kwargs):
        try:
            sub = json.loads(check_output(["az", "account", "show"]))["id"]
            name = json.loads(check_output(["az", "config", "get", "defaults.workspace"]))["value"]
            group = json.loads(check_output(["az", "config", "get", "defaults.group"]))["value"]
        except CalledProcessError:
            return None
        return cls(subscription_id=sub, workspace_name=name, resource_group=group, **kwargs)

    @classmethod
    def create(cls, yaml_path: Optional[Path] = None, **kwargs):
        """
        Creates an instance of the class first by trying to load the configuration from a YAML file, then by trying to load
        the configuration from environment variables, and finally by trying to load the configuration from the Azure CLI.

        Args:
            yaml_path (Optional[Path]): Path to a YAML file containing configuration settings. If provided, the class instance
                                        will be created using the settings from the YAML file.
            **kwargs: Additional keyword arguments that can be used to customize the class instance creation.

        Returns:
            An instance of the class.

        Raises:
            None.

        """
        if yaml_path is not None:
            return cls.from_yaml(yaml_path)
        if (
            {"AZUREML_ARM_SUBSCRIPTION", "AZUREML_ARM_WORKSPACE_NAME", "AZUREML_ARM_RESOURCE_GROUP"}.issubset(
                set(os.environ.keys())
            )
        ):
            return cls.from_env_vars(**kwargs)
        return cls.from_az_cli(**kwargs)


T = TypeVar("T")


def dictconfig_to_dataclass(cfg: DictConfig, dataclass_type: Type[T]) -> T:
    def convert_nested(cfg_obj: Any, dataclass_obj: Type[T]) -> T:
        if is_dataclass(dataclass_obj):
            fields = {field.name: field.type for field in dataclass_obj.__dataclass_fields__.values()}
            mutable_dict = OmegaConf.to_container(cfg_obj, resolve=True) if isinstance(cfg_obj, DictConfig) else cfg_obj
            for key, value in fields.items():
                if key in mutable_dict:
                    if get_origin(value) == list and len(get_args(value)) == 1 and is_dataclass(get_args(value)[0]):
                        mutable_dict[key] = [convert_nested(item, get_args(value)[0]) for item in mutable_dict[key]]
                    if isinstance(mutable_dict[key], DictConfig) or isinstance(mutable_dict[key], dict):
                        mutable_dict[key] = convert_nested(mutable_dict[key], fields[key])
            try:
                return dataclass_obj(**mutable_dict)
            except TypeError as e:
                raise TypeError(
                    f"Could not create {dataclass_obj}. {e}. Expected arguments: {sorted(fields.keys())}. "
                    f"Passed arguments: {sorted(mutable_dict.keys())}"
                )
        else:
            return cfg_obj

    return convert_nested(cfg, dataclass_type)


def is_url(path: str) -> bool:
    return str(path).startswith("http://") or str(path).startswith("https://")


def get_latest_version(client: MLClient, name: str) -> str:
    latest_version = None
    created_at = None
    for component in client.components.list(name=name):
        if created_at is None or component.creation_context.created_at > created_at:
            latest_version = component.version
            created_at = component.creation_context.created_at
    return latest_version


class PrivacyEstimatesComponentLoader(metaclass=SingletonMeta):
    def __init__(self, client: Optional[MLClient] = None, override_version: Optional[str] = None):
        """
        Singleton class to load privacy_estimates AML components.

        This class is used to have universal way of loading components for the privacy estimates package.
        Most of the time, you will not need to interact with this class directly, since the default is to load the components
        locally i.e. from your local privacy_estimates installation. However, other use cases require signed AML components
        in this case we want to globally select the version of the AML components to load.

        Args:
            client (Optional[MLClient]): An optional MLClient instance to interact with Azure Machine Learning services.
            override_version (Optional[str]): An optional string to override the component version. If not provided, 
                                              the version will be fetched from the PRIVACY_ESTIMATES_COMPONENT_VERSION 
                                              environment variable.
        """
        self.client = client
        self.override_version = override_version

        if self.override_version is None:
            logger.info("Using component version from PRIVACY_ESTIMATES_COMPONENT_VERSION environment variable")
            self.override_version = os.environ.get("PRIVACY_ESTIMATES_COMPONENT_VERSION", None)

    @classmethod
    def set_client(cls, client: MLClient, version: str = version("privacy_estimates")) -> None:
        """
        Sets the global MLClient and version for the PrivacyEstimatesComponentLoader.

        Args:
            client (MLClient): The MLClient instance to be set globally.
            version (str, optional): The version of the privacy estimates to override. Defaults to the version of the 
                                     "privacy_estimates" package. It is also possible to pass 'default' which will use
                                     the default version of the components in the workspace.
        """
        loader = PrivacyEstimatesComponentLoader(client=client, override_version=version)
        if loader.client is not client or loader.override_version != version:
            raise ValueError("Could not set global client and version. It probably has been set already.")

    def load_from_function(self, func: Callable[..., Component], version: str = "local") -> Callable[..., Component]:
        version = self.override_version or version

        if version == "local":
            return func
        else:
            return self.load_by_name(name=func.component.name, version=version)

    def load_from_component_spec(self, path: Path, version: str = "local") -> Callable[..., Component]:
        version = self.override_version or version

        if path.exists():
            return self._load_from_local_component_spec(path, version)
        elif is_url(path):
            return self._load_from_remote_component_spec(path, version)
        else:
            raise FileNotFoundError(f"Could not find component spec at {path}. Path does not exist.")

    def _load_from_local_component_spec(self, path: Path, version: str = "local") -> Callable[..., Component]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find component spec at {path}. Path does not exist.")

        if not path.is_file():
            path = path / "component_spec.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Could not find component spec at {path}")

        if version == "local":
            return load_component(source=path)
        else:
            with path.open() as f:
                spec = safe_load(f)
            name = spec["name"]
            return self.load_by_name(name=name, version=version)

    def _load_from_remote_component_spec(self, url: str, version: str = "local") -> Callable[..., Component]:
        raise NotImplementedError("Loading components from remote URLs is not yet supported")

    def load_by_name(self, name: str, version: str) -> Callable[..., Component]:
        if self.client is None:
            raise ValueError("Cannot load component by name without a client")
        if version == "local":
            raise ValueError("Cannot load component by name with version 'local'")
        if version == "latest":
            version = get_latest_version(self.client, name=name)
        if version == "default":
            return self.client.components.get(name)
        return self.client.components.get(name=name, version=version)


class ExperimentBase:
    def __init__(self, workspace: WorkspaceConfig):
        self.workspace = workspace

        # initialize AML logging
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
        logging.getLogger("azure.identity._credentials.chained").setLevel(logging.WARNING)
        logging.getLogger("azure.identity").setLevel(logging.WARNING)

    def submit(self, display_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> PipelineJob:
        self.validate()
        job: PipelineJob = self.pipeline(**self.pipeline_parameters)

        if display_name is not None:
            job.display_name = display_name
        elif self.job_name is not None:
            job.display_name = self.job_name
        else:
            job.display_name = HydraConfig.get().job.config_name

        if tags is None:
            tags = dict()
        for group_name, param_group in self.pipeline_parameters.items():
            for param_name, param_value in param_group.__dict__.items():
                tags[f"{group_name}.{param_name}"] = str(param_value)

        logger.info("Submitting job...")
        if not isinstance(self.default_compute, ClusterComputeConfig):
            raise ValueError("default_compute must be an instance of ClusterComputeConfig")
        job.settings.default_compute = self.default_compute.cluster_name
        submitted_job = self.workspace.ml_client.jobs.create_or_update(
            job, experiment_name=self.experiment_name, tags=tags, skip_validation=False
        )
        return submitted_job

    @property
    def experiment_name(self) -> Optional[str]:
        """Name of the experiment. This will be shown in AML studio"""
        return None

    @property
    def job_name(self) -> Optional[str]:
        """Name of the job. This will be shown in AML studio"""
        return None

    @property
    def default_compute(self) -> str:
        """Default compute target for components if not otherwise specified"""
        raise NotImplementedError(f"Must implement {self.__class__.__name__}.default_compute")

    @property
    def pipeline_parameters(self) -> Dict:
        """Dictionary of parameter name and value that will be passed to the pipeline"""
        raise NotImplementedError(f"Must implement {self.__class__.__name__}.pipeline_parameters")

    @property
    def pipeline(self) -> Callable[..., PipelineJob]:
        """Returns a `azure.ml.component.dsl.pipeline` decorated function that represents the pipeline

        See https://componentsdk.azurewebsites.net/concepts/pipeline.html for more information 
        """
        raise NotImplementedError(f"Must implement {self.__class__.__name__}.pipeline")

    def validate(self):
        pipeline_parameters_dict = dict(**self.pipeline_parameters)
        self.pipeline(**pipeline_parameters_dict)
        logger.info("Pipeline validated")

    @classmethod
    def main(cls, config_path):
        """
        Runs a machine learning experiment pipeline with a flexible and hierarchical configuration system.

        Special arguments in config:
            submit: bool = False  If true, submit the pipeline, otherwise validate it

        Special arguments can be passed to the pipeline by either adding to the config file or by passing them on the command
        line using Hydra syntax. For example, to submit the pipeline to AML, you can run:

        python my_experiment.py +submit=true

        Args:
            config_path: Path to the config directory
        """
        @hydra.main(config_path=os.path.abspath(config_path), version_base="1.2")
        def hydra_run(config: OmegaConf):
            type_hints = get_type_hints(cls.__init__)
            config = OmegaConf.to_container(config, resolve=True)

            submit = config.pop("submit", False)
            name = config.pop("name", None)

            if "return" in type_hints:
                type_hints.pop("return")
            if set(type_hints.keys()) != set(config.keys()):
                missing_keys_in_config = set(type_hints.keys()) - set(config.keys())
                if len(missing_keys_in_config) > 0:
                    raise ValueError(f"The config YAML has missing keys that are required by {cls.__name__}. "
                                     f"Missing keys: {missing_keys_in_config}")
                extra_keys_in_config = set(config.keys()) - set(type_hints.keys())
                if len(extra_keys_in_config) > 0:
                    raise ValueError(f"The config YAML has extra keys that are not required by {cls.__name__}. "
                                     f"Extra keys: {extra_keys_in_config}")

            config_dc = {
                config_name: dictconfig_to_dataclass(config[config_name], type_hints[config_name])
                for config_name
                in config
            }
            e = cls(**config_dc)

            if submit:
                job = e.submit(display_name=name)
                logger.info(f"Job submitted. AML URL: {job.studio_url}")
            else:
                e.validate()
        hydra_run()


class DatastoreURI(str):
    def __new__ (cls, uri: str):
        pattern = re.compile(
            r"azureml://subscriptions/([^\/]+)/resourcegroups/([^\/]+)/"
            r"(?:Microsoft.MachineLearningServices/)?workspaces/([^\/]+)/datastores/([^\/]+)/paths/(.*)"
        )
        if not pattern.match(uri):
            raise ValueError(f"URI {uri} does not match pattern {pattern}")
        return str.__new__(cls, uri)

    @classmethod
    def from_datastore_uri(cls, uri: str, workspace: WorkspaceConfig) -> "DatastoreURI":
        if "subscriptions" in uri:
            return cls(uri)
        workspace_path = f"azureml://subscriptions/{workspace.subscription_id}/resourcegroups/{workspace.resource_group}/" \
                         f"workspaces/{workspace.workspace_name}/"
        uri = uri.replace("azureml://", workspace_path)
        return cls(uri)

    @classmethod
    def from_asset_uri(cls, uri: str, workspace: WorkspaceConfig) -> "DatastoreURI":
        pattern = re.compile(r"/data/([^/]+)")
        match = re.search(pattern, uri)
        if match:
            data = match.group(1)
        else:
            raise ValueError(f"URI {uri} does not match pattern {pattern}. Verify that the URI is a valid Asset URI.")

        pattern = re.compile(r"/versions/([^/]+)")
        match = re.search(pattern, uri)
        if match:
            version = match.group(1)

        asset = workspace.ml_client.data.get(name=data, version=version)
        return cls(asset.path)

    def download_content(self, path: Union[str, Path], match_pattern: str = "*") -> Path:
        try:
            from azureml.fsspec import AzureMachineLearningFileSystem
        except ImportError as e:
            raise ImportError("Please install azureml-fsspec to use this function") from e
        fs = AzureMachineLearningFileSystem(self)
        if fs.isfile(self):
            fs.get(rpath=self, lpath=path)
            return Path(path) / os.path.basename(self)
        else:
            local_path = str(path) + os.path.sep
            files = fs.ls()
            files = fnmatch.filter(files, match_pattern)
            for f in files:
                if fs.isdir(f):
                    local_dir = os.path.join(local_path, os.path.basename(os.path.dirname(str(f) + os.path.sep)))
                    fs.get(rpath=f, lpath=local_dir, recursive=True)
                else:
                    fs.get(rpath=f, lpath=local_path, recursive=False)
            return Path(path)


class Job:
    def __init__(
        self, aml_job, workspace: WorkspaceConfig, local_name: Optional[str] = None, add_tags: Optional[Dict[str, str]] = None
    ):
        self.aml_job = aml_job
        self.ws = workspace

        self.aml_run = self.ws.workspace.get_run(aml_job.name)
        if add_tags is None:
            add_tags = dict()
        self.details = self.aml_run.get_details()
        self.local_name = local_name

        existing_tags = self.aml_run.get_tags()
        duplicate_keys = set(existing_tags.keys()).intersection(add_tags.keys())
        if len(duplicate_keys) > 0:
            raise ValueError(f"Tags {duplicate_keys} already exist in the job. Cannot add tags with duplicate keys.")
        self.aml_run.set_tags({**existing_tags, **add_tags})

    @classmethod
    def from_url(cls, url: str, local_name: Optional[str] = None, add_tags: Optional[Dict[str, str]] = None) -> "Job":
        parsed_url = urlparse(url)

        query_dict = parse_qs(parsed_url.query)
        wsid_value = query_dict['wsid'][0]
        wsid_segments = wsid_value.strip('/').split('/')
        ws_config = dict((k.lower(), v.lower()) for k, v in zip(wsid_segments[::2], wsid_segments[1::2]))
        ws = WorkspaceConfig(
            subscription_id=ws_config['subscriptions'],
            resource_group=ws_config['resourcegroups'],
            workspace_name=ws_config['workspaces'],
        )

        path_segments = parsed_url.path.strip('/').split('/')
        run_value_index = path_segments.index('runs') + 1
        run_id = path_segments[run_value_index]

        aml_job = ws.ml_client.jobs.get(run_id)

        node_path, = query_dict.get('nsq', [None])
        if node_path is not None:
            node_path = node_path.replace("nodePath == ", "").replace("\\/", "/")
            node_path_ls = [ p for p in node_path.split("/") if p ]
            job = Job(aml_job=aml_job, workspace=ws)
            while len(node_path_ls) > 0:
                job = job.get_node(node_path_ls.pop(0))
            aml_job = job.aml_job
        return Job(aml_job=aml_job, workspace=ws, local_name=local_name, add_tags=add_tags)

    @classmethod
    def from_id(cls, workspace: WorkspaceConfig, run_id: str, local_name: Optional[str] = None) -> "Job":
        aml_job = workspace.ml_client.jobs.get(run_id)
        return Job(aml_job=aml_job, local_name=local_name)

    def get_node(self, name: str) -> "Job":
        children = self.ws.ml_client.jobs.list(parent_job_name=self.aml_job.name)
        node = None
        for c in children:
            if c.display_name == name:
                node = c
                break
        if node is None:
            raise ValueError(
                f"Node {name} not found in job {self.aml_job.name}. "
                f"Available nodes: {', '.join([c.display_name for c in children])}"
            )
        if node.properties["StepType"] == "SubGraphCloudStep":
            children = list(self.ws.ml_client.jobs.list(parent_job_name=node.name))
            assert len(children) == 1
            node = self.ws.ml_client.jobs.get(children[0].name)
        return Job(aml_job=node, workspace=self.ws)

    def download_input(self, name: str, path: str, match_pattern: str = "*") -> Path:
        input_details = self.details['runDefinition']['inputAssets'][name]
        uri = DatastoreURI.from_asset_uri(uri=input_details["asset"]["assetId"],
                                          workspace=self.ws)
        local_path = uri.download_content(path=path, match_pattern=match_pattern)
        return local_path

    def download_output(self, name: str, path: str, match_pattern: str = "*") -> Path:
        run_id = self.aml_run.id
        if self.aml_job.properties.get("azureml.isreused", False):
            run_id = self.aml_job.properties["azureml.reusedrunid"]
        uri_path = self.details['runDefinition']['outputData'][name]["outputLocation"]["uri"]["path"]
        uri_path = uri_path.replace("${{name}}", run_id)
        uri = DatastoreURI.from_datastore_uri(uri=uri_path, workspace=self.ws)
        local_path = uri.download_content(path=path, match_pattern=match_pattern)
        return local_path

    def resubmit(self):
        breakpoint()
        return self.ws.ml_client.jobs.create_or_update(self.aml_job) 

    def get_command(self, input_paths: Optional[Dict[str, Path]] = None, output_path: Optional[Path] = None) -> str:
        """
        Get the command that was run in the job. This is useful for debugging.
        """
        rd = self.details["runDefinition"]
        if rd.get("command", None) is not None:
            cmd = shlex.split(rd["command"])
        else:
            cmd = ["python", rd["script"]] + rd["arguments"]
        if input_paths is None:
            input_paths = {}
        env = os.environ.copy()
        os.environ = rd["environmentVariables"]
        data_paths = {"AZUREML_DATAREFERENCE_"+k: str(v) for k, v in input_paths.items()}
        data_paths.update({"AZURE_ML_INPUT"+k: str(v) for k, v in input_paths.items()})
        os.environ.update(data_paths)
        cmd = [os.path.expandvars(c) for c in cmd]
        if output_path is not None:
            cmd = [c.replace("DatasetOutputConfig:", output_path + "/") for c in cmd]
        cmd = [c.replace("\\\n", " ").replace("\n", " ") for c in cmd]
        os.environ = env
        return " ".join(cmd)

    @property
    def simple_status(self) -> str:
        status = self.details["status"]
        if status in {"NotStarted", "Starting", "Provisioning", "Queued", "Preparing"}:
            return "Pending"
        if status in {"Running", "Finalizing"}:
            return "Running"
        if status in {"Completed"}:
            return "Completed"
        if status in {"CancelRequested", "Canceled"}:
            return "Canceled"
        if status in {"Failed"}:
            return "Failed"
        raise NotImplementedError(f"Status {status} not implemented")

    @property
    def inputs(self) -> List[str]:
        return list(self.details["runDefinition"]["inputAssets"].keys())

    @property
    def parameters(self) -> Dict[str, Any]:
        properties = self.details['properties']
        if 'azureml.parameters' in properties:
            return json.loads(properties['azureml.parameters'])
        else:
            return {}

    @property
    def environment_variables(self) -> Dict[str, Any]:
        return self.details["runDefinition"]["environmentVariables"]

    @property
    def input_parameters(self) -> Dict[str, Any]:
        env_vars = self.details["runDefinition"]["environmentVariables"]
        input_params = {
            k.replace("AZUREML_PARAMETER_", ""): v for k, v in env_vars.items() if k.startswith("AZUREML_PARAMETER_")
        }
        return input_params

    @property
    def experiment_name(self) -> str:
        return self.aml_run.experiment.name

    @property
    def job_name(self) -> str:
        return self.aml_run.display_name

    @property
    def description(self) -> str:
        return self.aml_run.description

    @property
    def tags(self) -> Dict[str, str]:
        return self.aml_run.get_tags()

    def get_metrics(self) -> Dict[str, Any]:
        return self.aml_run.get_metrics()

    @property
    def url(self) -> str:
        return self.aml_run.get_portal_url()

    def __repr__(self) -> str:
        return f"Job({self.experiment_name}/{self.job_name})"

    def as_row(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if columns is None:
            columns = ["local_name", "experiment_name", "job_name", "simple_status"] + \
                      list(self.parameters.keys()) + ["url", "description"]
        row = {}
        for c in columns:
            if c == "local_name" and self.local_name is not None:
                row[c] = self.local_name
            elif hasattr(self, c):
                row[c] = getattr(self, c)
            elif c in self.parameters:
                row[c] = self.parameters[c]
            elif c in self.tags:
                row[c] = self.tags[c]
            else:
                raise ValueError(
                    f"Column {c} not found. Available columns: {list(self.parameters.keys()) + list(self.tags.keys())}"
                )
        return pd.DataFrame([row])


YELLOW = "#ffeeba"
GREEN = "#d4edda"
RED = "#f5c6cb"
GREY = "#e2e3e5"


class JobList:
    def __init__(self, jobs: Iterable[Job]):
        self.jobs = list(jobs)

    def as_table(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        table = pd.concat((j.as_row(columns) for j in self.jobs), ignore_index=True, join="outer")
        return table

    def as_pretty_table(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        table = self.as_table(columns)

        def make_clickable(link):
            return f'<a target="_blank" href="{link}">AML</a>'

        styled_table = table.style

        if "url" in table.columns:
            styled_table = styled_table.format({'url': make_clickable})

        def status_colour(row):
            return [{
                "Pending": f"background-color: {YELLOW}",
                "Running": f"background-color: {YELLOW}",
                "Completed": f"background-color: {GREEN}",
                "Canceled": f"background-color: {GREY}",
                "Failed": f"background-color: {RED}",
            }[row["simple_status"]]]*len(row)

        if "simple_status" in table.columns:
            styled_table = styled_table.apply(status_colour, axis=1) 

        return styled_table

    @classmethod
    def from_table_with_urls(cls, table: pd.DataFrame, url_column: Optional[str] = "AML",
                             add_tags_from_columns: Optional[List[str]] = None) -> "JobList":
        """
        Creates a JobList object from a pandas DataFrame.
        """
        if add_tags_from_columns is None:
            tags = None
        else:
            tags = {c: table[c] for c in add_tags_from_columns}
        return cls.from_urls(table[url_column], add_tags=tags)

    @classmethod
    def from_table_with_ids(cls, table: pd.DataFrame, id_column: str, workspace: WorkspaceConfig,
                            filter_by_column: Optional[str] = None) -> "JobList":
        if filter_by_column is not None:
            table = table[bool(table[filter_by_column])]
        return cls.from_ids(workspace=workspace, ids=table[id_column])

    @classmethod
    def from_urls(cls, urls: Union[List[str], OrderedDict[str, str]], disable_pbar: Optional[bool] = True,
                  add_tags: Optional[Dict[str, List[str]]] = None) -> "JobList":
        """
        Creates a JobList object from a list of job URLs.

        Args:
            urls (Union[List[str], OrderedDict[str, str]]): A list of job URLs or a list of tuples containing the job name and
                                                            URL.

        Returns:
            JobList: A JobList object containing the jobs specified in the list of URLs.
        """
        if add_tags is not None:
            tags = [{k: v} for j in range(len(urls)) for i, (k, v) in enumerate(add_tags.items()) if i == j]
        else:
            tags = [{} for _ in range(len(urls))]

        if isinstance(urls, Mapping):
            jobs = [
                Job.from_url(url=url, local_name=name, add_tags=tags)
                for (name, url), tags
                in tqdm(zip(urls.items(), tags), disable=disable_pbar)
            ]
        else:
            jobs = [Job.from_url(url=url, add_tags=tag) for url, tag in tqdm(zip(urls, tags), disable=disable_pbar)]
        return cls(jobs=list(jobs))

    @classmethod
    def from_ids(cls, workspace: WorkspaceConfig, ids: Union[List[str], OrderedDict[str, str]]) -> "JobList":
        """
        Creates a JobList object from a list of job IDs.

        Args:
            workspace (WorkspaceConfig): The workspace configuration to use when retrieving jobs by ID.
            ids (List[str]): A list of job IDs.

        Returns:
            JobList: A JobList object containing the jobs specified in the list of IDs.
        """
        if isinstance(ids, Mapping):
            jobs = pmap(Job.from_id, [workspace]*len(ids), ids.values(), ids.keys())
        else:
            jobs = pmap(Job.from_id, [workspace]*len(ids), ids)
        return cls(jobs=list(jobs))

    def filter(self, property_name: str, property_value: Any) -> "JobList":
        """
        Filters the JobList by a given property name and value.

        Args:
            property_name (str): The name of the property to filter by.
            property_value (Any): The value of the property to filter by.

        Returns:
            JobList: A new JobList object containing only the jobs that match the given property name and value.
        """
        return JobList(j for j in self.jobs if getattr(j, property_name) == property_value)

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, index: int) -> Job:
        return self.jobs[index]
