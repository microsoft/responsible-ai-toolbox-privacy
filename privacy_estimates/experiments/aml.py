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
from azure.ai.ml.entities import Component, PipelineJob
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.core.exceptions import ClientAuthenticationError
from dataclasses import dataclass
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import TypeVar, Type, Any, get_type_hints, Dict, Union, List
from dataclasses import is_dataclass
from yaml import safe_load
from tqdm import tqdm
from typing import Optional, Callable, Optional, Iterable

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceConfig:
    workspace_name: str
    resource_group: str
    subscription_id: str
    cpu_compute: Optional[str] = None
    gpu_compute: Optional[str] = None
    large_memory_cpu_compute: Optional[str] = None

    def __post_init__(self) -> None:
        self.ml_client = self._get_ml_client()
        if self.large_memory_cpu_compute is None:
            self.large_memory_cpu_compute = self.cpu_compute

    def _get_ml_client(self) -> MLClient:
        try:
            credential = DefaultAzureCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except ClientAuthenticationError:
            # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
            credential = InteractiveBrowserCredential()
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
        )


T = TypeVar("T")
def dictconfig_to_dataclass(cfg: DictConfig, dataclass_type: Type[T]) -> T:
    def convert_nested(cfg_obj: Any, dataclass_obj: Type[T]) -> T:
        if is_dataclass(dataclass_obj):
            fields = {field.name: field.type for field in dataclass_obj.__dataclass_fields__.values()}
            mutable_dict = OmegaConf.to_container(cfg_obj, resolve=True) if isinstance(cfg_obj, DictConfig) else cfg_obj
            for key, value in fields.items():
                if key in mutable_dict:
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


class AMLComponentLoader:
    def __init__(self, workspace: WorkspaceConfig):
        self.workspace = workspace
        self.override_version = None

    def load_from_component_spec(self, path: Path, version: str = "local") -> Callable[..., Component]:
        version = self.override_version or version

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
            return load_component(client=self.workspace.ml_client, name=name, version=version)
        
    def load_by_name(self, name: str, version: str) -> Callable[..., Component]:
        return load_component(client=self.workspace.ml_client, name=name, version=version)


class ExperimentBase:
    def __init__(self, workspace: WorkspaceConfig):
        self.aml_component_loader = AMLComponentLoader(workspace=workspace)
        self.workspace=workspace

        # initialize AML logging
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
        logging.getLogger("azure.identity._credentials.chained").setLevel(logging.WARNING)

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
        submitted_job = self.workspace.ml_client.jobs.create_or_update(
            job, compute=self.default_compute, experiment_name=self.experiment_name, tags=tags,
            skip_validation=False
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
        job: PipelineJob = self.pipeline(**pipeline_parameters_dict)
        logger.info("Pipeline validated")

    def load_component_from_spec(self, path: Path, version: str = "local") -> Callable[..., Component]:
        return self.aml_component_loader.load_from_component_spec(path, version)
    
    def load_component_by_name(self, name: str, version: str) -> Callable[..., Component]:
        return self.aml_component_loader.load_by_name(name, version)
    
    def set_compute_target(self, component: Component, target: str):
        component.runsettings.configure(target=target)
    
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

            type_hints.pop("return")
            if set(type_hints.keys()) != set(config.keys()):
                missing_keys_in_config = set(type_hints.keys()) - set(config.keys())
                if len(missing_keys_in_config) > 0:
                    raise ValueError(f"The config YAML has missing keys that are required by {cls.__name__}. Missing keys: {missing_keys_in_config}")
                extra_keys_in_config = set(config.keys()) - set(type_hints.keys())
                if len(extra_keys_in_config) > 0:
                    raise ValueError(f"The config YAML has extra keys that are not required by {cls.__name__}. Extra keys: {extra_keys_in_config}")

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
            r"azureml://subscriptions/([^\/]+)/resourcegroups/([^\/]+)/(?:Microsoft.MachineLearningServices/)?workspaces/([^\/]+)/datastores/([^\/]+)/paths/(.*)"
        )
        if not pattern.match(uri):
            raise ValueError(f"URI {uri} does not match pattern {pattern}")
        return str.__new__(cls, uri)
    
    @classmethod
    def from_asset_uri(cls, uri: str, workspace: WorkspaceConfig) -> "DatastoreURI":
        pattern = re.compile(r"/data/([^/]+)")
        match = re.search(pattern, uri)
        if match:
            data = match.group(1)

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
    def __init__(self, aml_run):
        from azureml.core import Run
        self.aml_run: Run = aml_run
        self.details = aml_run.get_details()
        self.ws = WorkspaceConfig(
            workspace_name=aml_run.experiment.workspace.name,
            resource_group=aml_run.experiment.workspace.resource_group,
            subscription_id=aml_run.experiment.workspace.subscription_id,
        )

    @classmethod
    def from_url(cls, url: str) -> "Job":
        from azureml.core import Run
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

        run = Run.get(ws.workspace, run_id=run_id)

        return Job(aml_run=run)
    
    @classmethod
    def from_id(cls, workspace: WorkspaceConfig, run_id: str) -> "Job":
        from azureml.core import Run
        run = Run.get(workspace.workspace, run_id=run_id)
        return Job(aml_run=run)

    def get_node(self, name: str) -> "Job":
        children = self.aml_run.get_children()
        node = next(filter(lambda x: x.display_name == name, children))
        return Job(aml_run=node)

    def download_input(self, name: str, path: str, match_pattern: str = "*") -> Path:
        input_details = self.details['runDefinition']['inputAssets'][name]
        uri = DatastoreURI.from_asset_uri(uri=input_details["asset"]["assetId"],
                                          workspace=self.ws)
        local_path = uri.download_content(path=path, match_pattern=match_pattern)
        return local_path

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
        cmd = [c.replace("\\\n", " ") for c in cmd]
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
            columns = ["experiment_name", "job_name", "simple_status"] + list(self.parameters.keys()) + ["url", "description"]
        row = {}
        for c in columns:
            if hasattr(self, c):
                row[c] = getattr(self, c)
            elif c in self.parameters:
                row[c] = self.parameters[c]
            elif c in self.tags:
                row[c] = self.tags[c]
            else:
                raise ValueError(f"Column {c} not found. Available columns: {list(self.parameters.keys()) + list(self.tags.keys())}")
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
                             filter_by_column: Optional[str] = None) -> "JobList":
        """
        Creates a JobList object from a pandas DataFrame.
        """
        if filter_by_column is not None:
            table = table[bool(table[filter_by_column])]
        return cls.from_urls(table[url_column])
    
    @classmethod
    def from_table_with_ids(cls, table: pd.DataFrame, id_column: str, workspace: WorkspaceConfig,
                            filter_by_column: Optional[str] = None) -> "JobList":
        if filter_by_column is not None:
            table = table[bool(table[filter_by_column])]
        return cls.from_ids(workspace=workspace, ids=table[id_column])

    @classmethod
    def from_urls(cls, urls: List[str], disable_pbar: Optional[bool] = True) -> "JobList":
        """
        Creates a JobList object from a list of job URLs.

        Args:
            urls (List[str]): A list of job URLs.

        Returns:
            JobList: A JobList object containing the jobs specified in the list of URLs.
        """
        return cls(jobs=list(tqdm((Job.from_url(url) for url in urls), disable=disable_pbar, total=len(urls))))
    
    @classmethod
    def from_ids(cls, workspace: WorkspaceConfig, ids: List[str]) -> "JobList":
        """
        Creates a JobList object from a list of job IDs.

        Args:
            workspace (WorkspaceConfig): The workspace configuration to use when retrieving jobs by ID.
            ids (List[str]): A list of job IDs.

        Returns:
            JobList: A JobList object containing the jobs specified in the list of IDs.
        """
        return cls(jobs=[Job.from_id(workspace=workspace, run_id=id) for id in ids])
    
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
