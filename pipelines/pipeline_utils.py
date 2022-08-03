from pydantic import BaseModel, Field
from pydantic_cli import DefaultConfig
from typing import Optional, Callable
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication, ServicePrincipalAuthentication
from azure.ml.component import Component
from pathlib import Path
from ruamel import yaml


REPO_ROOT = Path(__file__).parent.parent
BIN = REPO_ROOT / "bin"
COMPONENT_DIR = REPO_ROOT / "components"


class PipelineArguments(BaseModel):
    class Config(DefaultConfig):
        CLI_JSON_ENABLE = True
 
    workspace_config: Optional[Path] = Field(default=None, description="Path to the AML workspace configuration file")
    run_name: Optional[str] = Field(default=None, description="AML experiment name")

    @property
    def workspace(self) -> Workspace:
        auth=AzureCliAuthentication()
        return Workspace.from_config(self.workspace_config, auth=auth)

    @property
    def gpu_cluster(self) -> Optional[str]:
        with open(self.workspace_config) as f:
            data = yaml.safe_load(f)
        target = data["gpu_cluster"]
        if isinstance(target, str):
            return target
        else:
            return target["name"]

    @property
    def cpu_cluster(self) -> Optional[str]:
        with open(self.workspace_config) as f:
            data = yaml.safe_load(f)
        target = data["cpu_cluster"]
        if isinstance(target, str):
            return target
        else:
            return target["name"]

    def set_compute_settings(self, component: Component, compute_target: str) -> Component:
        """
        Set the compute settings for a component.
        """
        with open(self.workspace_config) as f:
            data = yaml.safe_load(f)

        target = data[compute_target]

        if isinstance(target, str):
            component.runsettings.target = target
        elif isinstance(target, dict):
            if target.get("type", None) == "amlk8s":
                component.runsettings.target = target["name"]
                component.k8srunsettings.resource_configuration.gpu_count = target.get("gpu_count", 0)
            else:
                raise ValueError(f"Unsupported compute target {compute_target}")
        else:
            raise ValueError(f"Unsupported compute target {compute_target}")
        return component

    @property
    def datastore(self) -> Optional[str]:
        with open(self.workspace_config) as f:
            data = yaml.safe_load(f)
        return data.get("datastore", None)



def get_component_factory(yaml_file: Path, version: str = "local", workspace: Workspace = None) -> Callable[..., Component]:
    """
    Load an AML component factory.

    The `version` string can either be "local" which will load the component from the local repository and submit it to the
    workspace. This is useful for debugging or development to quickly iterate on a component without polluting the component
    repository. Alternatively, you can specify a version string which will load the component from the workspace. This is
    useful for running experiments and helps reproducibility and reusing past cached outputs

    :param Path yaml_file: Path to the component specification YAML file. Typically in `components/<component_name>/component_spec.yaml`
    :param str version: Either "local" or a version string. (See above)
    :param Workspace workspace: Optional workspace to use. If not specified, the workspace version needs to be "local"
    :return: The component factory that can instantiate a component given values for its parameters.
    """
    if version == "local":
        return Component.from_yaml(workspace=workspace, yaml_file=yaml_file)
    else:
        with open(yaml_file) as f:
            spec = yaml.safe_load(f)
        return Component.load(workspace=workspace, name=spec["name"], version=version)

