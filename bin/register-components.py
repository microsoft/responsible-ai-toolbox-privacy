import json
from pydantic import BaseModel, Field
from pydantic_cli import run_and_exit
from pathlib import Path
from typing import List, Dict, Optional
from subprocess import check_call, DEVNULL, CalledProcessError
from ruamel import yaml


REPO_ROOT = Path(__file__).parent.parent
COMPONENTS_DIR = REPO_ROOT / "components"

class Arguments(BaseModel):
    workspace_config: Path
    component_yaml: Optional[Path] = Field(default=None)


def is_component_spec(file_path: Path) -> bool:
    with file_path.open() as f:
        content = yaml.safe_load(f)
    return content.get("type", None) in ["CommandComponent"]


def get_all_component_yaml_files(component_dir: str = COMPONENTS_DIR) -> List[Path]:
    files = [
        f.absolute() for f in Path(component_dir).glob("**/*.yaml")
    ]
    files = filter(is_component_spec, files)
    return list(files)


def get_version() -> str:
    with (REPO_ROOT / "VERSION").open() as f:
        return f.read().strip()


def read_workspace_config(workspace_config: Path = REPO_ROOT / "config.json") -> Dict[str, str]:
    with workspace_config.open() as f:
        d = json.load(f)
    assert {"workspace_name", "subscription_id", "resource_group"}.issubset(d.keys())
    return d


def check_component_cli() -> None:
    help_url = "https://componentsdk.azurewebsites.net/getting_started.html#install-component-cli"
    try:
        check_call(["az", "ml", "component", "build", "--help"], stdout=DEVNULL, stderr=DEVNULL)
    except CalledProcessError:
        raise EnvironmentError(f"az ml component build is not installed. Check {help_url} for more details")


def main(args: Arguments):
    check_component_cli()

    workspace_config = read_workspace_config(args.workspace_config)

    cmd_base = [
        "az", "ml", "component", "create",
        "--version", get_version(),
        "--subscription-id", workspace_config["subscription_id"],
        "--resource-group", workspace_config["resource_group"],
        "--workspace-name", workspace_config["workspace_name"],
        "--fail-if-exists"
    ]

    if args.component_yaml is None:
        component_yamls = get_all_component_yaml_files()
    else:
        component_yamls = [args.component_yaml]


    for component_yaml in component_yamls:
        print(f"Registering {component_yaml}...")
        if not component_yaml.is_file():
            raise FileNotFoundError(f"{component_yaml} does not exist or is not a file")
        cmd = cmd_base + ["--file", str(component_yaml)]
        check_call(cmd)


if __name__ == "__main__":
    run_and_exit(Arguments, main)
