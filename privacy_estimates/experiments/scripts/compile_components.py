import shutil
from argparse_dataclass import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from mldesigner import compile
from tqdm import tqdm
from typing import Optional, Callable, List
from tempfile import TemporaryDirectory
from yaml import safe_load as yaml_load
from yaml import safe_dump as yaml_dump
from importlib.metadata import version


from privacy_estimates.experiments.components import (
    create_in_out_data_for_shadow_artifact_statistics, prepare_data,
    prepare_data_for_aml_parallel, filter_aux_data, reinsert_aux_data, filter_aux_data_aml_parallel,
    reinsert_aux_data_aml_parallel, append_column_constant_int, append_column_constant_str, append_column_incrementing,
    select_columns, rename_columns, postprocess_dpd_data,
    create_in_out_data_for_membership_inference_challenge, random_split_dataset, create_challenge_bits_aml_parallel,
    create_artifact_indices_for_aml_parallel, compute_shadow_artifact_statistics, select_cross_validation_challenge_points,
    append_artifact_index_column_aml_parallel, move_dataset, select_top_k_rows, create_empty_dataset, 
    convert_in_out_to_challenge, compute_mi_signals, convert_chat_jsonl_to_hfd, convert_hfd_to_jsonl, generate_canaries_with_secrets, convert_uri_file_to_int,
    aggregate_2_output_dirs, aggregate_2_output_files, aggregate_16_output_dirs, aggregate_16_output_files
)


DESCRIPTION = """
Compile components for the privacy estimates experiments pipeline.

This script compiles all components for the privacy estimates experiments pipeline.
The compiled components are stored in the output directory.
The output directory will contain a directory for each component, which contains the compiled component code and a component spec file.
The component spec file is a YAML file that describes the component and its dependencies.
"""


EXPERIMENT_DIR = Path(__file__).parent.parent


PY_COMPONENTS = [
    aggregate_2_output_dirs,
    aggregate_2_output_files,
    aggregate_16_output_dirs,
    aggregate_16_output_files,
    create_in_out_data_for_shadow_artifact_statistics,
    prepare_data,
    prepare_data_for_aml_parallel,
    filter_aux_data,
    reinsert_aux_data,
    filter_aux_data_aml_parallel,
    reinsert_aux_data_aml_parallel,
    append_column_constant_int,
    append_column_constant_str,
    append_column_incrementing,
    select_columns,
    rename_columns,
    postprocess_dpd_data,
    create_in_out_data_for_membership_inference_challenge,
    random_split_dataset,
    create_challenge_bits_aml_parallel,
    create_artifact_indices_for_aml_parallel,
    compute_shadow_artifact_statistics,
    append_artifact_index_column_aml_parallel,
    move_dataset,
    select_top_k_rows,
    create_empty_dataset,
    convert_in_out_to_challenge,
    convert_chat_jsonl_to_hfd,
    convert_hfd_to_jsonl,
    generate_canaries_with_secrets,
    convert_uri_file_to_int,
]


YAML_COMPONENTS = [
    EXPERIMENT_DIR / "attacks" / "lira" / "component_spec.yaml",
    EXPERIMENT_DIR / "attacks" / "rmia" / "component_spec.yaml",
    EXPERIMENT_DIR / "scorers" / "dataset" / "component_spec.yaml",
    EXPERIMENT_DIR / "components" / "compute_privacy_estimates" / "component_spec.yaml",
]


@dataclass
class Arguments:
    output_dir: Path = field(metadata={"help": "Output directory for compiled components"})
    version: str = field(default=version("privacy_estimates"), metadata={"help": "Version to override the component versions with"})
    disable_tqdm: bool = field(default=False, metadata={"help": "Disable tqdm progress bars"})


def compile_py_component(component: Callable, output_dir: Path, override_version: Optional[str] = None) -> Path:
    with TemporaryDirectory() as temp_dir:
        try:
            compile(source=component, output=temp_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to compile component {component.__name__}") from e
        
        component_name = component.component.name

        if not component_name.startswith("privacy_estimates__"):
            raise ValueError(f"Component name {component_name} does not start with 'privacy_estimates__'")

        component_spec_path = Path(temp_dir) / component_name / (component_name + ".yaml")
        with component_spec_path.open("r") as f:
            component_spec = yaml_load(f)
        component_spec_path.unlink()

        if override_version is not None:
            component_spec["version"] = override_version

        # add '_component_spec' suffix to the path
        component_spec_path = Path(temp_dir) / component_name / (component_name + "_component_spec.yaml")
        with component_spec_path.open("w") as f:
            yaml_dump(component_spec, f)
        rel_component_spec_path = component_spec_path.relative_to(temp_dir)
        
        # copy the compiled component to the output directory
        compiled_component_path = output_dir / rel_component_spec_path.parent.parent
        compiled_component_path.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp_dir, compiled_component_path, dirs_exist_ok=True)
    return rel_component_spec_path


def compile_yaml_component(component: Path, output_dir: Path, override_version: Optional[str] = None) -> Path:
    with component.open("r") as f:
        component_spec = yaml_load(f)

    if not component.name.endswith("spec.yaml"):
        raise ValueError(f"Component spec file {component} does not end with 'spec.yaml' and may not be picked up by the build pipeline")

    component_name = component_spec["name"]
    if not component_name.startswith("privacy_estimates__"):
        raise ValueError(f"Component name {component_name} does not start with 'privacy_estimates__'")

    if override_version is not None:
        component_spec["version"] = override_version

    component_output_dir = output_dir / component.parent.name
    shutil.copytree(component.parent / component_spec["code"], component_output_dir, dirs_exist_ok=True)

    additional_includes = [component.parent / s for s in component_spec.get("additional_includes", [])]
    for s in additional_includes:
        # might be a file or a dir
        if s.is_dir():
            shutil.copytree(s, component_output_dir / s.name, dirs_exist_ok=True)
        else:
            shutil.copy(s, component_output_dir / s.name)

    component_spec["code"] = "."
    component_spec["additional_includes"] = []

    compiled_component_spec_path = component_output_dir / component.name

    with compiled_component_spec_path.open("w") as f:
        yaml_dump(component_spec, f)
    
    return compiled_component_spec_path


def main(args: Arguments):
    component_paths = [
        compile_yaml_component(c, args.output_dir, override_version=args.version)
        for c in tqdm(YAML_COMPONENTS, disable=args.disable_tqdm, desc="Compiling YAML components")
    ]

    component_paths += [
        compile_py_component(c, args.output_dir, override_version=args.version)
        for c in tqdm(PY_COMPONENTS, disable=args.disable_tqdm, desc="Compiling Python components")
    ]

    with open(args.output_dir / "component_paths.yaml", "w") as f:
        yaml_dump({"component_paths": [str(p) for p in component_paths]}, f)


def run_main():
    parser = ArgumentParser(Arguments, description=DESCRIPTION)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run_main()
