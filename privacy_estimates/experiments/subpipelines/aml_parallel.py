import os
import shutil
from pathlib import Path
from tempfile import mkdtemp
from shutil import copytree, copy2
from typing import List, Optional
from azure.ai.ml import Input, Output
from azure.ai.ml.parallel import parallel_run_function, RunFunction

from privacy_estimates.experiments.loaders import ComponentLoader
from privacy_estimates.experiments.components import aml_parallel


class AMLParallelLoader(ComponentLoader):
    def __init__(self, component_loader: ComponentLoader, num_concurrent_jobs_per_node: int, outputs_folder: List[str],
                 outputs_file: List[str], inputs_to_distribute: List[str],
                 inputs_to_distribute_on_command_line: Optional[List[str]] = None, num_nodes: int = 1):
        """
        TODO
        inputs_to_distribute_on_command_line will be read and the content of the file will be passed on the command line
        """
        self.component_loader = component_loader
        self.num_concurrent_jobs_per_node = num_concurrent_jobs_per_node
        self.num_nodes = num_nodes

        self.code_dir = mkdtemp()

        self.batched_input = "model_indices"

        self.outputs_folder = outputs_folder
        self.outputs_file = outputs_file
        self.inputs_to_distribute = inputs_to_distribute
        self.inputs_to_disribute_on_command_line = inputs_to_distribute_on_command_line or []

        self.run_function = self.generate_run_function(code_dir=self.code_dir)

    def __del__(self):
        if os.path.exists(self.code_dir):
            shutil.rmtree(self.code_dir)

    def generate_run_function(self, code_dir: str) -> RunFunction:
        component = self.component_loader.component

        base_path = Path(component.base_path)
        code_dir = Path(code_dir)
        for f in (base_path/str(component.code)).iterdir():
            if f.is_dir():
                copytree(f, code_dir)
            else:
                copy2(f, code_dir/f.name)

        if component.additional_includes:
            for additional_include in component.additional_includes:
                additional_include = (base_path/additional_include).resolve()
                if additional_include.is_dir():
                    copytree(additional_include, code_dir/additional_include.name)
                else:
                    copy2(additional_include, code_dir/additional_include.name)

        copy2(aml_parallel.CODE_DIR/"aml_parallel_run.py", code_dir/"aml_parallel_run.py")

        cmd = component.command

        aml_parallel_args = "--aml_parallel_batched_input " + self.batched_input + " "
        for i in self.inputs_to_distribute:
            aml_parallel_args += "--aml_parallel_inputs_to_distribute " + i + " "
        for o in self.outputs_folder:
            aml_parallel_args += "--aml_parallel_outputs_folder " + o + " "
        for o in self.outputs_file:
            aml_parallel_args += "--aml_parallel_outputs_file " + o + " "
        for i in self.inputs_to_disribute_on_command_line:
            aml_parallel_args += "--aml_parallel_pass_on_command_line " + i + " "

        self.timeout_seconds = 3600*24*3 - 600

        run_function = RunFunction(
            code=code_dir,
            entry_script="aml_parallel_run.py",
            environment=component.environment,
            program_arguments=" [CMD_START] " + cmd + " [CMD_END] " + aml_parallel_args,
        )
        return run_function

    def load(self, model_indices: Input, **other_inputs):
        comp_fn = self.component_loader.component

        inputs = comp_fn.inputs
        inputs.update({i: Input(type="uri_folder") for i in self.inputs_to_distribute})
        inputs.update({self.batched_input: Input(type="uri_folder")})

        parallel_run = parallel_run_function(
            inputs=inputs,
            outputs={o: Output(type="uri_folder", mode="rw_mount") for o in self.outputs_folder + self.outputs_file},
            input_data="${{inputs." + self.batched_input + "}}",
            instance_count=self.num_nodes,
            max_concurrency_per_instance=self.num_concurrent_jobs_per_node,
            error_threshold=0,
            mini_batch_size="1",
            mini_batch_error_threshold=0,
            logging_level="DEBUG",
            retry_settings={"max_retries": 1, "timeout": self.timeout_seconds},
            task=self.run_function,
            compute=self.component_loader.compute,
        )
        return parallel_run(model_indices=model_indices, **self.component_loader.parameter_dict, **other_inputs)
