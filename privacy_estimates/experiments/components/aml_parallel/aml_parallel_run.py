import os
import sys
import json
import subprocess
import shlex
import argparse
import pandas as pd

from dataclasses import dataclass
from typing import List


CMD_START_TOKEN = "[CMD_START]"
CMD_END_TOKEN = "[CMD_END]"


def init():
    print("Calling init()")

    process_data = json.loads(sys.argv[-1])
    gpu_index = process_data["gpu_index"]

    cuda_visible_devices = str(gpu_index)
    print(f"Set CUDA_VISIBLE_DEVICES to {cuda_visible_devices} on process {os.getpid()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices 

    # print environment variables each on a new line
    print("[start] Environment variables")
    for k, v in os.environ.items():
        print(f"{k}={v}")
    print("[end] Environment variables")


def get_cmd() -> List[str]:
    cmd_start = sys.argv.index(CMD_START_TOKEN)
    cmd_end = sys.argv.index(CMD_END_TOKEN)
    cmd = sys.argv[cmd_start+1:cmd_end]
    return cmd


@dataclass
class AMLParallelArgs:
    batched_input: str
    inputs_to_distribute: List[str]
    outputs_folder: List[str]
    outputs_file: List[str]
    pass_on_command_line: List[str]


def get_aml_parallel_args() -> AMLParallelArgs:
    aml_parallel_parser = argparse.ArgumentParser()
    aml_parallel_parser.add_argument("--aml_parallel_batched_input", type=str, required=True)
    aml_parallel_parser.add_argument("--aml_parallel_inputs_to_distribute", type=str, action="append")
    aml_parallel_parser.add_argument("--aml_parallel_outputs_folder", type=str, action="append")
    aml_parallel_parser.add_argument("--aml_parallel_outputs_file", type=str, action="append")
    aml_parallel_parser.add_argument("--aml_parallel_pass_on_command_line", type=str, action="append")
    aml_parallel_args, _ = aml_parallel_parser.parse_known_args()

    return AMLParallelArgs(
        batched_input=aml_parallel_args.aml_parallel_batched_input,
        inputs_to_distribute=aml_parallel_args.aml_parallel_inputs_to_distribute,
        pass_on_command_line=aml_parallel_args.aml_parallel_pass_on_command_line or [],
        outputs_folder=aml_parallel_args.aml_parallel_outputs_folder or [],
        outputs_file=aml_parallel_args.aml_parallel_outputs_file or [],
    )


def get_model_index(mini_batch: List[str]) -> int:
    assert len(mini_batch) == 1
    batched_input_df = pd.read_csv(mini_batch[0])
    assert "model_index" in batched_input_df.columns
    model_index = batched_input_df["model_index"][0]
    print(f"model_index: {model_index}")
    return model_index


def replace(data: List[str], old_value: str, new_value: str) -> List[str]:
    return [new_value if x == old_value else x for x in data]


def modify_cmd(cmd: List[str], model_index: int, aml_parallel_args: AMLParallelArgs) -> List[str]:
    for input in aml_parallel_args.inputs_to_distribute:
        original_input = os.environ["AZURE_ML_INPUT_" + input]
        cmd = replace(cmd, original_input, os.path.join(original_input, f"model_index-{model_index:04}"))

    if aml_parallel_args.pass_on_command_line:
        for input in aml_parallel_args.pass_on_command_line:
            original_input = os.path.join(os.environ["AZURE_ML_INPUT_" + input], f"model_index-{model_index:04}")
            with open(os.path.join(original_input, "data"), 'r') as f:
                data = f.read().strip()
            cmd = replace(cmd, original_input, data)

    for output in aml_parallel_args.outputs_folder:
        original_output = os.environ["AZURE_ML_OUTPUT_" + output]
        new_output_dir = os.path.join(original_output, f"model_index-{model_index:04}")
        os.makedirs(new_output_dir)
        cmd = replace(cmd, original_output, new_output_dir)
    for output in aml_parallel_args.outputs_file:
        original_output = os.environ["AZURE_ML_OUTPUT_" + output]
        new_output_dir = os.path.join(original_output, f"model_index-{model_index:04}")
        os.makedirs(new_output_dir)
        cmd = replace(cmd, original_output, os.path.join(new_output_dir, "data"))

    return cmd


def run_cmd(cmd: List[str]):
    process = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while process.poll() is None:
        print(process.stdout.readline().rstrip())

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def run(mini_batch: List[str]) -> List[int]:
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    os.chdir(os.environ["AZ_BATCHAI_JOB_WORK_DIR"])

    cmd = get_cmd()
    aml_parallel_args = get_aml_parallel_args()
    model_index = get_model_index(mini_batch)

    print(f"cmd: {cmd}")
    print(f"aml_parallel_args: {aml_parallel_args}")
    print(f"model_index: {model_index}")

    cmd = [item for item in cmd if item.strip() and item.strip() != '\\']

    cmd = modify_cmd(cmd, model_index=model_index, aml_parallel_args=aml_parallel_args)

    print(f"Command: {cmd}")

    run_cmd(cmd)

    return [0]*len(mini_batch)
