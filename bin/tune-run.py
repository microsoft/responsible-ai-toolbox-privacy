#!/usr/bin/env python3
import os
import argparse
import subprocess
import pyparsing as pp
from seval import safe_eval
from ray import tune, init
from pyparsing import pyparsing_common as ppc
from typing import Tuple, Any
from transformers import set_seed
from azureml.core import Run


def rangeint(lower: int, upper: int):
    return tune.grid_search(list(range(lower, upper)))


SUPPORTED_SAMPLING_RULES = {
    "uniform": tune.uniform,
    "randint": tune.randint,
    "rangeint": rangeint
}


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('COMMAND', nargs=argparse.REMAINDER, help="Trial command")
    parser.add_argument("-p", "--parameter", action='append', type=str, help="Parameters to search. E.g. '--batch_size~randint(100,200)' will randomly sample a batch size and pass it as the '--batch_size' command line parameter to the trial")
    parser.add_argument("--gpus-per-trial", type=float, default=1, help="Number of GPUs required by one trial")
    parser.add_argument("--cpus-per-trial", type=float, default=1, help="Number of CPU cores required by one trial.")
    parser.add_argument("--num-trials", required=True, type=int, help="Number of trials in total")
    parser.add_argument("--output-dir", required=True, help="Path to the directory where all the outputs of the trials will be stored.")
    parser.add_argument("--seed", type=int, default=2394, help="Seed for the hyper-parameter search. This is independent of the seed for the training.")
    parser.add_argument("--output-directory-flag", help="Name of the command line argument the trial takes for writing output.")
    parser.add_argument("--dry-run", type=bool, default=False, help="Only run one trial with increased logging.")
    parser.add_argument("--max-failures", type=int, default=1, help="Maximum number of retries per trial")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a ray tune directory")
    return parser


def parse_parameter(s: str) -> Tuple[str, Any]:
    """
    :return: Return a tuple with the name of the parameter and the ray.tune sampling definition
    :rtype: Tuple[str, Any]
    """
    rule = (
        pp.Word(pp.alphanums + "_" + "-")("parameter_name") + "~" + 
        pp.oneOf(SUPPORTED_SAMPLING_RULES.keys())("distribution") + 
            "(" + pp.delimitedList(pp.Word(pp.nums + "/*%+-"))("arguments") + ")"
    )
    s_parsed = rule.parseString(s).asDict()
    dist = s_parsed["distribution"]
    args = []
    for a in s_parsed["arguments"]:
        args.append(safe_eval(a))
    return s_parsed["parameter_name"], SUPPORTED_SAMPLING_RULES[dist](*args)

def get_abspath_if_path(path: str, wd: str = ".") -> str:
    if os.path.exists(os.path.join(wd, path)):
        return os.path.abspath(os.path.join(wd, path))
    return path

def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    subprocess.call(["ray", "start", "--head"])
    init(address="auto")

    if args.parameter:
        parameters = dict(parse_parameter(p) for p in args.parameter)
    else:
        parameters = []

    wd = os.path.abspath(os.getcwd())

    def run_trial(config):
        parent_run = Run.get_context()
        with parent_run.child_run() as child_run:
            command = args.COMMAND

            command = [get_abspath_if_path(c, wd) for c in command]

            if args.output_directory_flag:
                command.append("--" + args.output_directory_flag)
                command.append(tune.get_trial_dir())

            for key in config:
                command.append("--" + key)
                command.append(config[key])

            command = [str(c) for c in command]

            env = os.environ.copy()
            env["AZUREML_RUN_ID"] = child_run.get_details()["runId"]

            with open(os.path.join(tune.get_trial_dir(), "trial.cmd"), "w") as cmd_f:
                cmd_f.write(" ".join(command))

            with open(os.path.join(tune.get_trial_dir(), "stdout"), "w") as stdout:
                with open(os.path.join(tune.get_trial_dir(), "stderr"), "w") as stderr:
                    retcode = subprocess.call(command, stdout=stdout, stderr=stderr,
                                              env=env)

            if retcode == 0:
                child_run.complete()
            else:
                child_run.fail(error_code=retcode)

        if retcode != 0:
            raise RuntimeError("Trial did not complete successfullly.")

    num_trials = args.num_trials
    if args.dry_run:
        print("--dry-run selected. Only running one trial...")
        num_trials = 1

    if args.resume_from:
        raise NotImplementedError()

    tune.run(
        run_trial,
        config = parameters,
        num_samples=num_trials,
        max_failures=args.max_failures,
        resources_per_trial={
            "gpu": args.gpus_per_trial,
            "cpu": args.cpus_per_trial
        },
        local_dir=args.output_dir
    )


if __name__ == "__main__":
    args = arg_parser().parse_args()
    main(args)

