import pandas as pd
import numpy as np

from mldesigner import command_component, Input, Output
from datasets import Dataset, load_from_disk
from tempfile import TemporaryDirectory
from multiprocessing import cpu_count
from pathlib import Path


def drop_null_rows(ds: Dataset) -> Dataset:
    # Save to a temporary directory and reload to ensure that we cache to disk
    # The original location may be read-only so we can't cache there
    # The input before filtering may be fairly large so we memory is a constraint and
    # caching to disk is necessary
    with TemporaryDirectory() as tmp_dir:
        ds.save_to_disk(tmp_dir)
        ds = load_from_disk(tmp_dir)
        filtered_ds = ds.filter(lambda row: all(row[k] is not None for k in ["split", "sample_index"]), num_proc=cpu_count())
        return filtered_ds


def log_mean_exp(log_column: pd.Series):
    """
    Compute the mean of a column of log values. This is done by exponentiating the log values, computing the mean of
    the exponentiated values, and then taking the log of the mean.

    The exponantiationg is done in longdouble. This can be useful for very small values, where the log values are very
    negative and the exponentiated values are very close to zero. In this case, the exponentiated values will be rounded
    """
    return np.log(np.mean(np.exp(np.array(log_column.values, dtype=np.longdouble)))).astype(np.double)


def _compute_shadow_artifact_statistics(in_scores: Dataset, out_scores: Dataset) -> Dataset:
    print(f"Computing loss statistics for {len(in_scores)} in-sample scores and {len(out_scores)} out-sample "
          "scores...")
    in_scores = drop_null_rows(in_scores)
    out_scores = drop_null_rows(out_scores)
    print(f"Computing loss statistics for {len(in_scores)} in-sample scores and {len(out_scores)} out-sample "
          "scores...")

    in_samples = set(zip(in_scores["sample_index"], in_scores["split"]))
    out_samples = set(zip(out_scores["sample_index"], out_scores["split"]))
    all_samples = in_samples.union(out_samples)
    missing_in_samples = all_samples - in_samples
    missing_out_samples = all_samples - out_samples

    in_df = in_scores.to_pandas()
    out_df = out_scores.to_pandas()

    group_by = ["sample_index", "split"]
    aggregate = {"artifact_index": [list]}

    if "label" in in_df.columns:
        group_by.append("label")

    if "logits" in in_df.columns:
        aggregate["logits"] = [list]

    if "loss" in in_df.columns:
        aggregate["loss"] = ["mean", "std", "max", "min", "median", "count", list]

    if "mi_signal" in in_df.columns:
        aggregate["mi_signal"] = ["mean", "std", "max", "min", "median", "count", list]

    if "log_mi_signal" in in_df.columns:
        aggregate["log_mi_signal"] = [log_mean_exp, list]

    in_df = in_df.groupby(group_by).agg(aggregate)
    out_df = out_df.groupby(group_by).agg(aggregate)

    # add missing in samples. If the column is count set it to 0, if list set it to [] otherwise set it to nan
    for sample_index, split in missing_in_samples:
        sample_dict = {col: [] if "list" in col else 0 if "count" in col else float('nan') for col in in_df.columns}
        sample_dict["sample_index"] = sample_index
        sample_dict["split"] = split
        in_df = in_df.append(sample_dict, ignore_index=True)

    # add missing out samples. If the column is count set it to 0, if list set it to [] otherwise set it to nan
    for sample_index, split in missing_out_samples:
        sample_dict = {col: [] if "list" in col else 0 if "count" in col else float('nan') for col in out_df.columns}
        sample_dict["sample_index"] = sample_index
        sample_dict["split"] = split
        out_df = out_df.append(sample_dict, ignore_index=True)

    in_df.columns = ['_'.join(col) for col in in_df.columns.values]
    out_df.columns = ['_'.join(col) for col in out_df.columns.values]
    in_df = in_df.rename(columns={"artifact_index_list": "artifact_index"})
    out_df = out_df.rename(columns={"artifact_index_list": "artifact_index"})

    samples = in_df.join(out_df, on=group_by, lsuffix="_in", rsuffix="_out").reset_index()

    return Dataset.from_pandas(df=samples, preserve_index=False)


@command_component(
    name="privacy_estimates__compute_shadow_artifact_statistics",
    environment={
        "conda_file": Path(__file__).parent/"environment.conda.yaml",
        "image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
    }
)
def compute_shadow_artifact_statistics(scores_in: Input, scores_out: Input, statistics: Output):
    stats = _compute_shadow_artifact_statistics(
        in_scores=load_from_disk(scores_in),
        out_scores=load_from_disk(scores_out)
    )

    print(f"Writing {len(stats)} loss statistics to {statistics}...")
    stats.save_to_disk(statistics)
