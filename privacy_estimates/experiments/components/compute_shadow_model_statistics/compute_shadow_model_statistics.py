from mldesigner import command_component, Input, Output
from datasets import Dataset, load_from_disk
from tempfile import TemporaryDirectory
from multiprocessing import cpu_count


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


def _compute_shadow_model_statistics(in_predictions: Dataset, out_predictions: Dataset) -> Dataset:
    print(f"Computing loss statistics for {len(in_predictions)} in-sample predictions and {len(out_predictions)} out-sample "
          "predictions...")
    in_predictions = drop_null_rows(in_predictions)
    out_predictions = drop_null_rows(out_predictions)
    print(f"Computing loss statistics for {len(in_predictions)} in-sample predictions and {len(out_predictions)} out-sample "
          "predictions...")

    in_samples = set(zip(in_predictions["sample_index"], in_predictions["split"]))
    out_samples = set(zip(out_predictions["sample_index"], out_predictions["split"]))
    all_samples = in_samples.union(out_samples)
    missing_in_samples = all_samples - in_samples
    missing_out_samples = all_samples - out_samples

    in_df = in_predictions.to_pandas()
    out_df = out_predictions.to_pandas()

    group_by = ["sample_index", "split"]
    aggregate = {"model_index": [list]}

    if "label" in in_df.columns:
        group_by.append("label")

    if "logits" in in_df.columns:
        aggregate["logits"] = [list]

    if "loss" in in_df.columns:
        aggregate["loss"] = ["mean", "std", "max", "min", "median", "count", list]

    if "mi_signal" in in_df.columns:
        aggregate["mi_signal"] = ["mean", "std", "max", "min", "median", "count", list]

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
    in_df = in_df.rename(columns={"model_index_list": "model_index"})
    out_df = out_df.rename(columns={"model_index_list": "model_index"})

    samples = in_df.join(out_df, on=group_by, lsuffix="_in", rsuffix="_out").reset_index()

    return Dataset.from_pandas(df=samples, preserve_index=False)


@command_component(environment="environment.aml.yaml")
def compute_shadow_model_statistics(predictions_in: Input, predictions_out: Input, statistics: Output):
    stats = _compute_shadow_model_statistics(
        in_predictions=load_from_disk(predictions_in),
        out_predictions=load_from_disk(predictions_out)
    )

    print(f"Writing {len(stats)} loss statistics to {statistics}...")
    stats.save_to_disk(statistics)
