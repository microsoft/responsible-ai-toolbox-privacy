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

    if in_samples != out_samples:
        all_samples = in_samples.union(out_samples)
        if all_samples != in_samples:
            raise ValueError(f"Missing samples in in_predictions. Missing samples: {all_samples - in_samples}")
        if all_samples != out_samples:
            raise ValueError(f"Missing samples in out_predictions. Missing samples: {all_samples - out_samples}")

    in_df = in_predictions.to_pandas()
    out_df = out_predictions.to_pandas()

    in_df = in_df.groupby(["sample_index", "split", "label"]).agg(
        {
            "loss": ["mean", "std", "max", "min", "median", "count", list],
            "logits": [list],
            "model_index": [list],
        }
    )
    out_df = out_df.groupby(["sample_index", "split", "label"]).agg(
        {
            "loss": ["mean", "std", "max", "min", "median", "count", list],
            "logits": [list],
            "model_index": [list],
        }
    )
    in_df.columns = ['_'.join(col) for col in in_df.columns.values]
    out_df.columns = ['_'.join(col) for col in out_df.columns.values]
    in_df = in_df.rename(columns={"logits_list": "logits", "model_index_list": "model_index"})
    out_df = out_df.rename(columns={"logits_list": "logits", "model_index_list": "model_index"})

    samples = in_df.join(out_df, on=["sample_index", "split", "label"], lsuffix="_in", rsuffix="_out").reset_index()

    return Dataset.from_pandas(df=samples, preserve_index=False)


@command_component(environment="environment.aml.yaml")
def compute_shadow_model_statistics(predictions_in: Input, predictions_out: Input, statistics: Output):
    stats = _compute_shadow_model_statistics(
        in_predictions=load_from_disk(predictions_in),
        out_predictions=load_from_disk(predictions_out)
    )

    print(f"Writing {len(stats)} loss statistics to {statistics}...")
    stats.save_to_disk(statistics)
