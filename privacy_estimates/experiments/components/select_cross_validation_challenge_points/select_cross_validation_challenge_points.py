import pandas as pd
import numpy as np
import mlflow
from datasets import load_from_disk, Dataset
from mldesigner import command_component, Input, Output
from sklearn.metrics import auc, roc_curve
from tqdm_loggable.auto import tqdm


@command_component(environment="environment.aml.yaml", name="select_cross_validation_challenge_points_preprocess")
def preprocess(
    shadow_model_statistics: Input,
    shadow_model_statistics_for_cross_validation: Output,
    challenge_points_for_cross_validation: Output,
    train_test_ratio: float = 0.6,
):
    """

    Assume we have `num_models` models trained for shadow model statistics.
    We want to split the num_models into a train and a test split and then find the points that had the highest 
    test vulnerability
    """

    stats_ds = load_from_disk(shadow_model_statistics)
    print(f"Loaded {len(stats_ds)} shadow model statistics. Schema: {stats_ds.features}")
    stats: pd.DataFrame = stats_ds.to_pandas()

    stats_in = stats[[
        "sample_index", "split", "logits_in", "model_index_in", "label"
    ]].rename(columns={"logits_in": "logits", "model_index_in": "model_index"})
    stats_out = stats[[
        "sample_index", "split", "logits_out", "model_index_out", "label"
    ]].rename(columns={"logits_out": "logits", "model_index_out": "model_index"})
    stats_in = stats_in.explode(["logits", "model_index"])
    stats_out = stats_out.explode(["logits", "model_index"])

    num_models = max(stats_in["model_index"].max(), stats_out["model_index"].max()) + 1
    print(f"Found {num_models} models")

    train_models = list(range(num_models))[:int(train_test_ratio * num_models)]
    test_models = list(range(num_models))[int(train_test_ratio * num_models):]

    loss_statistics_in_train = stats_in[stats_in["model_index"].isin(train_models)]
    loss_statistics_in_test = stats_in[stats_in["model_index"].isin(test_models)]
    loss_statistics_out_train = stats_out[stats_out["model_index"].isin(train_models)]
    loss_statistics_out_test = stats_out[stats_out["model_index"].isin(test_models)]

    loss_statistics_train = loss_statistics_in_train.groupby(["split", "sample_index", "label"]).agg({"logits": list}).join(
        loss_statistics_out_train.groupby(["split", "sample_index", "label"]).agg({"logits": list}),
        on=["split", "sample_index", "label"], lsuffix="_in", rsuffix="_out"
    ).reset_index()

    loss_statistics_out_test = loss_statistics_out_test.assign(is_in=0)
    loss_statistics_in_test = loss_statistics_in_test.assign(is_in=1)
    loss_statistics_test = pd.concat([loss_statistics_in_test, loss_statistics_out_test])

    Dataset.from_pandas(loss_statistics_train, preserve_index=False).save_to_disk(shadow_model_statistics_for_cross_validation)
    Dataset.from_pandas(loss_statistics_test, preserve_index=False).save_to_disk(challenge_points_for_cross_validation)


@command_component(environment="environment.aml.yaml", name="select_cross_validation_challenge_points_postprocess")
def postprocess(scores: Input, challenge_points_for_cross_validation: Input, data: Input, num_challenge_points: int,
                mi_challenge_points: Output, criterion: str = "auc"):
    """
    Select the challenge points with the highest vulnerability

    Args:
        scores: The scores for the challenge points
        challenge_points: The challenge points
        select_from_split: The split to select from. If None, select from all splits
        criterion: The criterion to select the challenge points by
    """
    if criterion not in ["auc", "acc"]:
        raise ValueError(f"Criterion must be one of 'auc' or 'acc', but was {criterion}")

    ds = load_from_disk(data)
    available_indices = set(zip(ds["split"], ds["sample_index"]))

    scores_ds = load_from_disk(scores)
    challenge_points_ds = load_from_disk(challenge_points_for_cross_validation)

    df = challenge_points_ds.add_column("score", scores_ds["score"]).to_pandas()

    results_dict = {"split": [], "sample_index": [], "auc": [], "acc": []}
    for (split, sample_index), group in tqdm(list(df.groupby(["split", "sample_index"]))):
        if (split, sample_index) in available_indices:
            results_dict["split"].append(split)
            results_dict["sample_index"].append(sample_index)

            fpr, tpr, _ = roc_curve(y_true=group["is_in"], y_score=group["score"])
            results_dict["auc"].append(auc(fpr, tpr))
            results_dict["acc"].append(np.max(1-(fpr+(1-tpr))/2))
    results = pd.DataFrame(results_dict)
    results = results.sort_values(by=criterion, ascending=False).head(num_challenge_points)

    if len(results) < num_challenge_points:
        print(f"Only found {len(results)} challenge points, but requested {num_challenge_points}.")

    metrics = {
        "mia_auc_mean": results["auc"].mean(),
        "mia_auc_std": results["auc"].std(),
        "mia_auc_max": results["auc"].max(),
        "mia_auc_min": results["auc"].min(),
        "mia_acc_mean": results["acc"].mean(),
        "mia_acc_std": results["acc"].std(),
        "mia_acc_max": results["acc"].max(),
        "mia_acc_min": results["acc"].min(),
    }
    for key, value in metrics.items():
        if mlflow.active_run():
            mlflow.log_metric(key, value)
        print(f"{key}: {value}")

    most_vulnerable_points_index = set(zip(results["split"], results["sample_index"]))

    ds.filter(
        lambda x: (x["split"], x["sample_index"]) in most_vulnerable_points_index, keep_in_memory=True
    ).save_to_disk(mi_challenge_points)
