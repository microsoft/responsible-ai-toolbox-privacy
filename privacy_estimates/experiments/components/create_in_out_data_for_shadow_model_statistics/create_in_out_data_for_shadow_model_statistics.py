import logging
import datetime
from mldesigner import command_component, Input, Output
from datasets import load_from_disk, Dataset, features, concatenate_datasets
from sklearn.model_selection import train_test_split
from enum import Enum
from tqdm_loggable.auto import tqdm
from tqdm_loggable.tqdm_logging import tqdm_logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SplitSizeType(Enum):
    EQUAL_IN_OUT = "equal_in_out"
    SAME_AS_TRAIN_TEST = "same_as_train_test"
    SPECIFY_TRAIN_FRACTION = "specify_train_fraction"


def empty_dataset(features: features.Features) -> Dataset:
    return Dataset.from_dict({k: [] for k in features}, features=features)


@command_component(environment="environment.aml.yaml")
def create_in_out_data_for_shadow_model_statistics(
    train_data: Input,
    validation_data: Input,
    train_base_data: Output,
    validation_base_data: Output,
    in_out_data: Output,
    in_indices: Output,
    out_indices: Output,
    num_points_per_model: Output(type="integer"),
    seed: int,
    split_size_type: str,
    train_size_fraction: float = 0.0,
    max_num_models: int = 5_000,
):
    """Create in and out indices for loss statistics computation.

    This component creates in and out indices for loss statistics computation. It takes in train and test data, and
    outputs train and test base data, in and out indices, and the component itself. The in and out indices are used to
    split the data into in and out samples for loss statistics computation. The split size can be specified using the
    split_size_type parameter, which can take one of three values: "equal_in_out", "same_as_train_test", or
    "specify_train_fraction". If "equal_in_out" is selected, the in and out samples will be of equal size. If
    "same_as_train_test" is selected, the in and out samples will be split in the same way as the train and test data.
    If "specify_train_fraction" is selected, the train_size_fraction parameter must be set to a value between 0.0 and
    1.0, and the in and out samples will be split according to this fraction. The max_num_models parameter specifies
    the maximum number of models to train on the data.

    Args:
        train_data (Input): Train data.
        validation_data (Input): Validation data.
        train_base_data (Output): Train base data.
        validation_base_data (Output): Validation base data.
        in_out_data (Output): In out data that contains in and out samples.
        in_indices (Output): In indices.
        out_indices (Output): Out indices.
        seed (int): Seed for random number generator.
        split_size_type (str): Split size type.
        num_points_per_model (Output): Number of points per model.
        train_size_fraction (float, optional): Train size fraction. Defaults to 0.0.
        max_num_models (Integer, optional): Maximum number of models to train. Defaults to 5_000.
    """
    fmt = f"%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[logging.StreamHandler()])
    tqdm_logging.set_level(logging.INFO)
    tqdm_logging.set_log_rate(datetime.timedelta(seconds=10))  


    train_ds = load_from_disk(train_data)
    validation_ds = load_from_disk(validation_data)
    logger.info(f"Loaded train data with {len(train_ds)} points, test data with {len(validation_ds)} points")

    assert "sample_index" in train_ds.column_names, "train_ds must have a column named 'sample_index'"
    assert "split" in train_ds.column_names, "train_ds must have a column named 'split'"
    assert "sample_index" in validation_ds.column_names, "validation_ds must have a column named 'sample_index'"
    assert "split" in validation_ds.column_names, "validation_ds must have a column named 'split'"

    indices = (
        [(split, i) for split, i in zip(train_ds["split"], train_ds["sample_index"])] + 
        [(split, i) for split, i in zip(validation_ds["split"], validation_ds["sample_index"])]
    )
        
    split_size_type = SplitSizeType(split_size_type)

    if split_size_type == SplitSizeType.EQUAL_IN_OUT:
        train_size_fraction = 0.5
    elif split_size_type == SplitSizeType.SAME_AS_TRAIN_TEST:
        train_size_fraction = len(train_ds) / (len(train_ds) + len(validation_ds))
    elif split_size_type == SplitSizeType.SPECIFY_TRAIN_FRACTION:
        assert train_size_fraction > 0.0, "train_size_fraction must be greater than 0.0"
        train_size_fraction = train_size_fraction

    in_indices_dict = {"split": [], "sample_index": []}
    out_indices_dict = {"split": [], "sample_index": []}
    for i in tqdm(range(max_num_models), desc="Creating in and out indices", unit="model", total=max_num_models):
        in_indices_i, out_indices_i = train_test_split(indices, train_size=train_size_fraction, random_state=seed+9239+i)
        points_per_model = len(in_indices_i)
        
    	# Ensure in_indices and out_indices are of the same length and fill with None if not
        if len(in_indices_i) > len(out_indices_i):
            out_indices_i += [(None, None)] * (len(in_indices_i) - len(out_indices_i))
        elif len(out_indices_i) > len(in_indices_i):
            in_indices_i += [(None, None)] * (len(out_indices_i) - len(in_indices_i))

        in_indices_dict["split"] += [x[0] for x in in_indices_i]
        in_indices_dict["sample_index"] += [x[1] for x in in_indices_i]
        out_indices_dict["split"] += [x[0] for x in out_indices_i]
        out_indices_dict["sample_index"] += [x[1] for x in out_indices_i]

    logger.info(f"Number of in samples per model: {points_per_model}")
    with open(num_points_per_model, "w") as f:
        f.write(str(points_per_model))

    index_features = features.Features({
        "split": features.Value(dtype="string"),
        "sample_index": features.Value(dtype="int32")
    })
    in_indices_ds = Dataset.from_dict(mapping=in_indices_dict, features=index_features)
    in_indices_ds.save_to_disk(in_indices)
    out_indices_ds = Dataset.from_dict(mapping=out_indices_dict, features=index_features)
    out_indices_ds.save_to_disk(out_indices)
    logger.info(f"Saved {len(in_indices_ds)} in indices and {len(out_indices_ds)} out indices")

    in_out_ds = concatenate_datasets([train_ds, validation_ds])
    in_out_ds.save_to_disk(in_out_data)

    # train_base_data and validation_base_data are empty since every model is trained on different
    # data
    empty_dataset(features=train_ds.features).save_to_disk(train_base_data)
    empty_dataset(features=validation_ds.features).save_to_disk(validation_base_data)
