from azure.ai.ml import dsl, Input
from azure.ai.ml.entities import PipelineComponent

from privacy_estimates.experiments.components import append_column_incrementing, append_column_constant_str
from privacy_estimates.experiments.aml import PrivacyEstimatesComponentLoader


@dsl.pipeline(name="add_index_to_dataset")
def add_index_to_dataset(data: Input, split: str) -> PipelineComponent:
    load_from_function = PrivacyEstimatesComponentLoader().load_from_function
    d = load_from_function(append_column_incrementing)(data=data, name="sample_index").outputs.output
    d = load_from_function(append_column_constant_str)(data=d, name="split", value=split).outputs.output
    return {"output": d}
