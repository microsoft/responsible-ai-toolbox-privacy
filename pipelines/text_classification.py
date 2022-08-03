from azure.ml.component import dsl, Pipeline
from typing import Dict, Any
from pydantic_cli import run_and_exit
from pipeline_utils import PipelineArguments, get_component_factory, COMPONENT_DIR


class TextClassificationArguments(PipelineArguments):
    learning_rate: float
    base_dataset: str
    extra_dataset: str
    test_dataset: str
    N: int
    m: int
    num_train_epochs: float
    model_name: str
    target_epsilon: float
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    per_sample_max_grad_norm: float
    max_sequence_length: int
    num_classes: int
    seed: int
    delta: float

    def dict(self) -> Dict[str, Any]:
        return super().dict(exclude={"base_dataset", "extra_dataset", "test_dataset", "workspace_config", "run_name"})


def main(args: TextClassificationArguments):
    ws = args.workspace

    base_data = ws.datasets[args.base_dataset]
    extra_data = ws.datasets[args.extra_dataset]
    test_data = ws.datasets[args.test_dataset]

    # Load components
    create_challenge_from_extra_factory = get_component_factory(
        yaml_file=COMPONENT_DIR / "create-challenge-from-extra" / "component_spec.yaml", version="0.1.0dev11", workspace=ws
    )
    convert_dataset_factory = get_component_factory(
        yaml_file=COMPONENT_DIR / "convert-tabular-dataset-to-parquet" / "component_spec.yaml", version="0.1.0dev11", workspace=ws
    )
    train_models_factory = get_component_factory(
        yaml_file=COMPONENT_DIR / "fine-tune-many-transformer-classifier" / "component_spec.yaml", version="0.1.0dev11", workspace=ws
    )
    compute_cp_predictions_factory = get_component_factory(
        yaml_file=COMPONENT_DIR / "compute-predictions-for-challenge-points-with-many-transformer-classifier" / "component_spec.yaml",
        version="0.1.0dev11", workspace=ws
    )
    compute_predictions_factory = get_component_factory(
        yaml_file=COMPONENT_DIR / "compute-predictions-with-many-transformer-classifier" / "component_spec.yaml",
        version="0.1.0dev11", workspace=ws
    )
    attack_loss_threshold_factory = get_component_factory(
        yaml_file=COMPONENT_DIR / "attack-loss-threshold" / "component_spec.yaml",
        version="0.1.0dev11", workspace=ws
    )
    compute_confusion_matrix_factory = get_component_factory(
        yaml_file=COMPONENT_DIR / "compute-confusion-matrix" / "component_spec.yaml",
        version="0.1.0dev11", workspace=ws
    )
    estimate_epsilon_factory = get_component_factory(
        yaml_file=COMPONENT_DIR / "estimate-epsilon" / "component_spec.yaml",
        version="0.1.0dev11", workspace=ws
    )

    # Define pipeline
    @dsl.pipeline(
        name=args.run_name,
        default_compute_target=args.cpu_cluster,
        default_datastore=args.datastore,
    )
    def create_pipeline(base_data_id, extra_data_id, test_data_id, N: int, m: int, num_train_epochs: float, model_name: str,
                        target_epsilon: float, gradient_accumulation_steps: int, learning_rate: float,
                        per_device_train_batch_size: int, per_sample_max_grad_norm: float, max_sequence_length: int,
                        seed: int, num_classes: int, delta: float) -> Pipeline:
        convert_base = convert_dataset_factory(tabular_dataset_id=base_data_id)
        convert_extra = convert_dataset_factory(tabular_dataset_id=extra_data_id)
        convert_test = convert_dataset_factory(tabular_dataset_id=test_data_id)
        create_challenge_from_extra = create_challenge_from_extra_factory(base_data=convert_base.outputs.output,
                                                                          extra_data=convert_extra.outputs.output, N=N,
                                                                          seed=seed)
        train_models = train_models_factory(
            train_base_data=create_challenge_from_extra.outputs.train_base_data,
            in_samples=create_challenge_from_extra.outputs.in_samples,
            test_data=convert_test.outputs.output,
            N=N,
            num_train_epochs=num_train_epochs,
            model_name=model_name,
            m=m,
            target_epsilon=target_epsilon,
            delta=delta,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_sample_max_grad_norm=per_sample_max_grad_norm,
            max_sequence_length=max_sequence_length,
            num_classes=num_classes
        )

        compute_predictions_first = compute_cp_predictions_factory(
            experiment_dir=train_models.outputs.output,
            challenge_points_per_model=m,
            challenge_points=create_challenge_from_extra.outputs.first_samples
        )
        compute_predictions_second = compute_cp_predictions_factory(
            experiment_dir=train_models.outputs.output,
            challenge_points=create_challenge_from_extra.outputs.second_samples,
            challenge_points_per_model=m
        )
        compute_predictions_train = compute_predictions_factory(
            experiment_dir=train_models.outputs.output,
            dataset=create_challenge_from_extra.outputs.train_base_data,
        )
        attack_loss_threshold = attack_loss_threshold_factory(
            first_samples_predictions=compute_predictions_first.outputs.predictions,
            second_samples_predictions=compute_predictions_second.outputs.predictions,
            train_base_predictions=compute_predictions_train.outputs.predictions,
            challenge_bits=create_challenge_from_extra.outputs.challenge_bits,
            delta=delta,
        )
        compute_confusion_matrix = compute_confusion_matrix_factory(
            attack_guesses=attack_loss_threshold.outputs.attack_guesses,
            challenge_bits=create_challenge_from_extra.outputs.challenge_bits
        )
        estimate_epsilon = estimate_epsilon_factory(
            attack_confusion_matrix=compute_confusion_matrix.outputs.confusion_matrix,
            delta=delta
        )

        convert_base.inputs.tabular_dataset_id.configure(mode="direct")
        convert_extra.inputs.tabular_dataset_id.configure(mode="direct")
        convert_test.inputs.tabular_dataset_id.configure(mode="direct")
        args.set_compute_settings(component=train_models, compute_target="gpu_cluster")
        args.set_compute_settings(component=compute_predictions_first, compute_target="gpu_cluster")
        args.set_compute_settings(component=compute_predictions_second, compute_target="gpu_cluster")
        args.set_compute_settings(component=compute_predictions_train, compute_target="gpu_cluster")

        return dict()

    # Instantiate pipeline
    p = create_pipeline(
        base_data_id=base_data, extra_data_id=extra_data, test_data_id=test_data, **args.dict()
    )

    # Submit to cluster
    p.validate(workspace=ws)
    p.submit(workspace=ws, experiment_name="MI_game_text_classification")
    

if __name__ == "__main__":
    run_and_exit(TextClassificationArguments, main)
