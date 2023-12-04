import numpy as np
import random
import os
import warnings
import mlflow
import numpy as np
import torch
import json
import datasets

from torch import nn
from torch import optim 
from torch.utils.data import DataLoader
from pydantic_cli import run_and_exit
from pydantic import Field, BaseModel
from pathlib import Path
from torch.utils.data import TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm
from typing import Callable, Optional

from models.cnn import CNN, compute_prediction_metrics, compute_accuracy, compute_loss
from dpd import CanaryGradient, CanaryTrackingOptimizer, DPDistinguishingData
from utils import DPParameters


class Arguments(BaseModel):
    total_train_batch_size: int = Field(
        description="Total number of samples seen between gradient updates"
    )
    max_physical_batch_size: int = Field(
        description="Maximum number of samples seen in a single batch on a device"
    )
    output_dir: Path = Field(
        description="Output directory where the model checkpoints will be written"
    )
    train_data_path: Path = Field(
        description="Path to the training data"
    )
    test_data_path: Path = Field(
        description="Path to the test data"
    )
    seed: int = Field(
        description="Random seed that will be set at the beginning of the script."
    )
    num_train_epochs: float = Field(
        description="Total number of training epochs to perform."
    )
    target_epsilon: float = Field(
        description="Target epsilon for the privacy loss"
    )
    delta: float = Field(
        description="Delta for the privacy loss"
    )
    learning_rate: float = Field(
        description="The initial learning rate."
    )
    per_sample_max_grad_norm: float = Field(
        description="The maximum gradient norm for clipping"
    )
    dataloader_num_workers: int = Field(
        default=4, description="Number of workers for data loading. 0 means that the data will be loaded in the main process"
    )
    logging_steps: int = Field(
        default=100, description="Prints accuracy, loss, and privacy accounting information during training every k logical batches"
    )
    output_dir: Path = Field(
        description="Output directory. If none given, will pick one based on hyperparameters"
    )
    metrics: Path = Field(
        description="Path to the metrics and parameters file"
    )
    dp_parameters: Path = Field(
        description="Path[out] to the DP parameters file"
    )
    dpd_data: Path = Field(
        description="Path[out] to the DPD data file"
    )
    lr_scheduler_gamma: float = Field(
        default=1.0, description="gamma parameter for exponential learning rate scheduler"
    )
    canary_gradient: str = Field(
        default="dirac", description="Insert a canary gradient into the optimizer"
    )
    lr_scheduler_step: int = Field(
        default=1, description="step size for exponential learning rate scheduler"
    )
    disable_dp: bool = Field(
        default=False, description="Disable differential privacy"
    )
    use_cpu: int = Field(
        default=0, description="Use CPU instead of GPU"
    )
    disable_ml_flow: int = Field(
        default=0, description="Disable ML Flow"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if np.isinf(self.target_epsilon):
            self.disable_dp = True

        if not float(int(self.num_train_epochs)) == self.num_train_epochs:
            raise ValueError(f"num_train_epochs must be an integer, got {self.num_train_epochs}")


def train(args: Arguments,
          model: nn.Module,
          device: torch.device,
          train_loader: DataLoader,
          optimizer: optim.Optimizer,
          epoch: int,
          compute_epsilon: Optional[Callable[[int], float]] = None):
    model.train()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.max_physical_batch_size,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        if args.disable_dp:
            data_loader = train_loader
        else:
            data_loader = memory_safe_data_loader

        # BatchSplittingSampler.__len__() approximates (badly) the length in physical batches
        # See https://github.com/pytorch/opacus/issues/516
        # We instead heuristically keep track of logical batches processed
        pbar = tqdm(data_loader, desc="Batch", unit="batch", position=1, leave=True, total=len(train_loader), disable=None)
        logical_batch_len = 0
        for i, (inputs, target) in enumerate(data_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            logical_batch_len += len(target)
            if logical_batch_len >= args.total_train_batch_size:
                pbar.update(1)
                logical_batch_len = logical_batch_len % args.max_physical_batch_size

            optimizer.zero_grad()
            logits = model(inputs)
            loss = compute_loss(logits, target).mean()

            # check for nan in logits
            if torch.isnan(loss):
                raise ValueError(f"Loss is NaN at step {i}. inputs: {inputs}, target: {target}")

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = compute_accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i + 1) % args.logging_steps == 0 or (i + 1) == len(data_loader):
                if not args.disable_dp:
                    epsilon = compute_epsilon(delta=args.delta)
                    pbar.set_postfix(
                        epoch=f"{epoch:02}",
                        train_loss=f"{np.mean(losses):.3f}",
                        accuracy=f"{np.mean(top1_acc) * 100:.3f}",
                        dp=f"(ε={epsilon:.2f}, δ={args.delta})"
                    )
                    if not args.disable_ml_flow:
                        mlflow.log_metric("epsilon", float(epsilon))
                else:
                    pbar.set_postfix(
                        epoch=f"{epoch:02}",
                        train_loss=f"{np.mean(losses):.3f}",
                        accuracy=f"{np.mean(top1_acc) * 100:.3f}",
                        dp="(ε = ∞, δ = 0)"
                    )
                    if not args.disable_ml_flow:
                        mlflow.log_metric("epsilon", float(np.inf))
                if not args.disable_ml_flow:
                    mlflow.log_metric("epoch", (epoch))
                    mlflow.log_metric("train_loss", (np.mean(losses)))
                    mlflow.log_metric("train_accuracy", (np.mean(top1_acc) * 100))
        pbar.update(pbar.total - pbar.n)


def main(args: Arguments):
    print(args)

    train_data = datasets.load_from_disk(args.train_data_path)
    test_data = datasets.load_from_disk(args.test_data_path)

    num_classes = len(train_data.features["label"].names)

    if not args.use_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = 'cpu' if args.use_cpu else 'cuda'

    # Following the advice on https://pytorch.org/docs/1.8.1/notes/randomness.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    noise_generator = torch.Generator(device=device).manual_seed(args.seed)

    # Required to get deterministic batches because Opacus uses secure_rng as a generator for
    # train_loader when poisson_sampling = True even though secure_mode = False, which sets secure_rng = None
    # https://github.com/pytorch/opacus/blob/5e632cdb8d497aade29e5555ad79921c239c78f7/opacus/privacy_engine.py#L206
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with train_data.formatted_as(type="torch"):
        assert not torch.isnan(train_data["image"]).any()
        train_dataset = TensorDataset(train_data["image"], train_data["label"])
    with test_data.formatted_as(type="torch"):
        assert not torch.isnan(test_data["image"]).any()
        test_dataset  = TensorDataset(test_data["image"], test_data["label"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.total_train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.max_physical_batch_size,
        num_workers=args.dataloader_num_workers
    )

    # Supress warnings
    warnings.filterwarnings(action="ignore", module="opacus", message=".*Secure RNG turned off")
    warnings.filterwarnings(action="ignore", module="torch", message=".*Using a non-full backward hook")

    model = CNN()
    assert ModuleValidator.is_valid(model)

    print(f"{torch.cuda.memory_summary(device=device)}")
    print(f"Sending model to {device}")
    model = model.to(device)
    print(f"Model sent to {device}")
    print(f"{torch.cuda.memory_summary(device=device)}")

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0)

    assert float(int(args.num_train_epochs)) == args.num_train_epochs, "num_train_epochs must be an integer"
    epochs = int(args.num_train_epochs)
    num_steps = int(len(train_loader) * epochs)

    canary_gradient = CanaryGradient.from_optimizer(optimizer, method=args.canary_gradient, norm=args.per_sample_max_grad_norm)
    optimizer = CanaryTrackingOptimizer(optimizer=optimizer, canary_gradient=canary_gradient)

    sample_rate = 1 / len(train_loader)
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(accountant="prv")


        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            target_delta=args.delta,
            target_epsilon=args.target_epsilon,
            data_loader=train_loader,
            epochs=epochs,
            max_grad_norm=args.per_sample_max_grad_norm,
            poisson_sampling=True,
            noise_generator=noise_generator
        )

        print(f"Training using DP-SGD with {optimizer.original_optimizer.__class__.__name__} optimizer\n"
             f"  noise multiplier σ = {optimizer.noise_multiplier},\n"
             f"  sampling rate q = {sample_rate},\n"
             f"  clipping norm C = {optimizer.max_grad_norm:},\n"
             f"  average batch size L = {args.total_train_batch_size},\n"
             f"  for {epochs} epochs ({num_steps} steps)\n"
             f"  to target ε = {args.target_epsilon}, δ = {args.delta}")

        compute_epsilon: Optional[Callable[[float], float]] = lambda delta: privacy_engine.get_epsilon(delta=delta)
    else:
        print(f"Training using SGD with {optimizer.__class__.__name__} optimizer\n"
             f"  sampling rate q = {sample_rate},\n"
             f"  batch size L = {args.total_train_batch_size},\n"
             f"  for {epochs} epochs ({num_steps} steps)")
        compute_epsilon = None

    # Must be initialized after attaching the privacy engine.
    # See https://discuss.pytorch.org/t/how-to-use-lr-scheduler-in-opacus/111718
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma)

    for epoch in range(epochs):
        train(args=args, model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch + 1,
              compute_epsilon=compute_epsilon)
        print(f"Epoch {epoch + 1} finished. learning_rate: {scheduler.get_last_lr()}")
        scheduler.step()

    metrics = compute_prediction_metrics(model=model, data_loader=test_loader, device=device)

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    with open(os.path.join(args.output_dir, "accuracy"), "w") as f:
        print(f"{metrics.accuracy:.3f}", file=f)
    if not args.disable_ml_flow:
        mlflow.log_metric("test_accuracy", metrics.accuracy)
        mlflow.log_metric("test_loss", metrics.loss)

    if not args.disable_dp:
        final_epsilon = compute_epsilon(args.delta)
        print(f"The trained model is (ε = {final_epsilon}, δ = {args.delta})-DP")
        with open(os.path.join(args.output_dir, "epsilon"), "w") as f:
            print(f"{final_epsilon:.3f}", file=f)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    DPParameters.from_opacus(privacy_engine=privacy_engine).save_to_disk(args.dp_parameters)
    DPDistinguishingData.from_opacus(optimizer=optimizer).save_to_disk(args.dpd_data)

    with args.metrics.open("w+") as f:
        json.dump({
            "test_accuracy": float(metrics.accuracy),
            "test_loss": float(metrics.loss)
        }, f)

    return 0


if __name__ == "__main__":
    run_and_exit(Arguments, main)

