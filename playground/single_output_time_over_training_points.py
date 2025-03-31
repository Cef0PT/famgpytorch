#!/usr/bin/env python3
import math
import time
import gc
import logging
from itertools import product
from pathlib import Path

import torch
import gpytorch
import numpy as np

import famgpytorch


DIRECTORY_PATH = Path("temp/time_complexity")
DATA_FILE_PATH = Path(DIRECTORY_PATH, "single_time_over_sample_nb.csv")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else 'CPU'}.")
torch.cuda.empty_cache()
torch.manual_seed(42)


class BaseKernel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(BaseKernel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ConventionalGPModel(BaseKernel):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ConventionalGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = gpytorch.kernels.RBFKernel()


class ApproxGPModel10(BaseKernel):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel10, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=10)


class ApproxGPModel20(BaseKernel):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel20, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=20)


def train_gp(model_type, nb_training_points):
    train_x = torch.linspace(0, 1, nb_training_points, device=DEVICE)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size(), device=DEVICE) * math.sqrt(0.04)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(device=DEVICE)
    model = model_type(train_x, train_y, likelihood)
    model.to(DEVICE)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    torch.cuda.synchronize()

    try:
        start = time.perf_counter()
        for i in range(20):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        end = time.perf_counter()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        logging.error(repr(e)[:140] + "...")
        return None, None, nb_training_points

    return end - start, -mll(model(train_x), train_y).item(), None


def repeat_train(model_types, training_sample_nbs, truncate_file=False):
    # create directory for csv file if it does not exist yet
    if not DATA_FILE_PATH.parent.exists():
        DATA_FILE_PATH.parent.mkdir(parents=True)
    if not DATA_FILE_PATH.is_file() or truncate_file:
        # create / truncate file and add headers
        with open(DATA_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(
                "sample_nbs," +
                ",".join([name + "_" + suffix for name, suffix in product(
                    [m.__name__.lower() for m in model_types],
                    ["train_time", "train_nll"]
                )])
            )

    training_times = np.empty(len(model_types), dtype=float)
    training_losses = np.empty(len(model_types), dtype=float)

    failed = set()
    marked = set()

    for train_sample_nb in training_sample_nbs:
        for row, model_type in enumerate(model_types):
            if model_type in failed:
                # skip already failed models
                training_times[row], training_losses[row] = None, None
                continue

            # Make sure everything unnecessary is deleted from memory
            gc.collect()
            torch.cuda.empty_cache()

            if model_type not in marked:
                training_times[row], training_losses[row], failed_sample_nb = train_gp(
                    model_type, nb_training_points=train_sample_nb
                )
                if training_times[row] > 900:
                    # training took > 900s -> mark to fail in next iteration
                    marked.add(model_type)
            else:
                training_times[row], training_losses[row], failed_sample_nb = None, None, train_sample_nb

            if failed_sample_nb:
                failed.add(model_type)
                print(f"{model_type.__name__} failed at {failed_sample_nb} data points.")
                if set(model_types).issubset(failed):
                    # break after all models failed
                    break

            print(f"Writing data for {train_sample_nb} data points for {model_type.__name__}...")

        else:
            # training did not fail for at least one model, write data
            with open(DATA_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n{train_sample_nb}," + ",".join([f"{a},{b}" for a, b in zip(training_times, training_losses)]))
            continue

        # training did fail for all models, break out
        break


def main():
    sample_nbs = np.arange(2000, 100001, 2000)
    repeat_train(
        [
            ConventionalGPModel,
            ApproxGPModel10,
            ApproxGPModel20
        ],
        sample_nbs,
        truncate_file=True
    )

if __name__ == "__main__":
    main()