import math
import gc
import shutil

import torch
from torch.profiler import profile, ProfilerActivity
import gpytorch
from linear_operator import settings

import famgpytorch

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}.")


class ExactGPMulti(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactGPMulti, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=2
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class ConventionalGPModel(ExactGPMulti):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ConventionalGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )


class ConventionalGPModelLinearCG(ExactGPMulti):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ConventionalGPModelLinearCG, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = famgpytorch.kernels.MultitaskKernelApprox(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )


class ApproxGPModel15(ExactGPMulti):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel15, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=15), num_tasks=2, rank=1
        )


class ApproxGPModel15LinearCG(ExactGPMulti):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel15LinearCG, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = famgpytorch.kernels.MultitaskKernelApprox(
            famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=15), num_tasks=2, rank=1
        )

class ApproxGPModel5(ExactGPMulti):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel5, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=5), num_tasks=2, rank=1
        )


class ApproxGPModel5Lazy(ExactGPMulti):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel5Lazy, self).__init__(train_inputs, train_targets, likelihood)
        self.covar_module = famgpytorch.kernels.MultitaskKernelApprox(
            famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=5), num_tasks=2, rank=1
        )


def profile_gp(model_type, nb_training_points):
    train_x = torch.linspace(0, 1, nb_training_points, device=device)
    train_y = torch.stack([
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size(), device=device) * math.sqrt(0.04),
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size(), device=device) * math.sqrt(0.04),
    ], -1)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    likelihood.to(device)
    model = model_type(train_x, train_y, likelihood)
    model.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if device.type == 'cuda':
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        torch.cuda.synchronize()
    else:
        activities = [ProfilerActivity.CPU]

    with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./temp/tensorboard/multi/train/{device.type}_linear_cg/" + model_type.__name__
            ),
    ):
        # force to use computing method for large matrices
        with settings.max_cholesky_size(1), settings.min_preconditioning_size(1):
            for _ in range(5):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()


def main():
    train_x_count = 2000

    shutil.rmtree(f"./temp/tensorboard/multi/train/{device.type}_linear_cg/", ignore_errors=True)

    for m in [ApproxGPModel15]:
        gc.collect()
        torch.cuda.empty_cache()

        profile_gp(m, train_x_count)

if __name__ == "__main__":
    main()