import math
import gc
import shutil

import torch
from torch.profiler import profile, ProfilerActivity
import gpytorch
from linear_operator import settings

import famgpytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}.")

class ConventionalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ConventionalGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ApproxGPModel15(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel15, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=15)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ApproxGPModel5(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel5, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def profile_gp(model_type, nb_training_points):
    train_x = torch.linspace(0, 1, nb_training_points, device=device)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size(), device=device) * math.sqrt(0.04)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(device=device)
    model = model_type(train_x, train_y, likelihood)
    model.to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == 'cuda' else [ProfilerActivity.CPU]

    with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./temp/tensorboard/single/test/{device.type}_dsadsa/" + model_type.__name__
            ),
    ):
        # force to use method for large matrices
        with settings.max_cholesky_size(1), settings.min_preconditioning_size(1):
            for _ in range(5):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()


def main():
    train_x_count = 5000

    shutil.rmtree(f"./temp/tensorboard/single/test/{device.type}_dsadsa/", ignore_errors=True)

    for m in [ConventionalGPModel, ApproxGPModel15, ApproxGPModel5]:
        gc.collect()
        torch.cuda.empty_cache()

        profile_gp(m, train_x_count)

if __name__ == "__main__":
    main()