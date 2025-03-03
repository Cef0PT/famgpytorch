import math
import gc
import os.path
import shutil

import torch
from torch.profiler import profile, ProfilerActivity
import gpytorch

import famgpytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}.")

class ConventionalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ConventionalGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class ApproxGPModel15(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel15, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=15), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class ApproxGPModel5(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ApproxGPModel5, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            famgpytorch.kernels.RBFKernelApprox(number_of_eigenvalues=5), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def train_gp(model_type, nb_training_points):
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

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./temp/tensorboard/multi_{device.type}_{nb_training_points}/" + model_type.__name__
            ),
    ) as prof:
        for i in range(3):
            prof.step()
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

    return prof


def main():
    sort_by_keyword = str(device) + "_time_total"
    train_x_count = 5000

    shutil.rmtree(f"./temp/tensorboard/multi_{device.type}_{train_x_count}/", ignore_errors=True)

    for m in [ConventionalGPModel, ApproxGPModel15, ApproxGPModel5]:
        gc.collect()
        torch.cuda.empty_cache()

        prof = train_gp(m, train_x_count)

        stdout_file_path = f"./temp/tensorboard/multi_{device.type}_{train_x_count}/out.txt"
        if not os.path.isfile(stdout_file_path):
            mode = "w"
        else:
            mode = "a"
        with open(stdout_file_path, mode) as f:
            print(m.__name__, file=f)
            print(prof.key_averages().table(sort_by=sort_by_keyword), file=f)
        del prof, stdout_file_path, mode

if __name__ == "__main__":
    main()