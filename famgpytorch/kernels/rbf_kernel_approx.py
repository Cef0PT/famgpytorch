#!/usr/bin/env python3

from gpytorch.kernels import Kernel


class RBFKernelApprox(Kernel):
    def forward(self, x1, x2, diag = False, last_dim_is_batch = False, **params):
        pass