#!/usr/bin/env python3

from gpytorch.kernels import MultitaskKernel
from linear_operator import to_linear_operator

from ..lazy import KroneckerProductLinearOperatorLinearCG


class MultitaskKernelLinearCG(MultitaskKernel):
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        covar_i = self.task_covar_module.covar_matrix
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        covar_x = to_linear_operator(self.data_covar_module.forward(x1, x2, **params))
        res = KroneckerProductLinearOperatorLinearCG(covar_x, covar_i)
        return res.diagonal(dim1=-1, dim2=-2) if diag else res