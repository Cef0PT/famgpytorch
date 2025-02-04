#!/usr/bin/env python3
from typing import Union

from gpytorch.kernels import Kernel
from linear_operator import LinearOperator
from torch import Tensor


class ApproxRBFKernel(Kernel):
    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params) -> Union[
        Tensor, LinearOperator]:
        pass