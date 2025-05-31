#!/usr/bin/env python3
from typing import Optional
import math
import warnings

import torch
from torch import Tensor
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.priors import Prior
from linear_operator import to_linear_operator
from linear_operator.operators import DiagLinearOperator, LinearOperator

from ..functions import ChebyshevHermitePolynomials


class RBFKernelApprox(Kernel):
    r"""
    Computes an approximate covariance matrix based on the Mercer's expansion of the RBF (squared exponential) kernel
    based on `Fast Approximate Multioutput Gaussian Processes`_.

    Mercer's theorem states the existence of an orthonormal basis consisting of eigenfunctions
    :math:`\phi_i(\mathbf{x})` and nonincreasing eigenvalues :math:`\lambda_i`

    .. math::
        \begin{equation}
            k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2}) =
            \sum^{\infty}_{i=0}\phi_i(\mathbf{x_1})\lambda_i\phi_i(\mathbf{x_2})
        \end{equation}

    and an approximate covariance matrix can be computed by utilizing only :math:`n` eigenvalues such that

    .. math::
        \begin{equation}
            K_{\mathbf{X_1X_2}} \approx \Phi_{\mathbf{X_1}}\Lambda\Phi_{\mathbf{X_2}}^\top
        \end{equation}

    where

    * :math:`\Phi_{\mathbf{X}i,j} = \phi_j(\mathbf{x_i}) | j \in 1, 2, \ldots, n`
    * :math:`\Lambda` is a diagonal matrix of the eigenvalues :math:`[\lambda_1,\lambda_2,\ldots,\lambda_n]`

    The Mercer's expansion of the RBF kernel is given by `Stable Evaluation of Gaussian RBF Interpolants`_

    .. math::
        \begin{align}
            \lambda_i &= \sqrt{ \frac{\alpha^2}{\alpha^2+\delta^2+\eta^2} }
            \left( \frac{\eta^2}{\alpha^2+\delta^2+\eta^2} \right)^i \\
            \phi_i(\mathbf{x}) &= \sqrt{\frac{\beta}{i!}} \exp \left( -\delta^2\mathbf{x}^2 \right)
            \mathrm{He}_i \left(\sqrt{2} \alpha \beta \mathbf{x} \right),
        \end{align}

    where

    * :math:`\eta^2=\frac{1}{2}\Theta^{-2}` with the lengthscale parameter :math:`\Theta`
    * :math:`\beta = \left( 1 + 4 \frac{\eta^2}{\alpha^2} \right)^{\frac{1}{4}}`
    * :math:`\delta^2 = \frac{\alpha^2}{2} (\beta^2 - 1)`
    * :math:`\alpha` is a global scaling parameter
    * :math:`\mathrm{He}_i(\cdot)` denotes the :math:`i\text{th}` Chebyshev-Hermite polynomial

    .. _Fast Approximate Multioutput Gaussian Processes:
        https://doi.org/10.1109/MIS.2022.3169036

    .. _Stable Evaluation of Gaussian RBF Interpolants:
        https://www.math.iit.edu/~fass/StableGaussianRBFs.pdf
    """

    has_lengthscale = True

    def __init__(
            self,
            number_of_eigenvalues: Optional[int]=15,
            alpha_prior: Optional[Prior]=None,
            alpha_constraint: Optional[Interval]=None,
            **kwargs
    ):
        super(RBFKernelApprox, self).__init__(**kwargs)

        if number_of_eigenvalues <= 0:
            raise ValueError("number_of_eigenvalues must be positive.")
        if not isinstance(number_of_eigenvalues, int):
                raise TypeError("Expected int but got " + type(number_of_eigenvalues).__name__)
        self.number_of_eigenvalues = number_of_eigenvalues

        # register parameter
        self.register_parameter(
            name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # set parameter constraint
        if alpha_constraint is None:
            alpha_constraint = GreaterThan(1e-4)

        # set parameter prior
        if alpha_prior is not None:
            if not isinstance(alpha_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(alpha_prior).__name__)
            # noinspection PyProtectedMember
            self.register_prior(
                "alpha_prior",
                alpha_prior,
                lambda m: m.alpha,
                lambda m, v: m._set_alpha(v)
            )

        # register constraint
        self.register_constraint("raw_alpha", alpha_constraint)

    @property
    def alpha(self):
        # apply constraint when accessing the parameter
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self._set_alpha(value)

    def _set_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)

        # when setting the parameter, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))


    def forward(self, x1, x2, diag=False, **params):
        if diag:
            # not worth to approximate here
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return self.covar_dist(x1_, x2_, square_dist=True, diag=True, **params).div_(-2).exp_()

        return approx_rbf_covariance(x1, x2, self.lengthscale, self.alpha, self.number_of_eigenvalues)


def approx_rbf_covariance(
        x1: Tensor,
        x2: Tensor,
        lengthscale: Tensor,
        alpha: Tensor,
        number_of_eigenvalues: int
) -> LinearOperator:
    """
    This is a helper function for computing the covariance matrix for the approximated RBF kernel using Mercer's
    expansion, which can be called without the need of creating a :class:`~famgpytorch.kernels.RBFKernelApprox`
    object.

    :param x1: First set of data.
    :param x2: Second set of data.
    :param lengthscale: The lengthscale parameter.
    :param alpha: The alpha parameter.
    :param number_of_eigenvalues: The number of eigenvalues n to use for approximation.
    :return: The resulting covariance matrix.
    """
    out_dtype = x1.dtype
    x1, x2 = x1.type(torch.float64),  x2.type(torch.float64)  # convert to double to improve numerical stability

    alpha_sq = alpha.pow(2)
    eta_sq = lengthscale.pow(-2).div(2)
    beta = eta_sq.mul(4).div(alpha_sq).add(1).pow(0.25)
    delta_sq = alpha_sq.div(2).mul(beta.pow(2).sub(1))

    # compute eigenvalues
    denominator = alpha_sq.add(delta_sq).add(eta_sq)
    eigenvalue_a = torch.sqrt(alpha_sq.div(denominator))
    eigenvalue_b = eta_sq.div(denominator)
    eigenvalues = torch.arange(number_of_eigenvalues, dtype=torch.float64, device=x1.device)
    eigenvalues = DiagLinearOperator(eigenvalue_a.mul(eigenvalue_b.pow(eigenvalues))[0, :])

    # define eigenfunctions
    def _eigenfunctions(x, n):
        # compute sqrt factor
        # computing the factorial of i would result in extremely large interim values, however, since we need to
        # calculate the reciprocal of the factorial, we make use of the natural log of the gamma function where
        # lgamma(i+1) = ln(i!) and e^(-ln(i!)) = 1 / i!
        range_ = torch.arange(n, dtype=torch.float64, device=x.device)
        sqrt_exp = torch.sqrt(beta).mul(torch.exp(-torch.lgamma(range_ + 1).div(2)-delta_sq.mul(x.pow(2))))
        if not sqrt_exp.all():
            # warn about zero
            warnings.warn("Interim results are zero. Try to reduce the number of eigenvalues.")

        # compute hermite polynomials
        hermites = ChebyshevHermitePolynomials.apply(alpha.mul(beta).mul(math.sqrt(2) * x), n)

        eigenfunctions = sqrt_exp.mul(hermites)

        if torch.isnan(eigenfunctions).any() or torch.isinf(eigenfunctions).any():
            # raise exception for nan or inf
            raise ValueError("Interim results too high. Try to reduce the number of eigenvalues.")

        return eigenfunctions

    eigenfunctions1 = to_linear_operator(_eigenfunctions(x1, number_of_eigenvalues))

    if torch.equal(x1, x2):
        eigenfunctions2 = eigenfunctions1
    else:
        eigenfunctions2 = to_linear_operator(_eigenfunctions(x2, number_of_eigenvalues))

    return eigenfunctions1.matmul(eigenvalues).matmul(eigenfunctions2.mT).type(out_dtype)