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


def _dim_helper(tens):
    while tens.dim() < 2:
        tens = tens.unsqueeze(-1)
    return tens

def approx_rbf_covariance(
        x1: Tensor,
        x2: Tensor,
        lengthscale: Tensor,
        alpha: Tensor,
        number_of_eigenvalues: int,
        diff_order_x1=0,
        diff_order_x2=0
) -> LinearOperator:
    """
    This is a helper function for computing the covariance matrix for the approximated RBF kernel using Mercer's
    expansion, which can be called without the need of creating a :class:`~famgpytorch.kernels.RBFKernelApprox`
    object.
    Additionally, can compute the k-th derivative of the kernel w.r.t to x1 and x2.

    :param x1: First set of data.
    :param x2: Second set of data.
    :param lengthscale: The lengthscale parameter.
    :param alpha: The alpha parameter.
    :param number_of_eigenvalues: The number of eigenvalues n to use for approximation.
    :param diff_order_x1: The number of times the kernel is differentiated w.r.t to x1.
    :param diff_order_x2: The number of times the kernel is differentiated w.r.t to x2.
    :return: The resulting covariance matrix.
    """
    # get inputs in desired dimensions
    x1, x2, lengthscale, alpha = tuple(_dim_helper(tens) for tens in (x1, x2, lengthscale, alpha))

    # save input dtype and convert to double to improve numerical stability
    out_dtype = x1.dtype
    x1, x2 = x1.type(torch.float64), x2.type(torch.float64)

    alpha_sq = alpha ** 2
    eta_sq = lengthscale ** (-2) / 2
    beta = (4 * eta_sq / alpha_sq + 1) ** 0.25
    delta_sq = (beta ** 2 - 1) * alpha_sq / 2

    # compute eigenvalues
    denominator = alpha_sq + delta_sq + eta_sq
    eigenvalue_a = torch.sqrt(alpha_sq / denominator)
    eigenvalue_b = eta_sq / denominator
    eigenvalues = torch.arange(number_of_eigenvalues, dtype=torch.float64, device=x1.device)
    eigenvalues = DiagLinearOperator((eigenvalue_a * eigenvalue_b ** eigenvalues).squeeze(0))

    # define eigenfunctions
    def _eigenfunctions(x, n, k=0):
        # compute sqrt factor
        # computing the factorial of i would result in extremely large interim values, however, since we need to
        # calculate the reciprocal of the factorial, we make use of the natural log of the gamma function where
        # lgamma(i+1) = ln(i!) and e^(-ln(i!)) = 1 / i!
        i = torch.arange(n, dtype=torch.float64, device=x.device)
        sqrt_exp = beta.sqrt() * torch.exp(-torch.lgamma(i + 1) / 2 - delta_sq * x ** 2)
        if not sqrt_exp.all():
            # warn about zero
            warnings.warn("Interim results are zero due to numerical stability issues. "
                          "Try to reduce the number of eigenvalues.")

        # compute hermite polynomials
        hermites = ChebyshevHermitePolynomials.apply(math.sqrt(2) * alpha * beta * x, n)

        if k == 0:
            eigenfunctions = sqrt_exp * hermites

        else:
            # computing the k-th derivative by...
            # create summation index j
            j = torch.arange(k + 1, dtype=torch.float64, device=x.device).unsqueeze(-1)

            # compute everything independent of x
            k_tens = torch.tensor(k)
            binoms = torch.exp(
                torch.lgamma(i + 1) -
                torch.lgamma(i - j + 1) +
                torch.lgamma(k_tens + 1) -
                torch.lgamma(j + 1) -
                torch.lgamma(k_tens - j + 1)
            )
            delta = delta_sq.sqrt()
            factors_ = (alpha * beta) ** j * (-delta) ** (k_tens - j) * binoms

            # compute everything independent of i
            hermites_kj = ChebyshevHermitePolynomials.apply(math.sqrt(2) * delta * x, k + 1)

            # sum over j
            sum_ = hermites * hermites_kj[:, -1:] * factors_[0, :]
            for j_idx in range(1, k + 1):
                hermites_ij = torch.zeros_like(hermites, dtype=torch.float64, device=x.device)
                hermites_ij[:, j_idx:] = hermites[:, :-j_idx]
                sum_ += hermites_ij * hermites_kj[:, -(j_idx + 1):-j_idx] * factors_[j_idx, :]

            # compute final product of independent stuff and sum
            eigenfunctions = math.sqrt(2) ** k * sqrt_exp * sum_

        if torch.isnan(eigenfunctions).any() or torch.isinf(eigenfunctions).any():
            # raise exception for nan or inf
            raise ValueError("Interim results too high due to numerical stability issues. "
                             "Try to reduce the number of eigenvalues.")

        return to_linear_operator(eigenfunctions)

    eigenfunctions1 = _eigenfunctions(x1, number_of_eigenvalues, diff_order_x1)

    if torch.equal(x1, x2) and diff_order_x1 == diff_order_x2:
        eigenfunctions2 = eigenfunctions1
    else:
        eigenfunctions2 = _eigenfunctions(x2, number_of_eigenvalues, diff_order_x2)

    return (eigenfunctions1 @ eigenvalues @ eigenfunctions2.mT).type(out_dtype)