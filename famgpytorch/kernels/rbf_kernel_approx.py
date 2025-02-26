#!/usr/bin/env python3
import math
from typing import Optional

import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.priors import Prior


class RBFKernelApprox(Kernel):
    r"""
    Computes an approximate covariance matrix based on the Mercer expansion (eigen decomposition)
    of the RBF (squared exponential) kernel based on `Fast Approximate Multioutput Gaussian Processes`_.

    Mercer's theorem states the existence of an orthonormal basis consisting of eigenfunctions
    :math:`\phi_i(\mathbf{x})` and nonincreasing eigenvalues :math:`\lambda_i`

    .. math::
        \begin{equation}
            k_{\text{RBF}}(\mathbf{x}, \mathbf{x'}) =
            \sum^{\infty}_{i=1}\lambda_i\phi_i(\mathbf{x})\phi_i(\mathbf{x'})
        \end{equation}

    and an approximate covariance matrix can be computed by utilizing only :math:`n` eigenvalues such that

    .. math::
        \begin{equation}
            K_{\mathbf{XX'}} \approx \Phi_{\mathbf{X}}\Lambda\Phi_{\mathbf{X'}}^\top
        \end{equation}

    where

    * :math:`\Phi_{\mathbf{X}i,j} = \phi_j(\mathbf{x_i})|j \in 1, 2, \ldots, n`
    * :math:`\Lambda` is a diagonal matrix of the eigenvalues :math:`[\lambda_1,\lambda_2,\ldots,\lambda_n]`

    The Mercer expansion (eigen decomposition) of the RBF kernel (https://doi.org/10.1137/110824784) is given by

    TODO: correct eigenfunction equation

    .. math::
        \begin{align}
            \lambda_i &= \sqrt{ \frac{\alpha^2}{\alpha^2+\delta^2+\eta^2} }
            \left( \frac{\eta^2}{\alpha^2+\delta^2+\eta^2} \right)^i \\
            \phi_i(\mathbf{x}) &= \sqrt{\frac{\beta}{i!}} \exp \left( -\alpha^2\mathbf{x}^2 \right)
            H_i \left( \sqrt{2} \alpha \beta \mathbf{x} \right)
        \end{align}

    where

    * :math:`\eta^2=\frac{1}{2}\Theta^{-2}` with the lengthscale parameter :math:`\Theta`
    * :math:`\beta = \left( 1 + 4 \frac{\eta^2}{\alpha^2} \right)^{\frac{1}{4}}`
    * :math:`\delta^2 = \frac{\alpha^2}{2} (\beta^2 - 1)`
    * :math:`\alpha` is a global scaling parameter
    * :math:`H_i` denotes the :math:`i\text{th}` Hermite polynomial

    .. _Fast Approximate Multioutput Gaussian Processes:
        https://doi.org/10.1109/MIS.2022.3169036
    """

    has_lengthscale = True

    ## noinspection PyProtectedMember
    def __init__(
            self,
            number_of_eigenvalues: Optional[int]=15,
            alpha_prior: Optional[Prior]=None,
            alpha_constraint: Optional[Interval]=None,
            **kwargs
    ):
        super(RBFKernelApprox, self).__init__(**kwargs)

        if number_of_eigenvalues < 1:
            raise ValueError("number_of_eigenvalues must be >= 1")
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


    def forward(self, x1, x2, diag = False, last_dim_is_batch = False, **params):
        if diag:
            if torch.equal(x1, x2):
                return torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                raise NotImplementedError("Approximate RBF Kernel can't handle diag for x1 not equal to x2.")

        alpha_sq = self.alpha.pow(2)
        eta_sq = self.lengthscale.pow(-2).div(2)
        beta = eta_sq.mul(4).div(alpha_sq).add(1).pow(0.25)
        delta_sq = alpha_sq.div(2).mul(beta.pow(2).sub(1))

        # compute eigenvalues
        denominator = alpha_sq.add(delta_sq).add(eta_sq)
        eigenvalue_a = torch.sqrt(alpha_sq.div(denominator))
        eigenvalue_b = eta_sq.div(denominator)
        eigenvalues = torch.empty(0, dtype=x1.dtype, device=x1.device)
        for i in range(self.number_of_eigenvalues):
            eigenvalues = torch.cat((eigenvalues, eigenvalue_a.mul(eigenvalue_b.pow(i))), 1)
        eigenvalues = torch.diag(eigenvalues.squeeze())

        # define eigenfunctions
        def _eigenfunctions(n, x):
            herm_inp = self.alpha.mul(beta).mul(x)
            hermites = torch.cat(
                (
                    torch.zeros(herm_inp.size(), dtype=x1.dtype, device=x1.device),
                    torch.ones(herm_inp.size(), dtype=x1.dtype, device=x1.device)
                ),
                1
            )
            def _next_hermite(j, v):
                # use recursive relation of hermite polynomials H_{j}(x) = 2x * H_{j-1}(x) - 2 * (j-1) * H_{j-2}(x)
                r = v.mul(2).mul(hermites[:,1].unsqueeze(1)).sub(hermites[:,0].unsqueeze(1).mul(2*(j-1)))
                return r

            exp = torch.exp(delta_sq.mul(x.pow(2)).neg())
            func_values = torch.empty((0,), dtype=x1.dtype, device=x1.device)
            for i in range(n):
                if i == 0:
                    # H_0(x) = 1
                    next_hermite = torch.ones(herm_inp.size(), dtype=x1.dtype, device=x1.device)
                else:
                    next_hermite = _next_hermite(i, herm_inp)
                    # save the last two hermite polynomials
                    hermites = torch.cat((hermites, next_hermite), 1).split((1, 2), 1)[1]

                # compute eigenfunction values
                # computing the factorial of i would result in an overflow for large i
                # since we need to calculate the reciprocal of the factorial, we make use of the natural log
                # of the gamma function where lgamma(i+1) = ln(i!) and e^(-ln(i!)) = 1 / i!
                value = (
                    torch.sqrt(beta * 2**-i * math.exp(-math.lgamma(i+1)))
                    .mul(exp)
                    .mul(next_hermite)
                )

                if torch.isnan(value).any():
                    raise ValueError("NaN values detected. Try to reduce the number of eigenvalues.")

                func_values = torch.cat((func_values, value), 1)

            return func_values


        return (
            _eigenfunctions(self.number_of_eigenvalues, x1)
            .matmul(eigenvalues)
            .matmul(_eigenfunctions(self.number_of_eigenvalues, x2).transpose(-2, -1))
        )