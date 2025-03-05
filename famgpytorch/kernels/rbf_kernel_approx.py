#!/usr/bin/env python3
from typing import Optional

import torch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.priors import Prior

#from ..functions import HermitePolynomial


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
        eigenvalues = torch.arange(self.number_of_eigenvalues, dtype=x1.dtype, device=x1.device)
        eigenvalues =  eigenvalue_a.mul(eigenvalue_b.pow(eigenvalues))

        # define eigenfunctions
        def _eigenfunctions(n, x):
            # compute sqrt factor
            # computing the factorial of i would result in an overflow for large i, however, since we need to calculate
            # the reciprocal of the factorial, we make use of the natural log of the gamma function where
            # lgamma(i+1) = ln(i!) and e^(-ln(i!)) = 1 / i!
            range_ = torch.arange(n, dtype=x.dtype, device=x.device)
            sqrt = torch.sqrt(beta.mul(2**(-range_)).mul(torch.exp(-torch.lgamma(range_ + 1))))

            # compute exp factor
            exp = torch.exp(-delta_sq.mul(x.pow(2)))

            # compute hermite polynomials
            # H_0(v) = 1 and H_1(v) = 2v with v = alpha * beta * x
            herm_inp_2 = self.alpha.mul(beta).mul(x * 2)
            hermites = torch.ones((x.size(0), n), dtype=x.dtype, device=x.device)
            hermites[:,1:2] = herm_inp_2
            # use recursive relation of hermite polynomials H_i(v) = 2v * H_{i-1}(v) - 2 * (i-1) * H_{i-2}(v)
            for i in range(2, n):
                last_hermites = hermites[:, i-2:i].clone()  # clone to avoid inplace operations (autograd needs this)
                hermites[:,i] = (
                    herm_inp_2.squeeze().mul(last_hermites[:,1]).sub(last_hermites[:,0].mul(2*(i-1)))
                )

            #hermites = HermitePolynomial.apply(self.alpha.mul(beta).mul(x), n)

            eigenfunctions = sqrt.mul(exp).mul(hermites)

            if torch.isnan(eigenfunctions).any() or torch.isinf(eigenfunctions).any():
                raise ValueError("Interim results too high. Try to reduce the number of eigenvalues.")

            return eigenfunctions

        eigenfunctions1 = _eigenfunctions(self.number_of_eigenvalues, x1)

        if torch.equal(x1, x2):
            eigenfunctions2 = eigenfunctions1
        else:
            eigenfunctions2 = _eigenfunctions(self.number_of_eigenvalues, x2)

        return (
            eigenfunctions1
            .mul(eigenvalues)
            .mm(eigenfunctions2.transpose(-2, -1))
        )