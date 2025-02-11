#!/usr/bin/env python3

from gpytorch.kernels import Kernel, RBFKernel


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

    .. math::
        \begin{align}
            \lambda_i &= \sqrt{ \frac{\alpha^2}{\alpha^2+\delta^2+\eta^2} }
            \left( \frac{\eta^2}{\alpha^2+\delta^2+\eta^2} \right)^i \\
            \phi_i(\mathbf{x}) &= \sqrt{\frac{\beta}{i!}} \exp \left( -\alpha^2\mathbf{x}^2 \right)
            H_i \left( \sqrt{2} \alpha \beta \mathbf{x} \right)
        \end{align}

    where

    * :math:`\eta = \frac{1}{\sqrt{2} \Theta}` with the lengthscale parameter :math:`\Theta`
    * :math:`\beta = (1 + (\frac{2\eta}{\alpha})^2 )^{\frac{1}{4}}`
    * :math:`\delta^2 = \frac{\alpha^2}{2} (\beta^2 - 1)`
    * :math:`\alpha` is a global scaling parameter
    * :math:`H_i` denotes the :math:`i\text{th}` Hermite polynomial

    .. _Fast Approximate Multioutput Gaussian Processes:
        https://doi.org/10.1109/MIS.2022.3169036
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag = False, last_dim_is_batch = False, **params):
        def eigenfunction(i, x):
            return

        def eigenvalue(i):
            return

        return RBFKernel.forward(self, x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)