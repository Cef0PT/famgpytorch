#!/usr/bin/env python3

import torch

class ChebyshevHermitePolynomials(torch.autograd.Function):
    """
    Computes the first :math:`n` Chebyshev-Hermite polynomials for a given input :math:`x` as described in
    `A digression on Hermite polynomials`_.

    .. _A digression on Hermite polynomials:
        https://doi.org/10.48550/arXiv.1901.01648
    """
    @staticmethod
    def forward(ctx, x, n):
        if n <= 0:
            raise ValueError('n must be a positive.')
        if not isinstance(n, int):
                raise TypeError("Expected int but got " + type(n).__name__)

        # He_0(x) = 1
        he = torch.ones((x.size(0), n), dtype=x.dtype, device=x.device)

        # He_1(x) = x
        he[:, 1:2] = x

        # use recursive relation of Chebyshev-Hermite polynomials He_{i}(x) = x * He_{i-1}(x) - (i-1) * He_{i-2}(x)
        for i in range(2, n):
            he[:, i] = (
                x.squeeze().mul(he[:, i-1]).sub(he[:, i-2].mul(i - 1))
            )

        if any(ctx.needs_input_grad):
            # compute dHe_i(x) / dx = i * He_{i-1}(x)
            range_ = torch.arange(n, dtype=x.dtype, device=x.device)
            d_he_d_x = torch.zeros((x.size(0), n), dtype=x.dtype, device=x.device)
            d_he_d_x[:, 1:] = he[:, :-1]
            d_he_d_x = range_.mul(d_he_d_x)
            # save for backward
            ctx.save_for_backward(d_he_d_x)

        return he

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.saved_tensors[0], None