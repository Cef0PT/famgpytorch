#!/usr/bin/env python3

import torch

class HermitePolynomials(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n):
        # H_0(x) = 1 and H_1(x) = 2x
        x_2 = x * 2
        hermites = torch.ones((x.size(0), n), dtype=x.dtype, device=x.device)
        hermites[:, 1:2] = x_2
        # use recursive relation of hermite polynomials H_{i}(x) = 2x * H_{i-1}(x) - 2 * (i-1) * H_{i-2}(x)
        for i in range(2, n):
            hermites[:, i] = (
                x_2.squeeze().mul(hermites[:, i-1]).sub(hermites[:, i-2].mul(2 * (i - 1)))
            )

        if any(ctx.needs_input_grad):
            # compute dH_i(x) / dx = 2i * H_{i-1}(x)
            range_ = torch.arange(n, dtype=x.dtype, device=x.device)
            d_h_d_x = torch.zeros((x.size(0), n), dtype=x.dtype, device=x.device)
            d_h_d_x[:, 1:] = hermites[:, :-1]
            d_h_d_x = range_.mul(2).mul(d_h_d_x)
            # save for backward
            ctx.save_for_backward(d_h_d_x)

        return hermites

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.saved_tensors[0], None