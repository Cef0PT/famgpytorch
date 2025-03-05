import torch

class HermitePolynomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, herm_inp, n):
        # H_0(v) = 1 and H_1(v) = 2v with v = alpha * beta * x
        herm_inp_2 = herm_inp * 2
        hermites = torch.ones((herm_inp.size(0), n), dtype=herm_inp.dtype, device=herm_inp.device)
        hermites[:, 1:2] = herm_inp_2
        # use recursive relation of hermite polynomials H_{i}(v) = 2v * H_{i-1}(v) - 2 * (i-1) * H_{i-2}(v)
        for i in range(2, n):
            hermites[:, i] = (
                herm_inp_2.squeeze().mul(hermites[:, i-1]).sub(hermites[:, i-2].mul(2 * (i - 1)))
            )

        if any(ctx.needs_input_grad):
            # compute dH_i(v) / dv = 2i * H_{i-1}(v)
            range_ = torch.arange(n, dtype=herm_inp.dtype, device=herm_inp.device)
            d_output_d_input = torch.zeros((herm_inp.size()[0], n), dtype=herm_inp.dtype, device=herm_inp.device)
            d_output_d_input[:, 1:] = hermites[:, :-1]
            d_output_d_input = range_.mul(2).mul(d_output_d_input)
            # save for backward
            ctx.save_for_backward(d_output_d_input)

        return hermites

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.saved_tensors[0], None