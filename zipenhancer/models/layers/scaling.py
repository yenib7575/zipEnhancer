# ------ coding : utf-8 ------
# @FileName     : scaling.py
# @Author       : lxc
# @Time         : 2025/3/4 11:42


import logging
import random
from typing import Optional, Union

# import k2
import torch
import torch.nn as nn
from torch import Tensor


class PiecewiseLinear(object):
    """
    Piecewise linear function, from float to float, specified as nonempty list of (x,y) pairs with
    the x values in order.  x values <[initial x] or >[final x] are map to [initial y], [final y]
    respectively.
    """

    def __init__(self, *args):
        assert len(args) >= 1, len(args)
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [(float(x), float(y)) for x, y in args]
        for x, y in self.pairs:
            assert isinstance(x, (float, int)), type(x)
            assert isinstance(y, (float, int)), type(y)

        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0], (
                i,
                self.pairs[i],
                self.pairs[i + 1],
            )

    def __str__(self):
        # e.g. 'PiecewiseLinear((0., 10.), (100., 0.))'
        return f'PiecewiseLinear({str(self.pairs)[1:-1]})'

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (
                            next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            assert False


class ScheduledFloat(torch.nn.Module):
    """
    This object is a torch.nn.Module only because we want it to show up in [top_level module].modules();
    it does not have a working forward() function.  You are supposed to cast it to float, as
    in, float(parent_module.whatever), and use it as something like a dropout prob.

    It is a floating point value whose value changes depending on the batch count of the
    training loop.  It is a piecewise linear function where you specify the (x,y) pairs
    in sorted order on x; x corresponds to the batch index.  For batch-index values before the
    first x or after the last x, we just use the first or last y value.

    Example:
       self.dropout = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0.0)

    `default` is used when self.batch_count is not set or not in training mode or in
     torch.jit scripting mode.
    """

    def __init__(self, *args, default: float = 0.0):
        super().__init__()
        # self.batch_count and self.name will be written to in the training loop.
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return (
            f'batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}'
        )

    def __float__(self):
        batch_count = self.batch_count
        if (batch_count is None or not self.training
                or torch.jit.is_scripting() or torch.jit.is_tracing()):
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:
                logging.info(
                    f'ScheduledFloat: name={self.name}, batch_count={self.batch_count}, ans={ans}'
                )
            return ans

FloatLike = Union[float, ScheduledFloat]


def softmax(x: Tensor, dim: int):
    return x.softmax(dim=dim)


class BiasNormFunction(torch.autograd.Function):
    # This computes:
    #   scales = (torch.mean((x - bias) ** 2, keepdim=True)) ** -0.5 * log_scale.exp()
    #   return x * scales
    # (after unsqueezing the bias), but it does it in a memory-efficient way so that
    # it can just store the returned value (chances are, this will also be needed for
    # some other reason, related to the next operation, so we can save memory).
    @staticmethod
    def forward(
            ctx,
            x: Tensor,
            bias: Tensor,
            log_scale: Tensor,
            channel_dim: int,
            store_output_for_backprop: bool,
    ) -> Tensor:
        assert bias.ndim == 1
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        _x_bias_square = torch.mean(
            (x - bias) ** 2, dim=channel_dim, keepdim=True)
        scales = (_x_bias_square ** -0.5) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            # recompute scales from x, bias and log_scale.
            _x_bias_square = torch.mean(
                (x - bias) ** 2, dim=ctx.channel_dim, keepdim=True)
            scales = (_x_bias_square ** -0.5) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    Instead, we give the BiasNorm a trainable bias that it can use when
    computing the scale for normalization.  We also give it a (scalar)
    trainable scale on the output.


    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
      log_scale: the initial log-scale that we multiply the output by; this
         is learnable.
      log_scale_min: FloatLike, minimum allowed value of log_scale
      log_scale_max: FloatLike, maximum allowed value of log_scale
      store_output_for_backprop: only possibly affects memory use; recommend
         to set to True if you think the output of this module is more likely
         than the input of this module to be required to be stored for the
         backprop.
    """

    def __init__(
            self,
            num_channels: int,
            channel_dim: int = -1,  # CAUTION: see documentation.
            log_scale: float = 1.0,
            log_scale_min: float = -1.5,
            log_scale_max: float = 1.5,
            store_output_for_backprop: bool = False,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(
            torch.empty(num_channels).normal_(mean=0, std=1e-4))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            _x_bias_square = torch.mean(
                (x - bias) ** 2, dim=channel_dim, keepdim=True)
            scales = (_x_bias_square ** -0.5) * self.log_scale.exp()
            return x * scales

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )

        return BiasNormFunction.apply(x, self.bias, log_scale,
                                      self.channel_dim,
                                      self.store_output_for_backprop)


def ScaledLinear(*args, initial_scale: float = 1.0, **kwargs) -> nn.Linear:
    """
    Behaves like a constructor of a modified version of nn.Linear
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale,
                                   0.1 * initial_scale)
    return ans


def limit_param_value(x: Tensor,
                      min: float,
                      max: float,
                      prob: float = 0.6,
                      training: bool = True):
    # Inference-only: training is always False, just return x unchanged.
    return x


def _no_op(x: Tensor) -> Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        # a no-op function that will have a node in the autograd graph,
        # to avoid certain bugs relating to backward hooks
        return x.chunk(1, dim=-1)[0]


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return _no_op(x)


# Dropout2 is just like normal dropout, except it supports schedules on the dropout rates.
class Dropout2(nn.Module):

    def __init__(self, p: FloatLike):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(
            x, p=float(self.p), training=self.training)


# simple version of SwooshL that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshLForward(x: Tensor):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float('inf'), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035


# simple version of SwooshR that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshRForward(x: Tensor):
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float('inf'), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.313261687


class ActivationDropoutAndLinear(torch.nn.Module):
    """
     This merges an activation function followed by dropout and then a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwooshL and dropout_shared_dim != None, this will be
     equivalent to:
       nn.Sequential(SwooshL(),
                     Dropout3(dropout_p, shared_dim=dropout_shared_dim),
                     ScaledLinear(in_channels, out_channels, bias=bias,
                                  initial_scale=initial_scale))
    If dropout_shared_dim is None, the dropout would be equivalent to
    Dropout2(dropout_p).  Note: Dropout3 will be more memory efficient as the dropout
    mask is smaller.

     Args:
        in_channels: number of input channels, e.g. 256
        out_channels: number of output channels, e.g. 256
        bias: if true, have a bias
        activation: the activation function, for now just support SwooshL.
        dropout_p: the dropout probability or schedule (happens after nonlinearity).
        dropout_shared_dim: the dimension, if any, across which the dropout mask is
             shared (e.g. the time dimension).  If None, this may be less memory
             efficient if there are modules before this one that cache the input
             for their backprop (e.g. Balancer or Whiten).
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            activation: str = 'SwooshL',
            dropout_p: FloatLike = 0.0,
            dropout_shared_dim: Optional[int] = -1,
            initial_scale: float = 1.0,
    ):
        super().__init__()
        # create a temporary module of nn.Linear that we'll steal the
        # weights and bias from
        linear_module = ScaledLinear(
            in_channels, out_channels, bias=bias, initial_scale=initial_scale)

        self.weight = linear_module.weight
        # register_parameter properly handles making it a parameter when l.bias
        # is None. I think there is some reason for doing it this way rather
        # than just setting it to None but I don't know what it is, maybe
        # something to do with exporting the module..
        self.register_parameter('bias', linear_module.bias)

        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor):
        if self.activation == 'SwooshL':
            x = SwooshLForward(x)
        elif self.activation == 'SwooshR':
            x = SwooshRForward(x)
        else:
            assert False, self.activation
        if self.training:
            x = torch.nn.functional.dropout(x, p=float(self.dropout_p), training=True)
        return torch.nn.functional.linear(x, self.weight, self.bias)


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    """

    :param x: (b, c, t, f)
    :param num_channels:
    :return: x: (b, num_channels, t, f)
    """
    if num_channels <= x.shape[1]:
        return x[:, :num_channels, :, :]
    else:
        shape = list(x.shape)
        shape[1] = num_channels - shape[1]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return torch.cat((x, zeros), dim=1)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
