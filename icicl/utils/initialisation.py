"""https://github.com/YannDubs/Neural-Process-Family/blob/master/npf/utils/initialization.py"""
from typing import Optional, Union

from torch import nn

__all__ = ["weights_init"]

CONV = [nn.Conv1d, nn.Conv2d, nn.Conv3d]


def weights_init(module: nn.Module, **kwargs):
    """Initialize a module and all its descendents.

    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    module.is_resetted = True
    for m in module.modules():
        try:
            if hasattr(module, "reset_parameters") and module.is_resetted:
                continue
        except AttributeError:
            pass

        if any(isinstance(m, conv) for conv in CONV):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", **kwargs)
        elif isinstance(m, nn.Linear):
            linear_init(m, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def get_activation_name(activation: Union[nn.Module, str]) -> str:
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return v

    raise ValueError(f"Unkown given activation type : {activation}")


def get_gain(activation: nn.Module) -> float:
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(
    module: nn.Module, activation: Optional[Union[str, nn.Module]] = "relu"
):
    """Initialize a linear layer.

    Parameters
    ----------
    module : nn.Module
       module to initialize.

    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight

    if module.bias is not None:
        module.bias.data.zero_()

    if activation is None:
        nn.init.xavier_uniform_(x)
        return

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
        return
    if activation_name == "relu":
        nn.init.kaiming_uniform_(x, nonlinearity="relu")
        return
    if activation_name in ["sigmoid", "tanh"]:
        nn.init.xavier_uniform_(x, gain=get_gain(activation))
        return
