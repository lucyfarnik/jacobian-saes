"""Metrics partially taken from this paper https://arxiv.org/pdf/0811.4706"""

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

type Jacobian = Float[Tensor, "batch seq_pos k2 k1"]


def l1(x: Jacobian):
    return x.abs().mean()


def l1_over_l2_norm_per_batch(x: Jacobian, eps: float = 1e-20):
    norm_inv = x.pow(2).mean().rsqrt() + eps
    return (x * norm_inv).abs().mean()


def l1_over_l2_norm_per_token(x: Jacobian, eps: float = 1e-20):
    norm_inv = x.pow(2).mean(dim=(-2, -1), keepdim=True).rsqrt() + eps
    return (x * norm_inv).abs().mean()


def tanh(x: Jacobian, scale: float = 0.8, scale_x: float = 10.0, power: float = 2.0):
    return (scale * torch.tanh((scale_x * x) ** power)).mean()


def scad(x: Jacobian, scale: float = 2.2, knot1: float = 0.5, knot2: float = 4.3):
    """Smoothly Clipped Absolute Deviation (https://www.mayo.edu/research/documents/scad-documentationpdf/DOC-10026887)"""
    x_abs = x.abs()
    if_small = knot1 * x_abs
    if_medium = -((x_abs**2 - 2 * knot2 * knot1 * x_abs + knot1**2) / (2 * (knot2 - 1)))
    if_large = 0.5 * (knot2 + 1) * knot1**2

    return (
        scale
        * torch.where(
            x_abs <= knot1,
            if_small,
            torch.where(x_abs <= knot1 * knot2, if_medium, if_large),
        ).mean()
    )


def log_x2(x: Jacobian, scale: float = 2.2, translate: float = 0.7, eps: float = 1e-20):
    return torch.clamp(scale * torch.log((x + eps) ** 2) + translate, min=0).mean()


def exp_neg_x(x: Jacobian, scale: float = 1.0, scale_x: float = 2.5):
    return (scale * (1 - torch.exp(-scale_x * x.abs()))).mean()


def _gini(x: Jacobian, eps: float = 1e-20):
    return torch.zeros_like(x)  #! TODO fix this (right now it OOMs for big batches)
    x_2d = rearrange(x, "... s k2 k1 -> (... s) (k2 k1)")
    x_dim = x_2d.size(-1)
    sum_diff = (x_2d.unsqueeze(2) - x_2d.unsqueeze(1)).abs().sum(dim=(1, 2))
    denom = 2 * (x_dim**2 - x_dim) * x_2d.abs().mean(dim=1) + eps
    return sum_diff / denom


def neg_gini(x: Jacobian, scale: float = 2.8, eps: float = 1e-20):
    return scale * (1 - _gini(x, eps).mean())


def _cap(x: Jacobian, cap: float = 0.5):
    return torch.clamp(x, min=-cap, max=cap)


def capped_l1(x: Jacobian, scale: float = 1.8, cap: float = 0.5):
    return scale * l1(_cap(x, cap))


def capped_l1_over_l2_norm_per_batch(
    x: Jacobian, scale: float = 0.9, cap: float = 0.5, eps: float = 1e-20
):
    return scale * l1_over_l2_norm_per_batch(_cap(x, cap), eps)


def capped_l1_over_l2_norm_per_token(
    x: Jacobian, scale: float = 0.9, cap: float = 0.5, eps: float = 1e-20
):
    return scale * l1_over_l2_norm_per_token(_cap(x, cap), eps)


def log_exp(x: Jacobian, scale: float = 1.3, power: float = 0.4, eps: float = 1e-20):
    return (scale * torch.log(1 + (x.abs() + eps) ** power)).mean()


def lp(x: Jacobian, p: float = 0.4, scale: float = 1.3, eps: float = 1e-20):
    return scale * ((x.abs() + eps) ** p).mean() ** (1 / p)


def _kurtosis(x: Jacobian, eps: float = 1e-20):
    x = x.flatten(start_dim=-2)
    return (x ** 4).mean(dim=-1) / ((x ** 2).mean(dim=-1) ** 2 + eps)


def kurtosis(x: Jacobian, scale: float = 1.0, eps: float = 1e-20):
    return scale * _kurtosis(x, eps).mean()


sparsity_metrics = {
    "l1": l1,
    "l1nb": l1_over_l2_norm_per_batch,
    "l1nt": l1_over_l2_norm_per_token,
    "tanh": tanh,
    "scad": scad,
    "log2": log_x2,
    "exp": exp_neg_x,
    # "gini": neg_gini,
    "cl1": capped_l1,
    "cl1nb": capped_l1_over_l2_norm_per_batch,
    "cl1nt": capped_l1_over_l2_norm_per_token,
    "loge": log_exp,
    "lp": lp,
    "l4": kurtosis,
}
