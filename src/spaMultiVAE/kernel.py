import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math


def sq_dist(x1, x2, x1_eq_x2=False):
    """Equivalent to the square of `torch.cdist` with p=2."""
    # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

    # Zero out negative values
    return res.clamp_min_(0)


def dist(x1, x2, x1_eq_x2=False):
    """
    Equivalent to `torch.cdist` with p=2, but clamps the minimum element to 1e-15.
    """
    if not x1_eq_x2:
        res = torch.cdist(x1, x2)
        return res.clamp_min(1e-15)
    res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
    return res.clamp_min_(1e-30).sqrt_()


class MaternKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, nu=1.5, dtype=torch.float32, device="cpu"):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernel, self).__init__()
        self.nu = nu
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        mean = x.mean(dim=-2, keepdim=True)

        x_ = (x - mean).div(self.scale)
        y_ = (y - mean).div(self.scale)
        distance = dist(x_, y_)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return constant_component * exp_component

    def forward_diag(self, x, y):
        mean = x.mean(dim=-2, keepdim=True)

        x_ = (x - mean).div(self.scale)
        y_ = (y - mean).div(self.scale)
        distance = ((x_-y_)**2).sum(dim=1)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return constant_component * exp_component


class MultiMaternKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, nu=1.5, dim=1, dtype=torch.float32, device="cpu"):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MultiMaternKernel, self).__init__()
        self.nu = nu
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(np.repeat(scale, dim), dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y, l):
        mean = x.mean(dim=-2, keepdim=True)

        x_ = (x - mean).div(self.scale[l])
        y_ = (y - mean).div(self.scale[l])
        distance = dist(x_, y_)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return constant_component * exp_component

    def forward_diag(self, x, y, l):
        mean = x.mean(dim=-2, keepdim=True)

        x_ = (x - mean).div(self.scale[l])
        y_ = (y - mean).div(self.scale[l])
        distance = ((x_-y_)**2).sum(dim=1)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        return constant_component * exp_component


class EQKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(EQKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = torch.exp(-d/self.scale)
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = torch.exp(-d/self.scale)
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)


class MultiEQKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dim=1, dtype=torch.float32, device="cpu"):
        super(MultiEQKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(np.repeat(scale, dim), dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y, l):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = torch.exp(-d/self.scale[l])
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale[l]), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y, l):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = torch.exp(-d/self.scale[l])
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale[l]), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)



class CauchyKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(CauchyKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = 1/(1+d/self.scale)
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = 1/(1+d/self.scale)
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)


class MultiCauchyKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dim=1, dtype=torch.float32, device="cpu"):
        super(MultiCauchyKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(np.repeat(scale, dim), dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(np.repeat(scale, dim), dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y, l):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            res = 1/(1+d/self.scale[l])
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale[l]), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y, l):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            res = 1/(1+d/self.scale[l])
        else:
            res = 1/(1+d/torch.clamp(F.softplus(self.scale[l]), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)


class LaplacianKernel(nn.Module):
    def __init__(self, scale=1., fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(LaplacianKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor([scale], dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor([scale], dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y):
        d = (x.unsqueeze(1) - y.unsqueeze(0).repeat(x.shape[0], 1, 1)).abs().sum(dim=-1)
        if self.fixed_scale:
            res = torch.exp(-d/self.scale)
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def forward_diag(self, x, y):
        d = ((x-y).abs()).sum(dim=1)
        if self.fixed_scale:
            res = torch.exp(-d/self.scale)
        else:
            res = torch.exp(-d/torch.clamp(F.softplus(self.scale), min=1e-10, max=1e4))
        return res

    def print_scale(self):
        print(self.scale)


class BatchedCauchyKernel(nn.Module):
    def __init__(self, scale=[], fixed_scale=True, dtype=torch.float32, device="cpu"):
        super(BatchedCauchyKernel, self).__init__()
        self.fixed_scale = fixed_scale
        if fixed_scale:
            self.scale = torch.tensor(scale, dtype=dtype).to(device)
        else:
            self.scale = nn.Parameter(torch.tensor(scale, dtype=dtype).to(device), requires_grad=True)

    def forward(self, x, y, sample_x, sample_y):
        sq_norms_x = torch.sum(x**2, dim=-1, keepdim=True)
        sq_norms_y = torch.transpose(torch.sum(y**2, dim=-1, keepdim=True), -2, -1)
        dotprods = torch.matmul(x, torch.transpose(y, -2, -1))
        d = sq_norms_x + sq_norms_y - 2. * dotprods
        if self.fixed_scale:
            scale_x = torch.matmul(sample_x, self.scale.unsqueeze(dim=1))
            scale_y = torch.matmul(sample_y, self.scale.unsqueeze(dim=1))
            scale_xy = torch.sqrt(torch.matmul(scale_x, scale_y.T))
            res = 1/(1+d/scale_xy)
        else:
            scale_x = torch.clamp(F.softplus(torch.matmul(sample_x, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4)
            scale_y = torch.clamp(F.softplus(torch.matmul(sample_y, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4)
            scale_xy = torch.sqrt(torch.matmul(scale_x, scale_y.T))
            res = 1/(1+d/scale_xy)
        return res

    def forward_diag(self, x, y, sample_x, sample_y):
        d = ((x-y)**2).sum(dim=1)
        if self.fixed_scale:
            scale_x = torch.matmul(sample_x, self.scale.unsqueeze(dim=1)).squeeze()
            scale_y = torch.matmul(sample_y, self.scale.unsqueeze(dim=1)).squeeze()
            scale_xy = torch.sqrt(scale_x * scale_y)
            res = 1/(1+d/scale_xy)
        else:
            scale_x = torch.clamp(F.softplus(torch.matmul(sample_x, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4).squeeze()
            scale_y = torch.clamp(F.softplus(torch.matmul(sample_y, self.scale.unsqueeze(dim=1))), min=1e-10, max=1e4).squeeze()
            scale_xy = torch.sqrt(scale_x * scale_y)
            res = 1/(1+d/scale_xy)
        return res

    def print_scale(self):
        print(self.scale)


class SampleKernel(nn.Module):
    def __init__(self):
        super(SampleKernel, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y.T)

    def forward_diag(self, x, y):
        return (x*y).sum(dim=1)
