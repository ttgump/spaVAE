import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


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
