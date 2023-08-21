import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from kernel import CauchyKernel


def _add_diagonal_jitter(matrix, jitter=1e-8):
    Eye = torch.eye(matrix.size(-1), device=matrix.device).expand(matrix.shape)
    return matrix + jitter * Eye


class SVGP(nn.Module):
    def __init__(self, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, jitter, N_train, dtype, device):
        super(SVGP, self).__init__()
        self.N_train = N_train
        self.jitter = jitter
        self.dtype = dtype
        self.device = device

        # inducing points
        if fixed_inducing_points:
            self.inducing_index_points = torch.tensor(initial_inducing_points, dtype=dtype).to(device)
        else:
            self.inducing_index_points = nn.Parameter(torch.tensor(initial_inducing_points, dtype=dtype).to(device), requires_grad=True)

        # length scale of the kernel
        self.kernel = CauchyKernel(scale=kernel_scale, fixed_scale=fixed_gp_params, dtype=dtype, device=device).to(device)

    def kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y).
        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        if diag_only:
            matrix = self.kernel.forward_diag(x, y)
        else:
            matrix = self.kernel(x, y)
        return matrix

    def variational_loss(self, x, y, noise, mu_hat, A_hat):
        """
        Computes L_H for the data in the current batch.
        :param x: auxiliary data for current batch (batch, 1 + 1 + M)
        :param y: mean vector for current latent channel, output of the encoder network (batch, 1)
        :param noise: variance vector for current latent channel, output of the encoder network (batch, 1)
        :param mu_hat:
        :param A_hat:
        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (1,)
        """
        b = x.shape[0]
        m = self.inducing_index_points.shape[0]

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points) # (m,m)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter)) # (m,m)

        K_nn = self.kernel_matrix(x, x, x_inducing=False, y_inducing=False, diag_only=True) # (b)

        K_nm = self.kernel_matrix(x, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mn = torch.transpose(K_nm, 0, 1)

#        S = A_hat

        # KL term
        mean_vector = torch.matmul(K_nm, torch.matmul(K_mm_inv, mu_hat))

        K_mm_chol = torch.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
        S_chol = torch.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))
        K_mm_log_det = 2 * torch.sum(torch.log(torch.diagonal(K_mm_chol)))
        S_log_det = 2 * torch.sum(torch.log(torch.diagonal(S_chol)))

        KL_term = 0.5 * (K_mm_log_det - S_log_det - m +
                             torch.trace(torch.matmul(K_mm_inv, A_hat)) +
                             torch.sum(mu_hat * torch.matmul(K_mm_inv, mu_hat)))

        # diag(K_tilde), (b, )
        precision = 1 / noise

        K_tilde_terms = precision * (K_nn - torch.diagonal(torch.matmul(K_nm, torch.matmul(K_mm_inv, K_mn))))

        # k_i \cdot k_i^T, (b, m, m)
        lambda_mat = torch.matmul(K_nm.unsqueeze(2), torch.transpose(K_nm.unsqueeze(2), 1, 2))

        # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
        lambda_mat = torch.matmul(K_mm_inv, torch.matmul(lambda_mat, K_mm_inv))

        # Trace terms, (b,)
        trace_terms = precision * torch.einsum('bii->b', torch.matmul(A_hat, lambda_mat))

        # L_3 sum part, (1,)
        L_3_sum_term = -0.5 * (torch.sum(K_tilde_terms) + torch.sum(trace_terms) +
                                torch.sum(torch.log(noise)) + b * np.log(2 * np.pi) +
                                torch.sum(precision * (y - mean_vector) ** 2))

        return L_3_sum_term, KL_term

    def approximate_posterior_params(self, index_points_test, index_points_train=None, y=None, noise=None):
        """
        Computes parameters of q_S.
        :param index_points_test: X_*
        :param index_points_train: X_Train
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP
        :return: posterior mean at index points,
                 (diagonal of) posterior covariance matrix at index points
        """
        b = index_points_train.shape[0]

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points) # (m,m)
        K_mm_inv = torch.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter)) # (m,m)

        K_xx = self.kernel_matrix(index_points_test, index_points_test, x_inducing=False,
                                  y_inducing=False, diag_only=True)  # (x)
        K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points, x_inducing=False)  # (x, m)
        K_mx = torch.transpose(K_xm, 0, 1)  # (m, x)

        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points, x_inducing=False)  # (N, m)
        K_mn = torch.transpose(K_nm, 0, 1)  # (m, N)

        sigma_l = K_mm + (self.N_train / b) * torch.matmul(K_mn, K_nm / noise[:,None])
        sigma_l_inv = torch.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))
        mean_vector = (self.N_train / b) * torch.matmul(K_xm, torch.matmul(sigma_l_inv, torch.matmul(K_mn, y/noise)))

        K_xm_Sigma_l_K_mx = torch.matmul(K_xm, torch.matmul(sigma_l_inv, K_mx))
        B = K_xx + torch.diagonal(-torch.matmul(K_xm, torch.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_l_K_mx)

        mu_hat = (self.N_train / b) * torch.matmul(torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mn)), y / noise)
        A_hat = torch.matmul(K_mm, torch.matmul(sigma_l_inv, K_mm))

        return mean_vector, B, mu_hat, A_hat
