import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions import kl_divergence
import numpy as np
import pandas as pd
from SVGP import SVGP
from VAE_utils import *


class SPAPEAKLDVAE(nn.Module):
    def __init__(self, input_dim, z_dim, encoder_layers, encoder_dropout, 
                    fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, 
                    beta, dtype, device):
        super(SPAPEAKLDVAE, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points, dim=z_dim,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.input_dim = input_dim
        self.beta = beta        # beta controls the weight of reconstruction loss
        self.dtype = dtype
        self.z_dim = z_dim
        self.device = device
        self.encoder = DenseEncoder(input_dim=input_dim, hidden_dims=encoder_layers, output_dim=2*z_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = nn.Parameter(torch.empty((2*z_dim, input_dim)), requires_grad=True)
        nn.init.kaiming_normal_(self.decoder, a=math.sqrt(5))

        self.prior_mu = nn.Parameter(torch.zeros(z_dim), requires_grad=True)
        self.prior_var = nn.Parameter(torch.zeros(z_dim), requires_grad=True)

        self.BCE_loss = nn.BCELoss(reduction="none").to(self.device)
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def forward(self, x, y, num_samples=1):
        """
        Forward pass.

        Parameters:
        -----------
        x: mini-batch of positions.
        y: mini-batch of preprocessed counts.s
        """ 

        self.train()

        b = y.shape[0]
        qnet_mu, qnet_var = self.encoder(y)

        gp_mu = qnet_mu[:, 0:self.z_dim]
        gp_var = qnet_var[:, 0:self.z_dim]

        gaussian_mu = qnet_mu[:, self.z_dim:]
        gaussian_var = qnet_var[:, self.z_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.z_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l], l)
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l],
                                                                    noise=gp_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l, l=l)

            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            gp_p_m.append(gp_p_m_l)
            gp_p_v.append(gp_p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)

        inside_elbo = inside_elbo_recon - (b / self.svgp.N_train) * inside_elbo_kl

        gp_p_m = torch.stack(gp_p_m, dim=1)
        gp_p_v = torch.stack(gp_p_v, dim=1)

        # cross entropy term
        gp_ce_term = gauss_cross_entropy(gp_p_m, gp_p_v, gp_mu, gp_var)
        gp_ce_term = torch.sum(gp_ce_term)

        # KL term of GP prior
        gp_KL_term = gp_ce_term - inside_elbo

        # KL term of Gaussian prior
        gaussian_prior_dist = Normal(self.prior_mu, torch.exp(self.prior_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        gaussian_KL_term = kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = LogNormal(p_m, torch.sqrt(p_v))
        latent_samples = []
        mean_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        recon_loss = 0
        for f in latent_samples:
            mean_samples_ = torch.matmul(f, torch.exp(self.decoder))
            mean_samples_ = mean_samples_ / (1+mean_samples_) # scale [0, infinity) to [0, 1)

            recon_loss += self.BCE_loss(mean_samples_, y).sum()
        recon_loss = recon_loss / num_samples

        # ELBO
        elbo = recon_loss + self.beta * (gp_KL_term + gaussian_KL_term)

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, mean_samples, latent_samples


    def spatial_score(self, X, Y, batch_size=256, n_samples=25, peak_name=None):
        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        Y = torch.tensor(Y, dtype=self.dtype)

        latent_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            ybatch = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(ybatch)

            gp_mu = qnet_mu[:, 0:self.z_dim]
            gp_var = qnet_var[:, 0:self.z_dim]

            gaussian_mu = qnet_mu[:, self.z_dim:]
            gaussian_var = qnet_var[:, self.z_dim:]
            gp_p_m, gp_p_v = [], []
            for l in range(self.z_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l], l)
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)
            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = LogNormal(p_m, torch.sqrt(p_v))
            latent_samples_ = []
            for _ in range(n_samples):
                latent_samples_.append(latent_dist.sample())
            latent_samples_ = torch.stack(latent_samples_, dim=0)
            latent_samples_ = torch.mean(latent_samples_, dim=0)
            latent_samples.append(latent_samples_.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)

        diag_latent = torch.sum(latent_samples, dim=0)
        decoder_weight = self.decoder.data.cpu().detach()
        decoder_weight = torch.matmul(torch.diag(diag_latent), torch.exp(decoder_weight))

        spatial_dep_score = torch.sum(decoder_weight[:self.z_dim, :], dim=0)
        spatial_ind_score = torch.sum(decoder_weight[self.z_dim:, :], dim=0)
        spatial_score = (spatial_dep_score / (spatial_dep_score+spatial_ind_score)).numpy()
        non_spatial_score = (spatial_ind_score / (spatial_dep_score+spatial_ind_score)).numpy()

        res_dat = pd.DataFrame(data={'spatial_score': spatial_score, 'non_spatial_score': non_spatial_score}, 
                        index=peak_name)

        return res_dat


    def train_model(self, pos, counts, lr=0.001, weight_decay=0.001, batch_size=256, num_samples=1, 
            maxiter=2000, save_model=True, model_weights="model.pt", print_kernel_scale=True):
        """
        Model training.

        Parameters:
        -----------
        pos: array_like, shape (n_spots, 2)
            Location information.
        counts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        lr: float, defalut = 0.001
            Learning rate for the opitimizer.
        weight_decay: float, defalut = 0.001
            Weight decay for the opitimizer.
        maxiter: int, default = 2000
            Maximum number of iterations.
        model_weights: str
            File name to save the model weights.
        print_kernel_scale: bool
            Whether to print current kernel scale during training steps.
        """

        self.train()

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(counts, dtype=self.dtype))

        if counts.shape[0] > batch_size:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            gaussian_KL_term_val = 0
            num = 0
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                elbo, recon_loss, gp_KL_term, gaussian_KL_term, mean_samples, latent_samples = \
                    self.forward(x=x_batch, y=y_batch, num_samples=num_samples)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                gaussian_KL_term_val += gaussian_KL_term.item()

                num += x_batch.shape[0]

            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            gaussian_KL_term_val = gaussian_KL_term_val/num

            print('Training epoch {}, ELBO:{:.8f}, BCE loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f},'.format(epoch+1, elbo_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val))
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

        if save_model:
            torch.save(self.state_dict(), model_weights)

