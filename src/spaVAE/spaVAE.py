import math
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
import pandas as pd
from SVGP import SVGP
from VAE_utils import *


class SPAVAE(nn.Module):
    def __init__(self, input_dim, z_dim, encoder_layers, decoder_layers, noise, encoder_dropout, decoder_dropout, 
                    fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, 
                    beta, dtype, device):
        super(SPAVAE, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.input_dim = input_dim
        self.beta = beta        # beta controls the weight of reconstruction loss
        self.dtype = dtype
        self.z_dim = z_dim
        self.noise = noise      # intensity of random noise
        self.device = device
        self.encoder = DenseEncoder(input_dim=input_dim, hidden_dims=encoder_layers, output_dim=z_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = buildNetwork([z_dim]+decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), MeanAct())
        else:
            self.dec_mean = nn.Sequential(nn.Linear(z_dim, input_dim), MeanAct())
        self.dec_disp = nn.Parameter(torch.randn(self.input_dim), requires_grad=True)       # trainable dispersion parameter for NB loss

        self.NB_loss = NBLoss().to(self.device)
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def forward(self, x, y, raw_y, size_factors, num_samples=1):
        """
        Forward pass.

        Parameters:
        -----------
        x: mini-batch of positions.
        y: mini-batch of preprocessed counts.
        raw_y: mini-batch of raw counts.
        size_factor: mini-batch of size factors.
        num_samples: number of samplings of the posterior distribution of latent embedding.

        raw_y and size_factor are used for NB likelihood.
        """ 

        self.train()
        b = y.shape[0]
        qnet_mu, qnet_var = self.encoder(y)

        inside_elbo_recon, inside_elbo_kl = [], []
        p_m, p_v = [], []
        for l in range(self.z_dim):
            p_m_l, p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    qnet_mu[:, l], qnet_var[:, l])
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=qnet_mu[:, l],
                                                                    noise=qnet_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l)

            inside_elbo_recon.append(inside_elbo_recon_l)
            inside_elbo_kl.append(inside_elbo_kl_l)
            p_m.append(p_m_l)
            p_v.append(p_v_l)

        inside_elbo_recon = torch.stack(inside_elbo_recon, dim=-1)
        inside_elbo_kl = torch.stack(inside_elbo_kl, dim=-1)
        inside_elbo_recon = torch.sum(inside_elbo_recon)
        inside_elbo_kl = torch.sum(inside_elbo_kl)

        inside_elbo = inside_elbo_recon - (b / self.svgp.N_train) * inside_elbo_kl

        p_m = torch.stack(p_m, dim=1)
        p_v = torch.stack(p_v, dim=1)

        # cross entropy term
        ce_term = gauss_cross_entropy(p_m, p_v, qnet_mu, qnet_var)
        ce_term = torch.sum(ce_term)

        # KL term of GP prior
        KL_term = (ce_term - inside_elbo) / self.z_dim

        # SAMPLE
        latent_samples = []
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = p_m + torch.randn_like(p_m) * torch.sqrt(p_v)
            latent_samples.append(latent_samples_)

        recon_loss = 0
        for f in latent_samples:
            hidden_samples = self.decoder(f)
            mean_samples_ = self.dec_mean(hidden_samples)
            disp_samples_ = (torch.exp(torch.clamp(self.dec_disp, -15., 15.))).unsqueeze(0)

            mean_samples.append(mean_samples_)
            disp_samples.append(disp_samples_)
            recon_loss += self.NB_loss(x=raw_y, mean=mean_samples_, disp=disp_samples_, scale_factor=size_factors)
        recon_loss = recon_loss / num_samples / self.input_dim

        noise_reg = 0
        if self.noise > 0:
            for _ in range(num_samples):
                qnet_mu_, qnet_var_ = self.encoder(y + torch.randn_like(y)*self.noise)
                p_m_, p_v_ = [], []
                for l in range(self.z_dim):
                    p_m_l_, p_v_l_, _, _ = self.svgp.approximate_posterior_params(x, x,
                                                                    qnet_mu_[:, l], qnet_var_[:, l])
                    p_m_.append(p_m_l_)
                    p_v_.append(p_v_l_)

                p_m_ = torch.stack(p_m_, dim=1)
                p_v_ = torch.stack(p_v_, dim=1)
                noise_reg += torch.sum((p_m - p_m_)**2) / self.z_dim
            noise_reg = noise_reg / num_samples


        # ELBO
        if self.noise > 0 :
            elbo = self.beta * recon_loss + self.beta * noise_reg + KL_term
        else:
            elbo = self.beta * recon_loss + KL_term

        return elbo, recon_loss, KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, \
            mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg


    def batching_latent_samples(self, X, Y, batch_size=512):
        """
        Output latent embedding.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        Y: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        """ 

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

            p_m, p_v = [], []
            for l in range(self.z_dim):
                p_m_l, p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, qnet_mu[:, l], qnet_var[:, l])
                p_m.append(p_m_l)
                p_v.append(p_v_l)

            p_m = torch.stack(p_m, dim=1)
            p_v = torch.stack(p_v, dim=1)

            # SAMPLE
            latent_samples.append(p_m.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)

        return latent_samples.numpy()


    def batching_denoise_counts(self, X, Y, n_samples=1, batch_size=512):
        """
        Output denoised counts.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        Y: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        Y = torch.tensor(Y, dtype=self.dtype)

        mean_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            ybatch = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(ybatch)

            p_m, p_v = [], []
            for l in range(self.z_dim):
                p_m_l, p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, qnet_mu[:, l], qnet_var[:, l])
                p_m.append(p_m_l)
                p_v.append(p_v_l)

            p_m = torch.stack(p_m, dim=1)
            p_v = torch.stack(p_v, dim=1)

            # SAMPLE
            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = p_m + torch.randn_like(p_m) * torch.sqrt(p_v)
                latent_samples.append(latent_samples_)

            mean_samples_ = []
            for f in latent_samples:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.data.cpu().detach())

        mean_samples = torch.cat(mean_samples, dim=0)

        return mean_samples.numpy()


    def batching_recon_samples(self, Z, batch_size=512):
        self.eval()

        Z = torch.tensor(Z, dtype=self.dtype)

        recon_samples = []

        num = Z.shape[0]
        num_batch = int(math.ceil(1.0*Z.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            zbatch = Z[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            h = self.decoder(zbatch)
            mean_batch = self.dec_mean(h)
            recon_samples.append(mean_batch.data.cpu().detach())

        recon_samples = torch.cat(recon_samples, dim=0)

        return recon_samples.numpy()


    def batching_predict_samples(self, X_test, X_train, Y_train, n_samples=1, batch_size=512):
        """
        Impute latent representations and denoised counts on unseen testing locations.

        Parameters:
        -----------
        X_test: array_like, shape (n_test_spots, 2)
            Location information of testing set.
        X_train: array_like, shape (n_train_spots, 2)
            Location information of training set.
        Y_train: array_like, shape (n_train_spots, n_genes)
            Preprocessed count matrix of training set.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        X_test = torch.tensor(X_test, dtype=self.dtype)
        X_train = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        Y_train = torch.tensor(Y_train, dtype=self.dtype).to(self.device)

        latent_samples = []
        mean_samples = []

        train_num = X_train.shape[0]
        train_num_batch = int(math.ceil(1.0*X_train.shape[0]/batch_size))
        test_num = X_test.shape[0]
        test_num_batch = int(math.ceil(1.0*X_test.shape[0]/batch_size))

        qnet_mu, qnet_var = [], []
        for batch_idx in range(train_num_batch):
            Y_train_batch = Y_train[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device)
            qnet_mu_, qnet_var_ = self.encoder(Y_train_batch)
            qnet_mu.append(qnet_mu_)
            qnet_var.append(qnet_var_)
        qnet_mu = torch.cat(qnet_mu, dim=0)
        qnet_var = torch.cat(qnet_var, dim=0)

        for batch_idx in range(test_num_batch):
            x_test_batch = X_test[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)

            p_m, p_v = [], []
            for l in range(self.z_dim):
                p_m_l, p_v_l, _, _ = self.svgp.approximate_posterior_params(index_points_test=x_test_batch, index_points_train=X_train, 
                                        y=qnet_mu[:, l], noise=qnet_var[:, l])
                p_m.append(p_m_l)
                p_v.append(p_v_l)

            p_m = torch.stack(p_m, dim=1)
            p_v = torch.stack(p_v, dim=1)

            latent_samples.append(p_m.data.cpu().detach())

            # SAMPLE
            latent_samples_ = []
            for _ in range(n_samples):
                f = p_m + torch.randn_like(p_m) * torch.sqrt(p_v)
                latent_samples_.append(f)

            mean_samples_ = []
            for f in latent_samples_:
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)
        mean_samples = torch.cat(mean_samples, dim=0)

        return latent_samples.numpy(), mean_samples.numpy()


    def differential_expression(self, group1_idx, group2_idx, num_denoise_samples=10000, batch_size=512, pos=None, ncounts=None, 
            gene_name=None, raw_counts=None, n_samples=1, estimate_pseudocount=True):
        """
        Differential expression analysis.

        Parameters:
        -----------
        group1_idx: array_like, shape (n_group1)
            Index of group1.
        group2_idx: array_like, shape (n_group2)
            Index of group2.
        num_denoise_samples: Number of samplings in each group.
        pos: array_like, shape (n_spots, 2)
            Location information.
        ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        raw_counts: array_like, shape (n_spots, n_genes)
            Raw count matrix.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        group1_idx_sampling = group1_idx[np.random.randint(group1_idx.shape[0], size=num_denoise_samples)]
        group2_idx_sampling = group2_idx[np.random.randint(group2_idx.shape[0], size=num_denoise_samples)]

        group1_denoised_counts = self.batching_denoise_counts(X=pos[group1_idx_sampling], Y=ncounts[group1_idx_sampling], batch_size=batch_size, n_samples=n_samples)
        group2_denoised_counts = self.batching_denoise_counts(X=pos[group2_idx_sampling], Y=ncounts[group2_idx_sampling], batch_size=batch_size, n_samples=n_samples)

        if estimate_pseudocount:
            group1_where_zero = np.quantile(raw_counts[group1_idx], q=0.95, axis=0) == 0
            group2_where_zero = np.quantile(raw_counts[group2_idx], q=0.95, axis=0) == 0
            group1_max_denoised_counts = np.max(group1_denoised_counts, axis=0)
            group2_max_denoised_counts = np.max(group2_denoised_counts, axis=0)

            if group1_where_zero.sum() >= 1:
                group1_atefact_count = group1_max_denoised_counts[group1_where_zero]
                group1_eps = np.quantile(group1_atefact_count, q=0.9)
            else:
                group1_eps = 1e-10
            if group2_where_zero.sum() >= 1:
                group2_atefact_count = group2_max_denoised_counts[group2_where_zero]
                group2_eps = np.quantile(group2_atefact_count, q=0.9)
            else:
                group2_eps = 1e-10

            eps = np.maximum(group1_eps, group2_eps)
            print("Estimated pseudocounts", eps)
        else:
            eps = 1e-10

        group1_denoised_mean = np.mean(group1_denoised_counts, axis=0)
        group2_denoised_mean = np.mean(group2_denoised_counts, axis=0)

        lfc = np.log2(group1_denoised_mean + eps) - np.log2(group2_denoised_mean + eps)

        group1_raw_mean = np.mean(raw_counts[group1_idx], axis=0)
        group2_raw_mean = np.mean(raw_counts[group2_idx], axis=0)

        res_dat = pd.DataFrame(data={'LFC': lfc, 'denoised_mean1': group1_denoised_mean, 'denoised_mean2': group2_denoised_mean,
                                    'raw_mean1': group1_raw_mean, 'raw_mean2': group2_raw_mean}, index=gene_name)
            
        return res_dat


    def train_model(self, pos, ncounts, raw_counts, size_factors, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, 
            maxiter=2000, save_model=True, model_weights="model.pt", print_kernel_scale=True):
        """
        Model training.

        Parameters:
        -----------
        pos: array_like, shape (n_spots, 2)
            Location information.
        ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        raw_counts: array_like, shape (n_spots, n_genes)
            Raw count matrix.
        size_factor: array_like, shape (n_spots)
            The size factor of each spot, which need for the NB loss.
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

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(ncounts, dtype=self.dtype), 
                        torch.tensor(raw_counts, dtype=self.dtype), torch.tensor(size_factors, dtype=self.dtype))

        if ncounts.shape[0] > batch_size:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            KL_term_val = 0
            noise_reg_val = 0
            num = 0
            for batch_idx, (x_batch, y_batch, y_raw_batch, sf_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_raw_batch = y_raw_batch.to(self.device)
                sf_batch = sf_batch.to(self.device)

                elbo, recon_loss, KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg = \
                    self.forward(x=x_batch, y=y_batch, raw_y=y_raw_batch, size_factors=sf_batch, num_samples=num_samples)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                KL_term_val += KL_term.item()
                if self.noise > 0:
                    noise_reg_val += noise_reg.item()

                num += x_batch.shape[0]

            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            KL_term_val = KL_term_val/num
            noise_reg_val = noise_reg_val/num

            print('Training epoch {}, ELBO:{:.8f}, NB loss:{:.8f}, KLD loss:{:.8f}, noise regularization:{:8f}'.format(epoch+1, elbo_val, recon_loss_val, KL_term_val, noise_reg_val))
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

        if save_model:
            torch.save(self.state_dict(), model_weights)

