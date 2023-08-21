import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.normal import Normal
from torch.distributions import LogNormal, Bernoulli
from torch.distributions.kl import kl_divergence
import numpy as np
import pandas as pd
from SVGP import SVGP
from I_PID import PIDControl
from VAE_utils import *
from collections import deque


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class SPAMULTIVAE(nn.Module):
    def __init__(self, gene_dim, protein_dim, GP_dim, Normal_dim, encoder_layers, gene_decoder_layers, protein_decoder_layers, gene_noise, protein_noise, 
                    encoder_dropout, decoder_dropout, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, 
                    KL_loss, init_beta, min_beta, max_beta, protein_back_mean, protein_back_scale, dtype, device):
        super(SPAMULTIVAE, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.gene_dim = gene_dim
        self.protein_dim = protein_dim
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss      # expected KL loss value
        self.beta = init_beta       # beta controls the weight of reconstruction loss
        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.gene_noise = gene_noise
        self.protein_noise = protein_noise
        self.device = device
        self.encoder = DenseEncoder(input_dim=gene_dim+protein_dim, hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)

        ### gene network
        self.gene_decoder = buildNetwork([GP_dim+Normal_dim]+gene_decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(gene_decoder_layers) > 0:
            self.gene_dec_mean = nn.Sequential(nn.Linear(gene_decoder_layers[-1], gene_dim), MeanAct())
        else:
            self.gene_dec_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, gene_dim), MeanAct())
        self.gene_dec_disp = nn.Parameter(torch.randn(self.gene_dim), requires_grad=True)

        ### protein network
        self.protein_decoder = buildNetwork([GP_dim+Normal_dim]+protein_decoder_layers, activation="elu", dropout=decoder_dropout)
        self.protein_back_decoder = buildNetwork([GP_dim+Normal_dim]+protein_decoder_layers, activation="elu", dropout=decoder_dropout)
        self.protein_back_prop_logit_decoder = buildNetwork([GP_dim+Normal_dim]+protein_decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(protein_decoder_layers) > 0:
            self.protein_fore_mean = nn.Sequential(nn.Linear(protein_decoder_layers[-1], protein_dim), nn.Softplus())
            self.protein_back_log_mean_dec = nn.Linear(protein_decoder_layers[-1], protein_dim)
            self.protein_back_log_scale_dec = nn.Linear(protein_decoder_layers[-1], protein_dim)
            self.protein_back_prop_logit = nn.Linear(protein_decoder_layers[-1], protein_dim)
        else:
            self.protein_fore_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, protein_dim), nn.Softplus())
            self.protein_back_log_mean_dec = nn.Linear(GP_dim+Normal_dim, protein_dim)
            self.protein_back_log_scale_dec = nn.Linear(GP_dim+Normal_dim, protein_dim)
            self.protein_back_prop_logit = nn.Linear(GP_dim+Normal_dim, protein_dim)
        self.protein_back_log_mean = nn.Parameter(torch.tensor(protein_back_mean), requires_grad=True)
        self.protein_back_log_scale = nn.Parameter(torch.tensor(np.log(protein_back_scale)), requires_grad=True)
        self.protein_dec_disp = nn.Parameter(torch.randn(self.protein_dim), requires_grad=True)

        self.NB_loss = NBLoss().to(self.device)
        self.MixtureNB_loss = MixtureNBLoss().to(self.device)
        self.to(device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)


    def forward(self, x, gene_y, protein_y, raw_gene_y, raw_protein_y, gene_size_factors, num_samples=1):
        """
        Forward pass.

        Parameters:
        -----------
        x: mini-batch of positions.
        gene_y: mini-batch of preprocessed gene counts.
        protein_y: mini-batch of preprocessed protein counts.
        raw_gene_y: mini-batch of raw gene counts.
        raw_protein_y: mini-batch of raw protein counts.
        gene_size_factors: mini-batch of gene size factors.
        num_samples: number of samplings of the posterior distribution of latent embedding.

        raw_gene_y and gene_size_factors are used for NB likelihood of genes.
        raw_protein_y is used for NB mixture likelihood of proteins.
        """ 

        self.train()
        b = gene_y.shape[0]
        qnet_mu, qnet_var = self.encoder(torch.cat((gene_y, protein_y), dim=-1))

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l])
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=qnet_mu[:, l],
                                                                    noise=gp_var[:, l], mu_hat=mu_hat_l,
                                                                    A_hat=A_hat_l)

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
        gaussian_prior_dist = Normal(torch.zeros_like(gaussian_mu), torch.ones_like(gaussian_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        gaussian_KL_term = kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))
        latent_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        gene_mean_samples = []
        gene_disp_samples = []
        gene_recon_loss = 0
        for f in latent_samples:
            gene_hidden_samples = self.gene_decoder(f)
            gene_mean_samples_ = self.gene_dec_mean(gene_hidden_samples)
            gene_disp_samples_ = (torch.exp(torch.clamp(self.gene_dec_disp, -15., 15.))).unsqueeze(0)

            gene_mean_samples.append(gene_mean_samples_)
            gene_disp_samples.append(gene_disp_samples_)
            gene_recon_loss += self.NB_loss(x=raw_gene_y, mean=gene_mean_samples_, disp=gene_disp_samples_, scale_factor=gene_size_factors)
        gene_recon_loss = gene_recon_loss / num_samples


        protein_mean_samples = []
        protein_disp_samples = []
        protein_recon_loss = 0
        protein_back_KL = 0
        for f in latent_samples:
            protein_hidden_samples = self.protein_decoder(f)
            protein_foreground_mean_samples_f = self.protein_fore_mean(protein_hidden_samples)
            protein_logit_f = self.protein_back_prop_logit(self.protein_back_prop_logit_decoder(f))
            protein_back_prior = LogNormal(self.protein_back_log_mean, torch.exp(self.protein_back_log_scale))
            protein_back_hidden_samples = self.protein_back_decoder(f)
            protein_back_log_mean_f = self.protein_back_log_mean_dec(protein_back_hidden_samples)
            protein_back_log_scale_f = torch.exp(self.protein_back_log_scale_dec(protein_back_hidden_samples))
            protein_back_postier = LogNormal(protein_back_log_mean_f, protein_back_log_scale_f)
            protein_background_mean_f = protein_back_postier.rsample()
            protein_forground_mean_f = (1+protein_foreground_mean_samples_f) * protein_background_mean_f
            protein_disp_samples_f = (torch.exp(torch.clamp(self.protein_dec_disp, -15., 15.))).unsqueeze(0)

            protein_mean_samples.append(protein_forground_mean_f)
            protein_disp_samples.append(protein_disp_samples_f)

            protein_recon_loss += self.MixtureNB_loss(x=raw_protein_y, mean1=protein_background_mean_f, mean2=protein_forground_mean_f,
                        disp=protein_disp_samples_f, pi_logits=protein_logit_f)
            protein_back_KL += kl_divergence(protein_back_postier, protein_back_prior).sum()
        protein_recon_loss = protein_recon_loss / num_samples
        protein_back_KL = protein_back_KL / num_samples

        noise_reg = 0
        if self.gene_noise > 0 or self.protein_noise > 0:
            for _ in range(num_samples):
                qnet_mu_, qnet_var_ = self.encoder(torch.cat((gene_y + torch.randn_like(gene_y)*self.gene_noise,
                                                                protein_y + torch.randn_like(protein_y)*self.protein_noise), dim=-1))
                gp_mu_ = qnet_mu_[:, 0:self.GP_dim]
                gp_var_ = qnet_var_[:, 0:self.GP_dim]

                gp_p_m_, gp_p_v_ = [], []
                for l in range(self.GP_dim):
                    gp_p_m_l_, gp_p_v_l_, _, _ = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu_[:, l], gp_var_[:, l])
                    gp_p_m_.append(gp_p_m_l_)
                    gp_p_v_.append(gp_p_v_l_)

                gp_p_m_ = torch.stack(gp_p_m_, dim=1)
                gp_p_v_ = torch.stack(gp_p_v_, dim=1)
                noise_reg += torch.sum((gp_p_m - gp_p_m_)**2)
            noise_reg = noise_reg / num_samples


        # ELBO
        if self.gene_noise > 0 or self.protein_noise > 0:
            elbo = gene_recon_loss + protein_recon_loss + protein_back_KL + noise_reg * (self.gene_dim+self.protein_dim) / self.GP_dim  + self.beta * gp_KL_term + self.beta * gaussian_KL_term
        else:
            elbo = gene_recon_loss + protein_recon_loss + protein_back_KL + self.beta * gp_KL_term + self.beta * gaussian_KL_term

        return elbo, gene_recon_loss, protein_recon_loss, protein_back_KL, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
            gene_mean_samples, gene_disp_samples, protein_mean_samples, protein_disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg


    def batching_latent_samples(self, X, gene_Y, protein_Y, batch_size=512):
        """
        Output latent embedding.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        gene_Y: array_like, shape (n_spots, n_genes)
            Preprocessed gene count matrix.
        protein_Y: array_like, shape (n_spots, n_proteins)
            Preprocessed protein count matrix.
        """ 

        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        gene_Y = torch.tensor(gene_Y, dtype=self.dtype)
        protein_Y = torch.tensor(protein_Y, dtype=self.dtype)

        latent_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            gene_y_batch = gene_Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            protein_y_batch = protein_Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(torch.cat((gene_y_batch, protein_y_batch), dim=-1))

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
#            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_samples.append(p_m.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)

        return latent_samples.numpy()


    def batching_denoise_counts(self, X, gene_Y, protein_Y, n_samples=1, sample_protein_mixing=True, batch_size=512):
        """
        Output denoised counts.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        gene_Y: array_like, shape (n_spots, n_genes)
            Preprocessed gene count matrix.
        protein_Y: array_like, shape (n_spots, n_proteins)
            Preprocessed protein count matrix.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        gene_Y = torch.tensor(gene_Y, dtype=self.dtype)
        protein_Y = torch.tensor(protein_Y, dtype=self.dtype)

        gene_mean_samples = []
        protein_mean_samples = []
        protein_sigmoid = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            gene_y_batch = gene_Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            protein_y_batch = protein_Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(torch.cat((gene_y_batch, protein_y_batch), dim=-1))

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            latent_samples = []
            for _ in range(n_samples):
                latent_samples_ = latent_dist.sample()
                latent_samples.append(latent_samples_)

            gene_mean_samples_ = []
            protein_mean_samples_ = []
            protein_sigmoid_ = []
            for f in latent_samples:
                gene_hidden_samples = self.gene_decoder(f)
                gene_mean_samples_f = self.gene_dec_mean(gene_hidden_samples)
                gene_mean_samples_.append(gene_mean_samples_f)

                protein_hidden_samples = self.protein_decoder(f)
                protein_foreground_mean_samples_f = self.protein_fore_mean(protein_hidden_samples)
                protein_logit_f = self.protein_back_prop_logit(self.protein_back_prop_logit_decoder(f))
                protein_sigmoid_f = torch.sigmoid(protein_logit_f)
                protein_sigmoid_.append(protein_sigmoid_f)
                protein_back_hidden_samples = self.protein_back_decoder(f)
                protein_back_log_mean_f = self.protein_back_log_mean_dec(protein_back_hidden_samples)
                protein_back_log_scale_f = torch.exp(self.protein_back_log_scale_dec(protein_back_hidden_samples))
                protein_back_postier = LogNormal(protein_back_log_mean_f, protein_back_log_scale_f)
                protein_background_mean = protein_back_postier.sample()
                if sample_protein_mixing:
                    protein_sigmoid_f = Bernoulli(protein_sigmoid_f).sample()
                protein_mean_samples_f = (1-protein_sigmoid_f) * (1+protein_foreground_mean_samples_f) *\
                                        protein_background_mean
                protein_mean_samples_.append(protein_mean_samples_f)

            gene_mean_samples_ = torch.stack(gene_mean_samples_, dim=0)
            protein_mean_samples_ = torch.stack(protein_mean_samples_, dim=0)
            protein_sigmoid_ = torch.stack(protein_sigmoid_, dim=0)

            gene_mean_samples_ = torch.mean(gene_mean_samples_, dim=0)
            protein_mean_samples_ = torch.mean(protein_mean_samples_, dim=0)
            protein_sigmoid_ = torch.mean(protein_sigmoid_, dim=0)

            gene_mean_samples.append(gene_mean_samples_.data.cpu().detach())
            protein_mean_samples.append(protein_mean_samples_.data.cpu().detach())
            protein_sigmoid.append(protein_sigmoid_.data.cpu().detach())

        gene_mean_samples = torch.cat(gene_mean_samples, dim=0)
        protein_mean_samples = torch.cat(protein_mean_samples, dim=0)
        protein_sigmoid = torch.cat(protein_sigmoid, dim=0)

        return gene_mean_samples.numpy(), protein_mean_samples.numpy(), protein_sigmoid.numpy()


    def batching_recon_samples(self, Z, batch_size=512):
        self.eval()

        Z = torch.tensor(Z, dtype=self.dtype)

        gene_mean_samples = []
        protein_mean_samples = []

        num = Z.shape[0]
        num_batch = int(math.ceil(1.0*Z.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            zbatch = Z[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            gene_hidden_samples = self.gene_decoder(zbatch)
            gene_mean_samples_z = self.gene_dec_mean(gene_hidden_samples)
            gene_mean_samples.append(gene_mean_samples_z.data.cpu())

            protein_hidden_samples = self.protein_decoder(zbatch)
            protein_foreground_mean_samples_z = self.protein_fore_mean(protein_hidden_samples)
            protein_logit_z = self.protein_back_prop_logit(self.protein_back_prop_logit_decoder(zbatch))
            protein_sigmoid_z = torch.sigmoid(protein_logit_z)
            protein_back_hidden_samples = self.protein_back_decoder(zbatch)
            protein_back_log_mean_z = self.protein_back_log_mean_dec(protein_back_hidden_samples)
            protein_back_log_scale_z = torch.exp(self.protein_back_log_scale_dec(protein_back_hidden_samples))
            protein_back_postier = LogNormal(protein_back_log_mean_z, protein_back_log_scale_z)
            protein_background_mean = protein_back_postier.sample()
            protein_mean_samples_z = (1-protein_sigmoid_z) * (1+protein_foreground_mean_samples_z) * protein_background_mean
            protein_mean_samples.append(protein_mean_samples_z.data.cpu().detach())

        gene_mean_samples = torch.cat(gene_mean_samples, dim=0)
        protein_mean_samples = torch.cat(protein_mean_samples, dim=0)

        return gene_mean_samples.numpy(), protein_mean_samples.numpy()


    def batching_predict_samples(self, X_test, X_train, gene_Y_train, protein_Y_train, n_samples=1, sample_protein_mixing=True, batch_size=512):
        """
        Impute latent representations and denoised counts on unseen testing locations.

        Parameters:
        -----------
        X_test: array_like, shape (n_test_spots, 2)
            Location information of testing set.
        X_train: array_like, shape (n_train_spots, 2)
            Location information of training set.
        gene_Y_train: array_like, shape (n_train_spots, n_genes)
            Preprocessed gene count matrix of training set.
        protein_Y_train: array_like, shape (n_train_spots, n_proteins)
            Preprocessed protein count matrix of training set.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        X_test = torch.tensor(X_test, dtype=self.dtype)
        X_train = torch.tensor(X_train, dtype=self.dtype).to(self.device)
        gene_Y_train = torch.tensor(gene_Y_train, dtype=self.dtype).to(self.device)
        protein_Y_train = torch.tensor(protein_Y_train, dtype=self.dtype).to(self.device)

        latent_samples = []
        gene_mean_samples = []
        protein_mean_samples = []

        train_num = X_train.shape[0]
        train_num_batch = int(math.ceil(1.0*X_train.shape[0]/batch_size))
        test_num = X_test.shape[0]
        test_num_batch = int(math.ceil(1.0*X_test.shape[0]/batch_size))

        qnet_mu, qnet_var = [], []
        for batch_idx in range(train_num_batch):
            gene_Y_train_batch = gene_Y_train[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device)
            protein_Y_train_batch = protein_Y_train[batch_idx*batch_size : min((batch_idx+1)*batch_size, train_num)].to(self.device)
            qnet_mu_, qnet_var_ = self.encoder(torch.cat((gene_Y_train_batch, protein_Y_train_batch), dim=-1))
            qnet_mu.append(qnet_mu_)
            qnet_var.append(qnet_var_)
        qnet_mu = torch.cat(qnet_mu, dim=0)
        qnet_var = torch.cat(qnet_var, dim=0)

        def find_nearest(array, value):
            idx = torch.argmin(torch.sum((array - value)**2, dim=1))
            return idx

        for batch_idx in range(test_num_batch):
            x_test_batch = X_test[batch_idx*batch_size : min((batch_idx+1)*batch_size, test_num)].to(self.device)

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            # x_train_select_batch represents the nearest X_train spots to x_test_batch
            x_train_select_batch = []
            for e in range(x_test_batch.shape[0]):
                x_train_select_batch.append(find_nearest(X_train, x_test_batch[e]))
            x_train_select_batch = torch.stack(x_train_select_batch)
            gaussian_mu = qnet_mu[x_train_select_batch.long(), self.GP_dim:]
            gaussian_var = qnet_var[x_train_select_batch.long(), self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(index_points_test=x_test_batch, index_points_train=X_train, 
                                        y=gp_mu[:, l], noise=gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.data.cpu().detach())
            p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
            latent_dist = Normal(p_m, torch.sqrt(p_v))
            # SAMPLE
            latent_samples_ = []
            for _ in range(n_samples):
                f = latent_dist.sample()
                latent_samples_.append(f)

            gene_mean_samples_ = []
            protein_mean_samples_ = []
            for f in latent_samples_:
                gene_hidden_samples = self.gene_decoder(f)
                gene_mean_samples_f = self.gene_dec_mean(gene_hidden_samples)
                gene_mean_samples_.append(gene_mean_samples_f)

                protein_hidden_samples = self.protein_decoder(f)
                protein_foreground_mean_samples_f = self.protein_fore_mean(protein_hidden_samples)
                protein_logit_f = self.protein_back_prop_logit(self.protein_back_prop_logit_decoder(f))
                protein_sigmoid_f = torch.sigmoid(protein_logit_f)
                protein_back_hidden_samples = self.protein_back_decoder(f)
                protein_back_log_mean_f = self.protein_back_log_mean_dec(protein_back_hidden_samples)
                protein_back_log_scale_f = torch.exp(self.protein_back_log_scale_dec(protein_back_hidden_samples))
                protein_back_postier = LogNormal(protein_back_log_mean_f, protein_back_log_scale_f)
                protein_background_mean = protein_back_postier.sample()
                if sample_protein_mixing:
                    protein_sigmoid_f = Bernoulli(protein_sigmoid_f).sample()
                protein_mean_samples_f = (1-protein_sigmoid_f) * (1+protein_foreground_mean_samples_f) * protein_background_mean
                protein_mean_samples_.append(protein_mean_samples_f)

            gene_mean_samples_ = torch.stack(gene_mean_samples_, dim=0)
            protein_mean_samples_ = torch.stack(protein_mean_samples_, dim=0)

            gene_mean_samples_ = torch.mean(gene_mean_samples_, dim=0)
            protein_mean_samples_ = torch.mean(protein_mean_samples_, dim=0)

            gene_mean_samples.append(gene_mean_samples_.data.cpu().detach())
            protein_mean_samples.append(protein_mean_samples_.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)
        gene_mean_samples = torch.cat(gene_mean_samples, dim=0)
        protein_mean_samples = torch.cat(protein_mean_samples, dim=0)

        return latent_samples.numpy(), gene_mean_samples.numpy(), protein_mean_samples.numpy()


    def differential_expression(self, group1_idx, group2_idx, num_denoise_samples=10000, batch_size=512, pos=None, gene_ncounts=None, protein_ncounts=None,
            gene_name=None, raw_gene_counts=None, protein_name=None, raw_protein_counts=None, estimate_pseudocount=True):
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
        gene_ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed gene count matrix.
        protein_ncounts: array_like, shape (n_spots, n_proteins)
            Preprocessed protein count matrix.
        gene_name: array_like, shape (n_genes)
            gene names.
        raw_gene_counts: array_like, shape (n_spots, n_genes)
            Raw gene count matrix.
        protein_name: array_like, shape (n_proteins)
            protein names.
        raw_protein_counts: array_like, shape (n_spots, n_proteins)
            Raw protein count matrix.
        estimate_pseudocount: Whether to estimate pseudocount from data, otherwise use default values 0.05 for genes and 0.5 for proteins.
        """

        group1_idx_sampling = group1_idx[np.random.randint(group1_idx.shape[0], size=num_denoise_samples)]
        group2_idx_sampling = group2_idx[np.random.randint(group2_idx.shape[0], size=num_denoise_samples)]

        group1_gene_denoised_counts, group1_protein_denoised_counts, _ = self.batching_denoise_counts(X=pos[group1_idx_sampling], 
                                    gene_Y=gene_ncounts[group1_idx_sampling], protein_Y=protein_ncounts[group1_idx_sampling], 
                                    batch_size=batch_size, n_samples=1)
        group2_gene_denoised_counts, group2_protein_denoised_counts, _ = self.batching_denoise_counts(X=pos[group2_idx_sampling], 
                                    gene_Y=gene_ncounts[group2_idx_sampling], protein_Y=protein_ncounts[group2_idx_sampling], 
                                    batch_size=batch_size, n_samples=1)

        if estimate_pseudocount:
#            gene_group1_where_zero = np.quantile(raw_gene_counts[group1_idx], q=0.95, axis=0) == 0
#            gene_group2_where_zero = np.quantile(raw_gene_counts[group2_idx], q=0.95, axis=0) == 0
            gene_group1_where_zero = np.max(raw_gene_counts[group1_idx], axis=0) == 0
            gene_group2_where_zero = np.max(raw_gene_counts[group2_idx], axis=0) == 0
            gene_group1_max_denoised_counts = np.max(raw_gene_counts, axis=0)
            gene_group2_max_denoised_counts = np.max(raw_gene_counts, axis=0)

            if gene_group1_where_zero.sum() >= 1:
                gene_group1_atefact_count = gene_group1_max_denoised_counts[gene_group1_where_zero]
                gene_group1_eps = np.quantile(gene_group1_atefact_count, q=0.9)
            else:
                gene_group1_eps = 1e-10
            if gene_group2_where_zero.sum() >= 1:
                gene_group2_atefact_count = gene_group2_max_denoised_counts[gene_group2_where_zero]
                gene_group2_eps = np.quantile(gene_group2_atefact_count, q=0.9)
            else:
                gene_group2_eps = 1e-10

            gene_eps = np.maximum(gene_group1_eps, gene_group2_eps)
            gene_eps = np.clip(gene_eps, a_min=0.05, a_max=0.5)
            print("Estimated gene pseudocounts", gene_eps)
        else:
            gene_eps = 1e-10

        group1_gene_denoised_mean = np.mean(group1_gene_denoised_counts, axis=0)
        group2_gene_denoised_mean = np.mean(group2_gene_denoised_counts, axis=0)
        
        gene_lfc = np.log2(group1_gene_denoised_mean + gene_eps) - np.log2(group2_gene_denoised_mean + gene_eps)
        p_gene_lfc = np.log2(group1_gene_denoised_counts + gene_eps) - np.log2(group2_gene_denoised_counts + gene_eps)
        mean_gene_lfc = np.mean(p_gene_lfc, axis=0)
        median_gene_lfc = np.median(p_gene_lfc, axis=0)
        sd_gene_lfc = np.std(p_gene_lfc, axis=0)
        gene_delta = gmm_fit(data=mean_gene_lfc)
        print("Gene LFC delta:", gene_delta)
        is_de = (np.abs(p_gene_lfc) >= gene_delta).mean(0)
        not_de = (np.abs(p_gene_lfc) < gene_delta).mean(0)
        bayes_factor = np.log(is_de + 1e-10) - np.log(not_de + 1e-10)

        group1_gene_raw_mean = np.mean(raw_gene_counts[group1_idx], axis=0)
        group2_gene_raw_mean = np.mean(raw_gene_counts[group2_idx], axis=0)

        gene_res_dat = pd.DataFrame(data={'LFC': gene_lfc, 'mean_LFC': mean_gene_lfc, 'median_LFC': median_gene_lfc, 'sd_LFC': sd_gene_lfc,
                                     'prob_DE': is_de, 'prob_not_DE': not_de, 'bayes_factor': bayes_factor,
                                     'denoised_mean1': group1_gene_denoised_mean, 'denoised_mean2': group2_gene_denoised_mean,
                                    'raw_mean1': group1_gene_raw_mean, 'raw_mean2': group2_gene_raw_mean}, index=gene_name)
        
        
        protein_eps = 0.5

        group1_protein_denoised_mean = np.mean(group1_protein_denoised_counts, axis=0)
        group2_protein_denoised_mean = np.mean(group2_protein_denoised_counts, axis=0)

        protein_lfc = np.log2(group1_protein_denoised_mean + protein_eps) - np.log2(group2_protein_denoised_mean + protein_eps)
        p_protein_lfc = np.log2(group1_protein_denoised_counts + protein_eps) - np.log2(group2_protein_denoised_counts + protein_eps)
        mean_protein_lfc = np.mean(p_protein_lfc, axis=0)
        median_protein_lfc = np.median(p_protein_lfc, axis=0)
        sd_protein_lfc = np.std(p_protein_lfc, axis=0)
        protein_delta = gmm_fit(data=mean_protein_lfc)
        print("protein LFC delta:", protein_delta)
        is_de = (np.abs(p_protein_lfc) >= protein_delta).mean(0)
        not_de = (np.abs(p_protein_lfc) < protein_delta).mean(0)
        bayes_factor = np.log(is_de + 1e-10) - np.log(not_de + 1e-10)

        group1_protein_raw_mean = np.mean(raw_protein_counts[group1_idx], axis=0)
        group2_protein_raw_mean = np.mean(raw_protein_counts[group2_idx], axis=0)

        protein_res_dat = pd.DataFrame(data={'LFC': protein_lfc, 'mean_LFC': mean_protein_lfc, 'median_LFC': median_protein_lfc, 'sd_LFC': sd_protein_lfc,
                                     'prob_DE': is_de, 'prob_not_DE': not_de, 'bayes_factor': bayes_factor,
                                     'denoised_mean1': group1_protein_denoised_mean, 'denoised_mean2': group2_protein_denoised_mean,
                                    'raw_mean1': group1_protein_raw_mean, 'raw_mean2': group2_protein_raw_mean}, index=protein_name)
        return gene_res_dat, protein_res_dat


    def train_model(self, pos, gene_ncounts, gene_raw_counts, gene_size_factors, protein_ncounts, protein_raw_counts, 
            lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, train_size=0.95, maxiter=5000, patience=200, 
            save_model=True, model_weights="model.pt", print_kernel_scale=True):
        """
        Model training.

        Parameters:
        -----------
        pos: array_like, shape (n_spots, 2)
            Location information.
        gene_ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed gene count matrix.
        gene_raw_counts: array_like, shape (n_spots, n_genes)
            Raw gene count matrix.
        gene_size_factors: array_like, shape (n_spots)
            The gene size factor of each spot, which need for the NB loss.
        protein_ncounts array_like, shape (n_spots, n_proteins)
            Preprocessed protein count matrix.
        protein_raw_counts: array_like, shape (n_spots, n_proteins)
            Raw protein count matrix.
        lr: float, defalut = 0.001
            Learning rate for the opitimizer.
        weight_decay: float, defalut = 0.001
            Weight decay for the opitimizer.
        train_size: float, default = 0.95
            proportion of training size, the other samples are validations.
        maxiter: int, default = 5000
            Maximum number of iterations.
        patience: int, default = 200
            Patience for early stopping.
        model_weights: str
            File name to save the model weights.
        print_kernel_scale: bool
            Whether to print current kernel scale during training steps.
        """

        self.train()

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(gene_ncounts, dtype=self.dtype), 
                        torch.tensor(gene_raw_counts, dtype=self.dtype), torch.tensor(gene_size_factors, dtype=self.dtype),
                        torch.tensor(protein_ncounts, dtype=self.dtype), torch.tensor(protein_raw_counts, dtype=self.dtype))

        if train_size < 1:
            train_dataset, validate_dataset = random_split(dataset=dataset, lengths=[train_size, 1.-train_size])
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = dataset

        if gene_ncounts.shape[0] > batch_size:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        early_stopping = EarlyStopping(patience=patience, modelfile=model_weights)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        queue = deque()

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            gene_recon_loss_val = 0
            protein_recon_loss_val = 0
            protein_back_KL_val = 0
            gp_KL_term_val = 0
            gaussian_KL_term_val = 0
            noise_reg_val = 0
            num = 0
            for batch_idx, (x_batch, gene_batch, gene_raw_batch, gene_sf_batch, protein_batch, protein_raw_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                gene_batch = gene_batch.to(self.device)
                gene_raw_batch = gene_raw_batch.to(self.device)
                gene_sf_batch = gene_sf_batch.to(self.device)
                protein_batch = protein_batch.to(self.device)
                protein_raw_batch = protein_raw_batch.to(self.device)

                elbo, gene_recon_loss, protein_recon_loss, protein_back_KL, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    gene_mean_samples, gene_disp_samples, protein_mean_samples, protein_disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg = \
                    self.forward(x=x_batch, gene_y=gene_batch, protein_y=protein_batch, raw_gene_y=gene_raw_batch, raw_protein_y=protein_raw_batch, 
                                gene_size_factors=gene_sf_batch, num_samples=num_samples)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                gene_recon_loss_val += gene_recon_loss.item()
                protein_recon_loss_val += protein_recon_loss.item()
                protein_back_KL_val += protein_back_KL.item()
                gp_KL_term_val += gp_KL_term.item()
                gaussian_KL_term_val += gaussian_KL_term.item()
                if self.gene_noise > 0 or self.protein_noise > 0:
                    noise_reg_val += noise_reg.item()

                num += x_batch.shape[0]

                KL_val = (gp_KL_term.item() + gaussian_KL_term.item()) / x_batch.shape[0]
                queue.append(KL_val)
                avg_KL = np.mean(queue)
                self.beta, _ = self.PID.pid(self.KL_loss*(self.GP_dim+self.Normal_dim), avg_KL)
                if len(queue) >= 10:
                    queue.popleft()

            elbo_val = elbo_val/num
            gene_recon_loss_val = gene_recon_loss_val/num
            protein_recon_loss_val = protein_recon_loss_val/num
            protein_back_KL_val = protein_back_KL_val/num
            gp_KL_term_val = gp_KL_term_val/num
            gaussian_KL_term_val = gaussian_KL_term_val/num
            noise_reg_val = noise_reg_val/num

            print('Training epoch {}, ELBO:{:.8f}, Gene NB loss:{:.8f}, Protein NB mixture loss:{:.8f}, Protein background KL:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f}, noise regularization:{:8f}'.format(epoch+1, elbo_val, gene_recon_loss_val, protein_recon_loss_val, protein_back_KL_val, gp_KL_term_val, gaussian_KL_term_val, noise_reg_val))
            print('Current beta', self.beta)
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

            if train_size < 1:
                validate_elbo_val = 0
                validate_num = 0
                for _, (validate_x_batch, validate_gene_batch, validate_gene_raw_batch, validate_gene_sf_batch, validate_protein_batch, validate_protein_raw_batch) in enumerate(validate_dataloader):
                    validate_x_batch = validate_x_batch.to(self.device)
                    validate_gene_batch = validate_gene_batch.to(self.device)
                    validate_gene_raw_batch = validate_gene_raw_batch.to(self.device)
                    validate_gene_sf_batch = validate_gene_sf_batch.to(self.device)
                    validate_protein_batch = validate_protein_batch.to(self.device)
                    validate_protein_raw_batch = validate_protein_raw_batch.to(self.device)

                    validate_elbo, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                    self.forward(x=validate_x_batch, gene_y=validate_gene_batch, protein_y=validate_protein_batch, raw_gene_y=validate_gene_raw_batch, 
                                 raw_protein_y=validate_protein_raw_batch, gene_size_factors=validate_gene_sf_batch, num_samples=num_samples)

                    validate_elbo_val += validate_elbo.item()
                    validate_num += validate_x_batch.shape[0]

                validate_elbo_val = validate_elbo_val / validate_num

                print("Training epoch {}, validating ELBO:{:.8f}".format(epoch+1, validate_elbo_val))
                early_stopping(validate_elbo_val, self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} iteration'.format(epoch+1))
                    break

        if save_model:
            torch.save(self.state_dict(), model_weights)



