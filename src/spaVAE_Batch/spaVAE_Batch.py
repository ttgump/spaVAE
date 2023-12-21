import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import numpy as np
import pandas as pd
from SVGP_Batch import SVGP
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


class SPAVAE(nn.Module):
    def __init__(self, input_dim, GP_dim, Normal_dim, n_batch, encoder_layers, decoder_layers, noise, encoder_dropout, decoder_dropout, 
                    shared_dispersion, fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, allow_batch_kernel_scale,
                    N_train, KL_loss, dynamicVAE, init_beta, min_beta, max_beta, dtype, device):
        super(SPAVAE, self).__init__()
        torch.set_default_dtype(dtype)
        if allow_batch_kernel_scale:
            self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=[kernel_scale]*n_batch, allow_batch_kernel_scale=allow_batch_kernel_scale, 
                jitter=1e-8, N_train=N_train, dtype=dtype, device=device)

        else:
            self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, allow_batch_kernel_scale=allow_batch_kernel_scale,
                jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.input_dim = input_dim
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss          # expected KL loss value
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta           # beta controls the weight of reconstruction loss
        self.dtype = dtype
        self.GP_dim = GP_dim            # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.n_batch = n_batch
        self.noise = noise              # intensity of random noise
        self.device = device
        self.encoder = DenseEncoder(input_dim=input_dim+n_batch, hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = buildNetwork([GP_dim+Normal_dim+n_batch]+decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), MeanAct())
        else:
            self.dec_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, input_dim), MeanAct())
        self.shared_dispersion = shared_dispersion
        if self.shared_dispersion:
            self.dec_disp = nn.Parameter(torch.randn(self.input_dim), requires_grad=True)
        else:
            self.dec_disp = nn.Parameter(torch.randn(self.input_dim, n_batch), requires_grad=True)

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


    def forward(self, x, y, batch, raw_y, size_factors, num_samples=1):
        """
        Forward pass.

        Parameters:
        -----------
        x: mini-batch of positions.
        y: mini-batch of preprocessed counts.
        batch: mini-batch of one-hot encoded batch IDs.
        raw_y: mini-batch of raw counts.
        size_factor: mini-batch of size factors.
        num_samples: number of samplings of the posterior distribution of latent embedding.

        raw_y and size_factor are used for NB likelihood
        """ 

        b = y.shape[0]
        qnet_mu, qnet_var = self.encoder(torch.cat((y, batch), dim=1))

        gp_mu = qnet_mu[:, 0:self.GP_dim]
        gp_var = qnet_var[:, 0:self.GP_dim]

        gaussian_mu = qnet_mu[:, self.GP_dim:]
        gaussian_var = qnet_var[:, self.GP_dim:]

        inside_elbo_recon, inside_elbo_kl = [], []
        gp_p_m, gp_p_v = [], []
        for l in range(self.GP_dim):
            gp_p_m_l, gp_p_v_l, mu_hat_l, A_hat_l = self.svgp.approximate_posterior_params(x, x,
                                                                    gp_mu[:, l], gp_var[:, l])
            inside_elbo_recon_l,  inside_elbo_kl_l = self.svgp.variational_loss(x=x, y=gp_mu[:, l],
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
        gp_KL_term = (gp_ce_term - inside_elbo)

        # KL term of Gaussian prior
        gaussian_prior_dist = Normal(torch.zeros_like(gaussian_mu), torch.ones_like(gaussian_var))
        gaussian_post_dist = Normal(gaussian_mu, torch.sqrt(gaussian_var))
        gaussian_KL_term = kl_divergence(gaussian_post_dist, gaussian_prior_dist).sum()

        # SAMPLE
        p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
        p_v = torch.cat((gp_p_v, gaussian_var), dim=1)
        latent_dist = Normal(p_m, torch.sqrt(p_v))
        latent_samples = []
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        recon_loss = 0
        for f in latent_samples:
            hidden_samples = self.decoder(torch.cat((f, batch), dim=1))
            mean_samples_ = self.dec_mean(hidden_samples)
            if self.shared_dispersion:
                disp_samples_ = torch.exp(torch.clamp(self.dec_disp, -15., 15.)).T
            else:
                disp_samples_ = torch.exp(torch.clamp(torch.matmul(self.dec_disp, batch.T), -15., 15.)).T

            mean_samples.append(mean_samples_)
            disp_samples.append(disp_samples_)
            recon_loss += self.NB_loss(x=raw_y, mean=mean_samples_, disp=disp_samples_, scale_factor=size_factors)
        recon_loss = recon_loss / num_samples

        noise_reg = 0
        if self.noise > 0:
            for _ in range(num_samples):
                qnet_mu_, qnet_var_ = self.encoder(torch.cat((y + torch.randn_like(y)*self.noise, batch), dim=1))

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
        if self.noise > 0 :
            elbo = recon_loss + noise_reg * self.input_dim / self.GP_dim + self.beta * gp_KL_term + self.beta * gaussian_KL_term
        else:
            elbo = recon_loss + self.beta * gp_KL_term + self.beta * gaussian_KL_term

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
            mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg


    def batching_latent_samples(self, X, Y, B, batch_size=512):
        """
        Output latent embedding.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        Y: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        B: array_like, shape (n_spots, n_batches)
            One-hot encoded batch IDs.
        """ 

        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        Y = torch.tensor(Y, dtype=self.dtype)
        B = torch.tensor(B, dtype=self.dtype)

        latent_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            ybatch = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            bbatch = B[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(torch.cat((ybatch, bbatch), dim=1))

            gp_mu = qnet_mu[:, 0:self.GP_dim]
            gp_var = qnet_var[:, 0:self.GP_dim]

            gaussian_mu = qnet_mu[:, self.GP_dim:]
#            gaussian_var = qnet_var[:, self.GP_dim:]

            gp_p_m, gp_p_v = [], []
            for l in range(self.GP_dim):
                gp_p_m_l, gp_p_v_l, _, _ = self.svgp.approximate_posterior_params(xbatch, xbatch, gp_mu[:, l], gp_var[:, l])
                gp_p_m.append(gp_p_m_l)
                gp_p_v.append(gp_p_v_l)

            gp_p_m = torch.stack(gp_p_m, dim=1)
            gp_p_v = torch.stack(gp_p_v, dim=1)

            # SAMPLE
            p_m = torch.cat((gp_p_m, gaussian_mu), dim=1)
            latent_samples.append(p_m.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)

        return latent_samples.numpy()


    def batching_denoise_counts(self, X, Y, B, n_samples=1, batch_size=512):
        """
        Output denoised counts.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        Y: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        B: array_like, shape (n_spots, n_batches)
            One-hot encoded batch IDs.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        """ 

        self.eval()

        X = torch.tensor(X, dtype=self.dtype)
        Y = torch.tensor(Y, dtype=self.dtype)
        B = torch.tensor(B, dtype=self.dtype)

        mean_samples = []

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            ybatch = Y[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)
            bbatch = B[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)].to(self.device)

            qnet_mu, qnet_var = self.encoder(torch.cat((ybatch, bbatch), dim=1))

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

            mean_samples_ = []
            for f in latent_samples:
                hidden_samples = self.decoder(torch.cat((f, bbatch), dim=1))
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)
    
            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.cpu().detach())

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
            recon_samples.append(mean_batch.cpu().detach())

        recon_samples = torch.cat(recon_samples, dim=0)

        return recon_samples.numpy()


    def differential_expression(self, group1_idx, group2_idx, num_denoise_samples=10000, batch_size=512, pos=None, ncounts=None, batch=None, 
                                gene_name=None, raw_counts=None, estimate_pseudocount=True):
        """
        Differential expression analysis.

        Parameters:
        -----------
        group1_idx: array_like, shape (n_group1)
            Index of group1.
        group2_idx: array_like, shape (n_group2)
            Index of group2.
        num_denoise_samples: Number of samplings in each group
        pos: array_like, shape (n_spots, 2)
            Location information.
        ncounts: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        batch: array_like, shape (n_spots, n_batches)
            One-hot encoded batch IDs.
        raw_counts: array_like, shape (n_spots, n_genes)
            Raw count matrix.
        estimate_pseudocount: Whether to estimate pseudocount from data, otherwise use default value 0.05.
        """ 

        group1_idx_sampling_array = []
        group2_idx_sampling_array = []
        batch_array = np.argmax(batch, axis=1)
        avail_n_batches = 0
        avail_n_batches_array = []
        for batch_val in set(batch_array):
            if np.sum(batch_array[group1_idx]==batch_val) == 0 or np.sum(batch_array[group2_idx]==batch_val) == 0:
                continue

            batch_group1_idx = group1_idx[np.where(np.array(batch_array[group1_idx]==batch_val, dtype=bool))[0]]
            batch_group1_idx_sampling = batch_group1_idx[np.random.randint(batch_group1_idx.shape[0], size=num_denoise_samples)]
            group1_idx_sampling_array.append(batch_group1_idx_sampling)

            batch_group2_idx = group2_idx[np.where(np.array(batch_array[group2_idx]==batch_val, dtype=bool))[0]]
            batch_group2_idx_sampling = batch_group2_idx[np.random.randint(batch_group2_idx.shape[0], size=num_denoise_samples)]
            group2_idx_sampling_array.append(batch_group2_idx_sampling)

            avail_n_batches += 1
            avail_n_batches_array.append(batch_val)

        group1_idx_sampling = np.concatenate(group1_idx_sampling_array)
        group2_idx_sampling = np.concatenate(group2_idx_sampling_array)

        group1_denoised_counts = self.batching_denoise_counts(X=pos[group1_idx_sampling], Y=ncounts[group1_idx_sampling], B=batch[group1_idx_sampling], batch_size=batch_size, n_samples=1)
        group2_denoised_counts = self.batching_denoise_counts(X=pos[group2_idx_sampling], Y=ncounts[group2_idx_sampling], B=batch[group2_idx_sampling], batch_size=batch_size, n_samples=1)

        if estimate_pseudocount:
            group1_where_zero = np.max(raw_counts[group1_idx], axis=0) == 0
            group2_where_zero = np.max(raw_counts[group2_idx], axis=0) == 0
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
            eps = np.clip(eps, a_min=0.05, a_max=0.5)
            print("Estimated pseudocounts", eps)
        else:
            eps = 0.05

        group1_denoised_mean_array = []
        group2_denoised_mean_array = []
        lfc_array = []
        group1_raw_mean_array = []
        group2_raw_mean_array = []
        for b in range(avail_n_batches):
            group1_denoised_mean = np.mean(group1_denoised_counts[b*num_denoise_samples:(b+1)*num_denoise_samples], axis=0)
            group1_denoised_mean_array.append(group1_denoised_mean)
            group2_denoised_mean = np.mean(group2_denoised_counts[b*num_denoise_samples:(b+1)*num_denoise_samples], axis=0)
            group2_denoised_mean_array.append(group2_denoised_mean)

            lfc_ = np.log2(group1_denoised_mean + eps) - np.log2(group2_denoised_mean + eps)
            lfc_array.append(lfc_)

            group1_raw_mean = np.mean(raw_counts[group1_idx_sampling_array[b]], axis=0)
            group1_raw_mean_array.append(group1_raw_mean)
            group2_raw_mean = np.mean(raw_counts[group2_idx_sampling_array[b]], axis=0)
            group2_raw_mean_array.append(group2_raw_mean)

        lfc_array = np.stack(lfc_array, axis=0)
        lfc = np.mean(lfc_array, axis=0)
        p_lfc = np.log2(group1_denoised_counts + eps) - np.log2(group2_denoised_counts + eps)
        mean_lfc = np.mean(p_lfc, axis=0)
        median_lfc = np.median(p_lfc, axis=0)
        sd_lfc = np.std(p_lfc, axis=0)
        delta = gmm_fit(data=mean_lfc)
        print("LFC delta:", delta)
        is_de = (np.abs(p_lfc) >= delta).mean(0)
        not_de = (np.abs(p_lfc) < delta).mean(0)
        bayes_factor = np.log(is_de + 1e-10) - np.log(not_de + 1e-10)

        data = [lfc, mean_lfc, median_lfc, sd_lfc, is_de, not_de, bayes_factor]
        data += group1_denoised_mean_array
        data += group2_denoised_mean_array
        data += group1_raw_mean_array
        data += group2_raw_mean_array

        cols = ['LFC', 'mean_LFC', 'median_LFC', 'sd_LFC', 'prob_DE', 'prob_not_DE', 'bayes_factor']
        denoised_mean1_name, denoised_mean2_name, raw_mean1_name, raw_mean2_name = [], [], [], []
        for b in range(avail_n_batches):
            denoised_mean1_name.append('denoised_mean1_batch'+str(avail_n_batches_array[b]))
            denoised_mean2_name.append('denoised_mean2_batch'+str(avail_n_batches_array[b]))
            raw_mean1_name.append('raw_mean1_batch'+str(avail_n_batches_array[b]))
            raw_mean2_name.append('raw_mean2_batch'+str(avail_n_batches_array[b]))
        cols = cols+denoised_mean1_name+denoised_mean2_name+raw_mean1_name+raw_mean2_name

        res = {}
        for i in range(len(data)):
            res[cols[i]] = data[i]

        res_dat = pd.DataFrame(data=res, index=gene_name)
            
        return res_dat


    def train_model(self, pos, ncounts, raw_counts, size_factors, batch, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, 
        train_size=0.95, maxiter=2000, patience=50, save_model=True, model_weights="model.pt", print_kernel_scale=True):
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
        size_factor: array_like, shape (n_samples)
            The size factor of each sample, which need for the NB loss.
        batch: array_like, shape (n_spots, n_batches)
            One-hot encoded batch IDs.
        lr: float, defalut = 0.001
            Learning rate for the opitimizer.
        weight_decay: float, defalut = 0.001
            Weight decay for the opitimizer.
        num_samples: integer, default = 1
            number of samplings of the posterior distribution of latent embedding.
        maxiter: int, default = 5000
            Maximum number of iterations.
        model_weights: str
            File name to save the model weights.
        print_kernel_scale: bool
            Whether to print current kernel scale during training steps.
        """

        self.train()

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(ncounts, dtype=self.dtype), 
                        torch.tensor(raw_counts, dtype=self.dtype), torch.tensor(size_factors, dtype=self.dtype),
                        torch.tensor(batch, dtype=self.dtype))
        if train_size < 1:
            train_dataset, validate_dataset = random_split(dataset=dataset, lengths=[train_size, 1.-train_size])
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = dataset

        if ncounts.shape[0] > batch_size:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        early_stopping = EarlyStopping(patience=patience, modelfile=model_weights)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        queue = deque()

        print("Training")

        for epoch in range(maxiter):
            elbo_val = 0
            recon_loss_val = 0
            gp_KL_term_val = 0
            gaussian_KL_term_val = 0
            noise_reg_val = 0
            num = 0
            for batch_idx, (x_batch, y_batch, y_raw_batch, sf_batch, b_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                b_batch = b_batch.to(self.device)
                y_raw_batch = y_raw_batch.to(self.device)
                sf_batch = sf_batch.to(self.device)

                elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg = \
                    self.forward(x=x_batch, y=y_batch, batch=b_batch, raw_y=y_raw_batch, size_factors=sf_batch, num_samples=num_samples)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                gp_KL_term_val += gp_KL_term.item()
                gaussian_KL_term_val += gaussian_KL_term.item()
                if self.noise > 0:
                    noise_reg_val += noise_reg.item()

                num += x_batch.shape[0]

                if self.dynamicVAE:
                    KL_val = (gp_KL_term.item() + gaussian_KL_term.item()) / x_batch.shape[0]
                    queue.append(KL_val)
                    avg_KL = np.mean(queue)
                    self.beta, _ = self.PID.pid(self.KL_loss*(self.GP_dim+self.Normal_dim), avg_KL)
                    if len(queue) >= 10:
                        queue.popleft()

            elbo_val = elbo_val/num
            recon_loss_val = recon_loss_val/num
            gp_KL_term_val = gp_KL_term_val/num
            gaussian_KL_term_val = gaussian_KL_term_val/num
            noise_reg_val = noise_reg_val/num

            print('Training epoch {}, ELBO:{:.8f}, NB loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f}, noise regularization:{:8f}'.format(epoch+1, elbo_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val, noise_reg_val))
            print('Current beta', self.beta)
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

            if train_size < 1:
                validate_elbo_val = 0
                validate_num = 0
                for _, (validate_x_batch, validate_y_batch, validate_y_raw_batch, validate_sf_batch, validate_b_batch) in enumerate(validate_dataloader):
                    validate_x_batch = validate_x_batch.to(self.device)
                    validate_y_batch = validate_y_batch.to(self.device)
                    validate_b_batch = validate_b_batch.to(self.device)
                    validate_y_raw_batch = validate_y_raw_batch.to(self.device)
                    validate_sf_batch = validate_sf_batch.to(self.device)

                    validate_elbo, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                        self.forward(x=validate_x_batch, y=validate_y_batch, batch=validate_b_batch, raw_y=validate_y_raw_batch, size_factors=validate_sf_batch, num_samples=num_samples)

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
