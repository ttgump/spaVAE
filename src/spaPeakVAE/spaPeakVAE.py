import math
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
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


class SPAPEAKVAE(nn.Module):
    def __init__(self, input_dim, GP_dim, Normal_dim, encoder_layers, decoder_layers, noise, encoder_dropout, decoder_dropout, 
                    fixed_inducing_points, initial_inducing_points, fixed_gp_params, kernel_scale, N_train, 
                    KL_loss, dynamicVAE, init_beta, min_beta, max_beta, dtype, device):
        super(SPAPEAKVAE, self).__init__()
        torch.set_default_dtype(dtype)
        self.svgp = SVGP(fixed_inducing_points=fixed_inducing_points, initial_inducing_points=initial_inducing_points,
                fixed_gp_params=fixed_gp_params, kernel_scale=kernel_scale, jitter=1e-8, N_train=N_train, dtype=dtype, device=device)
        self.input_dim = input_dim
        self.PID = PIDControl(Kp=0.01, Ki=-0.005, init_beta=init_beta, min_beta=min_beta, max_beta=max_beta)
        self.KL_loss = KL_loss
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta        # beta controls the weight of reconstruction loss
        self.dtype = dtype
        self.GP_dim = GP_dim    # dimension of latent Gaussian process embedding
        self.Normal_dim = Normal_dim    # dimension of latent standard Gaussian embedding
        self.noise = noise      # intensity of random noise
        self.device = device
        self.encoder = DenseEncoder(input_dim=input_dim, hidden_dims=encoder_layers, output_dim=GP_dim+Normal_dim, activation="elu", dropout=encoder_dropout)
        self.decoder = buildNetwork([GP_dim+Normal_dim]+decoder_layers, activation="elu", dropout=decoder_dropout)
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(nn.Linear(decoder_layers[-1], input_dim), nn.Sigmoid())
        else:
            self.dec_mean = nn.Sequential(nn.Linear(GP_dim+Normal_dim, input_dim), nn.Sigmoid())

        self.l_encoder = buildNetwork([input_dim]+encoder_layers, activation="elu", dropout=encoder_dropout)
        self.l_encoder.append(nn.Linear(encoder_layers[-1], 1))

        self.peak_bias = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

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
        y: mini-batch of preprocessed counts.
        num_samples: number of samplings of the posterior distribution of latent embedding.
        """ 

        self.train()

        l_samples = self.l_encoder(y)

        b = y.shape[0]
        qnet_mu, qnet_var = self.encoder(y)

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
                                                                    noise=qnet_var[:, l], mu_hat=mu_hat_l,
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
        mean_samples = []
        disp_samples = []
        for _ in range(num_samples):
            latent_samples_ = latent_dist.rsample()
            latent_samples.append(latent_samples_)

        recon_loss = 0
        for f in latent_samples:
            hidden_samples = self.decoder(f)
            mean_samples_ = self.dec_mean(hidden_samples)

            recon_loss += self.BCE_loss(torch.sigmoid(l_samples)[:None] * mean_samples_ * torch.sigmoid(self.peak_bias).unsqueeze(0), y).sum()
        recon_loss = recon_loss / num_samples

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
                noise_reg += torch.sum((p_m - p_m_)**2)
            noise_reg = noise_reg / num_samples


        # ELBO
        if self.noise > 0 :
            elbo = recon_loss + noise_reg * self.input_dim / self.GP_dim + self.beta * gp_KL_term + self.beta * gaussian_KL_term
        else:
            elbo = recon_loss + self.beta * gp_KL_term + self.beta * gaussian_KL_term

        return elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, gp_ce_term, p_m, p_v, qnet_mu, qnet_var, \
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
            latent_samples.append(p_m.data.cpu().detach())

        latent_samples = torch.cat(latent_samples, dim=0)

        return latent_samples.numpy()


    def batching_denoise_counts(self, X, Y, n_samples=1, batch_size=512, binary=False, threshold=0.5):
        """
        Output denoised counts.

        Parameters:
        -----------
        X: array_like, shape (n_spots, 2)
            Location information.
        Y: array_like, shape (n_spots, n_genes)
            Preprocessed count matrix.
        num_samples: Number of samplings of the posterior distribution of latent embedding. The denoised counts are average of the samplings.
        binary: whether to binarizate the outputs.
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
                hidden_samples = self.decoder(f)
                mean_samples_f = self.dec_mean(hidden_samples)
                mean_samples_.append(mean_samples_f)

            mean_samples_ = torch.stack(mean_samples_, dim=0)
            mean_samples_ = torch.mean(mean_samples_, dim=0)
            mean_samples.append(mean_samples_.data.cpu().detach())

        mean_samples = torch.cat(mean_samples, dim=0)

        if binary:
#            mean_samples = (mean_samples.T > Y.mean(1).T).T & (mean_samples>Y.mean(0))
            mean_samples = (mean_samples >= threshold).int()

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


    def batching_predict_samples(self, X_test, X_train, Y_train, n_samples=1, batch_size=512, binary=False, threshold=0.5):
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
        binary: whether to binarizate the outputs.
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

        if binary:
            mean_samples = (mean_samples >= threshold).int()

        return latent_samples.numpy(), mean_samples.numpy()


    def differential_accessibility(self, group1_idx, group2_idx, num_denoise_samples=10000, batch_size=512, pos=None, counts=None, 
            peak_name=None):
        """
        Differential accessibility analysis.

        Parameters:
        -----------
        group1_idx: array_like, shape (n_group1)
            Index of group1.
        group2_idx: array_like, shape (n_group2)
            Index of group2.
        num_denoise_samples: Number of samplings in each group.
        pos: array_like, shape (n_spots, 2)
            Location information.
        counts: array_like, shape (n_spots, n_peakss)
            Preprocessed count matrix.
        peak_name: array_like, shape (n_peakss)
            Peak names.
        """ 

        group1_idx_sampling = group1_idx[np.random.randint(group1_idx.shape[0], size=num_denoise_samples)]
        group2_idx_sampling = group2_idx[np.random.randint(group2_idx.shape[0], size=num_denoise_samples)]

        group1_denoised_counts = self.batching_denoise_counts(X=pos[group1_idx_sampling], Y=counts[group1_idx_sampling], batch_size=batch_size, n_samples=1)
        group2_denoised_counts = self.batching_denoise_counts(X=pos[group2_idx_sampling], Y=counts[group2_idx_sampling], batch_size=batch_size, n_samples=1)

        group1_denoised_mean = np.mean(group1_denoised_counts, axis=0)
        group2_denoised_mean = np.mean(group2_denoised_counts, axis=0)

        eps = 1e-4 # small offset to avoid log(0)

        lfc = np.log2(group1_denoised_mean + eps) - np.log2(group2_denoised_mean + eps)
        p_lfc = np.log2(group1_denoised_counts + eps) - np.log2(group2_denoised_counts + eps)
        mean_lfc = np.mean(p_lfc, axis=0)
        median_lfc = np.median(p_lfc, axis=0)
        sd_lfc = np.std(p_lfc, axis=0)
        delta = gmm_fit(data=mean_lfc)
        print("LFC delta:", delta)
        is_de = (np.abs(p_lfc) >= delta).mean(0)
        not_de = (np.abs(p_lfc) < delta).mean(0)
        bayes_factor = np.log(is_de + 1e-10) - np.log(not_de + 1e-10)


        group1_raw_mean = np.mean(counts[group1_idx], axis=0)
        group2_raw_mean = np.mean(counts[group2_idx], axis=0)

        res_dat = pd.DataFrame(data={'LFC': lfc, 'mean_LFC': mean_lfc, 'median_LFC': median_lfc, 'sd_LFC': sd_lfc,
                                     'prob_DE': is_de, 'prob_not_DE': not_de, 'bayes_factor': bayes_factor,
                                     'denoised_mean1': group1_denoised_mean, 'denoised_mean2': group2_denoised_mean,
                                    'raw_mean1': group1_raw_mean, 'raw_mean2': group2_raw_mean}, index=peak_name)
            
        return res_dat


    def train_model(self, pos, counts, lr=0.001, weight_decay=0.001, batch_size=512, num_samples=1, 
            train_size=0.95, maxiter=5000, patience=200, save_model=True, model_weights="model.pt", print_kernel_scale=True):
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

        dataset = TensorDataset(torch.tensor(pos, dtype=self.dtype), torch.tensor(counts, dtype=self.dtype))

        if train_size < 1:
            train_dataset, validate_dataset = random_split(dataset=dataset, lengths=[train_size, 1.-train_size])
            validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        else:
            train_dataset = dataset

        if counts.shape[0] > batch_size:
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
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                elbo, recon_loss, gp_KL_term, gaussian_KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, \
                    mean_samples, disp_samples, inside_elbo_recon, inside_elbo_kl, latent_samples, noise_reg = \
                    self.forward(x=x_batch, y=y_batch, num_samples=num_samples)

                self.zero_grad()
                elbo.backward(retain_graph=True)
                optimizer.step()

                elbo_val += elbo.item()
                recon_loss_val += recon_loss.item()
                gp_KL_term_val += gp_KL_term.item()
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
            gp_KL_term_val = gp_KL_term/num
            gaussian_KL_term_val = gaussian_KL_term/num
            noise_reg_val = noise_reg_val/num

            print('Training epoch {}, ELBO:{:.8f}, BCE loss:{:.8f}, GP KLD loss:{:.8f}, Gaussian KLD loss:{:.8f}, noise regularization:{:8f}'.format(epoch+1, elbo_val, recon_loss_val, gp_KL_term_val, gaussian_KL_term_val, noise_reg_val))
            print('Current beta', self.beta)
            if print_kernel_scale:
                print('Current kernel scale', torch.clamp(F.softplus(self.svgp.kernel.scale), min=1e-10, max=1e4).data)

            if train_size < 1:
                validate_elbo_val = 0
                validate_num = 0
                for _, (validate_x_batch, validate_y_batch) in enumerate(validate_dataloader):
                    validate_x_batch = validate_x_batch.to(self.device)
                    validate_y_batch = validate_y_batch.to(self.device)

                    validate_elbo, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
                    self.forward(x=validate_x_batch, y=validate_y_batch, num_samples=num_samples)

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

