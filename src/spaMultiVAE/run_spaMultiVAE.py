import math, os
from time import time

import torch
from spaMultiVAE import SPAMULTIVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
import h5py
import scanpy as sc
from preprocess import normalize, geneSelection


# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Spatial dependency-aware variational autoencoder for spatial multi-omics data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--select_proteins', default=0, type=int)
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--lr', default=5e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--gene_noise', default=0, type=float)
    parser.add_argument('--protein_noise', default=0, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--GP_dim', default=2, type=int)
    parser.add_argument('--Normal_dim', default=18, type=int)
    parser.add_argument('--gene_decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--protein_decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=1, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_steps', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--gene_denoised_counts_file', default='gene_denoised_counts.txt')
    parser.add_argument('--protein_denoised_counts_file', default='protein_denoised_counts.txt')
    parser.add_argument('--protein_sigmoid_file', default='protein_sigmoid.txt')
    parser.add_argument('--gene_enhanced_denoised_counts_file', default='gene_enhanced_denoised_counts.txt')
    parser.add_argument('--protein_enhanced_denoised_counts_file', default='protein_enhanced_denoised_counts.txt')
    parser.add_argument('--enhanced_loc_file', default='enhanced_loc.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    data_mat = h5py.File(args.data_file, 'r')
    x1 = np.array(data_mat['X_gene']).astype('float64')     # gene count matrix
    x2 = np.array(data_mat['X_protein']).astype('float64')  # protein count matrix
    loc = np.array(data_mat['pos']).astype('float64')       # location information
    data_mat.close()

    if args.batch_size == "auto":
        if x1.shape[0] <= 1024:
            args.batch_size = 128
        elif x1.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)
        
    print(args)

    if args.select_genes > 0:
        importantGenes = geneSelection(x1, n=args.select_genes, plot=False)
        x1 = x1[:, importantGenes]
        np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

    if args.select_proteins > 0:
        importantProteins = geneSelection(x2, n=args.select_proteins, plot=False)
        x2 = x2[:, importantProteins]
        np.savetxt("selected_proteins.txt", importantProteins, delimiter=",", fmt="%i")

    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * args.loc_range

    print(x1.shape)
    print(x2.shape)
    print(loc.shape)

    eps = 1e-5
    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
    print(initial_inducing_points.shape)

    adata1 = sc.AnnData(x1, dtype="float64")
    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    adata2 = sc.AnnData(x2, dtype="float64")
    adata2 = normalize(adata2,
                      size_factors=False,
                      normalize_input=True,
                      logtrans_input=True)

    adata2_no_scale = sc.AnnData(x2, dtype="float64")
    adata2_no_scale = normalize(adata2_no_scale,
                      size_factors=False,
                      normalize_input=False,
                      logtrans_input=True)

    # Fit GMM model to the protein counts and use the smaller component as the initial values as protein background prior
    gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(adata2_no_scale.X)
    back_idx = np.argmin(gm.means_, axis=0)
    protein_log_back_mean = np.log(np.expm1(gm.means_[back_idx, np.arange(adata2_no_scale.n_vars)]))
    protein_log_back_scale = np.sqrt(gm.covariances_[back_idx, np.arange(adata2_no_scale.n_vars)])
    print("protein_back_mean shape", protein_log_back_mean.shape)

    model = SPAMULTIVAE(gene_dim=adata1.n_vars, protein_dim=adata2.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, 
        encoder_layers=args.encoder_layers, gene_decoder_layers=args.gene_decoder_layers, protein_decoder_layers=args.protein_decoder_layers,
        gene_noise=args.gene_noise, protein_noise=args.protein_noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata1.n_obs, KL_loss=args.KL_loss, init_beta=args.init_beta, min_beta=args.min_beta, 
        max_beta=args.max_beta, protein_back_mean=protein_log_back_mean, protein_back_scale=protein_log_back_scale, dtype=torch.float64, 
        device=args.device)

    print(str(model))

    t0 = time()

    model.train_model(pos=loc, gene_ncounts=adata1.X, gene_raw_counts=adata1.raw.X, gene_size_factors=adata1.obs.size_factors, 
                protein_ncounts=adata2.X, protein_raw_counts=adata2.raw.X,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
    print('Training time: %d seconds.' % int(time() - t0))

    final_latent = model.batching_latent_samples(X=loc, gene_Y=adata1.X, protein_Y=adata2.X, batch_size=args.batch_size)
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")


    gene_denoised_counts, protein_denoised_counts, protein_sigmoid = model.batching_denoise_counts(X=loc, gene_Y=adata1.X, protein_Y=adata2.X, batch_size=args.batch_size, n_samples=25)
    np.savetxt(args.gene_denoised_counts_file, gene_denoised_counts, delimiter=",")
    np.savetxt(args.protein_denoised_counts_file, protein_denoised_counts, delimiter=",")
    np.savetxt(args.protein_sigmoid_file, protein_sigmoid, delimiter=",")

