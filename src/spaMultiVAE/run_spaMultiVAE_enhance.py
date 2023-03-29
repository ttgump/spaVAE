import math, os
from time import time

import torch
from spaMultiVAE import SPAMULTIVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import h5py
import scanpy as sc
from preprocess import normalize, geneSelection


# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--select_proteins', default=0, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--gene_noise', default=0, type=float)
    parser.add_argument('--protein_noise', default=0, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64, 32], type=int)
    parser.add_argument('--z_dim', default=2, type=int)
    parser.add_argument('--gene_decoder_layers', nargs="+", default=[32, 64], type=int)
    parser.add_argument('--protein_decoder_layers', nargs="+", default=[32, 32], type=int)
    parser.add_argument('--beta', default=20, type=float,
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_steps', default=15, type=int)
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
    print(args)

    data_mat = h5py.File(args.data_file, 'r')
    x1 = np.array(data_mat['X_gene']).astype('float64')
    x2 = np.array(data_mat['X_protein']).astype('float64')
    loc = np.array(data_mat['pos']).astype('float64')
    loc = loc.T
    data_mat.close()

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

    gm = GaussianMixture(n_components=2, covariance_type="diag", n_init=20).fit(adata2_no_scale.X)
    back_idx = np.argmin(gm.means_, axis=0)
    protein_log_back_mean = np.log(np.expm1(gm.means_[back_idx, np.arange(adata2_no_scale.n_vars)]))
    protein_log_back_scale = np.sqrt(gm.covariances_[back_idx, np.arange(adata2_no_scale.n_vars)])
    print("protein_back_mean shape", protein_log_back_mean.shape)

    model = SPAMULTIVAE(gene_dim=adata1.n_vars, protein_dim=adata2.n_vars, z_dim=args.z_dim, encoder_layers=args.encoder_layers, 
        gene_decoder_layers=args.gene_decoder_layers, protein_decoder_layers=args.protein_decoder_layers,
        gene_noise=args.gene_noise, protein_noise=args.protein_noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata1.n_obs, beta=args.beta, 
        protein_back_mean=protein_log_back_mean, protein_back_scale=protein_log_back_scale, dtype=torch.float64, 
        device=args.device)

    print(str(model))

    model.load_model(args.model_file)

    # Enhancing spatial resolutions
    neigh = NearestNeighbors(n_neighbors=2).fit(loc)
    nearest_dist = neigh.kneighbors(loc, n_neighbors=2)[0]
    small_distance = np.median(nearest_dist[:,1])/4
    loc_new1 = np.empty_like(loc)
    loc_new2 = np.empty_like(loc)
    loc_new3 = np.empty_like(loc)
    loc_new4 = np.empty_like(loc)
    loc_new1[:] = loc
    loc_new2[:] = loc
    loc_new3[:] = loc
    loc_new4[:] = loc
    loc_new1[:,0] = loc_new1[:,0] - small_distance
    loc_new1[:,1] = loc_new1[:,1] + small_distance
    loc_new2[:,0] = loc_new2[:,0] + small_distance
    loc_new2[:,1] = loc_new2[:,1] + small_distance
    loc_new3[:,0] = loc_new3[:,0] - small_distance
    loc_new3[:,1] = loc_new3[:,1] - small_distance
    loc_new4[:,0] = loc_new4[:,0] + small_distance
    loc_new4[:,1] = loc_new4[:,1] - small_distance
    loc_enhance = np.concatenate((loc_new1, loc_new2, loc_new3, loc_new4, loc), axis=0)

    _, enhanced_gene_denoised_counts, enhanced_protein_denoised_counts = model.batching_predict_samples(X_test=loc_enhance, X_train=loc, gene_Y_train=adata1.X, protein_Y_train=adata2.X, batch_size=args.batch_size, n_samples=25)
    np.savetxt(args.gene_enhanced_denoised_counts_file, enhanced_gene_denoised_counts, delimiter=",")
    np.savetxt(args.protein_enhanced_denoised_counts_file, enhanced_protein_denoised_counts, delimiter=",")
    np.savetxt(args.enhanced_loc_file, loc_enhance, delimiter=",")