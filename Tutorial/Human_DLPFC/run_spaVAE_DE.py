import math, os
from time import time

import torch
from spaVAE import SPAVAE
import numpy as np
import pandas as pd
# import lhsmdu
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
import h5py
import scanpy as sc
from preprocess import normalize

import igraph

# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='sample_151673.h5')
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--noise', default=1, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64, 32], type=int)
    parser.add_argument('--z_dim', default=2, type=int)
    parser.add_argument('--decoder_layers', nargs="+", default=[32], type=int)
    parser.add_argument('--beta', default=20, type=float,
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--num_denoise_samples', default=10000, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_steps', default=6, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='151673_model.pt')
    parser.add_argument('--denoised_counts_file', default='sample_151673')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    print(args)

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).astype('float64')
    loc = np.array(data_mat['pos']).astype('float64')
    y = np.array(data_mat['Y']).astype('U26')
    gene = np.array(data_mat['gene']).astype('U26')
    data_mat.close()

    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * 20.

    print(x.shape)
    print(loc.shape)
    print(y.shape)

    eps = 1e-5
    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * 20.
    print(initial_inducing_points.shape)

    adata = sc.AnnData(x, dtype="float64")

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    model = SPAVAE(input_dim=adata.n_vars, z_dim=args.z_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata.n_obs, beta=args.beta, dtype=torch.float64, 
        device=args.device)

    print(str(model))

    model.load_model(args.model_file)

    # raw counts
    raw_adata = sc.AnnData(x, dtype="float64")

    raw_adata = normalize(raw_adata,
                      size_factors=True,
                      normalize_input=False,
                      logtrans_input=False)
    for layer in np.unique(y):
        if layer == 'NA':
            continue
        layer_idx = np.where(np.array(y==layer, dtype=bool))[0]
        not_layer_idx = np.where(np.logical_and(np.array(y!=layer, dtype=bool), np.array(y!='NA', dtype=bool)))[0]

        res_dat = model.differential_expression(group1_idx=layer_idx, group2_idx=not_layer_idx, num_denoise_samples=args.num_denoise_samples,
                            batch_size=args.batch_size, pos=loc, ncounts=adata.X, gene_name=gene, raw_counts=raw_adata.X)

        res_dat.to_csv(args.denoised_counts_file+"_"+layer+"_LFC.txt")
