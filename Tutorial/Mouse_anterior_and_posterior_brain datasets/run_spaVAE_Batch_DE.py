import math, os
from time import time

import torch
from spaVAE_Batch import SPAVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
    parser.add_argument('--data_file', default='anterior_section1.h5_anterior_section2.h5_posterior_section1.h5_posterior_section2.h5_union_all_sgene.h5')
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--patience', default=100, type=int)
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
    parser.add_argument('--shared_dispersion', default=False, type=bool)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_steps', default=6, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--allow_batch_kernel_scale', default=True, type=bool)
    parser.add_argument('--save_dir', default='ES_model/')
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--denoised_counts_file', default='denoised_counts')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    print(args)

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).astype('float64')
    loc = np.array(data_mat['pos']).astype('float64')
    loc = loc.T
    batch = np.array(data_mat['batch']).astype('float64')
    gene = np.array(data_mat['gene']).astype('U26')
    data_mat.close()

    n_batch = batch.shape[1]

    loc_scaled = np.zeros(loc.shape, dtype=np.float64)
    for i in range(n_batch):
        scaler = MinMaxScaler()
        b_loc = loc[batch[:,i]==1, :]
        b_loc = scaler.fit_transform(b_loc) * 20.
        loc_scaled[batch[:,i]==1, :] = b_loc
    loc = loc_scaled

    loc = np.concatenate((loc, batch), axis=1)

    np.savetxt("scaled_loc.txt", loc, delimiter=",")


    print(x.shape)
    print(loc.shape)
    print(batch.shape)

    # build inducing point matrix with batch index
    eps = 1e-4
    initial_inducing_points_0 = []
    initial_inducing_points_shape = []
    for i in range(n_batch):
        if i==0 or i==1:
            n_steps = 8
        else:
            n_steps = 15
        initial_inducing_points_0_ = np.mgrid[0:(1+eps):(1./n_steps), 0:(1+eps):(1./n_steps)].reshape(2, -1).T * args.loc_range
        initial_inducing_points_0.append(initial_inducing_points_0_)
        initial_inducing_points_shape.append(initial_inducing_points_0_.shape[0])
    initial_inducing_points_0 = np.concatenate(initial_inducing_points_0, axis=0)
    print(initial_inducing_points_shape)
    initial_inducing_points_1 = []
    for i in range(n_batch):
        initial_inducing_points_1_ = np.zeros((initial_inducing_points_shape[i], n_batch))
        initial_inducing_points_1_[:, i] = 1
        initial_inducing_points_1.append(initial_inducing_points_1_)
    initial_inducing_points_1 = np.concatenate(initial_inducing_points_1, axis=0)
    initial_inducing_points = np.concatenate((initial_inducing_points_0, initial_inducing_points_1), axis=1)
    print(initial_inducing_points.shape)

    np.savetxt("initial_inducing_points.txt", initial_inducing_points, delimiter=",")

    adata = sc.AnnData(x, dtype="float64")

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    model = SPAVAE(input_dim=adata.n_vars, z_dim=args.z_dim, n_batch=n_batch, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD, shared_dispersion=args.shared_dispersion,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, allow_batch_kernel_scale=args.allow_batch_kernel_scale,
        N_train=adata.n_obs, beta=args.beta, dtype=torch.float64, device=args.device)

    print(str(model))

    model.load_model(args.model_file)

    y = np.loadtxt("louvain_clustering_labels.txt").astype("int")

    # raw counts
    raw_adata = sc.AnnData(x, dtype="float64")

    raw_adata = normalize(raw_adata,
                      size_factors=True,
                      normalize_input=False,
                      logtrans_input=False)

    for i in range(n_batch):
        for j in np.unique(y):
            layer_idx = np.where(np.logical_and(y==j, batch[:,i]==1))[0]
            if len(layer_idx) < 50:
                continue
            not_layer_idx = np.where(np.logical_and(y!=j, batch[:,i]==1))[0]

            print("cluster:", j, ", batch:", i)

            res_dat = model.differentail_expression(group1_idx=layer_idx, group2_idx=not_layer_idx, num_denoise_samples=args.num_denoise_samples,
                            batch_size=args.batch_size, pos=loc, ncounts=adata.X, batch=batch, gene_name=gene, raw_counts=raw_adata.X)

            res_dat.to_csv(args.denoised_counts_file+"_cluster_"+str(j)+"_batch_"+str(i)+"_LFC.txt")
