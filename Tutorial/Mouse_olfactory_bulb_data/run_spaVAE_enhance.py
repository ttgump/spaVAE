import math, os
from time import time

import torch
from spaVAE import SPAVAE
import numpy as np
# import lhsmdu
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import h5py
import scanpy as sc
from preprocess import normalize, geneSelection

import igraph

# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='Rep1_MOB.h5')
    parser.add_argument('--select_genes', default=2000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--maxiter', default=2000, type=int)
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
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_steps', default=6, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--denoised_counts_file', default='denoised_counts.txt')
    parser.add_argument('--enhanced_counts_file', default='enhanced_counts.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    print(args)

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).astype('float64')
    loc = np.array(data_mat['pos']).astype('float64')
    data_mat.close()

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]
        np.savetxt(args.data_file[:-3]+"_selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * 20.

    print(x.shape)
    print(loc.shape)

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

    _, enhanced_counts = model.batching_predict_samples(X_test=loc_enhance, X_train=loc, Y_train=adata.X, batch_size=args.batch_size, n_samples=25)
    np.savetxt(args.data_file[:-3]+"_"+args.enhanced_counts_file, enhanced_counts, delimiter=",")
    np.savetxt(args.data_file[:-3]+"_enhanced_loc.txt", loc_enhance, delimiter=",")
