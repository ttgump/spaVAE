import math, os
from time import time

import torch
from spaPeakLDVAE import SPAPEAKLDVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import h5py
import scanpy as sc
from preprocess import normalize, preprocessing_atac, geneSelection


# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='Solid_E12_rep1_spatialATAC.h5')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-2, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[512, 256, 128], type=int)
    parser.add_argument('--z_dim', default=5, type=int,help='dimension of the latent embedding')
    parser.add_argument('--beta', default=1, type=float,
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=19, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--spatial_score_file', default='spatial_score.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    print(args)

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).astype('float64') # count matrix
    loc = np.array(data_mat['pos']).astype('float64') # location information
    peak_name = np.array(data_mat['Peakname']).astype('U30') # peak names
    data_mat.close()

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]
        np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * 20.

    print(x.shape)
    print(loc.shape)

    # We provide two ways to generate inducing point, argument "grid_inducing_points" controls whether to choice grid inducing or k-means
    # One way is grid inducing points, argument "inducing_point_steps" controls number of grid steps, the resulting number of inducing point is (inducing_point_steps+1)^2
    # Another way is k-means on the locations, argument "inducing_point_nums" controls number of inducing points
    if args.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * 20.
        print(initial_inducing_points.shape)
    else:
        loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(loc)
        np.savetxt("location_centroids.txt", loc_kmeans.cluster_centers_, delimiter=",")
        np.savetxt("location_kmeans_labels.txt", loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    adata = sc.AnnData(x, dtype="float64")
    adata.var["name"] = peak_name

    adata = preprocessing_atac(adata)
    peak_name = adata.var["name"].values.astype('U30')

    model = SPAPEAKLDVAE(input_dim=adata.n_vars, z_dim=args.z_dim, encoder_layers=args.encoder_layers, encoder_dropout=args.dropoutE,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata.n_obs, beta=args.beta, dtype=torch.float64, 
        device=args.device)

    print(str(model))

    t0 = time()

    model.train_model(pos=loc, counts=adata.X, lr=args.lr, weight_decay=args.weight_decay, 
                batch_size=args.batch_size, num_samples=args.num_samples, maxiter=args.maxiter, save_model=True, 
                model_weights=args.model_file)
    print('Training time: %d seconds.' % int(time() - t0))

    spatial_score = model.spatial_score(gene_name=peak_name)
    spatial_score.to_csv(args.spatial_score_file)