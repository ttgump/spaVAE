import math, os
from time import time

import torch
from spaLDVAE import SPALDVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
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
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--z_dim', default=15, type=int,help='dimension of the latent embedding')
    parser.add_argument('--beta', default=10, type=float,
                        help='coefficient of the reconstruction loss')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=None, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--permutate', default=0, type=int)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--spatial_score_file', default='spatial_score.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    print(args)

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).astype('float64') # count matrix
    loc = np.array(data_mat['pos']).astype('float64') # location information
    gene_name = np.array(data_mat['gene']).astype('U26') # gene names
    data_mat.close()

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]
        gene_name = gene_name[importantGenes]
        np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * args.loc_range

    print(x.shape)
    print(loc.shape)

    # We provide two ways to generate inducing point, argument "grid_inducing_points" controls whether to choice grid inducing or k-means
    # One way is grid inducing points, argument "inducing_point_steps" controls number of grid steps, the resulting number of inducing point is (inducing_point_steps+1)^2
    # Another way is k-means on the locations, argument "inducing_point_nums" controls number of inducing points
    if args.grid_inducing_points:
        eps = 1e-5
        initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
        print(initial_inducing_points.shape)
    else:
        loc_kmeans = KMeans(n_clusters=args.inducing_point_nums, n_init=100).fit(loc)
        np.savetxt("location_centroids.txt", loc_kmeans.cluster_centers_, delimiter=",")
        np.savetxt("location_kmeans_labels.txt", loc_kmeans.labels_, delimiter=",", fmt="%i")
        initial_inducing_points = loc_kmeans.cluster_centers_

    adata = sc.AnnData(x, dtype="float64")
    adata.var["name"] = gene_name
    sc.pp.filter_genes(adata, min_cells=100)
    gene_name = adata.var["name"].values.astype('U26')

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    model = SPALDVAE(input_dim=adata.n_vars, z_dim=args.z_dim, encoder_layers=args.encoder_layers, encoder_dropout=args.dropoutE,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata.n_obs, beta=args.beta, dtype=torch.float64, 
        device=args.device)

    print(str(model))

    if not os.path.isfile(args.model_file):
        t0 = time()
        model.train_model(pos=loc, ncounts=adata.X, raw_counts=adata.raw.X, size_factors=adata.obs.size_factors,
                    lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                    maxiter=args.maxiter, save_model=True, model_weights=args.model_file)
        print('Training time: %d seconds.' % int(time() - t0))
    else:
        model.load_model(args.model_file)

    spatial_score = model.spatial_score(X=loc, Y=adata.X, batch_size=args.batch_size, gene_name=gene_name)
    spatial_score.to_csv(args.spatial_score_file)

    if args.permutate > 0:
        for i in range(int(args.permutate)):
            model = SPALDVAE(input_dim=adata.n_vars, z_dim=args.z_dim, encoder_layers=args.encoder_layers, encoder_dropout=args.dropoutE,
                fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
                fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata.n_obs, beta=args.beta, dtype=torch.float64, 
                device=args.device)

            print("Permutate position iter", i+1)
            sample_idx = np.arange(loc.shape[0])
            np.random.shuffle(sample_idx)
            loc_permutate = loc[sample_idx]

            t0 = time()
            model.train_model(pos=loc_permutate, ncounts=adata.X, raw_counts=adata.raw.X, size_factors=adata.obs.size_factors,
                        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                        maxiter=args.maxiter, save_model=True, model_weights="Perturb_"+str(i+1)+"_"+args.model_file)
            print('Training time: %d seconds.' % int(time() - t0))

            spatial_score = model.spatial_score(X=loc, Y=adata.X, batch_size=args.batch_size, gene_name=gene_name)
            spatial_score.to_csv("Permutate"+str(i+1)+"_"+args.spatial_score_file)

