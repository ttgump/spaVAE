import math, os
from time import time

import torch
from spaPeakVAE import SPAPEAKVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import h5py
import scanpy as sc
from preprocess import preprocessing_atac


# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Differential accessibility analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--batch_size', default="auto")
    parser.add_argument('--maxiter', default=5000, type=int)
    parser.add_argument('--train_size', default=0.95, type=float)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--noise', default=0, type=float)
    parser.add_argument('--dropoutE', default=0, type=float,
                        help='dropout probability for encoder')
    parser.add_argument('--dropoutD', default=0, type=float,
                        help='dropout probability for decoder')
    parser.add_argument('--encoder_layers', nargs="+", default=[1024, 128], type=int)
    parser.add_argument('--GP_dim', default=4, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=8, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[128, 1024], type=int)
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=4, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--num_denoise_samples', default=10000, type=int)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--grid_inducing_points', default=True, type=bool, 
                        help='whether to generate grid inducing points or use k-means centroids on locations as inducing points')
    parser.add_argument('--inducing_point_steps', default=None, type=int)
    parser.add_argument('--inducing_point_nums', default=None, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--denoised_counts_file', default='denoised_counts.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['Peakcounts']).astype('float64')      # count matrix
    loc = np.array(data_mat['Pos']).astype('float64')           # location information
    peak_name = np.array(data_mat['Peaknames']).astype('U30')   # peak names
    data_mat.close()

    if args.batch_size == "auto":
        if x.shape[0] <= 1024:
            args.batch_size = 128
        elif x.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)

    print(args)

    labels = np.loadtxt("annotated_labels.txt", dtype='str', delimiter="\n").astype('U30')

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
    adata.var["name"] = peak_name

    adata = preprocessing_atac(adata)
    peak_name = adata.var["name"].values.astype('U30')

    model = SPAPEAKVAE(input_dim=adata.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata.n_obs, KL_loss=args.KL_loss, init_beta=args.init_beta, min_beta=args.min_beta, 
        max_beta=args.max_beta, dtype=torch.float64, device=args.device)

    print(str(model))

    model.load_model(args.model_file)

    layer_idx_1 = np.where(np.array(labels=="Forebrain", dtype=bool))[0]
    layer_idx_2 = np.where(np.array(labels=="Midbrain", dtype=bool))[0]

    res_dat = model.differential_accessibility(group1_idx=layer_idx_1, group2_idx=layer_idx_2, num_denoise_samples=args.num_denoise_samples,
                        batch_size=args.batch_size, pos=loc, counts=adata.X, peak_name=peak_name)

    res_dat.to_csv("Forebrain_vs_Midbrain_LFC.txt")


    layer_idx_1 = np.where(np.array(labels=="Forebrain", dtype=bool))[0]
    layer_idx_2 = np.where(np.array(labels=="Hindbrain", dtype=bool))[0]

    res_dat = model.differential_accessibility(group1_idx=layer_idx_1, group2_idx=layer_idx_2, num_denoise_samples=args.num_denoise_samples,
                        batch_size=args.batch_size, pos=loc, counts=adata.X, peak_name=peak_name)

    res_dat.to_csv("Forebrain_vs_Hindbrain_LFC.txt")


    layer_idx_1 = np.where(np.array(labels=="Midbrain", dtype=bool))[0]
    layer_idx_2 = np.where(np.array(labels=="Hindbrain", dtype=bool))[0]

    res_dat = model.differential_accessibility(group1_idx=layer_idx_1, group2_idx=layer_idx_2, num_denoise_samples=args.num_denoise_samples,
                        batch_size=args.batch_size, pos=loc, counts=adata.X, peak_name=peak_name)

    res_dat.to_csv("Midbrain_vs_Hindbrain_LFC.txt")
