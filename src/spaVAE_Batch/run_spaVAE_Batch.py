import math, os
from time import time

import torch
from spaVAE_Batch import SPAVAE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py
import scanpy as sc
from preprocess import normalize

# torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='Spatial dependency-aware variational autoencoder for integrating batches of data',
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
    parser.add_argument('--encoder_layers', nargs="+", default=[128, 64], type=int)
    parser.add_argument('--GP_dim', default=2, type=int,help='dimension of the latent Gaussian process embedding')
    parser.add_argument('--Normal_dim', default=8, type=int,help='dimension of the latent standard Gaussian embedding')
    parser.add_argument('--decoder_layers', nargs="+", default=[128], type=int)
    parser.add_argument('--init_beta', default=10, type=float, help='initial coefficient of the KL loss')
    parser.add_argument('--min_beta', default=4, type=float, help='minimal coefficient of the KL loss')
    parser.add_argument('--max_beta', default=25, type=float, help='maximal coefficient of the KL loss')
    parser.add_argument('--KL_loss', default=0.025, type=float, help='desired KL_divergence value')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--shared_dispersion', default=False, type=bool)
    parser.add_argument('--fix_inducing_points', default=True, type=bool)
    parser.add_argument('--inducing_point_steps', default=6, type=int)
    parser.add_argument('--fixed_gp_params', default=False, type=bool)
    parser.add_argument('--loc_range', default=20., type=float)
    parser.add_argument('--kernel_scale', default=20., type=float)
    parser.add_argument('--allow_batch_kernel_scale', default=True, type=bool)
    parser.add_argument('--model_file', default='model.pt')
    parser.add_argument('--final_latent_file', default='final_latent.txt')
    parser.add_argument('--denoised_counts_file', default='denoised_counts.txt')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()

    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).astype('float64')
    loc = np.array(data_mat['pos']).astype('float64')
    batch = np.array(data_mat['batch']).astype('float64')
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

    n_batch = batch.shape[1]

    # scale locations per batch
    loc_scaled = np.zeros(loc.shape, dtype=np.float64)
    for i in range(n_batch):
        scaler = MinMaxScaler()
        b_loc = loc[batch[:,i]==1, :]
        b_loc = scaler.fit_transform(b_loc) * args.loc_range
        loc_scaled[batch[:,i]==1, :] = b_loc
    loc = loc_scaled

    loc = np.concatenate((loc, batch), axis=1)

    print(x.shape)
    print(loc.shape)
    print(batch.shape)

    # build inducing point matrix with batch index
    eps = 1e-5
    initial_inducing_points_0_ = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * args.loc_range
    initial_inducing_points_0 = np.tile(initial_inducing_points_0_, (n_batch, 1))
    initial_inducing_points_1 = []
    for i in range(n_batch):
        initial_inducing_points_1_ = np.zeros((initial_inducing_points_0_.shape[0], n_batch))
        initial_inducing_points_1_[:, i] = 1
        initial_inducing_points_1.append(initial_inducing_points_1_)
    initial_inducing_points_1 = np.concatenate(initial_inducing_points_1, axis=0)
    initial_inducing_points = np.concatenate((initial_inducing_points_0, initial_inducing_points_1), axis=1)
    print(initial_inducing_points.shape)

    adata = sc.AnnData(x, dtype="float64")

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    model = SPAVAE(input_dim=adata.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, n_batch=n_batch, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD, shared_dispersion=args.shared_dispersion,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, allow_batch_kernel_scale=args.allow_batch_kernel_scale,
        N_train=adata.n_obs, KL_loss=args.KL_loss, init_beta=args.init_beta, min_beta=args.min_beta, max_beta=args.max_beta, dtype=torch.float64, device=args.device)

    print(str(model))

    t0 = time()

    model.train_model(pos=loc, ncounts=adata.X, raw_counts=adata.raw.X, size_factors=adata.obs.size_factors, batch=batch,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=True, model_weights=args.model_file)
    print('Training time: %d seconds.' % int(time() - t0))

    final_latent = model.batching_latent_samples(X=loc, Y=adata.X, B=batch, batch_size=args.batch_size)
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
