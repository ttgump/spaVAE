from cgi import test
import math, os
from time import time

import torch
from spaVAE import SPAVAE
import numpy as np
# import lhsmdu
from sklearn.preprocessing import MinMaxScaler
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
    parser.add_argument('--data_file', default='sample_151673.h5')
    parser.add_argument('--select_genes', default=0, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--mask_prob', default=0.1, type=float)
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
    parser.add_argument('--train_final_latent_file', default='train_final_latent.txt')
    parser.add_argument('--train_denoised_counts_file', default='train_denoised_counts.txt')
    parser.add_argument('--test_final_latent_file', default='test_final_latent.txt')
    parser.add_argument('--test_denoised_counts_file', default='test_denoised_counts.txt')
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
        np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * 20.

    print(x.shape)
    print(loc.shape)

    eps = 1e-5
    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * 20.
    print(initial_inducing_points.shape)

    sample_idx = np.arange(x.shape[0])
    np.random.shuffle(sample_idx)
    train_idx, test_idx = sample_idx[int(args.mask_prob*x.shape[0]):], sample_idx[:int(args.mask_prob*x.shape[0])]
    np.savetxt(args.data_file[:-3]+"_train_index.txt", train_idx, delimiter=",", fmt="%i")
    np.savetxt(args.data_file[:-3]+"_test_index.txt", test_idx, delimiter=",", fmt="%i")
    x_train, x_test = x[train_idx], x[test_idx]
    loc_train, loc_test = loc[train_idx], loc[test_idx]
    print(x_train.shape, x_test.shape)
    print(loc_train.shape, loc_test.shape)

    adata_train = sc.AnnData(x_train, dtype="float64")

    adata_train = normalize(adata_train,
                    size_factors=True,
                    normalize_input=True,
                    logtrans_input=True)

    model = SPAVAE(input_dim=adata_train.n_vars, z_dim=args.z_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata_train.n_obs, beta=args.beta, dtype=torch.float64, 
        device=args.device)

    print(str(model))

    t0 = time()

    model.train_model(pos=loc_train, ncounts=adata_train.X, raw_counts=adata_train.raw.X, size_factors=adata_train.obs.size_factors,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                maxiter=args.maxiter, save_model=True, model_weights=args.model_file)
    print('Training time: %d seconds.' % int(time() - t0))

    final_latent = model.batching_latent_samples(X=loc_train, Y=adata_train.X, batch_size=args.batch_size)
    np.savetxt(args.data_file[:-3]+"_"+args.train_final_latent_file, final_latent, delimiter=",")


    denoised_counts = model.batching_denoise_counts(X=loc_train, Y=adata_train.X, batch_size=args.batch_size, n_samples=25)
    np.savetxt(args.data_file[:-3]+"_"+args.train_denoised_counts_file, denoised_counts, delimiter=",")

    test_latent, test_denoised_counts = model.batching_predict_samples(X_test=loc_test, X_train=loc_train, Y_train=adata_train.X, batch_size=args.batch_size, n_samples=25)
    np.savetxt(args.data_file[:-3]+"_"+args.test_final_latent_file, test_latent, delimiter=",")
    np.savetxt(args.data_file[:-3]+"_"+args.test_denoised_counts_file, test_denoised_counts, delimiter=",")
