from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import scanpy as sc
import pylab as plt
import seaborn as sns
import pandas as pd
from scipy import sparse
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture


def read_dataset(adata, transpose=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def preprocessing_atac(
        adata, 
        min_genes=None, 
        min_cells=0.01, 
        n_top_genes=30000,
        target_sum=None,
        log=None
    ):
    """
    preprocessing
    """
    print('Raw dataset shape: {}'.format(adata.shape))
    if log: log.info('Preprocessing')
#    if not issparse(adata.X):
#        adata.X = sp.sparse.csr_matrix(adata.X)
        
    adata.X[adata.X>0] = 1
    
    if log: log.info('Filtering cells')
    if min_genes:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    
    if log: log.info('Filtering genes')
    if min_cells:
        if min_cells < 1:
            min_cells = min_cells * adata.shape[0]
        sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if n_top_genes:
        if log: log.info('Finding variable features')
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False, subset=True)
        # adata = epi.pp.select_var_feature(adata, nb_features=n_top_genes, show=False, copy=True)
    
    # if log: log.infor('Normalizing total per cell')
    # sc.pp.normalize_total(adata, target_sum=target_sum)
        
    if log: log.info('Batch specific maxabs scaling')
    
#     adata.X = maxabs_scale(adata.X)
#     adata = batch_scale(adata, chunk_size=chunk_size)
    
    print('Processed dataset shape: {}'.format(adata.shape))
    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6,3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    """
    Gene selection by mean-variance relationship
    """


    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
        A = data.multiply(data>threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (1-zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data>threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:,detected]>threshold
        logs = np.zeros_like(data[:,detected]) * np.nan
        logs[mask] = np.log2(data[:,detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan
            
    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low)/2
            else:
                low = xoffset
                xoffset = (xoffset + up)/2
        if verbose>0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
                
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold>0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1]+.1,.1)
        y = np.exp(-decay*(x - xoffset)) + yoffset
        if decay==1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected),xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected),decay,xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold==0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()
        
        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num,g in enumerate(markers):
                i = np.where(genes==g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i]+dx+.1, zeroRate[i]+dy, g, color='k', fontsize=labelsize)
    
    return selected
