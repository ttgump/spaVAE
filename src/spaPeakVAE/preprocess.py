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
