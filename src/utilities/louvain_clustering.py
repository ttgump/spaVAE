import scanpy as sc
import numpy as np

n_cluster = None # define the number of clusters here

latent = np.loadtxt("final_latent.txt", delimiter=",")
adata_latent = sc.AnnData(latent)
sc.pp.neighbors(adata_latent, n_neighbors=20, use_rep="X")
res = 0
while True:
    print("current resolution", res)
    sc.tl.louvain(adata_latent, resolution=res)
    y_pred = np.asarray(adata_latent.obs['louvain'], dtype=int)
    if np.unique(y_pred).shape[0] >= n_cluster:
        break
    res += 0.001

np.savetxt("louvain_clustering_labels.txt", y_pred, delimiter=",", fmt="%i")
