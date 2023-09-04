import numpy as np
from sklearn.neighbors import kneighbors_graph
import scanpy as sc
import h5py
from preprocess import normalize

def MoranI(y, A):
#y is the normalzied expression of a gene
#A is a knn graph from locations (loc)
    s = sum(sum(A))
    N = len(y)
    y_=np.mean(y)
    y_f = y - y_
    y_f_1 = y_f.reshape(1,-1)
    y_f_2 = y_f.reshape(-1,1)
    r = sum(sum(A*np.dot(y_f_2,y_f_1)))
    l = sum(y_f**2)
    s2 = r/l
    s1 = N/s
    return s1*s2

data_mat = h5py.File("data.h5")
x = np.array(data_mat['X'])
loc = np.array(data_mat["pos"])
data_mat.close()

#build graph
A = kneighbors_graph(loc, 5, mode="connectivity", metric="euclidean", include_self=False, n_jobs=-1)
A = A.toarray()

#normalized data
adata = sc.AnnData(x, dtype="float64")

adata = normalize(adata,
                    size_factors=True,
                    normalize_input=True,
                    logtrans_input=True)

score = []
for j in range(adata.X.shape[1]):
    gene = adata.X[:,j]
    I = MoranI(gene, A)
    score.append(I)

score = np.stack(score)
np.savetxt("Moran_I.txt", score, delimiter=",")