import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
import h5py


def refine(sample_id, pred, dis, shape="square"):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape=="hexagon":
        num_nbs=6 
    elif shape=="square":
        num_nbs=4
    else:
        print("Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data.")
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp.iloc[0:(num_nbs+1)]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<num_nbs/2) and (np.max(v_c)>num_nbs/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
        if (i+1) % 1000 == 0:
            print("Processed", i+1, "lines")
    return np.array(refined_pred)


dat_mat = h5py.File("data_file.h5", 'r')
pos = np.array(dat_mat['pos']).astype('float64')
dat_mat.close()

pred = np.loadtxt('clustering_labels.txt')

print(pos.shape)
print(pred.shape)

dis = pairwise_distances(pos, metric="euclidean", n_jobs=-1).astype(np.double)
print("dis done")

refine_pred = refine(np.arange(pred.shape[0]), pred, dis)

np.savetxt("refined_clustering_labels.txt", refine_pred, delimiter=",", fmt="%i")