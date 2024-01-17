import pandas as pd
import numpy as np
import pickle
import scipy
import torch
from utils import gen_adj_mat_tensor,cal_adj_mat_parameter
from torch_geometric.utils.convert import to_scipy_sparse_matrix,from_scipy_sparse_matrix

num_iter = 10000
thershold = 3
base_path = ''
dataset_name = 'sample_data'
data_path_node =  base_path + 'data/' + dataset_name +'/'
base = ['mRNA','DNA','miRNA']
new_base = ['n_mRNA','n_DNA','n_miRNA']
netw_base = ['1_te','1_tr','2_te','2_tr','3_te','3_tr']
lab_te = pd.read_csv('dataset/labels_te.csv', header=None)
lab_tr = pd.read_csv('dataset/labels_tr.csv', header=None)
labels = np.concatenate((lab_te.to_numpy(),lab_tr.to_numpy()),axis=0)[:,0]
k = 0
N = 0
ADJ = {}
for df_base in base:
    with open(base_path + 'dataset/' + netw_base[0+k] + '.csv') as f:
        df_te = pd.read_csv(f, header=None)
    with open(base_path + 'dataset/' + netw_base[1+k] + '.csv') as f:
        df_tr = pd.read_csv(f, header=None)
    k+=2
    df = torch.tensor(np.concatenate((df_te.to_numpy(),df_tr.to_numpy()),axis=0))
    PARA = cal_adj_mat_parameter(thershold, df, metric="cosine")
    param = PARA
    adj = gen_adj_mat_tensor(df, param, metric='cosine')
    adj = adj._indices()
    adj_mtx = to_scipy_sparse_matrix(adj).toarray()
    thr = np.sum(adj_mtx)/df.shape[0]
    dm1,dm2 = adj_mtx.shape
    ADJ[N] =adj_mtx
    Adj_Mtx_tr = from_scipy_sparse_matrix(scipy.sparse.coo_matrix(adj_mtx))[0]
    emb_file = data_path_node + 'edges_' + new_base[N] + '.pkl'
    with open(emb_file, 'wb') as f:
        pickle.dump(Adj_Mtx_tr, f)
    N+=1

