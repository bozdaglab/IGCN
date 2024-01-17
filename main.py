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
netw_base = ['1_','2_','3_']
labels = pd.read_csv('dataset/labels_.csv', header=None).to_numpy()[:,0]

N = 0
ADJ = {}
for df_base in base:
    with open(base_path + 'dataset/' + netw_base[1] + '.csv') as f:
        df_tr = pd.read_csv(f, header=None)
    df = torch.tensor(df_tr.values)
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

