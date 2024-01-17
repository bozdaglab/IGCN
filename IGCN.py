base_path = ''
dataset_name = 'sample_data'
data_path_node =  base_path + 'data/' + dataset_name +'/'
save_path = base_path + 'data/' + 'node_emd/' + ''
base = ['mRNA','DNA','miRNA']
new_base = ['n_mRNA','n_DNA','n_miRNA']
netw_base = ['1_te','1_tr','2_te','2_tr','3_te','3_tr']

max_epochs = 1000
min_epochs = 200
patience = 30
xtimes1 = 10
xtimes2 = 10
learning_rates = [0.01,0.005,0.001] # learning rates to tune GCN
hd_sizes = [64,128,256,512] # hidden sizes to tune GCN
#  run
print('setting up!')
import statistics
from library import combine_module2
import pickle
import time
from sklearn.metrics import f1_score, accuracy_score,matthews_corrcoef
from sklearn.model_selection import  train_test_split,RepeatedStratifiedKFold
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import os
import torch
import errno
import warnings
import matplotlib
import main
matplotlib.use('macosx')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

path = base_path + "data/" + dataset_name
if not os.path.exists(path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)

device = torch.device("cpu")

def train():
    model.train()
    optimizer.zero_grad()
    out1,coef1,coef2,coef3 = model(DATA[0],DATA[1],DATA[2])
    loss1 = criterion(out1[DATA[0].train_mask], DATA[0].y[DATA[0].train_mask])
    loss1.backward()
    optimizer.step()
    return out1,loss1

def validate():
    model.eval()
    with torch.no_grad():
        out1,coef1,coef2,coef3 = model(DATA[0],DATA[1],DATA[2])
        pred = (out1).argmax(dim=1)
        coef1 = coef1
        coef2 = coef2
        coef3 = coef3
        loss1 = criterion(out1[DATA[0].valid_mask], DATA[0].y[DATA[0].valid_mask])
    return pred,loss1,coef1,coef2,coef3
k = 0
DF = {}
for i in range(len(base)):
    with open(base_path + 'dataset/' + netw_base[0+k] + '.csv') as f:
        df_te = pd.read_csv(f, header=None)
    with open(base_path + 'dataset/' + netw_base[1+k] + '.csv') as f:
        df_tr = pd.read_csv(f, header=None)
    k+=2
    df = torch.tensor(np.concatenate((df_te.to_numpy(),df_tr.to_numpy()),axis=0),device=device,dtype=torch.float32)
    DF[i] = df


EDGES = {}
for n in range(int(len(new_base))):
    with open(data_path_node + 'edges_' + new_base[n] + '.pkl', 'rb') as f:
        edge_index = pickle.load(f)
    edge_index = torch.tensor(edge_index,device=device)
    EDGES[n] = edge_index
criterion = torch.nn.CrossEntropyLoss()
lab_te = pd.read_csv('dataset/labels_te.csv', header=None)
lab_tr = pd.read_csv('dataset/labels_tr.csv', header=None)
labels = np.concatenate((lab_te.to_numpy(),lab_tr.to_numpy()),axis=0)[:,0]
av_result_acc = list()
av_result_wf1 = list()
av_result_mf1 = list()
av_result_mcc = list()
av_time = list()
# rand_states = [3,12,26,39,44,64,66,75,87,91]
rand_states = [64]
for IT in rand_states:
    print('running..')
    start = time.time()
    alltrain_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, shuffle=True, stratify=labels,
                                           random_state=IT)
    train_idx, val_idx = train_test_split(alltrain_idx, test_size=0.25, shuffle=True,
                                          stratify=labels[alltrain_idx], random_state=IT)

    DATA = {}
    for i in range(len(base)):
        dim1,dim2 = EDGES[i].shape
        data = Data(x=DF[i],
                    edge_index=EDGES[i],
                    edge_attr=torch.tensor(np.ones((dim2)), device=device, dtype=torch.float32),
                    y=torch.tensor(labels, device=device).long())

        DATA[i] = data
    best_ValidLoss = np.Inf
    in_size1 = DATA[0].x.shape[1]
    in_size2 = DATA[1].x.shape[1]
    in_size3 = DATA[2].x.shape[1]
    out_size = torch.unique(DATA[0].y).shape[0]
    for learning_rate in learning_rates:
        for hd_size in hd_sizes:
            av_valid_losses = list()
            for ii in range(xtimes1):
                model = combine_module2.GCN(in_size1=in_size1,in_size2=in_size2,in_size3=in_size3,hid_size1=hd_size,out_size=out_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                train_mask = np.array([i in set(train_idx) for i in range(data.x.shape[0])])
                DATA[0].train_mask = torch.tensor(train_mask, device=device)
                valid_mask = np.array([i in set(val_idx) for i in range(data.x.shape[0])])
                DATA[0].valid_mask = torch.tensor(valid_mask, device=device)

                min_valid_loss = np.Inf
                patience_count = 0
                TR_loss = list()
                Te_loss = list()
                for epoch in range(max_epochs):
                    out,tr_loss = train()
                    TR_loss.append(tr_loss)
                    pred,this_valid_loss,coef1,coef2,coef3 = validate()
                    Te_loss.append(this_valid_loss)
                    if this_valid_loss < min_valid_loss:
                        min_valid_loss = this_valid_loss
                        patience_count = 0
                    else:
                        patience_count += 1

                    if min_epochs<= epoch and patience_count >= patience:
                        break
                av_valid_losses.append(min_valid_loss.item())
            av_valid_loss = round(statistics.median(av_valid_losses), 3)

            if av_valid_loss < best_ValidLoss:
                best_ValidLoss = av_valid_loss
                best_emb_lr = learning_rate
                best_emb_hs = hd_size
    train_mask = np.array([i in set(train_idx) for i in range(data.x.shape[0])])
    DATA[0].train_mask = torch.tensor(train_mask, device=device)
    valid_mask = np.array([i in set(val_idx) for i in range(data.x.shape[0])])
    DATA[0].valid_mask = torch.tensor(valid_mask, device=device)
    result_acc = list()
    result_wf1 = list()
    result_mf1 = list()
    result_mcc = list()
    for rns in range(xtimes2):
        model = combine_module2.GCN(in_size1=in_size1, in_size2=in_size2, in_size3=in_size3, hid_size1=best_emb_hs,
                                    out_size=out_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_emb_lr)
        min_valid_loss = np.Inf
        patience_count = 0
        TR_loss = list()
        Te_loss = list()
        for epoch in range(max_epochs):
            out, tr_loss = train()
            TR_loss.append(tr_loss)
            pred, this_valid_loss, coef1, coef2, coef3 = validate()
            Te_loss.append(this_valid_loss)
            if this_valid_loss < min_valid_loss:
                min_valid_loss = this_valid_loss
                patience_count = 0
            else:
                patience_count += 1

            if min_epochs <= epoch and patience_count >= patience:
                break
        valid_mask = np.array([i in set(test_idx) for i in range(data.x.shape[0])])
        DATA[0].valid_mask = torch.tensor(valid_mask, device=device)
        predictions, this_valid_loss, coef1, coef2, coef3 = validate()
        y_test = labels[valid_mask]
        pred_ = predictions[valid_mask]
        result_acc.append(accuracy_score(labels[valid_mask], pred_))
        result_wf1.append(f1_score(labels[valid_mask], pred_, average='weighted'))
        result_mf1.append(f1_score(labels[valid_mask], pred_, average='macro'))
        result_mcc.append(matthews_corrcoef(labels[valid_mask], pred_))
    av_result_acc.append(np.median(result_acc))
    av_result_wf1.append(np.median(result_wf1))
    av_result_mf1.append(np.median(result_mf1))
    av_result_mcc.append(np.median(result_mcc))
    end = time.time()
    av_time.append(round(end - start, 1))

print('acc:',np.round(np.mean(av_result_acc),decimals=3),'std:',np.round(np.std(av_result_acc),decimals=3))
print('wf1:',np.round(np.mean(av_result_wf1),decimals=3),'std:',np.round(np.std(av_result_wf1),decimals=3))
print('mf1:',np.round(np.mean(av_result_mf1),decimals=3),'std:',np.round(np.std(av_result_mf1),decimals=3))
print('mcc:',np.round(np.mean(av_result_mcc),decimals=3),'std:',np.round(np.std(av_result_mcc),decimals=3))
print('time:',np.round(np.mean(av_time),decimals=3),'std:',np.round(np.std(av_time),decimals=3))
# print('accuracy:',round(accuracy_score(labels[test_idx], pred_), 3))
# print('wf1:',round(f1_score(labels[test_idx], pred_, average='weighted'), 3))
# print('mf1:',round(f1_score(labels[test_idx], pred_, average='macro'), 3))
# print('It took ' + str(round(end - start, 1)) + ' seconds for all runs.')
coef_1 = coef1[valid_mask,:]
coef_2 = coef2[valid_mask,:]
coef_3 = coef3[valid_mask,:]
# results = [av_result_acc,av_result_wf1,av_result_mf1,av_result_mcc]
# with open(base_path + 'exp_results/' + 'ite_int' + '.pkl','wb') as f:
#     pickle.dump(results,f)
id_0 = np.where((y_test==0) & (pred_.numpy()==0))[0]
id_1 = np.where((y_test==1) & (pred_.numpy()==1))[0]
id_2 = np.where((y_test==2) & (pred_.numpy()==2))[0]
id_3 = np.where((y_test==3) & (pred_.numpy()==3))[0]
id_4 = np.where((y_test==4) & (pred_.numpy()==4))[0]
Coef1_0 = coef_1[id_0[:10],:]
Coef1_1 = coef_1[id_1[:10],:]
Coef1_2 = coef_1[id_2[:10],:]
Coef1_3 = coef_1[id_3[:10],:]
Coef1_4 = coef_1[id_4[:10],:]
Coef2_0 = coef_2[id_0[:10],:]
Coef2_1 = coef_2[id_1[:10],:]
Coef2_2 = coef_2[id_2[:10],:]
Coef2_3 = coef_2[id_3[:10],:]
Coef2_4 = coef_2[id_4[:10],:]
Coef3_0 = coef_3[id_0[:10],:]
Coef3_1 = coef_3[id_1[:10],:]
Coef3_2 = coef_3[id_2[:10],:]
Coef3_3 = coef_3[id_3[:10],:]
Coef3_4 = coef_3[id_4[:10],:]
Coef1 = torch.cat((Coef1_0,Coef1_1,Coef1_2,Coef1_3,Coef1_4),dim=0)
Coef2 = torch.cat((Coef2_0,Coef2_1,Coef2_2,Coef2_3,Coef2_4),dim=0)
Coef3 = torch.cat((Coef3_0,Coef3_1,Coef3_2,Coef3_3,Coef3_4),dim=0)
plt.figure(figsize=(50,6))
plt.plot(Coef1[:,0],'--k^',label='mRNA',markersize=8)
plt.plot(Coef2[:,0],'-ro',label='DNA meth.',markersize=8)
plt.plot(Coef3[:,0],'-go',label='miRNA',markersize=8)
plt.xlim(-0.5,65)
plt.legend(loc='center right',fontsize="19")
plt.ylabel('Attention coefficients',fontsize="19",fontweight='bold')
plt.xlabel('Samples',fontsize="19",fontweight='bold')
plt.yticks(fontsize=16,fontweight='bold')
plt.xticks(fontsize=16,fontweight='bold')
plt.title("TCGA-BRCA", fontsize=24, fontweight='bold')
plt.show()

