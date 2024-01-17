import os
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv,GATConv
import numpy as np
import torch.nn as nn
from openpyxl import load_workbook
from collections import Counter
class GCN(torch.nn.Module):
    def __init__(self, in_size1=16,in_size2=16,in_size3=16,hid_size1=8,out_size=2):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_size1, hid_size1)
        self.conv2 = GCNConv(in_size2, hid_size1)
        self.conv3 = GCNConv(in_size3, hid_size1)
        self.fc1 = nn.Linear(hid_size1, 1)
        self.conv11 = GCNConv(hid_size1, out_size)
        self.conv22 = GCNConv(hid_size1, out_size)
        self.conv33 = GCNConv(hid_size1, out_size)

    def forward(self, data1,data2,data3):
        x1, edge_index1, edge_weight1 = data1.x, data1.edge_index, data1.edge_attr
        x2, edge_index2, edge_weight2 = data2.x, data2.edge_index, data2.edge_attr
        x3, edge_index3, edge_weight3 = data3.x, data3.edge_index, data3.edge_attr

        x_emb1 = F.relu(self.conv1(x1, edge_index1, edge_weight1))
        x_emb1 = F.dropout(x_emb1,training=self.training)
        x_emb2 = F.relu(self.conv2(x2, edge_index2, edge_weight2))
        x_emb2 = F.dropout(x_emb2, training=self.training)
        x_emb3 = F.relu(self.conv3(x3, edge_index3, edge_weight3))
        x_emb3 = F.dropout(x_emb3, training=self.training)
        coef1_1 = torch.exp(F.leaky_relu(self.fc1(x_emb1)))
        coef1_2 = torch.exp(F.leaky_relu(self.fc1(x_emb2)))
        coef1_3 = torch.exp(F.leaky_relu(self.fc1(x_emb3)))
        coefd = coef1_1 + coef1_2 + coef1_3
        coef1 = torch.div(coef1_1,coefd)
        coef2 = torch.div(coef1_2,coefd)
        coef3 = torch.div(coef1_3,coefd)
        combined = ((x_emb1*coef1) + (x_emb2*coef2) + (x_emb3*coef3))
        out1 = (self.conv11(combined, edge_index1, edge_weight1))
        out2 = (self.conv22(combined, edge_index2, edge_weight2))
        out3 = (self.conv33(combined, edge_index3, edge_weight3))
        out4 = (out1 + out2 + out3)
        return out4,coef1,coef2,coef3
