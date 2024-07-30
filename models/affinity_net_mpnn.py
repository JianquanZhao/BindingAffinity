import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.nn import MessagePassing,global_max_pool
from torch_geometric.nn.pool import global_max_pool,max_pool,graclus,avg_pool,global_mean_pool
from torch_geometric.utils import add_self_loops, degree
import torch.nn.init as init


class MPNN(MessagePassing):
    def __init__(self, input_dim, hidden_dim, output_dim,drop_weight):
        super(MPNN, self).__init__(aggr='add', flow='source_to_target')
        self.lin = torch.nn.Sequential(
            nn.Linear(input_dim , hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(drop_weight),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # self.lin = nn.Linear(input_dim,output_dim)
        self.edge_emb = nn.Linear(12,1)

    def forward(self, x, edge_index, edge_attr):
        # 添加自环边
        edge_index, edge_attr = add_self_loops(edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))

        x = self.lin(x)

        # 归一化系数
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        edge_attr = self.edge_emb(edge_attr)
        # 节点特征传递
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j,edge_attr, norm):
        return norm.view(-1, 1) * edge_attr.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
    
    def aggregate(self, inputs, index, ptr, dim_size):
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        
        self.mpnn_0 = MPNN(input_dim=input_dim,hidden_dim=input_dim, output_dim=input_dim, drop_weight=0.4)
        self.mpnn_1 = MPNN(input_dim=input_dim,hidden_dim=hidden_dim, output_dim=output_dim, drop_weight=0.5)
        
        # self.energy_emb = nn.Linear(21,8)

        self.lin = torch.nn.Sequential(
            nn.Linear(output_dim, output_dim//2),
            nn.BatchNorm1d(output_dim//2),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(output_dim//2, 1)
        )


    def forward(self, data, origin):
        e = data.energy
        data.x = self.mpnn_0(data.x, data.edge_index, data.edge_attr)
        data.x = data.x + origin
        cluster = graclus(data.edge_index, num_nodes=data.x.shape[0])
        data.x = F.leaky_relu(data.x)
        data = max_pool(cluster,data)
        x = self.mpnn_1(data.x, data.edge_index, data.edge_attr)
        x = F.leaky_relu(x)
        x = global_max_pool(x, data.batch)

        # e = e.reshape(x.shape[0],21)
        # e = self.energy_emb(e)
        # x = torch.cat([x,e],dim=1)
        x = self.lin(x)
        return x
