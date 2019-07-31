import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, supports, dropout=0.):
        super(GraphConvolution, self).__init__()

        self.supports = supports # The adjacency matrices.
        self.n_supports = len(self.supports)
        
        self.lin_supports = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(len(self.supports))])
        self.lay_dropout = nn.Dropout(dropout)    

    def forward(self, x):
        
        x = self.lay_dropout(x)
        
        supports = [None] * self.n_supports
        for i in range(self.n_supports):
            pre_sup = self.lin_supports[i](x)
            support = torch.mm(self.supports[i], pre_sup)
            supports[i] = support
        
        x = sum(supports)
        
        return F.relu(x)

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1, supports, dropout):
        super(GCN, self).__init__()
        
        self.gcn_1 = GraphConvolution(input_dim, hidden1, supports, dropout)
        self.gcn_2 = GraphConvolution(hidden1, output_dim, supports, dropout)

    def forward(self, x):
        
        x = self.gcn_1(x)
        x = self.gcn_2(x)
        
        return F.softmax(x, dim=1)