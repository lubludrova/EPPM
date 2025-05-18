import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch

class RGCN(nn.Module):
    def __init__(self, in_feats, params, rel_names):
        super(RGCN, self).__init__()
        self.convs = nn.ModuleList([dglnn.HeteroGraphConv({rel: dglnn.GATv2Conv(in_feats=in_feats[rel], out_feats=params['hidden_dim'], num_heads=params['n_heads'], feat_drop=params['dropout'], share_weights=True, residual=True) for rel in rel_names}, aggregate='sum') for i in range(params['n_layers'])])
        self.conv_out = dglnn.HeteroGraphConv({rel: dglnn.GATv2Conv(in_feats=params['hidden_dim'] * params['n_heads'], out_feats=params['hidden_dim'], feat_drop=params['dropout'], num_heads=1, share_weights=True, residual=True) for rel in rel_names}, aggregate='sum')
        self.conv_out2 = dglnn.HeteroGraphConv({rel: dglnn.GATv2Conv(in_feats=in_feats[rel], out_feats=params['hidden_dim'], feat_drop=params['dropout'], num_heads=1, share_weights=True, residual=True) for rel in rel_names}, aggregate='sum')


    def forward(self, graph, inputs, edge_weight=None):
        h = inputs

        for f in self.convs:
            if edge_weight is not None:
                h = f(graph, h, mod_kwargs=edge_weight)
            else:
                h = f(graph, inputs)
            h = {k: torch.reshape(F.relu(v), (v.shape[0], -1)) for k, v in h.items()}


        if len(self.convs)==0:
            if edge_weight is not None:
                h = self.conv_out2(graph, inputs, mod_kwargs=edge_weight)
            else:
                h = self.conv_out2(graph, inputs)
            h = {k: torch.reshape(F.relu(v), (v.shape[0], -1)) for k, v in h.items()}
        else:
            if edge_weight is not None:
                h = self.conv_out(graph, h, mod_kwargs=edge_weight)
            else:
                h = self.conv_out(graph, h)
            h = {k: torch.reshape(F.relu(v), (v.shape[0], -1)) for k, v in h.items()}

        return h



