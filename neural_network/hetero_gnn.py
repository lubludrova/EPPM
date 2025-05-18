from neural_network.gatv2_nn import RGCN
import torch.nn as nn
import dgl

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, params, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, params, rel_names)
        self.classify = nn.Linear(params['hidden_dim'], n_classes)

    def forward(self, graph, feat, embed=False, edge_weight=None, eweight=None, **unused,):
        if eweight is not None and edge_weight is None:
            edge_weight = eweight
        if unused:
            raise TypeError(f"Unexpected kwargs: {unused.keys()}")
        if edge_weight is not None:
            h = self.rgcn(graph, feat, edge_weight=edge_weight)
        else:
            h = self.rgcn(graph, feat)
        if embed:
            return h

        # h = self.rgcn(graph, feat)
        with graph.local_scope():
            graph.ndata['h'] = h
            hg = 0
            for ntype in graph.ntypes:
                hg = hg + dgl.sum_nodes(graph, 'h', ntype=ntype)
            return self.classify(hg)


