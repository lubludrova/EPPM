from dgl import to_homogeneous, ETYPE
import dgl

from dgl.nn.pytorch.explain import HeteroGNNExplainer, HeteroPGExplainer
import torch
import torch.nn as nn
import torch.nn.functional as F



class TypeAwarePGExplainer(HeteroPGExplainer):
    def __init__(
        self,
        model,
        num_features,
        canonical_etypes,
        mode: str = "multi_head",
        budget_coef: float = 0.01,
        emb_dim: int = 16,
        **kwargs,
    ):
        super().__init__(model, num_features, **kwargs)
        self.mode = mode
        self.budget_coef = budget_coef
        self.canonical_etypes = canonical_etypes

        if mode == "multi_head":
            self.type_mlps = nn.ModuleDict({
                str(et): nn.Sequential(
                    nn.Linear(2 * num_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                ) for et in canonical_etypes
            })
        else:
            self.type_emb = nn.Embedding(len(canonical_etypes), emb_dim)
            self.elayers = nn.Sequential(
                nn.Linear(2 * num_features + emb_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )



    def _edge_logits(self, emb, etype_ids):

        if self.mode == "multi_head":
            logits = torch.empty(emb.size(0), device=emb.device)
            for t in torch.unique(etype_ids):
                idx = (etype_ids == t).nonzero(as_tuple=True)[0]
                logits[idx] = self.type_mlps[str(self.canonical_etypes[t])](emb[idx]).squeeze()
        else:
            t_emb = self.type_emb(etype_ids)               # (|E|, emb_dim)
            logits = self.elayers(torch.cat([emb, t_emb], dim=-1)).squeeze()
        return logits

    def explain_graph(self, graph, feat, temperature=1.0, training=False, **kw):
        assert self.graph_explanation, "explainer настроен на graph‑level"

        self.model = self.model.to(graph.device)
        embed = self.model(graph, feat, embed=True, **kw)
        for ntype, e in embed.items():
            graph.nodes[ntype].data["emb"] = e

        homo_g = to_homogeneous(graph, ndata=["emb"])
        homo_emb = homo_g.ndata["emb"]


        col, row = homo_g.edges()
        pair_emb = torch.cat([homo_emb[col], homo_emb[row]], dim=-1)
        etype_ids = homo_g.edata[ETYPE].long()
        # print('ETYPE in homo:', homo_g.edata[ETYPE].shape)

        logits = self._edge_logits(pair_emb, etype_ids)

        values = self.concrete_sample(logits, beta=temperature, training=training)
        self.sparse_mask_values = values
        reverse = homo_g.edge_ids(row, col).long()
        edge_mask = (values + values[reverse]) / 2
        self.set_masks(homo_g, edge_mask)

        hetero_edge_mask = self._edge_mask_to_heterogeneous(edge_mask, homo_g, graph)

        logits_model = self.model(graph, feat, edge_weight=hetero_edge_mask, **kw)
        probs = F.softmax(logits_model, dim=-1)
        if training:
            probs = probs.data
        else:
            self.clear_masks()

        self._last_etype_ids = etype_ids

        return (probs, hetero_edge_mask)


    def loss(self, prob, ori_pred):

        target_prob = prob.gather(-1, ori_pred.unsqueeze(-1)) + 1e-6
        pred_loss = torch.mean(-torch.log(target_prob))

        edge_mask = self.sparse_mask_values  # (|E|)
        et_ids = self._last_etype_ids  # (|E|)

        per_type_sums = []
        for t in torch.unique(et_ids):
            m_t = edge_mask[et_ids == t]
            per_type_sums.append(m_t.abs().sum())

        total_sum = sum(per_type_sums)
        size_loss = self.budget_coef * F.relu(total_sum - self.budget_coef)

        # entropy
        scale = 0.99
        em = self.edge_mask * (2 * scale - 1.0) + (1.0 - scale)
        ent = -em * torch.log(em) - (1 - em) * torch.log(1 - em)
        ent_loss = self.coff_connect * torch.mean(ent)

        return pred_loss + size_loss + ent_loss


    @torch.no_grad()
    def importance_by_type(self, edge_mask, etype_ids):
        out = {}
        for t in torch.unique(etype_ids):
            et = self.canonical_etypes[t]
            out[str(et)] = edge_mask[etype_ids == t].sum().item()
        return out