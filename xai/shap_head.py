import torch
import shap

class HeadWrapper(torch.nn.Module):
    """
    Принимает tuple (graph_emb  [B,g_dim], attr_seq [B,T,a_dim]),
    формирует concat‑последовательность так же, как это делается
    внутри GGRNNTorch → LSTMHead, и отдаёт logits.
    """
    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.head = head.eval()

    def forward(self, inputs):
        g_emb, attr_seq = inputs            # unpack
        B, T, _ = attr_seq.shape
        g_rep = g_emb.unsqueeze(1).expand(-1, T, -1)     # [B,T,g_dim]
        seq   = torch.cat([g_rep, attr_seq], dim=-1)     # [B,T,in_dim]
        return self.head(seq)
