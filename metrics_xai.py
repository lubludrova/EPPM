# metrics_xai.py
from random import random

import torch
import numpy as np
import dgl
import plotly.graph_objects as go
import plotly.io as pio

import os, ast
from PIL import Image
import math, subprocess, tempfile, pathlib

import networkx as nx
from dgl import to_networkx
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_agraph import to_agraph
import matplotlib as mpl  # ← иначе NameError на mpl.colors
import matplotlib.pyplot as plt  # уже есть
import matplotlib.colors as mcolors  # удобно: mcolors.to_hex
from scipy.stats import kendalltau
import torch.nn.functional as F

# metrics_xai.py
from typing import Tuple, Dict
import ast, os, numpy as np, matplotlib.pyplot as plt, networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from collections import defaultdict


from collections import defaultdict
import torch
import dgl
from typing import Dict, Tuple

def collapse_has_edges(
    g: dgl.DGLHeteroGraph,
    feat_mask: Dict[str, torch.Tensor],
    edge_mask: Dict[Tuple[str,str,str], torch.Tensor],
) -> Tuple[dgl.DGLHeteroGraph, Dict[str, torch.Tensor], Dict[Tuple[str,str,str], torch.Tensor]]:

    tmp = defaultdict(lambda: ([], [], []))  # src_list, dst_list, mask_list
    for etype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        m = edge_mask.get(etype, torch.ones(src.shape[0]))
        if etype[1]=='has':
            key = (etype[0],'has',etype[2]); s,d = src,dst
        elif etype[1]=='rev_has':
            key = (etype[2],'has',etype[0]); s,d = dst,src
        else:
            key = etype; s,d = src,dst

        tmp[key][0].append(s)
        tmp[key][1].append(d)
        tmp[key][2].append(m.detach().cpu())

    data_dict = {}
    new_edge_mask = {}
    for key, (ss, ds, ms) in tmp.items():
        if key[1]=='follow' and key[0]==key[2] and key[0] != 'Activity':
            continue  # отложим обработку follow T→T
        S = torch.cat(ss); D = torch.cat(ds); M = torch.cat(ms)
        data_dict[key]     = (S, D)
        new_edge_mask[key] = M

    for (T, rel, _), (ss, ds, ms) in tmp.items():
        if rel!='follow' or T=='Activity':
            continue
        S_all = torch.cat(ss).tolist()
        D_all = torch.cat(ds).tolist()
        M_all = torch.cat(ms).tolist()
        imp_sum = defaultdict(float)
        imp_count = defaultdict(int)
        for idx,u in enumerate(S_all):
            imp_sum[u] += M_all[idx]
            imp_count[u] += 1
        for idx,v in enumerate(D_all):
            imp_sum[v] += M_all[idx]
            imp_count[v] += 1
        for node, total in imp_sum.items():
           imp_sum[node] = total / imp_count[node]

        has_key = ('Activity','has',T)
        if has_key in new_edge_mask:
            # S_has, D_has для этой связи
            S_has, D_has = data_dict[has_key]
            # собираем дельту для каждого ребра
            delta = torch.tensor([ imp_sum[int(node)] for node in D_has.tolist() ])
            new_edge_mask[has_key] = new_edge_mask[has_key] + delta

        follow_key = ('Activity', 'follow', 'Activity')
        if follow_key in new_edge_mask:
            Sf, Df = data_dict[follow_key]
            Mf = new_edge_mask[follow_key]  # исходные маски follow
            delta = []
            for src_id, dst_id in zip(Sf.tolist(), Df.tolist()):
                sum_imp = 0.0
                count_imp = 0
                for _, _, T in [k for k in data_dict if k[0] == 'Activity' and k[1] == 'has' and k[2] != "Activity"]:
                    SHT, DHT = data_dict[('Activity', 'has', T)]
                    MHT = new_edge_mask[('Activity', 'has', T)]
                    mask_dst = MHT[(DHT == dst_id).nonzero(as_tuple=True)].sum().item()
                    mask_src = MHT[(DHT == src_id).nonzero(as_tuple=True)].sum().item()
                    sum_imp   += (mask_src + mask_dst)
                    count_imp += 1

                add_imp = sum_imp / count_imp if count_imp else 0
                # delta.append(add_imp)
                delta.append(add_imp)

            new_edge_mask[follow_key] = new_edge_mask[follow_key] * 0.8 + (torch.tensor(delta, dtype=Mf.dtype)) * 0.2

    num_nodes = {nt: g.num_nodes(nt) for nt in g.ntypes}
    g2 = dgl.heterograph(data_dict, num_nodes_dict=num_nodes)

    return g2, feat_mask, new_edge_mask


def dgl_to_typed_nx(g: dgl.DGLHeteroGraph) -> nx.MultiDiGraph:

    G = nx.MultiDiGraph()
    for ntype in g.ntypes:
        for nid in g.nodes(ntype).tolist():
            G.add_node((ntype, nid), ntype=ntype)

    for etype in g.canonical_etypes:
        srctype, reltype, dsttype = etype
        src, dst = g.edges(etype=etype)
        src_list, dst_list = src.tolist(), dst.tolist()
        for idx, (u, v) in enumerate(zip(src_list, dst_list)):
            G.add_edge(
                (srctype, u),
                (dsttype, v),
                etype=etype,
                idx=idx
            )
    return G
def visualize_explanation_matplotlib(
    g,
    feat_mask: Dict[str, 'Tensor'],
    edge_mask: Dict[tuple, 'Tensor'],
    *,
    case_idx: int,
    out_dir: str,
    figsize: tuple[int,int] = (8, 5),
    use_graphviz: bool = True,
    file_format: str = "png",
):
    """
    1) DGL → NetworkX
    2) Проставляем importance в nodes и edges
    3) Рисуем весь граф с размерами узлов и толщиной рёбер по важности
    """
    g, feat_mask, edge_mask = collapse_has_edges(g, feat_mask, edge_mask)
    print("G After:", g)
    nx_g = dgl_to_typed_nx(g)


    for ntype, mask in feat_mask.items():
        arr = mask.detach().cpu().numpy()
        for i, nid in enumerate(g.nodes(ntype).tolist()):
            nx_g.nodes[(ntype, nid)]['imp'] = float(arr[i])
    # 2.2. Рёбра (используя data['idx'], который ваш конвертер должен проставить)
    for etype, emask in edge_mask.items():
        arr = emask.detach().cpu().numpy()
        for u, v, data in nx_g.edges(keys=False, data=True):
            if data.get('etype') == etype:
                data['imp'] = float(arr[data['idx']])


    print("MULTI DI", nx_g)
    plot_importance_by_type(
        nx_g,
        out_path = os.path.join(out_dir, f"{case_idx}_importance_by_type.png")
    )

    #    – layout
    if use_graphviz:
        pos = graphviz_layout(nx_g, prog='dot', args='-Grankdir=LR')
    else:
        pos = nx.spring_layout(nx_g, seed=42)


    #    – рёбра: толщина ∝ imp
    edge_imps = np.array([d.get('imp', 0.0) for *_,d in nx_g.edges(data=True)])
    min_e, max_e = edge_imps.min(), edge_imps.max() or 1.0
    widths = 0.3 + 2 * (edge_imps - min_e) / (max_e - min_e + 1e-9)


    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_axis_off()

    # слоем: сначала все рёбра
    nx.draw_networkx_edges(
        nx_g, pos,
        width=widths,
        edge_color=edge_imps,
        edge_cmap=plt.cm.Reds,
        edge_vmin=min_e,
        edge_vmax=max_e,
        # edge_color='#888',
        alpha=0.6,
        arrowsize=10,
        arrowstyle='-|>',
        style='solid',
        connectionstyle='arc3,rad=0.1',
        ax=ax
    )
    # цветовая шкала для рёбер
    sm_e = plt.cm.ScalarMappable(
        cmap = plt.cm.Reds,
        norm = plt.Normalize(vmin=min_e, vmax=max_e)
    )
    sm_e.set_array([])
    cbar_e = plt.colorbar(sm_e, ax=ax, fraction=0.046, pad=0.04)
    cbar_e.set_label("Importance", rotation=270, labelpad=15)


    #    – узлы: размер ∝ imp
    node_imps = np.array([d.get('imp', 0.0) for _,d in nx_g.nodes(data=True)])
    min_imp, max_imp = node_imps.min(), node_imps.max() or 1.0
    sizes = 300 + 1000 * (node_imps - min_imp) / (max_imp - min_imp + 1e-9)

    nx.draw_networkx_nodes(
        nx_g, pos,
        node_size=sizes,
        # node_color='#1f78b4',
        cmap=plt.cm.Reds,
        node_color=node_imps,
        vmin=min_imp,
        vmax=max_imp,
        linewidths=1.5,
        alpha=0.9,
        ax=ax
    )
    sm_n = plt.cm.ScalarMappable(
        cmap = plt.cm.Reds,
        norm = plt.Normalize(vmin=min_imp, vmax=max_imp)
    )
    sm_n.set_array([])


    labels = {}
    for n, data in nx_g.nodes(data=True):
        node_id = n[1]
        imp = data.get('imp', 0.0)
        # labels[n] = f"{node_id}\n{imp:.2f}"
        labels[n] = f"{data['ntype']}: {node_id}"

    nx.draw_networkx_labels(
        nx_g, pos,
        labels=labels,
        font_size=8,
        ax=ax
    )

    # сохраняем
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"case_{case_idx}_graph.{file_format}")
    plt.tight_layout()
    plt.savefig(path, dpi=300, format=file_format, bbox_inches='tight')
    plt.close()
    print(f"✓ saved full graph → {path}")

def visualize_explanation(
        g,
        feat_mask: Dict[str, torch.Tensor],
        edge_mask: Dict[Tuple, torch.Tensor],
        *,
        case_idx: Tuple[str, int],
        out_dir: str,
        target_node: Tuple[str, int] | None = None,
        main_color: str = '#e41a1c',
        keep_top_edges: float = .20,
        prog: str = 'dot',  # Graphviz layout engine
):
    """
    Визуализация DFG через to_agraph:
    - строим NetworkX-диграф из follow-ребер,
    - разбиваем на normal/highlight по keep_top_edges,
    - настраиваем стили узлов и рёбер в AGraph и рисуем.
    """
    nx_g = dgl_to_typed_nx(g)
    print("nx_g: ", nx_g)
    print("Graph: ", g)
    print("Feat mask: ", feat_mask)
    print("Edge mask: ", edge_mask)

    for ntype, mask in feat_mask.items():
        arr = mask.detach().cpu().numpy()
        for i, nid in enumerate(g.nodes(ntype).tolist()):
            nx_g.nodes[(ntype, nid)]['imp'] = float(arr[i])

    for etype, emask in edge_mask.items():
        arr = emask.detach().cpu().numpy()
        for u, v, data in nx_g.edges(keys=False, data=True):
            if data.get('etype') == etype:
                data['imp'] = float(arr[data['idx']])



    follow = [
        (u, v, d['imp'])
        for u, v, _, d in nx_g.edges(keys=True, data=True)
        if d.get('etype', (None,))[0]=='Activity'
        and d['etype'][1]=='follow'
        and d['etype'][2]=='Activity'
    ]
    if not follow:
        raise RuntimeError("No Activity→follow→Activity edges found!")

    imps = np.array([imp for *_, imp in follow])
    thr = np.quantile(imps, 1 - keep_top_edges)
    highlight = [(u,v,imp) for u,v,imp in follow if imp >= thr]
    normal    = [(u,v,imp) for u,v,imp in follow if imp <  thr]


    D = nx.DiGraph()
    for u, v, imp in follow:
        label_u = str(u[1])
        label_v = str(v[1])
        D.add_node(label_u, imp=nx_g.nodes[u]['imp'])
        D.add_node(label_v, imp=nx_g.nodes[v]['imp'])
        D.add_edge(label_u, label_v, imp=imp)

    A = to_agraph(D)
    A.graph_attr.update(
        rankdir='LR',
        splines='ortho',
        nodesep='0.6',
        ranksep='0.75',
        margin='0.1'
    )

    print(D)
    print(A.nodes())

    maxn = max(data['imp'] for _, data in D.nodes(data=True)) or 1.0
    for n in A.nodes():
        imp = D.nodes[n]['imp']
        w = 0.3 + 0.7 * (imp / maxn)
        nd = A.get_node(n)
        nd.attr['shape'] = 'circle'
        nd.attr['fixedsize'] = 'true'
        nd.attr['width'] = str(w)
        nd.attr['height'] = str(w)
        # nd.attr['label'] = n
        nd.attr['label'] = f"{n}\\nimp={imp:.2f}"
        nd.attr['fontsize'] = '9'
        nd.attr['style'] = 'filled'
        nd.attr['fillcolor'] = main_color if n == str(case_idx) else '#dddddd'
        nd.attr['penwidth'] = '1'

    maxe = max(imp for *_, imp in follow) or 1.0
    for u, v in D.edges():
        e = A.get_edge(u, v)
        imp = D.edges[u, v]['imp']
        pen = 1 + 4 * (imp / maxe)
        if imp >= thr:
            e.attr['color'] = main_color
            e.attr['penwidth'] = str(pen)
        else:
            e.attr['color'] = '#888888'
            e.attr['penwidth'] = '1.0'

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'case_{case_idx}_dfg.svg')
    A.draw(out_path, format='svg', prog=prog)
    print(f"✓ saved → {out_path}")


def compute_xai_metrics(
        model,
        graph,
        edge_mask,
        label,
        *,
        threshold='mean',
        top_k=None,
        device=None,
        edge_mask_ref=None
):
    """
    Возвращает dict с:
        comprehensiveness, sufficiency, sparsity,
        stability  (τ‑intra‑type, NaN если edge_mask_ref=None)
    """

    model.eval()
    device = device or next(model.parameters()).device
    graph = graph.to(device)

    #  признаки узлов

    feat_dict = {}
    for ntype in graph.ntypes:
        feat_dict[ntype] = graph.ndata[ntype][ntype].to(device)

    # нормализуем label
    if isinstance(label, torch.Tensor):
        label_idx = int(label.argmax().item())
    else:
        label_idx = int(np.asarray(label).argmax())

    # полный граф
    with torch.no_grad():
        logits_full = model(graph, feat_dict)
        p_full = torch.softmax(logits_full, dim=1)[0, label_idx].item()

    # выбираем топ‑k рёбер
    total_edges = sum(m.shape[0] for m in edge_mask.values())
    keep_dict = {et: torch.tensor([], dtype=torch.int64, device=device)
                 for et in edge_mask}

    if top_k is not None:
        flat = [(float(w.item()), et, i)
                for et, mask in edge_mask.items()
                for i, w in enumerate(mask)]
        for _, et, i in sorted(flat, key=lambda x: x[0], reverse=True)[:top_k]:
            keep_dict[et] = torch.cat([keep_dict[et],
                                       torch.tensor([i], device=device)])
    else:
        for et, mask in edge_mask.items():
            thr = mask.mean().item() if threshold == 'mean' else float(threshold)
            idx = torch.nonzero(mask >= thr, as_tuple=False).view(-1)
            keep_dict[et] = idx.to(device)

    #  предсказание на kept
    g_kept = dgl.edge_subgraph(graph, keep_dict, relabel_nodes=False)
    with torch.no_grad():
        p_kept = torch.softmax(model(g_kept, feat_dict), dim=1)[0, label_idx].item()

    # предсказание на removed
    rem_dict = {}
    for et, mask in edge_mask.items():
        all_idx = torch.arange(mask.shape[0], device=device)
        rem = torch.tensor(np.setdiff1d(all_idx.cpu(), keep_dict[et].cpu()),
                           dtype=torch.int64, device=device)
        rem_dict[et] = rem

    g_removed = dgl.edge_subgraph(graph, rem_dict, relabel_nodes=False)
    with torch.no_grad():
        p_removed = torch.softmax(model(g_removed, feat_dict),
                                  dim=1)[0, label_idx].item()

    # метрики fidelity
    comp = p_full - p_removed  # comprehensiveness
    suff = p_full - p_kept  # sufficiency
    spars = sum(len(v) for v in keep_dict.values()) / total_edges if total_edges else 0.

    # intra‑type stability

    if edge_mask_ref is not None:
        orig = torch.cat([edge_mask[et].flatten() for et in edge_mask], dim=0)
        pert = torch.cat([edge_mask_ref[et].flatten() for et in edge_mask], dim=0)
        cos_sim = F.cosine_similarity(orig.unsqueeze(0), pert.unsqueeze(0), dim=1).item()
        stability = 1.0 - cos_sim  # 0 = идеально устойчиво, 1 = максимум нестабильности
    else:
        stability = float('nan')

    # Fidelity
    denom = (1 - spars) + comp
    f1_fidelity = (2 * (1 - spars) * comp / denom) if denom != 0 else float('nan')


    return {
        'comprehensiveness': comp,
        'sufficiency': suff,
        'sparsity': spars,
        'stability': stability,
        'f1_fidelity': f1_fidelity,
    }


def create_sample_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    types = ['Activity', 'Resource', 'case:creator']
    for t in types:
        for i in range(3):
            node = (t, i)
            G.add_node(node, ntype=t, imp=random.uniform(5, 15))
    for _ in range(15):
        u, v = random.sample(list(G.nodes()), 2)
        G.add_edge(u, v, imp=random.uniform(0, 5))
    return G


def plot_importance_by_type(nx_g: nx.Graph, out_path: str) -> None:
    type_vals = defaultdict(list)
    for n, d in nx_g.nodes(data=True):
        type_vals[d['ntype']].append(d['imp'])
    types = list(type_vals.keys())
    means = np.array([np.mean(type_vals[t]) for t in types])
    min_m, max_m = means.min(), means.max()
    norm_means = (means - min_m) / (max_m - min_m + 1e-9)

    # 3) Отрисовка
    plt.figure(figsize=(6,4))
    colors = plt.cm.viridis(np.linspace(0,1,len(types)))
    bars = plt.bar(types, norm_means, color=colors)

    plt.ylabel('Normalized Avg Importance')
    plt.title('Node Importance by Type (normalized)')
    plt.xticks(rotation=45)

    for bar, m in zip(bars, means):
        x = bar.get_x() + bar.get_width()/2
        y = bar.get_height()
        plt.text(x, y + 0.02, f"{m:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
