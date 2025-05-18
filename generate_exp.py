import sys
import pickle

import numpy as np
import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from torch import nn

from preprocessing.dgl_dataset import TextDataset
import time
import h5py
from dgl.nn.pytorch.explain import HeteroGNNExplainer, HeteroPGExplainer, HeteroSubgraphX
from metrics_xai import compute_xai_metrics, visualize_explanation, visualize_explanation_matplotlib
import torch.nn.functional as F
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from dgl import to_networkx
import os
import ast

from utility.TAPG import TypeAwarePGExplainer

# log_name = 'env_permit.xes.gz' #'Helpdesk.xes.gz'
# log_name = "env_permit.xes.gz"
# log_name = "BPI_Challenge_2012_A.xes.gz"
log_name = "BPI_Challenge_2013_closed_problems.xes.gz"
# log_name = "SEPSIS.xes.gz"

print(log_name, '-----')
device = torch.device("cpu")

def load_graphs_from_hdf5(filename):
    graphs = []
    label = []
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            pickled_graph = f[key][()]
            graph = pickle.loads(pickled_graph)
            graphs.append(graph['graph'])
            label.append(graph['label'])
    return graphs, label





if __name__ == '__main__':
    need_compute = True
    explain = 'subx' # pg, gnn, attention, tapg, subx


    base_dir = '/Users/ruslanageev/PycharmProjects/Prophet'
    dump_dir = '/Users/ruslanageev/PycharmProjects/Prophet/utility/files'
    number_of_examples = 13

    start_time = time.time()

    model = torch.load(f'{base_dir}/models/model_{log_name}.h5')
    # model.eval()

    X_test, y_test = load_graphs_from_hdf5(
        os.path.join(base_dir, 'heterographs_tracenode', f"{log_name}_test.db"))
    with open("/Users/ruslanageev/PycharmProjects/Prophet/preprocessing/w2v/" + log_name + "/relations.pkl", 'rb') as f:
        relation = pickle.load(f)
    # print(X_test[0], y_test[0])
    #
    # indices = [0, 4, 7, 9, 11]
    indices = [4, 7, 9]
    graphs = [X_test[i] for i in indices]
    labels = [y_test[i] for i in indices]
    # graphs = X_test[1:2]
    # labels = y_test[1:2]

    # вместо первых N подряд — случайная выборка для честного сравнения
    # random.seed(42)
    # total_cases = len(X_test)
    # num_cases = min(number_of_examples, total_cases)
    # selected = random.sample(range(total_cases), num_cases)
    # graphs = [X_test[i] for i in selected]
    # labels = [y_test[i] for i in selected]

    init_labels = labels.copy()


    if need_compute:
        print("Считаем заново")
        df_test = TextDataset(graphs, labels)
        test_loader = GraphDataLoader(df_test, batch_size=1, drop_last=False, shuffle=False)
        g0, _ = next(iter(test_loader))
        g0 = g0.to(device)
        # print(g0)
        feat0 = {nt: g0.ndata[nt][nt] for nt in g0.ntypes}
        # print(feat0)
        # вот тут hidden_dim — размер embedding‑а узла, он лежит во втором измерении тензора
        emb0 = model(g0, feat0, embed=True)
        first_ntype = next(iter(emb0.keys()))
        true_emb_dim = emb0[first_ntype].shape[1]

        dict_feature = {}
        dict_edge = {}
        for n in next(iter(test_loader))[0].ntypes:
            dict_feature[n] = []
        for e in relation:
            dict_edge[e] = []

        if explain == 'tapg':
            df_train = TextDataset(X_test, y_test)
            train_loader = GraphDataLoader(df_train, batch_size=1, drop_last=False, shuffle=False)

            bc = 4e-3
            final_temp = 0.65
            # explain = f"{explain}_{str(bc)}_{str(final_temp)}"

            cfg = f"tapg_bc{bc:.0e}_ft{final_temp}"
            explainer = TypeAwarePGExplainer(
                model.to(device),
                true_emb_dim,  # размер эмбеддингов
                canonical_etypes=g0.canonical_etypes,
                num_hops=1,
                mode='multi_head',  # или 'type_emb'
                budget_coef=bc
            ).to(device)
            init_temp = 5.0
            optimizer_exp = torch.optim.Adam(explainer.parameters(), lr=0.005)
            num_epochs = 100
            for epoch in range(num_epochs):
                temp = init_temp * (final_temp / init_temp) ** (epoch / num_epochs)
                for i, (g, labels) in enumerate(test_loader):
                    g, labels = g.to(device), labels.to(device)
                    # собираем фичи в dict по ntype
                    feat = {nt: g.ndata[nt][nt] for nt in g.ntypes}
                    # экспоненциальный спад температуры
                    loss = explainer.train_step(g, feat, temperature=float(temp))
                    optimizer_exp.zero_grad()
                    loss.backward()
                    optimizer_exp.step()


            for i, (g, labels) in enumerate(test_loader):
                g, labels = g.to(device), labels.to(device)
                feat = {nt: g.ndata[nt][nt] for nt in g.ntypes}

                probs, edge_mask = explainer.explain_graph(g, feat)
                # edge_mask: dict { (src_nt, etype, dst_nt) : mask_tensor }

                # инициализируем нулевые маски для каждого типа узлов
                feat_mask = {nt: torch.zeros(g.num_nodes(nt), device=g.device)
                             for nt in g.ntypes}

                # суммируем вклад каждого ребра в src и dst узлы
                for rel, em in edge_mask.items():
                    src_nt, _, dst_nt = rel
                    src_ids, dst_ids = g.edges(form='uv', etype=rel)
                    feat_mask[src_nt].index_add_(0, src_ids, em)
                    feat_mask[dst_nt].index_add_(0, dst_ids, em)
                # print("dict_edge", dict_edge)
                # сохраняем результаты
                for nt in g.ntypes:
                    dict_feature[nt].append(feat_mask[nt])
                # for rel in g.canonical_etypes:
                for rel, mask in edge_mask.items():
                    # dict_edge[rel].append(edge_mask[rel])
                    dict_edge.setdefault(rel, []).append(mask)



        elif explain == 'gnn':
            explainer = HeteroGNNExplainer(model.to(device), num_hops=1, log=False)
            for i, (g, l) in enumerate(test_loader):
                print(i)
                g = g.to(device)
                l = l.to(device)

                feat = {n: g.ndata[n][n] for n in g.ntypes}
                feat_mask, edge_mask = explainer.explain_graph(graph=g, feat=feat)



                for f in g.ntypes:
                    dict_feature[f].append(feat_mask[f])
                for e in relation:
                    dict_edge[e].append(edge_mask[e])


        elif explain == 'pg':

            explainer = HeteroPGExplainer(
                model.to(device),
                true_emb_dim,
                num_hops=1)

            init_tmp, final_tmp = 5.0, 1.0
            optimizer_exp = torch.optim.Adam(explainer.parameters(), lr=0.005)
            num_epochs = 3
            for epoch in range(50):
                tmp = init_tmp * (final_tmp / init_tmp) ** (epoch / 3)
                for i, (g, l) in enumerate(test_loader):
                    # g, l = g.to(device), l.to(device)
                    feat = {nt: g.ndata[nt][nt] for nt in g.ntypes}
                    # explainer.train_graph(graph=g, feat=feat, target=l)
                    loss = explainer.train_step(g, feat, tmp)
                    optimizer_exp.zero_grad()
                    loss.backward()
                    optimizer_exp.step()

            for i, (g, l) in enumerate(test_loader):
                g, l = g.to(device), l.to(device)
                feat = {nt: g.ndata[nt][nt] for nt in g.ntypes}
                prob, edge_mask = explainer.explain_graph(graph=g, feat=feat)

                # 2) агрегируем node‑mask из edge_mask
                feat_mask = {}
                for ntype in g.ntypes:
                    feat_mask[ntype] = torch.zeros(g.num_nodes(ntype),
                                                   device=g.device)

                for rel, em in edge_mask.items():
                    src_type, etype, dst_type = rel
                    src_ids, dst_ids = g.edges(form='uv', etype=rel)
                    feat_mask[src_type].index_add_(0, src_ids, em)
                    feat_mask[dst_type].index_add_(0, dst_ids, em)


                for nt in g.ntypes:
                    dict_feature[nt].append(feat_mask[nt])
                for rel in relation:
                    dict_edge[rel].append(edge_mask[rel])

        elif explain == 'attention':
            for i, (g, l) in enumerate(test_loader):
                g, l = g.to(device), l.to(device)

                # 1) Получаем node‑features (они не меняются)
                feat = {nt: g.ndata[nt][nt] for nt in g.ntypes}

                # 2) Прогоняем до последнего GAT‑слоя, чтобы взять attention
                h_dict = model(g, feat, embed=True)
                # 2) Manual forward through RGCN conv layers to get pre-attention h
                h = feat
                for conv_layer in model.rgcn.convs:
                    h = conv_layer(g, h)
                    # apply activation and flatten heads
                    h = {nt: torch.reshape(F.relu(v), (v.shape[0], -1)) for nt, v in h.items()}
                    # h = {nt: F.relu(v) for nt, v in h.items()}
                h_pre = h


                # 3) Compute attention scores per relation
                edge_alpha = {}
                for rel in relation:
                    conv_dict = model.rgcn.conv_out.mods

                    key_str = str(rel)
                    conv = conv_dict[key_str]

                    src_type, _, dst_type = rel
                    h_src, h_dst = h_pre[src_type], h_pre[dst_type]
                    # proj = nn.Linear(100, 128, bias=False).to(device)  # можно объявить один раз
                    # h_src = proj(h_src)
                    # h_dst = proj(h_dst)
                    # Extract attention weights
                    _, alpha = conv(g[rel], (h_src, h_dst), get_attention=True)
                    edge_alpha[rel] = alpha.mean(dim=1).detach()

                # 4) Из edge_alpha строим бинарную маску топ‑k
                total_edges = sum(g.num_edges(etype=rel) for rel in relation)
                k = max(1, int(0.2 * total_edges))  # или подбираем sparsity как r*|E|
                # собираем кортежи (rel, idx, score) и сортируем
                all_scores = [(rel, idx, s.item())
                              for rel, scores in edge_alpha.items()
                              for idx, s in enumerate(scores)]
                all_scores.sort(key=lambda x: -x[2])
                topk = set((rel, idx) for rel, idx, _ in all_scores[:k])

                edge_mask = {}
                for rel in relation:
                    m = torch.zeros(g.num_edges(etype=rel), dtype=torch.float, device=device)
                    for (_, idx) in topk:
                        if _ == rel:
                            m[idx] = 1.0
                    edge_mask[rel] = m

                feat_mask = {}
                for ntype in g.ntypes:
                    fm = torch.zeros(g.num_nodes(ntype), device=device)
                    feat_mask[ntype] = fm
                for rel, em in edge_mask.items():
                    src, _, dst = rel
                    u, v = g.edges(form='uv', etype=rel)
                    feat_mask[src].index_add_(0, u, em)
                    feat_mask[dst].index_add_(0, v, em)

                for nt in g.ntypes:
                    dict_feature[nt].append(feat_mask[nt].cpu())
                for rel in relation:
                    dict_edge[rel].append(edge_mask[rel].cpu())

        elif explain == 'subx':
            print("SubX began")
            # --- Инициализируем HeteroSubgraphX ---------------------------------
            num_ntypes = len(g0.ntypes)  # g0 — любой загруженный граф

            explainer = HeteroSubgraphX(
                model.to(device),
                num_hops=1,
                num_rollouts=30,  # можно увеличить при желании
                node_min=max(4, num_ntypes),  # ↓ объяснение «best_leaf=None»
                log=False
            )


            for i, (g, l) in enumerate(test_loader):
                g, l = g.to(device), l.to(device)

                # 1) исходные признаки узлов
                feat = {nt: g.ndata[nt][nt] for nt in g.ntypes}

                # 2) целевой класс нужен в формате int
                target_cls = int(l.argmax()) if l.ndim > 0 else int(l.item())
                print('l', l)
                print('target_cls', target_cls)

                # 3) получаем важнейший подграф
                #    node_dict: { ntype : Tensor[node_ids] }
                print('explain_graph')
                node_dict = explainer.explain_graph(
                    graph=g,
                    feat=feat,
                    target_class=target_cls
                )

                # 4) cтроим node- и edge-маски из полученного подграфа
                feat_mask = {nt: torch.zeros(g.num_nodes(nt), device=g.device)
                             for nt in g.ntypes}
                for nt, nids in node_dict.items():
                    feat_mask[nt][nids] = 1.0

                edge_mask = {}
                for rel in relation:  # relation загружен ранее
                    num_e = g.num_edges(etype=rel)
                    mask = torch.zeros(num_e, device=g.device)

                    if num_e:  # нет рёбер → пустая маска
                        u, v = g.edges(form='uv', etype=rel)
                        src_nt, _, dst_nt = rel
                        # ребро важно, если оба конца в отобранном подграфе
                        keep = (feat_mask[src_nt][u] == 1) & (feat_mask[dst_nt][v] == 1)
                        mask = keep.float()

                    edge_mask[rel] = mask

                for nt in g.ntypes:
                    dict_feature[nt].append(feat_mask[nt].cpu())
                for rel in relation:
                    dict_edge[rel].append(edge_mask[rel].cpu())

        os.makedirs(dump_dir, exist_ok=True)
        for f in next(iter(test_loader))[0].ntypes:
            with open(f'{dump_dir}/{log_name}_feature_{f}.pickle', 'wb') as fil:
                pickle.dump(dict_feature[f], fil)

        for e in relation:
            with open(f'{dump_dir}/{log_name}_edge_{e}.pickle', 'wb') as fil2:
                pickle.dump(dict_edge[e], fil2)
    else:
        print("Находим посчитанное")
        dict_feature = {}
        dict_edge = {}
        # feature
        for fname in os.listdir(dump_dir):
            if '_feature_' in fname:
                nt = fname.split('_feature_')[1].replace('.pickle', '')
                with open(os.path.join(dump_dir, fname), 'rb') as f:
                    dict_feature[nt] = pickle.load(f)

        # edge
        for fname in os.listdir(dump_dir):
            if '_edge_' in fname:
                key_str = fname.split('_edge_')[1].replace('.pickle', '')
                key = ast.literal_eval(key_str)  # строку → кортеж
                # rel = fname.split('_edge_')[1].replace('.pickle', '')
                with open(os.path.join(dump_dir, fname), 'rb') as f:
                    dict_edge[key] = pickle.load(f)
        print("Тип ключей dict_edge:", type(next(iter(dict_edge.keys()))))
        print("Первые 5 ключей:", list(dict_edge.keys())[:5])

    sparsity_grid = np.linspace(0.05, 0.6, 12)
    all_results = []

    for idx, g in enumerate(graphs):
        # вытаскиваем маски для i‑го графа в нужном формате:
        feat_i = {nt: dict_feature[nt][idx] for nt in dict_feature}
        edge_i = {rel: dict_edge[rel][idx] for rel in dict_edge}
        rows = []
        for sp in sparsity_grid:
            total_edges = sum(m.shape[0] for m in edge_i.values())
            top_k = max(1, int(sp * total_edges))
            metrics = compute_xai_metrics(model, g, edge_i, init_labels[idx], top_k=top_k)

            rows.append({
                'sparsity': sp,
                'comprehensiveness': metrics['comprehensiveness'],
                'sufficiency': metrics['sufficiency'],
                # 'stability': metrics['stability'],
                'f1_fidelity': metrics['f1_fidelity'],
            })
        rows = pd.DataFrame(rows)
        rows['case'] = idx
        all_results.append(rows)


        # top_k = max(1, int(0.25 * total_edges))
        # metrics = compute_xai_metrics(model, g, edge_i, y_test[idx], top_k=top_k)
        # print(f"XAI metrics for loaded case {idx}: {metrics}")
        # metrics_list.append({'case': idx, **metrics})

        print(f"Рисую объяснение для графа #{idx}")

        y_vec = np.asarray(init_labels[idx])
        node_id = int(y_vec.argmax())
        target_node = ('Activity', node_id)
        target_type = 'Activity'
        target_node_label = f"{target_type}_{int(node_id)}"

        out_dir = f'{base_dir}/pics/{log_name}'
        os.makedirs(out_dir, exist_ok=True)
        visualize_explanation_matplotlib(g, feat_i, edge_i, case_idx=idx, out_dir=out_dir)




    results_df = pd.concat(all_results, ignore_index=True)
    results_df = results_df.round(2)
    results_df.to_csv(os.path.join(f'{base_dir}/results/raw', f"{log_name}_metrics_{explain}.csv"), index=False)


    mean_df = results_df.groupby('sparsity').mean().reset_index()
    elapsed = time.time() - start_time
    mean_df['runtime_sec'] = elapsed
    mean_df = mean_df.round(2)
    mean_df.to_csv(os.path.join(f'{base_dir}/results/agg', f"{log_name}_metrics_mean_{explain}.csv"), index=False)
    print(f"Saved raw metrics to metrics_{explain}.csv and aggregated to metrics_mean_{explain}.csv")
