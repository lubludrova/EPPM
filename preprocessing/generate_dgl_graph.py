import sys
import pandas as pd
import torch
from preprocessing.read_event_log import GenerateTrace
import pickle
import dgl
from sklearn import preprocessing
from gensim.models import Word2Vec
from Orange.data.pandas_compat import table_from_frame
import Orange
from dgl import AddReverse
import h5py
from pm4py.objects.log.importer.xes import importer as xes_importer
import numpy as np
from pm4py.objects.conversion.log import converter as xes_converter
import os

def replace_char(ele):
    ele = ele.replace(' ', '')
    ele = ele.replace('-', '')
    ele = ele.replace('+', '')
    ele = ele.replace('_', '')
    ele = ele.replace('.', '')
    ele = ele.replace(':', '')
    ele = ele.replace('(', '')
    ele = ele.replace(')', '')
    return ele

def build_w2v(prefix_list, mean):
        temp_traces = []
        for k in prefix_list:
            listToStr = ' '.join([replace_char(str(elem)) for elem in k])
            temp_traces.append(listToStr)

        tokenized_words = []
        for s in temp_traces:
            tokenized_words.append(s.split(' '))

        w2v_model = Word2Vec(vector_size=mean, seed=0, min_count=1, sg=0, workers=-1)
        w2v_model.build_vocab(tokenized_words, min_count=1)
        total_examples = w2v_model.corpus_count
        w2v_model.train(tokenized_words, total_examples=total_examples, epochs=50)
        vocab = list(w2v_model.wv.index_to_key)
        word_vec_dict = {}
        for word in vocab:
            word_vec_dict[word] = w2v_model.wv.get_vector(word)
        return word_vec_dict

def clear_list(prefix_list):
    temp_traces = []
    for k in prefix_list:
        listToStr = ' '.join([replace_char(str(elem)) for elem in k])
        temp_traces.append(listToStr)

    tokenized_words = []
    for s in temp_traces:
        tokenized_words.append(s.split(' '))
    return tokenized_words


def gen_flow(id):
    id.insert(0, 'START')
    remove_dup = list(dict.fromkeys(id))
    remove_dup = [[a] for a in remove_dup]
    id = np.array(id)
    node_encoder = preprocessing.LabelEncoder()
    enc = node_encoder.fit_transform(id)

    return remove_dup, enc


def create_triangular_matrix(columns):
    # Convert set to sorted list for consistent indexing
    cols = list(columns)
    n = len(cols)

    # Create empty matrix filled with zeros
    list_relation = []
    # Fill upper triangular part
    for i in range(n):
        for j in range(i, n):
            # Create pair of column names
            if cols[i] == cols[j]:
                pair = (cols[i], 'follow', cols[j])
            else:
                pair = (cols[i], 'has', cols[j])
            list_relation.append(pair)
    return list_relation

def gen_edge_weigts(list1, list2):
    combined_tuples = list(zip(list1, list2))
    tuple_counts = {}
    for pair in combined_tuples:
        if pair in tuple_counts:
            tuple_counts[pair] += 1
        else:
            tuple_counts[pair] = 1
    return list(tuple_counts.values())


def apply_w2v(list_act, enc_act, mean):
    list_act = clear_list(list_act)
    x_act_ohe = []
    for l in list_act:
        list_emb_temp = []
        for t in l:
            embed_vector = enc_act.get(t)
            if embed_vector is not None:
                list_emb_temp.append(embed_vector)
            else:
                list_emb_temp.append(np.zeros(shape=(mean,)))
        x_act_ohe.append(list_emb_temp)
    x_act_ohe = np.array(x_act_ohe)
    x_act_ohe = x_act_ohe.reshape(x_act_ohe.shape[0], mean)
    return x_act_ohe


def unique_edge(list1, list2):
    unique_tuples = []
    seen_tuples = set()

    for pair in zip(list1, list2):
        if pair not in seen_tuples:
            unique_tuples.append(pair)
            seen_tuples.add(pair)

    return unique_tuples


def build_list_graphs(dict_view, dict_y, dict_enc, mean, c, event_attributes, case_attributes, relation):


    with h5py.File(c, "w") as f:
        k = 0
        list_graphs = []
        while k<len(dict_view['concept:name']):
                list_node = {}
                list_node_comp = {}
                list_node_feature = {}
                dgl_canonical_edge = {}
                weight_node_follow_node = {}
                for v in event_attributes:#lc.dict_log[log_name]['event_attr']:
                    list_node[v], list_node_comp[v] = gen_flow(dict_view[v][k])
                    list_node_feature[v] = apply_w2v(list_node[v], dict_enc[v], mean) #W2W

                list_att_trace = []
                for v in case_attributes:#lc.dict_log[log_name]['trace_attr_cat']:
                    embed_vector = dict_enc[v].get(replace_char(dict_view[v][k][0]))
                    if embed_vector is not None:
                        res = embed_vector
                    else:
                        res = np.zeros(shape=(mean,))
                    list_att_trace.append(res)

                #for v in lc.dict_log[log_name]['trace_attr_num']:
                #    list_att_trace.append(np.array(dict_view[v][k][0]).reshape(1))
                if list_att_trace !=[]:
                    list_node_comp['trace_att'] = [0]
                    list_node_feature['trace_att'] = np.array([np.concatenate(list_att_trace)])

                for rel in relation:#lc.dict_log[log_name]['relation']:
                    if rel[1] == 'follow':
                        edge_res = np.array([[list_node_comp[rel[0]][i], list_node_comp[rel[0]][i + 1]] for i in range(len(list_node_comp[rel[0]]) - 1)])
                    elif rel[1] == 'has_ta':
                        list_node_comp[rel[2]] = [0]*len(np.unique(list_node_comp[rel[0]]))
                        edge_res = list(map(lambda X: [X[0], X[1]], list(zip(np.unique(list_node_comp[rel[0]]), list_node_comp[rel[2]]))))
                    else:
                        edge_res = list(map(lambda X: [X[0], X[1]], list(zip(list_node_comp[rel[0]], list_node_comp[rel[2]]))))
                    src = [item[0] for item in edge_res]
                    dst = [item[1] for item in edge_res]
                    tuple_src_dst = unique_edge(src, dst)
                    dgl_canonical_edge[rel] = tuple_src_dst
                    weight_node_follow_node[rel] = gen_edge_weigts(src,dst)
                hetero_graph = dgl.heterograph(dgl_canonical_edge)


                for nn in list_node_feature:
                        hetero_graph.nodes[nn].data[nn] = torch.tensor(list_node_feature[nn], dtype=torch.float)

                for rel in weight_node_follow_node:
                    hetero_graph.edata['h'] = {rel:torch.tensor(weight_node_follow_node[rel])}
                transform = AddReverse(copy_edata=True)
                new_g = transform(hetero_graph)
                list_graphs.append(new_g)
                pickled_graph = pickle.dumps({'graph':new_g, 'label':dict_y[k]})
                f.create_dataset(f"array_{k}", data=np.void(pickled_graph))
                k = k + 1


def equifreq(view_train, view_test, n_bin):
        sort_v = np.append(view_train, view_test)
        df = pd.DataFrame(sort_v)
        df = table_from_frame(df)
        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EqualFreq(n=n_bin)
        df = disc(df)
        df = list(df)
        df = list(map(str, df))
        view_train = df[:len(view_train)]
        view_test = df[len(view_train):]
        return view_train, view_test

def add_time_column(group):
        timestamp_col = 'time:timestamp'
        group = group.sort_values(timestamp_col, ascending=True)
        # end_date = group[timestamp_col].iloc[-1]
        start_date = group[timestamp_col].iloc[0]

        timesincelastevent = group[timestamp_col].diff()
        timesincelastevent = timesincelastevent.fillna(pd.Timedelta(seconds=0))
        group["timesincelastevent"] = timesincelastevent.apply(
            lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

        elapsed = group[timestamp_col] - start_date
        elapsed = elapsed.fillna(pd.Timedelta(seconds=0))
        group["timesincecasestart"] = elapsed.apply(lambda x: float(x / np.timedelta64(1, 's')))  # s is for seconds

        return group

if __name__ == '__main__':

    # log_name = 'Helpdesk.xes.gz'

    # Load the XES log
    # log = xes_importer.apply('fold/'+log_name)
    log_folder = "fold"
    # log_name = "Helpdesk.xes.gz"
    # log_name = "BPI_Challenge_2012_A.xes.gz"
    # log_name = "BPI_Challenge_2012_W.xes.gz"
    # log_name = "SEPSIS.xes.gz"
    # log_name = "BPI_Challenge_2013_closed_problems.xes.gz"
    log_name = "SEPSIS.xes.gz"


    log_path = os.path.join(
        '/Users/ruslanageev/PycharmProjects/Prophet/',
        log_folder,
        log_name  # Пример: "Helpdesk.xes.gz"
    )
    log = xes_importer.apply(log_path, parameters={"timestamp_sort": True})

    # Extract case attributes
    case_attributes = []
    for trace in log:
        for attr in trace.attributes.keys():
            case_attributes.append('case:'+attr)

    case_attributes = list(np.unique(case_attributes))

    # Extract event attributes
    event_attributes = []
    for trace in log:
        for event in trace:
            for attr in event.keys():
                event_attributes.append(attr)

    event_attributes = list(np.unique(event_attributes))

    list_e_a = []
    for i in list(event_attributes):
        print('Event Attribute --> {}'.format(i))
        print('Do you want to keep the feature? yes/no')
        while True:
            # x = input()
            if i != "Unnamed: 0":
                x = 'yes'
            else:
                x = 'no'
            if x.lower() == 'yes':
                list_e_a.append(i)
                break
            elif x.lower() == 'no':
                break
            else:
                print("Please enter 'yes' or 'no'")

    list_e_a.remove('time:timestamp')
    list_e_a.append('timesincecasestart')

    list_c_a = []
    for i in list(case_attributes):
        if i != 'case:concept:name':
            print('Case Attribute --> {}'.format(i))
            print('Do you want to keep the feature? yes/no')
            while True:
                # x = input()
                x = 'yes'
                if x.lower() == 'yes':
                    list_e_a.append(i)
                    break
                elif x.lower() == 'no':
                    break
                else:
                    print("Please enter 'yes' or 'no'")


    result = create_triangular_matrix(list_e_a)
    result2 = []
    if len(list_c_a) > 0:
        for x in list_e_a:
                result2.append((x, 'has_ta', 'trace_att'))

    dict_card = {}
    dict_view_train = {}
    dict_view_test = {}
    dict_view_train_y = {}
    dict_view_test_y = {}
    dict_enc = {}

    relation = result + result2
    print('Event Attribute --> {}'.format(list_e_a))
    print('Case Attribute --> {}'.format(list_c_a))
    print('Relations --> {}'.format(relation))

    print('log-->', log_name)

    output = open("w2v/" + log_name + "/relations.pkl", 'wb')
    pickle.dump(relation, output)

    pm = GenerateTrace(log_name)
    log = xes_converter.apply(log, variant=xes_converter.Variants.TO_DATA_FRAME)

    mean = 100

    log = log.fillna('unk')
    log_train, log_test = pm.generate_split(log)

    log_train['time:timestamp'] = pd.to_datetime(log_train['time:timestamp'])
    log_test['time:timestamp'] = pd.to_datetime(log_test['time:timestamp'])

    log_train = log_train.groupby('case:concept:name', group_keys=False).apply(add_time_column)
    log_test = log_test.groupby('case:concept:name', group_keys=False).apply(add_time_column)
    log_train = log_train.reset_index(drop=True)
    log_test = log_test.reset_index(drop=True)

    if 'org:resource' in list_e_a:
        num_act = list(set(list(log_train['concept:name'].unique()) + list(log_test['concept:name'].unique())))
        num_res = list(set(list(log_train['org:resource'].unique()) + list(log_test['org:resource'].unique())))
        n_bin = (len(num_act) + len(num_res)) // 2
        log_train['org:resource'] = log_train['org:resource'].astype(str)
        log_test['org:resource'] = log_test['org:resource'].astype(str)
    else:
        num_act = list(set(list(log_train['concept:name'].unique()) + list(log_test['concept:name'].unique())))
        n_bin = len(num_act)

    if not os.path.exists("w2v/" + log_name):
        os.makedirs("w2v/" + log_name)

    for attr in list_e_a:
        is_numeric = pd.to_numeric(log_train[attr], errors='coerce').notnull().all()
        if is_numeric:
            log_train[attr], log_test[attr] = equifreq(log_train[attr], log_test[attr], n_bin)
        node_encoder = preprocessing.OneHotEncoder()
        dict_card[attr] = list(set(pd.concat([log_train[attr],log_test[attr]])))
        dict_card[attr].insert(0, 'START')
        dict_view_train[attr], dict_view_train_y[attr] = pm.get_sequence(pm.generate_prefix_trace(log=log_train, view=attr))
        dict_view_test[attr], dict_view_test_y[attr] = pm.get_sequence(pm.generate_prefix_trace(log=log_test, view=attr))
        dict_enc[attr] = build_w2v(dict_view_train[attr], mean)
        output = open("w2v/" + log_name + "/" + attr + ".pkl", 'wb')
        pickle.dump(dict_enc[attr], output)
        output.close()
        dict_att = [[a] for a in dict_card[attr]]

    for attr in list_c_a:
        is_numeric = pd.to_numeric(log_train[attr], errors='coerce').notnull().all()
        print(attr, is_numeric)
        if is_numeric:
            print('numerical column--->', attr)
            log_train[attr], log_test[attr] = equifreq(log_train[attr], log_test[attr], n_bin)
        dict_card[attr] = list(set(pd.concat([log_train[attr], log_test[attr]])))
        dict_view_train[attr] = pm.get_sequence_num(pm.generate_prefix_trace(log=log_train, view=attr))
        dict_view_test[attr] = pm.get_sequence_num(pm.generate_prefix_trace(log=log_test, view=attr))
        dict_enc[attr] = build_w2v(dict_view_train[attr], mean)
        output = open("w2v/" + log_name + "/" + attr + ".pkl", 'wb')
        pickle.dump(dict_enc[attr], output)
        output.close()
        dict_att = [[a] for a in dict_card[attr]]

    label_encoder = preprocessing.LabelEncoder()
    dict_card['concept:name'].remove('START')

    integer_encoded = label_encoder.fit_transform(dict_card['concept:name'])
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    train_integer_encoded = label_encoder.transform(dict_view_train_y['concept:name']).reshape(-1, 1)
    train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
    Y_train = np.asarray(train_onehot_encoded)

    test_integer_encoded = label_encoder.transform(dict_view_test_y['concept:name']).reshape(-1, 1)
    test_onehot_encoded = onehot_encoder.transform(test_integer_encoded)
    Y_test = np.asarray(test_onehot_encoded)
    Y_test_int = np.asarray(test_integer_encoded)

    build_list_graphs(dict_view_train, Y_train, dict_enc, mean, '../heterographs_tracenode/' + log_name + '_train.db', list_e_a, list_c_a, relation)
    print('end_train')

    build_list_graphs(dict_view_test, Y_test, dict_enc, mean, '../heterographs_tracenode/' + log_name + '_test.db', list_e_a, list_c_a, relation)
    print('end_test')