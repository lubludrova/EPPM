# src/vectorizer/EmbeddingGraphVectorizer.py
# from pm4py.objects.petri import networkx_graph
import networkx as nx
import numpy as np
from src.vectorizer.AbstractVectorizer import AbstractVectorizer

class EmbeddingGraphVectorizer(AbstractVectorizer):

    def __init__(self, net, initial_marking, final_marking, unique_attributes, unique_activities, max_len, activity_dict, avg_time_events=None, avg_time_start=None, is_inductive=True):
        super().__init__(net, initial_marking, final_marking, unique_attributes, unique_activities, max_len, activity_dict, is_inductive)

        self.delete_transitions(self.nx_graph)
        self.reachability_vectors = self.get_reachability_vectors(net)
        self.avg_time_events = avg_time_events
        self.avg_time_start = avg_time_start
        self.extra_features = 7  # временные признаки + node и activity id
        self.L = len(self.unique_activities)
        # self.F = self.extra_features + len(unique_attributes)
        self.F = self.extra_features + len(list(unique_attributes.keys()))
        self.replay_cache = {}
        self.log_cache = {}

    def delete_transitions(self, nx_graph):
        place_assignment = {}
        for i, place in enumerate(self.places):
            place_assignment[place] = i

        items = self.nodes.items()
        # for key, value in items:
        #     print(value)

        nodes = {str(key): value for key, value in items}
        reverse_nodes = {value: str(key) for key, value in items}
        self.nodes = place_assignment

        self.adjacency_matrix = np.zeros((len(self.places), len(self.places)))
        self.N = len(self.places)
        # print(nodes)
        self.transition_to_index = {}
        for idx, obj in self.nx_graph[1].items():
            if hasattr(obj, 'label'):
                label = obj.label if obj.label is not None else obj.name
                self.transition_to_index[label] = idx

        # print(self.nx_graph[1].items())
        # print(self.transition_to_index)

        for transition in self.transitions:
            # transition_id = nodes[transition]
            transition_id = self.transition_to_index.get(transition)
            if transition_id is None:
                raise KeyError(f"Transition {transition} not found in transition_to_index mapping.")

            # transition_id = 58
            in_edges = self.nx_graph[0].in_edges(transition_id)
            out_edges = self.nx_graph[0].out_edges(transition_id)

            for in_edge in in_edges:
                for out_edge in out_edges:
                    in_place = reverse_nodes[in_edge[0]]
                    out_place = reverse_nodes[out_edge[1]]
                    i = place_assignment[in_place]
                    j = place_assignment[out_place]
                    self.adjacency_matrix[i, j] = 1

    def get_reachability_vectors(self, net):
        reachability_vector = {}
        regular_transitions = [t for t in net.transitions if t.label]

        trans_dict = {v: k for k, v in self.nx_graph[1].items()}

        for place in net.places:
            reachability_vector[place] = [
                t.label for t in regular_transitions
                if nx.has_path(self.nx_graph[0], trans_dict[place], trans_dict[t])
            ]
        return reachability_vector

    def vectorize_batch(self, partial_trace_batch, next_events):
        avg_time_events = self.avg_time_events
        avg_time_start = self.avg_time_start

        X, y, y_next_timestamp = [], [], []

        for idx, (trace, next_event) in enumerate(zip(partial_trace_batch, next_events)):
            trace_events, marking = [], self.initial_marking
            node_activations = {n: 0 for n in self.nodes}

            for curr_event, event in enumerate(trace, 1):
                F_event = np.zeros((self.N, self.F), dtype="float32")
                curr_activity = event["concept:name"]

                time_feats = self.get_time_features(
                    event, trace, curr_event - 1, avg_time_events, avg_time_start
                )
                all_markings, marking, _ = self.replay_event(event, marking, node_activations)

                for place in all_markings:
                    place_id = self.nodes[str(place)]
                    F_event[place_id, 0] = self.activity_dict[curr_activity]
                    F_event[place_id, 1] = place_id
                    F_event[place_id, 2] = node_activations[str(place)] / self.max_len
                    F_event[place_id, 3:7] = time_feats
                    for attr_pos, attr in enumerate(self.unique_attributes, 1):
                        F_event[place_id, 6 + attr_pos] = self.unique_attributes[attr][str(event[attr])]

                trace_events.append(F_event[None, ...])

            # Pre-padding
            padded_trace_events = trace_events.copy()
            for _ in range(self.max_len - len(trace_events)):
                padded_trace_events.insert(0, np.zeros((1, self.N, self.F), dtype="float32"))

            if self.max_len < len(trace_events):
                padded_trace_events = padded_trace_events[-self.max_len:]

            X.append(np.concatenate(padded_trace_events, axis=0)[None, ...])

            if next_event != "[EOC]":
                delta_time = (next_event["time:timestamp"] - trace[-1]["time:timestamp"]).total_seconds() / avg_time_events
                y_next_timestamp.append(delta_time)
                y.append(self.activity_dict[next_event["concept:name"]])
            else:
                y_next_timestamp.append(0.0)
                y.append(self.activity_dict["[EOC]"])


        return (
            np.concatenate(X, axis=0),
            np.array(y),
            self.adjacency_matrix,
            self.N,
            self.F,
            np.expand_dims(np.array(y_next_timestamp), -1)
        )

    # replay_event и другие вспомогательные методы берутся полностью из оригинального файла без изменений
