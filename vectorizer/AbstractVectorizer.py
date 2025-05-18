import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
#ABC: Abstract Base Class
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.utils import networkx_graph
#from pm4py.algo.conformance.tokenreplay.versions.token_replay import *
from pm4py.algo.conformance.tokenreplay.algorithm import *
import networkx as nx

class AbstractVectorizer(ABC):
    def __init__(self, net, initial_marking, final_marking, unique_attributes, unique_activities, max_len, activity_dict, is_inductive=True):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.unique_attributes = unique_attributes
        self.unique_activities = unique_activities
        self.max_len = max_len
        self.activity_dict = activity_dict
        self.nx_graph = networkx_graph.create_networkx_directed_graph(self.net)
        self.adjacency_matrix, self.N, self.nodes, self.places, self.transitions, self.arcs = self.get_features_from_graph(self.nx_graph)
        self.activity_to_transition = {}
        for transition in self.net.transitions:
            self.activity_to_transition[transition.label] = transition

        self.is_inductive = is_inductive

    @abstractmethod
    def vectorize_batch(self, partial_trace_batch, next_events):
        pass

    def load_log(self, log_file):
        log = xes_importer.apply(log_file, parameters={"timestamp_sort": True})
        return log

    def replay_log(self, log_file):
        log = self.load_log(log_file)
        replay_result = apply(log, self.net, self.initial_marking, self.final_marking)
        return replay_result, log

    def get_features_from_graph(self, nx_graph):
        adjacency_matrix = nx.adjacency_matrix(nx_graph[0])
        number_of_nodes = adjacency_matrix.shape[0]
        nodes = {v: k for k, v in nx_graph[1].items()}
        places = [p.name for p in self.net.places]
        places = sorted(places)
        transitions = [t.label if t.label is not None else t.name for t in self.net.transitions]
        transitions = sorted(transitions)
        arcs = self.net.arcs
        return adjacency_matrix, number_of_nodes, nodes, places, transitions, arcs

    def visualize_net(self):
        from pm4py.visualization.petrinet import visualizer as pn_vis_factory
        gviz = pn_vis_factory.apply(self.net, self.initial_marking, self.final_marking)
        pn_vis_factory.view(gviz)

    def get_time_measures(self, log, bucket_1, bucket_2):
        time_since_last_events = []
        time_since_start = []
        start_seconds = None
        for trace in log:
            for i, event in enumerate(trace):
                if i == 0:
                    start_seconds = event["time:timestamp"]
                    time_since_last_events.append(0)
                    time_since_start.append(0)
                else:
                    curr_seconds = event["time:timestamp"]
                    prev_seconds = trace[i-1]["time:timestamp"]
                    time_since_last_events.append((curr_seconds - prev_seconds).total_seconds())
                    time_since_start.append((curr_seconds - start_seconds).total_seconds())

        mean_time_between_events = np.mean(time_since_last_events)
        mean_time_since_start = np.mean(time_since_start)
        std_time_between_events = np.std(time_since_last_events)
        std_time_since_start = np.std(time_since_start)

        max_time_between_events = np.max(time_since_last_events)
        max_time_since_start = np.max(time_since_start)
        min_time_between_events = np.min(time_since_last_events)
        min_time_since_start = np.min(time_since_start)

        _, bins_between_events = pd.qcut(time_since_last_events, q=bucket_1, duplicates="drop", retbins=True, labels=False)
        _, bins_since_start = pd.qcut(time_since_start, q=bucket_2, duplicates="drop", retbins=True, labels=False)

        return bins_between_events, bins_since_start, \
               (mean_time_between_events, std_time_between_events, max_time_between_events, min_time_between_events),\
               (mean_time_since_start, std_time_since_start, max_time_since_start, min_time_since_start)

    def quantize_timestamp(self, value, bins):
        # Add 1 to avoid the embedding of the 0 value
        # (0 values are ignored by the embedding layer)
        if value == 0:
            return 1 # Return 1 since the 0 value must be ignored
        quantization = pd.cut([value], bins=bins, labels=False)[0] + 1

        #print("Value: ", value)
        #print("Bins: ", bins)
        #print("Quantization: ", quantization)

        return quantization

    def get_time_features(self, event, trace, event_id):
        if event_id > 0:
            time_since_prev = (event["time:timestamp"] - trace[event_id - 1]["time:timestamp"]).total_seconds()
            time_since_start = (event["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds()
        else:
            time_since_prev = 0.0
            time_since_start = 0
        weekday = event["time:timestamp"].weekday() / 7
        midnight = event["time:timestamp"].replace(hour=0, minute=0, second=0, microsecond=0)
        time_since_midnight = (event["time:timestamp"] - midnight).total_seconds() / 86400

        return time_since_prev, time_since_start, weekday, time_since_midnight

    @staticmethod
    def get_prefixes_suffixes(log_file):
        X_prefix = []
        y_prefix = []
        y_suffix = []
        log = xes_importer.apply(log_file, parameters={"timestamp_sort": True})
        for trace in log:
            for i, event in enumerate(trace, 1):
                X_prefix.append(trace[0:i])
                try:
                    y_prefix.append(trace[i])
                except IndexError:
                    y_prefix.append("[EOC]")
                y_suffix.append(trace[i:])

        return X_prefix, y_prefix, y_suffix


