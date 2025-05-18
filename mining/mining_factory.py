import re
from pm4py.visualization.petri_net.variants.token_decoration_frequency import get_decorations
import numpy as np
# from pm4py.algo.discovery.inductive.algorithm import apply as inductive_miner, IM, IMf, IMd, Variants
# from pm4py.algo.discovery.inductive import algorithm as inductive_miner

import networkx as nx
# from pm4py.objects.petri_net import networkx_graph
from pm4py.objects.petri_net.utils import networkx_graph

import os
# from pm4py.algo.discovery.heuristics.algorithm import Parameters as heuristics_parameters)
from pathlib import Path

from src.mining.MiningUtils import MiningUtils
from src.mining.SplitMiner import SplitMiner
from src.utils.utils import Utils
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner


class MinerAlgorithmNotFoundException(Exception):
    def __init__(self, algorithm, message="Mining algorithm not found: "):
        self.algorithm = algorithm
        self.message = message + algorithm


class MiningFactory:
    def __init__(self, log, log_path):
        self.log = log
        self.log_path = log_path

    def mine(self, algorithm):
        data_results_store_dir = "./data/results/" + Path(self.log_path).name.replace("train_val_", "") + "/models/"

        if algorithm == "heuristics":
            # net, initial_marking, final_marking = heuristics_miner.apply(self.log, parameters={
            #     heuristics_parameters.DEPENDENCY_THRESH: 0.95})
            print("Этот метод не доделан, есть ошибки")
        elif algorithm == "inductive":
            # net, initial_marking, final_marking = inductive_miner.apply(self.log, parameters=None,
            #                                                       variant=IM)
            print("Этот метод не доделан, есть ошибки")
        elif algorithm == "inductive_infrequent":
            # net, initial_marking, final_marking = inductive_miner.apply(self.log, parameters=None,
            #                                                       variant=IMf)
            print("Этот метод не доделан, есть ошибки")
        elif algorithm == "inductive_directly":
            # net, initial_marking, final_marking = inductive_miner.apply(self.log, parameters=None,
            #                                                       variant=IMd)
            print("Этот метод не доделан, есть ошибки")
        elif algorithm == "split":
            net, initial_marking, final_marking = SplitMiner(self.log_path, "./data/split_miner_models",
                                                             "./data/split_miner_best_models").mine(
                store_results=data_results_store_dir)
        else:
            raise MinerAlgorithmNotFoundException(algorithm)

        Path(data_results_store_dir).mkdir(parents=True, exist_ok=True)
        save_file = os.path.join(data_results_store_dir, "best_model_" + algorithm + "_results.txt")
        MiningUtils.calculate_metrics(self.log, net, initial_marking, final_marking, save_file=save_file,
                                      model_name="best model ")

        print(f"[Mining] Transitions: {len(net.transitions)}, Places: {len(net.places)}")
        print(f"[Mining] Arcs: {len(net.arcs)}")
        print(f"[Mining] Silent transitions: {[t.label for t in net.transitions if t.label is None]}")

        return net, initial_marking, final_marking

    def _replay(self, net, initial_marking, final_marking):
        from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
        return token_replay.apply(self.log, net, initial_marking, final_marking)

    def get_path_statistics(self, net, initial_marking, final_marking, activity_dict, node_dict):
        # Parse the decorations from the pm4py image to get how many times a certain path is used
        path_stats = self._parse_pm4py_decorations(net, initial_marking, final_marking)
        path_statistics = np.zeros(shape=(len(node_dict.keys()), len(activity_dict.keys())))

        nx_net, inv_node_dict = networkx_graph.create_networkx_directed_graph(net)
        trans_dict = {value: key for key, value in inv_node_dict.items()}
        for place in net.places:
            for transition in net.transitions:
                if transition.label is not None:
                    id_place = node_dict[str(place)]
                    id_activity = activity_dict[str(transition)]
                    # Find the path between the place and the transition
                    # The frequency will be the minimum between the subpaths of the path
                    min_freq_value = np.inf
                    try:
                        shortest_path = nx.shortest_path(nx_net, trans_dict[place], trans_dict[transition])
                        for prev_node_id, node_id in zip(shortest_path, shortest_path[1:]):
                            prev_node = str(inv_node_dict[prev_node_id])
                            node = str(inv_node_dict[node_id])
                            freq = path_stats[prev_node][node]
                            if min_freq_value > freq:
                                min_freq_value = freq

                        path_statistics[id_place][id_activity] = min_freq_value

                        # print("Shortest path: ", shortest_path)

                    except:
                        #print("Path not found between: ", place, " and ", transition)
                        pass

        # Normalize the matrix here
        path_statistics /= np.amax(path_statistics)
        # print("Path statistics: ")
        # print(path_statistics)

        return path_statistics

    def _parse_pm4py_decorations(self, net, initial_marking, final_marking):
        decorations = get_decorations(self.log, net, initial_marking, final_marking)
        path_pattern = "\(.\)(.*)->\(.\)(.*)"
        path_stats = {}
        for key in decorations.keys():
            key_str = str(key)
            # print("Key: ", type(key), " ; label: ", decorations[key]["label"])
            search = re.search(path_pattern, key_str)
            if search is not None:
                source = search.group(1)
                dest = search.group(2)
                if source not in path_stats:
                    path_stats[source] = {}
                if dest not in path_stats[source]:
                    path_stats[source][dest] = {}
                path_stats[source][dest] = int(decorations[key]["label"])

        return path_stats

    def get_reachability_vectors(self, net):
        nx_graph, inv_node_dict = networkx_graph.create_networkx_directed_graph(net)
        # self.nodes: place assignment
        # self.transition_assigment: transition assigment
        reachability_vector = {}
        regular_transitions = []
        for transition in net.transitions:
            if transition.label is not None:
                regular_transitions.append(transition)

        # Invert the pm4py node to petri node dict
        trans_dict = {value: key for key, value in inv_node_dict.items()}

        for place in net.places:
            reachability_vector[place] = []
            for transition in regular_transitions:
                source = trans_dict[place]
                dest = trans_dict[transition]
                if nx.has_path(nx_graph, source, dest):
                    reachability_vector[place].append(transition.label)

        # print("Reachability vector: ", reachability_vector)
        return reachability_vector
