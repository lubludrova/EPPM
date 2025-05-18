from copy import copy
from pathlib import Path

# import pm4pycvxopt  # Fast alignments
import networkx as nx
import numpy as np
import time

from pm4py.algo.conformance.tokenreplay.variants.token_replay import apply_hidden_trans
# from pm4py.algo.conformance.tokenreplay.versions.token_replay import apply_hidden_trans
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.utils.petri_utils import get_places_shortest_path_by_hidden
from sklearn.metrics import accuracy_score
import tqdm, sys
from src.vectorizer.AbstractVectorizer import AbstractVectorizer


class EmbeddingGraphVectorizerPostMarkingNoTransitionsMultihotAdditionalFeatures(AbstractVectorizer):

    def __init__(self, net, initial_marking, final_marking, unique_attributes, unique_activities, max_len, activity_dict, bins_between_events=None, bins_since_start=None, normalization_between_events=None, normalization_since_start=None, is_inductive=True):

        super().__init__(net, initial_marking, final_marking, unique_attributes, unique_activities, max_len, activity_dict, is_inductive)
        self.delete_transitions(self.nx_graph)

        self.bins_between_events = bins_between_events
        self.bins_since_start = bins_since_start
        # self.reachability_vectors = mining_factory.get_reachability_vectors(net)
        self.normalization_between_events = normalization_between_events
        self.normalization_since_start = normalization_since_start
        #self.path_statistics = mining_factory.get_path_statistics(net, initial_marking, final_marking, activity_dict, self.nodes)
        self.static_event_features = 5
        #self.static_concatenable_features = len(net.transitions)
        self.static_concatenable_features = 0
        self.dynamic_features = len(list(unique_attributes.keys()))
        self.L = len(self.unique_activities)
        self.F = self.static_event_features + self.static_concatenable_features + self.dynamic_features
        self.replay_cache = {}
        self.log_cache = {}
        # print("N transitions: ", len(self.transitions))

    def delete_transitions(self, nx_graph):
        place_assignment = {}
        self.transition_assignment = {}

        for i, place in enumerate(self.places, start=1):
            place_assignment[place] = i



        for i, transition in enumerate(self.transitions, start=1):
            self.transition_assignment[str(transition)] = i

        items = self.nodes.items()
        nodes = {str(key) : value for key, value in items}
        reverse_nodes = {value : str(key) for key, value in items}
        self.reverse_nodes = reverse_nodes
        self.nodes = place_assignment

        #print("Nodes: ", nodes)

        self.adjacency_matrix = np.zeros(shape=(len(self.places), len(self.places)))
        self.N = len(self.places)
        #print("Transition list: ", self.transitions)
        #print("Nodes: ", nodes)
        self.transition_to_index = {}
        for idx, obj in self.nx_graph[1].items():
            if hasattr(obj, 'label'):
                label = obj.label if obj.label is not None else obj.name
                self.transition_to_index[label] = idx

        for transition in self.transitions:

            transition_id = self.transition_to_index.get(transition)
            if transition_id is None:
                raise KeyError(f"Transition {transition} not found in transition_to_index mapping.")


            in_edges = self.nx_graph[0].in_edges(transition_id)
            out_edges = self.nx_graph[0].out_edges(transition_id)

            for in_edge in in_edges:
                for out_edge in out_edges:
                    in_place = reverse_nodes[in_edge[0]]
                    out_place = reverse_nodes[out_edge[1]]
                    i = place_assignment[in_place]
                    j = place_assignment[out_place]
                    # Substract 1 since the ids start in 1
                    self.adjacency_matrix[i - 1, j - 1] = 1

        """
        import matplotlib.pyplot as plt
        reduced_graph = nx.from_numpy_matrix(self.adjacency_matrix, create_using=nx.DiGraph)
        nx.draw(reduced_graph, with_labels=True)
        log_name = Path(self.mining_factory.log_path).name.replace("train_val_", "")
        plt.savefig("./data/model_visualization/" + log_name + "_networkx_graph.png")
        plt.close()

        # Substract 1 so as to the ids match in the graph and in the dictionary
        reverse_place_assignment = {value - 1 : str(key) for key, value in place_assignment.items()}

        print("Nodes networkx: ", reduced_graph.nodes())
        print("Reverse place assignment: ", reverse_place_assignment)
        nx.relabel_nodes(reduced_graph, reverse_place_assignment, copy=False)
        print("Nodes networkx: ", reduced_graph.nodes())
        nx.draw(reduced_graph, with_labels=True)
        plt.savefig("./data/model_visualization/" + log_name + "_relabeled_networkx_graph.png")

        import json
        with open("./data/model_visualization/" + log_name + "_node_assignment.json", "w") as f:
            f.write(json.dumps(reverse_place_assignment, indent=2))


        import sys
        np.printoptions(threshold=sys.maxsize)
        #print(self.adjacency_matrix)
        #print(nx_graph)
        """

    def delete_more_transitions(self, nx_graph):
        place_assignment = {}
        self.transition_assignment = {}

        # Note that the assignment starts in 1 since the 0 is the padding value
        for i, place in enumerate(self.places, start=1):
            place_assignment[place] = i

        for i, transition in enumerate(self.transitions):
            self.transition_assignment[str(transition)] = i

        items = self.nodes.items()
        nodes = {str(key) : value for key, value in items}
        reverse_nodes = {value : str(key) for key, value in items}
        self.reverse_nodes = reverse_nodes
        self.nodes = place_assignment
        #print("Nodes: ", nodes)

        self.adjacency_matrix = np.zeros(shape=(len(self.places), len(self.places)))
        self.N = len(self.places)
        #print("Transition list: ", self.transitions)
        #print("Nodes: ", nodes)

        self.transition_to_index = {}
        for idx, obj in self.nx_graph[1].items():
            if hasattr(obj, 'label'):
                label = obj.label if obj.label is not None else obj.name
                self.transition_to_index[label] = idx

        for transition in self.transitions:
            # Get the id of the transition in the graph
            # transition_id = nodes[transition]

            transition_id = self.transition_to_index.get(transition)
            if transition_id is None:
                raise KeyError(f"Transition {transition} not found in transition_to_index mapping.")

            in_edges = self.nx_graph[0].in_edges(transition_id)
            out_edges = self.nx_graph[0].out_edges(transition_id)
            #print("Transition id: ", transition_id)
            #print("IN EDGES: ", in_edges)
            #print("OUT EDGES: ", out_edges)
            for in_edge in in_edges:
                for out_edge in out_edges:
                    in_place = reverse_nodes[in_edge[0]]
                    out_place = reverse_nodes[out_edge[1]]
                    i = place_assignment[in_place]
                    j = place_assignment[out_place]
                    self.adjacency_matrix[i, j] = 1

        import sys
        np.printoptions(threshold=sys.maxsize)
        #print(self.adjacency_matrix)
        #print(nx_graph)

    def detect_petri_net_constructs(self, net):
        # Matrix with rows = to n places and columns = n transitions
        choices = np.zeros(shape=(len(net.places), len(net.transitions)), dtype=np.float32)
        paralels = np.zeros(shape=(len(net.places), len(net.transitions)), dtype=np.float32)
        loops = np.zeros(shape=(len(net.places), len(net.transitions)), dtype=np.float32)
        for place in net.places:
            if len(place.out_arcs) > 1:
                for arc in place.out_arcs:
                    print("Found choice")
                    choices[self.nodes[str(place)]][self.transition_assignment[str(arc.target)] - 1] = 1

        # print("Choices: ", choices)

        for transition in net.transitions:
            if len(transition.out_arcs) > 1:
                places_in_paralel = []
                transitions_in_paralel = []
                for arc in transition.out_arcs: # This target is a place
                    for arc2 in arc.target.out_arcs: # This target is a transition
                        places_in_paralel.append(arc.target)
                        transitions_in_paralel.append(arc2.target)
                for place in places_in_paralel:
                    for transition in transitions_in_paralel:
                        paralels[self.nodes[str(place)]][self.transition_assignment[str(transition)] - 1] = 1

        # print("Paralels: ", paralels)

        detected_loops = list(nx.simple_cycles(self.nx_graph[0]))
        for loop in detected_loops:
            transitions_in_loop = []
            places_in_loop = []
            for node in loop:
                reverse_node = self.reverse_nodes[node]
                if str(reverse_node) in self.transition_assignment:
                    transitions_in_loop.append(reverse_node)
                else:
                    places_in_loop.append(reverse_node)
            for place in places_in_loop:
                for transition in transitions_in_loop:
                    loops[self.nodes[str(place)]][self.transition_assignment[str(transition)] - 1] = 1

        # print("Paralels: ", loops)

        #print("detected loops: ", detected_loops)
        return choices, paralels, loops



    def vectorize_batch(self, partial_trace_batch, next_events):
        # print('Vectorizing partial trace batch')
        # print(f"[DEBUG] TRACE: {partial_trace_batch}")
        # print(f"[DEBUG] PREFIX LENGTH: {len(partial_trace_batch)}")
        # print(f"[DEBUG] All activity_dict keys: {list(self.activity_dict.keys())}")

        y = []
        y_next_timestamp = []
        X = []
        X_attributes = []
        y_attributes = []
        for attr in list(self.unique_attributes.keys()):
            y_attributes.append([])

        self.DEBUG_activated_transitions = []
        self.DEBUG_activated_places = []
        self.DEBUG_event_attributes = []

        last_places_activated = []

        for idx, (trace, next_event) in enumerate(zip(partial_trace_batch, next_events)):
            # print(f"\n🔍 Trace #{idx}:")
            # print(f"    ➤ Events in trace: {[e['concept:name'] for e in trace]}")

            trace_attributes = []
            trace_events = []

            marking = self.initial_marking
            node_activations = {}
            for n in self.nodes:
                node_activations[n] = 0

            # TODO: limpiar la matriz para cada evento o no
            F_event = np.zeros(shape=(self.N, self.F - self.static_concatenable_features), dtype="float32")
            # Custom pad values
            #print("F_event: ", F_event[0])

            for place_raw in marking:
                place = str(place_raw)

                # print(place_raw, ": ", dir(place_raw))

                # print("Place: ", place)
                # Update activation since a token is in the place
                # node_activations[place] += 1
                # Create feature
                # resource_one_hot = np.eye(R)[unique_resources.index(event["org:resource"])].tolist()

                # Substract 1 in the matrix so as to get a correct reference on the node matrix (the id are 1..+ and the matrix is 1..+-1)
                F_event[self.nodes[place] - 1][0] = self.activity_dict["[EOC]"] + 1  # One hot of the activity
                F_event[self.nodes[place] - 1][1] = self.N + 1
                F_event[self.nodes[place] - 1][2] = 11
                F_event[self.nodes[place] - 1][3] = 11
                F_event[self.nodes[place] - 1][4] = len(self.transitions) + 1
                sorted_attribute_list = sorted(list(self.unique_attributes.keys()))
                for attr_pos, attr in enumerate(sorted_attribute_list, 0):
                    F_event[self.nodes[place] - 1][self.static_event_features + attr_pos] = \
                        len(self.unique_attributes[attr].keys()) + 1

            x_attr = [self.activity_dict["[EOC]"] + 1, 11, 11]
            sorted_attribute_list = sorted(list(self.unique_attributes.keys()))
            for attr_pos, attr in enumerate(sorted_attribute_list, 0):
                    x_attr.append(len(self.unique_attributes[attr].keys()) + 1)

            """
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            x_attr.append(0)
            """

            #trace_events.append(np.expand_dims(F_event.copy(), axis=0))
            #trace_attributes.append(x_attr)

            for curr_event, event in enumerate(trace, 1):
                #print("=====")
                #print("Event", event["concept:name"])
                # Generate an empty feature matrix for the current event
                curr_activity = event["concept:name"]

                # Init dictionary number of times a token has pased over a transition or a place.

                # Start replaying the prefix
                order = 0
                regular_transitions_found = 0
                time_since_prev, time_since_start, weekday, time_since_midnight = self.get_time_features(event, trace, curr_event-1)

                all_markings, marking, fired_transitions = self.replay_event(event, marking, node_activations)

                #print("All markings: ", all_markings)
                #print("=========")

                self.DEBUG_activated_places.append([str(p) for p in all_markings])
                self.DEBUG_activated_transitions.append([str(t) for t in fired_transitions])
                DEBUG_attr_list = []
                for attr_pos, attr in enumerate(list(self.unique_attributes.keys()), 0):
                    DEBUG_attr_list.append(str(event[attr]))
                self.DEBUG_event_attributes.append(DEBUG_attr_list)

                for transition_list, place_list in zip(fired_transitions, all_markings):
                    for place_raw in place_list:
                        place = str(place_raw)

                        #print(place_raw, ": ", dir(place_raw))

                        #print("Place: ", place)
                        # Update activation since a token is in the place
                        #node_activations[place] += 1
                        # Create feature
                        # resource_one_hot = np.eye(R)[unique_resources.index(event["org:resource"])].tolist()

                        # Substract 1 in the matrix so as to get a correct reference on the node matrix (the id are 1..+ and the matrix is 1..+-1)
                        F_event[self.nodes[place] - 1][0] = self.activity_dict[curr_activity]  # One hot of the activity
                        F_event[self.nodes[place] - 1][1] = self.nodes[place]  # One hot of the node
                        F_event[self.nodes[place] - 1][2] = self.quantize_timestamp(time_since_prev, bins=self.bins_between_events)
                        F_event[self.nodes[place] - 1][3] = self.quantize_timestamp(time_since_start, bins=self.bins_since_start)

                        # print(self.transitions, self.places, self.nodes)
                        # print(transition_list[0])
                        # Удаляем круглые скобки по краям
                        s = str(transition_list[0]).strip("()")
                        # Разбиваем по первой запятой
                        parts = s.split(",", 1)
                        if len(parts) < 2:
                            return s  # Если запятой нет – возвращаем всю строку
                        first = parts[0].strip()
                        second = parts[1].strip()
                        # Если вторая часть равна "None" (без учета регистра), возвращаем первую часть
                        if second.lower() == "none":
                            result = first
                        else:
                            # Если вторая часть заключена в апострофы или кавычки, удаляем их
                            if (second.startswith("'") and second.endswith("'")) or (
                                    second.startswith('"') and second.endswith('"')):
                                result = second[1:-1]
                            else:
                                result = second

                        F_event[self.nodes[place] - 1][4] = self.transition_assignment[result] + 1

                        #path_statistics_place = self.path_statistics[self.nodes[place]]
                        #for i, freq in enumerate(path_statistics_place):
                            #F_event[self.nodes[place]][7 + len(self.unique_activities) + i] = path_statistics_place[i]

                        # print("STATIC FEATURES: ", F_event[self.nodes[place] - 1][2: self.static_event_features])

                        sorted_attribute_list = sorted(list(self.unique_attributes.keys()))
                        for attr_pos, attr in enumerate(sorted_attribute_list, 0):
                            F_event[self.nodes[place] - 1][self.static_event_features + attr_pos] = 0 # self.unique_attributes[attr][str(event[attr])]

                        # Add the last places activated
                        last_place_activated = [self.nodes[str(place_iter)] - 1 for place_iter in place_list]
                        last_places_activated.append(last_place_activated)

                x_attr = []

                x_attr.append(self.activity_dict[curr_activity])
                x_attr.append(self.quantize_timestamp(time_since_prev, bins=self.bins_between_events))
                x_attr.append(self.quantize_timestamp(time_since_start, bins=self.bins_since_start))
                sorted_attribute_list = sorted(list(self.unique_attributes.keys()))
                for attr_pos, attr in enumerate(sorted_attribute_list, 0):
                       x_attr.append(0)# self.unique_attributes[attr][str(event[attr])])
                trace_attributes.append(x_attr)

                """
                x_attr.append((time_since_prev - self.normalization_between_events[0]) / self.normalization_between_events[1])
                x_attr.append((time_since_start - self.normalization_since_start[0]) / self.normalization_since_start[1])
                x_attr.append((time_since_prev - self.normalization_between_events[3]) / self.normalization_between_events[2])
                x_attr.append((time_since_start - self.normalization_since_start[3]) / self.normalization_since_start[2])
                x_attr.append(weekday)
                x_attr.append(np.sin(2 * np.pi * time_since_midnight))
                x_attr.append(np.cos(2 * np.pi * time_since_midnight))
                x_attr.append(np.sin(2 * np.pi * time_since_prev))
                x_attr.append(np.cos(2 * np.pi * time_since_prev))
                x_attr.append(np.sin(2 * np.pi * time_since_start))
                x_attr.append(np.cos(2 * np.pi * time_since_start))
                """

                """
                F_event = np.concatenate((
                    F_event[:, 0:self.static_event_features - 1],
                    self.loops,
                    F_event[:, self.static_event_features - 1:]),
                    axis=-1)
                """

                # Create the feature vector for the transition

                order += 1

                #print("Non zero F_event: ", np.count_nonzero(np.sum(F_event, axis=-1)))
                #print("=====")

                trace_events.append(np.expand_dims(F_event.copy(), axis=0))


            #print("-------")

            padded_trace_events = trace_events.copy()
            """
            for id_debug, X_debug in enumerate(padded_trace_events):
                print("ID DEBUG: ", id_debug)
                if id_debug == 0:
                    nodes_debug = np.nonzero(np.sum(padded_trace_events[id_debug], axis=-1))
                else:
                    nodes_debug = np.nonzero(np.sum(padded_trace_events[id_debug], axis=-1) - np.sum(padded_trace_events[id_debug - 1], axis=-1))
                print("NODES DEBUG: ", nodes_debug)
            """


            if self.max_len < len(trace_events):
                padded_trace_events = padded_trace_events[-self.max_len:]
                trace_attributes = trace_attributes[-self.max_len:]

            X.append(padded_trace_events)
            X_attributes.append(trace_attributes)

            #print("Activated transitions: ", self.DEBUG_activated_transitions)
            # print(next_event)
            # print(f"[DEBUG] NEXT_EVENT and its type: {next_event, type(next_event)}")

            if "concept:name" in next_event:
                # print(f"[DEBUG] next event['concept:name']: {next_event['concept:name']}")
                # print(f"[DEBUG] Full activity_dict: {self.activity_dict}")
                # print(f"[DEBUG] activity_dict.get = {self.activity_dict.get(next_event['concept:name'], '❌ Not found')}")
                y_act = self.activity_dict[next_event["concept:name"]]
                y.append(y_act - 1) # Substract
                # print(
                    # f"✅ [TRACE {idx}] Appended label: {y[-1]} ← class for event: {next_event['concept:name'] if isinstance(next_event, dict) else '[EOC]'}")

                for attr_pos, attr in enumerate(list(self.unique_attributes.keys()), 0):
                    attr_str = str(attr)
                    attr_id = self.unique_attributes[attr_str][str(next_event[attr_str])]
                    y_attributes[attr_pos].append(attr_id - 1)
            else:
                # In that case, add an artificial activity [END]
                # print(f"[DEBUG] next_event (must be EOC):{next_event}")
                y_next_timestamp.append(0.0)
                y_act = self.activity_dict["[EOC]"]
                y.append(y_act - 1)

                for attr_pos, attr in enumerate(list(self.unique_attributes.keys()), 0):
                    attr_str = str(attr)
                    attr_id = self.unique_attributes[attr_str][str(trace[-1][attr_str])]

                    y_attributes[attr_pos].append(attr_id - 1)

            # raise ValueError

        y_np = np.array(y)
        y_next_timestamp = np.array(y_next_timestamp)
        for i in range(len(y_attributes)):
            y_attributes[i] = np.array(y_attributes[i])
        y_next_timestamp = np.expand_dims(y_next_timestamp, axis=-1)
        #
        # print(f"\n📊 Итог vectorize_batch:")
        # print(f"    ➤ Кол-во примеров: {len(y_np)}")
        # print(f"    ➤ Уникальные метки: {np.unique(y_np)}")

        return X, y_np, self.adjacency_matrix, self.N, self.F, y_next_timestamp, y_attributes, last_places_activated, X_attributes

    def replay_event(self, event, marking, node_activations=None):
        # Calculate the marking for the partial trace
        n_regular_firings = 0
        fired_transitions = []
        activated_places = []
        all_markings = []
        places_shortest_path = get_places_shortest_path_by_hidden(self.net, 50)
        # This comprobation checks whether the event is contemplated in the model
        m_marking = copy(marking)
        if event["concept:name"] in self.activity_to_transition:
            transition_to_activate = self.activity_to_transition[event["concept:name"]]
            # If the transition we have to activate is not yet activated, try to activate every hidden transition
            # possible

            # TODO: añadir el marking anterior o no?
            #for m in m_marking:
                #print("Appending initial_marking: ", str(m))
                #all_markings.append(m)

            if not semantics.is_enabled(transition_to_activate, self.net, m_marking):
                _, _, act_trans, _ = apply_hidden_trans(transition_to_activate, self.net, copy(m_marking),
                                                        places_shortest_path, [], 0, set(), [copy(m_marking)])
                tmp_firing = []
                tmp_marking = []

                for act_tran in act_trans:
                    tmp_firing.append(act_tran)
                    for arc in act_tran.out_arcs:
                        activated_places.append(arc.target)
                    m_marking = semantics.execute(act_tran, self.net, m_marking)
                    for m in m_marking:
                        #print("Firing hidden transition: ", str(act_tran) + " for place ", str(m))
                        tmp_marking.append(m)

                all_markings.append(tmp_marking)
                fired_transitions.append(tmp_firing)

            if not semantics.is_enabled(transition_to_activate, self.net, m_marking):
                for arc in transition_to_activate.in_arcs:
                    if arc.source not in m_marking:
                        #print("M marking failed: ", arc.source)
                        m_marking[arc.source] += 1

            for arc in transition_to_activate.out_arcs:
                activated_places.append(arc.target)

            m_marking = semantics.execute(transition_to_activate, self.net, m_marking)
            tmp_firing = []
            tmp_marking = []
            for m in m_marking:
                #print("Firing transition ", str(transition_to_activate), " for place ", str(m))
                tmp_marking.append(m)
            tmp_firing.append(transition_to_activate)
            fired_transitions.append(tmp_firing)
            all_markings.append(tmp_marking)


            # Calculate the number of node activations
            for m in all_markings:
                for k in m:
                    if node_activations is not None:
                        node_activations[str(k)] += 1
        return all_markings, m_marking, fired_transitions
