# import pm4pycvxopt  # Fast alignments
import networkx as nx
import numpy as np
import time

from pm4py.objects.petri_net import semantics
from sklearn.metrics import accuracy_score
import tqdm, sys
from src.vectorizer.AbstractVectorizer import AbstractVectorizer


class EnhancedLastStateVectorizer(AbstractVectorizer):

    def __init__(self, net, initial_marking, final_marking, n_resources, unique_activities, max_len, activity_dict, avg_time_events=None, avg_time_start=None):

        super().__init__(net, initial_marking, final_marking, n_resources, unique_activities, max_len, activity_dict)
        self.avg_time_events = avg_time_events
        self.avg_time_start = avg_time_start

    def vectorize_batch(self, log_file, partial_trace_batch, next_events):

        replay_result, nx_graph, log = self.replay_log(log_file)

        # print("Replay result: ", replay_result)
        """
        fitness = replay_factory.apply(log, net, initial_marking, final_marking)
        try:
            fitness = fitness["averageFitness"]
        except:
            # Depending on the version is one or another
            fitness = fitness["average_trace_fitness"]

        print("AVG FITNESS: ", fitness)
        fitnesses = []
        for r in replay_result:
            fitnesses.append(r["trace_fitness"])
        print("Avg fitness: ", np.mean(fitnesses))
        """

        # alignments = align_factory.apply_log(log, net, initial_marking, final_marking)

        # for replay, alignment in zip(replay_result, alignments):
        #  print("Replay: ", replay, " | Alignment: ", alignment)

        adjacency_matrix, N, nodes, places, transitions, arcs = self.get_features_from_graph(nx_graph)
        if self.avg_time_events is not None:
            avg_time_events = self.avg_time_events
            avg_time_start = self.avg_time_start
        else:
            avg_time_events, avg_time_start = self.get_time_measures(log)


        print(nodes)
        print(N)
        print("Places: ", places)
        print("Transitions: ", self.net.transitions)
        print("Arcs: ", arcs)

        event_list = [event for trace in log for event in trace]

        # Set up the matrix dimensions of the features
        L = len(self.unique_activities)
        types_of_nodes = 8
        extra_features = 6
        F = types_of_nodes + N + extra_features + self.R + L # 6 for the different types of transitions. 2 for the extra features

        y = []
        X = []

        for idx, trace in enumerate(log):
            replay_trace = replay_result[idx]
            last_event = 0
            for curr_event, event in enumerate(trace, 1):
                # Generate an empty feature matrix for the current event
                F_event = np.zeros(shape=(N, F), dtype="float32")
                curr_activity = event["concept:name"]
                marking = self.initial_marking

                # Init dictionary number of times a token has pased over a transition or a place.
                node_activations = {}
                for n in nodes:
                    node_activations[n] = 0

                # Start replaying the prefix
                order = 0
                regular_transitions_found = 0
                time_since_prev, time_since_start, weekday, time_since_midnight = self.get_time_features(event, trace, curr_event-1, avg_time_events, avg_time_start)
                for transition in replay_trace["activated_transitions"]:
                    for place in marking:
                        # Update activation since a token is in the place
                        node_activations[place] += 1
                        # Create feature
                        # resource_one_hot = np.eye(R)[unique_resources.index(event["org:resource"])].tolist()

                        F_event[nodes[place]][0] = 1.0  # Node type
                        F_event[nodes[place]][types_of_nodes + self.R + self.activity_dict[curr_activity] + 1 - 1] = 1.0  # One hot of the activity
                        F_event[nodes[place]][types_of_nodes + self.R + L + nodes[place] + 1 - 1] = 1.0  # One hot of the node
                        F_event[nodes[place]][types_of_nodes + self.R + L + N + 1 - 1] = float(node_activations[place]) / self.max_len  # how many activations
                        F_event[nodes[place]][types_of_nodes + self.R + L + N + 2 - 1] = time_since_prev
                        F_event[nodes[place]][types_of_nodes + self.R + L + N + 3 - 1] = time_since_start
                        F_event[nodes[place]][types_of_nodes + self.R + L + N + 4 - 1] = weekday
                        F_event[nodes[place]][types_of_nodes + self.R + L + N + 5 - 1] = time_since_midnight
                        F_event[nodes[place]][types_of_nodes + self.R + L + N + 6 - 1] = float(order) / self.max_len   # ordering

                    if semantics.is_enabled(transition, self.net, marking):
                        marking = semantics.execute(transition, self.net, marking)
                    else:
                        # If the trace is misaligned, fire the transition anyway
                        # filling whatever place is necesary for that
                        marking = semantics.weak_execute(transition, marking)

                    order += 1

                    # Create the feature vector for the transition

                    if transition._Transition__label is None:
                        if "tauSplit" in transition._Transition__name:
                            F_event[nodes[transition]][2] = 1.0
                        elif "tauJoin" in transition._Transition__name:
                            F_event[nodes[transition]][3] = 1.0
                        elif "init_loop" in transition._Transition__name:
                            F_event[nodes[transition]][4] = 1.0
                        elif "loop" in transition._Transition__name:
                            F_event[nodes[transition]][5] = 1.0
                        elif "skip" in transition._Transition__name:
                            F_event[nodes[transition]][6] = 1.0
                        elif "tau" in transition._Transition__name:
                            F_event[nodes[transition]][7] = 1.0
                        else:
                            print("Unrecognized transition type: ", transition._Transition__name)
                            sys.exit(1)
                    else:
                        F_event[nodes[transition]][1] = 1.0
                        regular_transitions_found += 1

                    node_activations[transition] += 1

                    F_event[nodes[transition]][types_of_nodes + self.R + self.activity_dict[curr_activity] + 1 - 1] = 1.0 # One hot of the activity
                    F_event[nodes[transition]][ types_of_nodes + self.R + L + nodes[transition] + 1 - 1] = 1.0  # One hot of the node
                    F_event[nodes[transition]][types_of_nodes + self.R + L + N + 1 - 1] = float(node_activations[transition]) / self.max_len
                    F_event[nodes[transition]][types_of_nodes + self.R + L + N + 2 - 1] = time_since_prev
                    F_event[nodes[transition]][types_of_nodes + self.R + L + N + 3 - 1] = time_since_start
                    F_event[nodes[transition]][types_of_nodes + self.R + L + N + 4 - 1] = weekday
                    F_event[nodes[transition]][types_of_nodes + self.R + L + N + 5 - 1] = time_since_midnight
                    F_event[nodes[transition]][types_of_nodes + self.R + L + N + 6 - 1] = float(order) / self.max_len

                    # Check whether we finished processing the prefix
                    if curr_event == regular_transitions_found and transition._Transition__label == curr_activity:
                        break

                X.append(np.expand_dims(F_event, axis=0))
                # Dirty check to find whether we replayed the entire trace
                try:
                    y.append(self.activity_dict[trace[curr_event]["concept:name"]])
                except IndexError:
                    # In that case, add an artificial activity [END]
                    y.append(self.activity_dict["[EOC]"])

            # raise ValueError

        X_np = np.concatenate(X, axis=0)
        # print(X_np.shape)
        y_np = np.array(y)
        y_np = np.expand_dims(y_np, axis=-1)
        # print(y_np.shape)

        return X_np, y_np, adjacency_matrix, N, F
