import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
# from similarity.normalized_levenshtein import NormalizedLevenshtein
import Levenshtein as NormalizedLevenshtein
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# from model.Interpretability.DebuggerEvent import DebuggerEvent
# from model.Interpretability.DebuggerTrace import DebuggerTrace
# from model.Interpretability.interpretability import Interpretability
# from model.sampler.Samplers import ArgmaxSampler
# import wandb


class Tester:
    def __init__(self, vectorizer, generator, attributes):
        self.vectorizer = vectorizer
        self.generator = generator
        self.attributes = attributes

    def sample(self, probas):
        return np.argmax(probas, axis=-1)

    def test_next_activity_timestamp(self, model, train_val_log_file, test_prefixes, test_suffixes, activity_dict, architecture,
                                     next_act, log_name):
        model.eval()
        from sklearn.metrics import accuracy_score, mean_absolute_error
        activity_reverse_dict = {value: key for key, value in activity_dict.items()}
        y_pred = []
        y_pred_mae = []
        y_true_mae = []
        y_true = []
        id_trace = 0

        reverse_activity_dict = {value: key for key, value in activity_dict.items()}

        prefix_length_acc = {}
        with torch.no_grad():
            for prefix, suffix, n_act in tqdm(zip(test_prefixes, test_suffixes, next_act), total=len(test_prefixes)):
                id_trace += 1
                try:
                    next_act = n_act
                    next_act_name = n_act["concept:name"]
                except:
                    next_act = n_act
                    next_act_name = n_act

                # print("Prefix: ", prefix)
                # print("Next act: ", next_act)
                # print("X: ", X.shape, ". Adj: ", adj.shape)
                # print("Preds: ", predictions[0])

                X, y_np, adj, _, _, y_next_timestamp, y_attributes, last_places_activated, X_attributes = self.vectorizer.vectorize_batch(
                    [prefix], [next_act])
                adj = self.generator.localpooling_filter(adj, symmetric=False)
                adj = np.expand_dims(adj, axis=0)
                #print("X: ", X)
                
                if len(prefix) not in prefix_length_acc:
                    prefix_length_acc[len(prefix)] = {
                        "y_true": [],
                        "y_pred": []
                    }

                X_np = []
                for item in X:
                    # For some reason the rnn requires padding with zeros the input.
                    # Pad with the maximum length so as to be sure the trace is in the same
                    # format as shown in training.
                    #for i in range(self.vectorizer.max_len - len(item)):
                    #    item.insert(0, np.zeros(shape=(1, self.vectorizer.N, self.vectorizer.F), dtype="float32"))
                    c = np.concatenate(item, axis=0)
                    X_np.append(np.expand_dims(c, axis=0))

                X_np = np.concatenate(X_np, axis=0)

                # print("X_NP: ", json.dumps(X_np.tolist(), indent=2))
                # print("X test shape post: ", X_np.shape)
                # print("Len: ", X_np.shape[1])
                predictions = model(X_np, adj, X_np.shape[1], last_places_activated, np.array(X_attributes))
                predicted_next_activity = np.argmax(predictions[0].cpu().numpy(), axis=-1)

                trace_array = [event["concept:name"] for event in prefix]
                # print("IG prefix: ", trace_array)
                """
                Interpretability().ig(
                    model, X_np, adj, self.vectorizer.max_len, self.vectorizer.N,
                    self.vectorizer.F, activity_dict[next_act_name] - 1,
                    None, trace_array, id_trace, log_name, self.vectorizer,
                    (next_act_name, reverse_activity_dict[predicted_next_activity[0] + 1])
                )
                """
                # DebuggerEvent(
                #     (next_act_name, reverse_activity_dict[predicted_next_activity[0] + 1]),
                #     self.vectorizer,
                #     trace_array,
                #     log_name + "_" + architecture, id_trace
                # ).print_debug()

                """
                predicted_next_timestamp = (predictions[1][0][0].cpu().numpy() * self.vectorizer.std_time_events) + self.vectorizer.mean_time_events
                y_pred_mae.append(predicted_next_timestamp)
                try:
                    real_next_tm = suffix[0]["time:timestamp"]
                    real_prev_tm = prefix[-1]["time:timestamp"]
                    y_true_mae.append((real_next_tm - real_prev_tm).total_seconds())

                except:
                    y_true_mae.append(0.0)
                """


                #print("Prediction: ", predicted_next_activity[0])
                #print("True: ", activity_dict[next_act_name] - 1)

                # y_pred.append(predicted_next_activity)
                y_pred.append(predicted_next_activity[0])
                y_true.append(activity_dict[next_act_name] - 1)
                
                prefix_length_acc[len(prefix)]["y_true"].append(predicted_next_activity[0])
                prefix_length_acc[len(prefix)]["y_pred"].append(activity_dict[next_act_name] - 1)

        accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
        print("Next activity accuracy: ", accuracy)
        
        accuracies_prefix = []
        n_prefixes = []
        for prefix_length in prefix_length_acc.keys():
            accuracy_prefix = accuracy_score(
                prefix_length_acc[prefix_length]["y_true"],
                prefix_length_acc[prefix_length]["y_pred"]
            )
            accuracies_prefix.append(accuracy_prefix)
            n_prefixes.append(len(prefix_length_acc[prefix_length]["y_true"]))
            print("Length: ", prefix_length, " acc: ", accuracy_prefix)

        x_values = prefix_length_acc.keys()

        accuracies_data_plot = [[x, y] for (x, y) in zip(x_values, accuracies_prefix)]
        n_prefixes_data_plot = [[x, y] for (x, y) in zip(x_values, n_prefixes)]

        labels_sorted = [l for l in sorted(set(y_true) | set(y_pred)) if l in activity_reverse_dict]

        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        cm_df = pd.DataFrame(
            cm,
            index=[f"T:{activity_reverse_dict[l]}" for l in labels_sorted],
            columns=[f"P:{activity_reverse_dict[l]}" for l in labels_sorted],
        )

        print("\n🧩 Confusion Matrix:")
        print(cm_df.to_string())

        # table_1 = wandb.Table(data=accuracies_data_plot, columns=["prefix_length", "accuracy"])
        # wandb.log({"accuracies_against_prefix_length" : wandb.plot.line(table_1, "prefix_length" , "accuracy", title="Test accuracy for each prefix length")})

        # table_2 = wandb.Table(data=n_prefixes_data_plot, columns=["prefix_length", "n_prefixes"])
        # wandb.log({"n_prefixes_for_prefix_length" : wandb.plot.line(table_2, "prefix_length" , "n_prefixes", title="Test number of prefixes for each prefix length")})

        # trace_debugger = DebuggerTrace(y_true, y_pred, reverse_activity_dict, test_prefixes, log_name + "_" + architecture)
        # trace_debugger.print_debug()
        return accuracy
        #mae_seconds = mean_absolute_error(y_pred=y_pred_mae, y_true=y_true_mae)
        #mae_days = mean_absolute_error(y_pred=y_pred_mae, y_true=y_true_mae) / 86400
        #print("Acc: ", accuracy, " MAE: ", mae_days)
        #return accuracy, mae_seconds, mae_days




    def predict_suffix(self, vectorizer, model, attribute_reverse_dict, activity_reverse_dict, attributes, real_suffix, prefix, max_trace_log_length, sampler, file=None):
        curr_prefix = prefix.copy()
        predicted_suffix = []
        with torch.no_grad():
            while True:
                # Use [EOC] as a dummy value
             
                X, _, adj, _, _, _, _, last_places_activated, X_attributes = vectorizer.vectorize_batch([curr_prefix], ["[EOC]"])
                adj = self.generator.localpooling_filter(adj, symmetric=False)
                adj = np.expand_dims(adj, axis=0)

                
                
                X_np = []
                for item in X:
                    for i in range(self.vectorizer.max_len - len(item)):
                        item.append(np.zeros(shape=(1, self.vectorizer.N, self.vectorizer.F), dtype="float32"))
                    c = np.concatenate(item, axis=0)
                    X_np.append(np.expand_dims(c, axis=0))

                X_np = np.concatenate(X_np, axis=0)

                predictions = model(X_np, adj, X_np.shape[1],last_places_activated, np.array(X_attributes))

                predicted_next_activity = sampler.sample(predictions[0][0].cpu().numpy())
                predicted_next_timestamp = (predictions[1][0][0].cpu().numpy() * self.vectorizer.std_time_events) + self.vectorizer.mean_time_events

                predicted_event = {}
                for attribute, attribute_pred in zip(attributes, predictions[2:]):
                    # print("Attribute pred: ", attribute_pred, " attr ", attribute)
                    predicted_attribute = self.sample(attribute_pred[0].cpu().numpy())
                    predicted_event[attribute] = attribute_reverse_dict[attribute][predicted_attribute + 1]
                predicted_event["concept:name"] = activity_reverse_dict[predicted_next_activity + 1]
                predicted_event["time:timestamp"] = curr_prefix[-1]["time:timestamp"] + datetime.timedelta(seconds = predicted_next_timestamp)

                curr_prefix.append(predicted_event)
                predicted_suffix.append(predicted_event["concept:name"])

                # Set a hard stop condition on the prediction
                if predicted_event["concept:name"] == "[EOC]" or (len(prefix) + len(predicted_suffix)) > max_trace_log_length + 1:
                    break

            return predicted_suffix, real_suffix

    def test_suffix(self, model, train_val_log_file, test_prefixes, test_suffixes, activity_dict, max_trace_log_length, sampler):
        model.eval()
    
        levenshtein = NormalizedLevenshtein()
        levenshtein_distances = []
        A_IN_ASCII = 65
        activity_reverse_dict = {value : key for key, value in activity_dict.items()}
        attribute_reverse_dict = {}
        for attr in self.vectorizer.unique_attributes.keys():
            attribute_reverse_dict[attr] = {value : key for key, value in self.vectorizer.unique_attributes[attr].items()}

        real_suffixes = []
        predicted_suffixes = []

        for prefix, suffix in tqdm(zip(test_prefixes, test_suffixes), total=len(test_prefixes)):
            predicted_suffix, real_suffix = self.predict_suffix(self.vectorizer, model, attribute_reverse_dict, activity_reverse_dict, self.attributes, suffix, prefix, max_trace_log_length, sampler)
            predicted_suffixes.append(predicted_suffix)
            real_suffixes.append(real_suffix)

        for real_suffix, predicted_suffix in zip(real_suffixes, predicted_suffixes):
            converted_real_suffix = ""
            converted_predicted_suffix = ""

            for event in real_suffix:
                converted_real_suffix += chr(activity_dict[event["concept:name"]] + A_IN_ASCII)
            converted_real_suffix += chr(activity_dict["[EOC]"] + A_IN_ASCII)

            for event in predicted_suffix:
                converted_predicted_suffix += chr(activity_dict[event] + A_IN_ASCII)

            levenshtein_distances.append(levenshtein.similarity(converted_real_suffix, converted_predicted_suffix))

        return np.mean(levenshtein_distances)

