import pandas as pd
import numpy as np

class GenerateTrace:
    def __init__(self, eventlog):
        self._eventlog = eventlog

    def import_log(self,name):

        dataframe = pd.read_csv('fold/'+name+'.csv', sep=',')
        return dataframe

    def generate_split(self, log):
        grouped = log.groupby("case:concept:name")
        start_timestamps = grouped["time:timestamp"].min().reset_index()
        start_timestamps = start_timestamps.sort_values("time:timestamp", ascending=True, kind="mergesort")
        train_ids = list(start_timestamps["case:concept:name"])[:int(0.66 * len(start_timestamps))]
        train = log[log["case:concept:name"].isin(train_ids)].sort_values("time:timestamp", ascending=True,kind='mergesort')
        test = log[~log["case:concept:name"].isin(train_ids)].sort_values("time:timestamp", ascending=True,kind='mergesort')
        return train, test

    def generate_prefix_trace(self, log, view):
        act = log.groupby('case:concept:name', sort=False).agg({view: lambda x: list(x)})
        return act

    def get_act(self):
        return self.__act

    def get_sequence(self, sequence):
        i = 0
        list_seq = []
        list_label = []
        while i < len(sequence):
            list_temp = []
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                list_seq.append(list_temp.copy())
                list_label.append(sequence.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_seq, list_label

    def get_sequence_num(self, sequence):
        i = 0
        list_seq = []
        while i < len(sequence):
            list_temp = []
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                list_seq.append(list_temp.copy())
                j = j + 1
            i = i + 1
        return list_seq

    @staticmethod
    def dataset_summary(log):
        print("Activity Distribution\n", log['activity'].value_counts())
        n_caseid = log['case'].nunique()
        n_activity = log['activity'].nunique()
        print("Number of CaseID", n_caseid)
        print("Number of Unique Activities", n_activity)
        print("Number of Activities", log['activity'].count())
        cont_trace = log['case'].value_counts(dropna=False)
        max_trace = max(cont_trace)
        mean = np.mean(cont_trace)
        print("Max lenght trace", max_trace)
        print("Mean lenght trace", np.mean(cont_trace))
        print("Min lenght trace", min(cont_trace))
        return max_trace,  int(round(mean)), n_caseid, n_activity
