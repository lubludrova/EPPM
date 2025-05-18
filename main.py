# src/main.py
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader

import pm4py

from src.model.MainModel import MainModel
from src.model.Trainer import Trainer
from src.model.Tester import Tester
from src.model.ggrnn import GGRNN

from src.mining.mining_factory import MiningFactory

from src.vectorizer.EmbeddingGraphVectorizer import EmbeddingGraphVectorizer
from src.vectorizer.EmbeddingGraphVectorizerPostMarkingNoTransitionsMultihotAdditionalFeatures import \
    EmbeddingGraphVectorizerPostMarkingNoTransitionsMultihotAdditionalFeatures
from src.vectorizer.EnhancedLastStateVectorizer import EnhancedLastStateVectorizer
from src.utils.utils import Utils
from src.generator.Generator import Generator
from src.generator.GeneratorMultioutput import GeneratorMultioutput
from src.generator.TorchGenerator import TorchGenerator
import pickle, gzip, json, datetime as dt

from pm4py.objects.log.importer.xes import importer as xes_importer
import os
import sys

# src/utils.py
import pm4py
import yaml
import os
import sys
import datetime


class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def custom_collate(batch):
    # batch — это список кортежей, возвращённых __getitem__
    Xs, ys, times, As = zip(*batch)

    # Предполагается, что все X имеют одинаковую форму: [max_len, N, F]
    Xs = torch.stack(Xs, dim=0)
    # ys — скаляры (или векторы размерности [1]); делаем их 1-D
    ys = torch.tensor(ys)
    # times — скаляры, переводим их в тензор
    times = torch.tensor(times)
    # As должно иметь одинаковую форму для всех примеров: [N, N]
    # Если матрица смежности одинакова для всех примеров, можно её не объединять,
    # но если возвращается отдельный экземпляр для каждого примера, то собираем:
    As = torch.stack(As, dim=0)

    return Xs, ys, times, As


def load_log_and_extract_params(log_path=None, log_name=None):
    # 1. Загрузка лога
    log = xes_importer.apply(log_path, parameters={"timestamp_sort": True})

    # 3. Загрузка атрибутов из YAML
    attribute_dict = {}
    attributes = []
    attributes_path = os.path.join(os.path.dirname(__file__), "attributes.yaml")
    with open(attributes_path, "r") as attr_f:
        attribute_file = yaml.safe_load(attr_f)
        log_name = log_name.split(".")[0]
        if log_name in attribute_file:
            attributes = attribute_file[log_name]

    # 4. Извлечение уникальных активностей
    # unique_activities = sorted(set(
    #     event["concept:name"] for trace in log for event in trace
    # ))
    # unique_activities = sorted(unique_activities)
    unique_activities = list(
        set([event["concept:name"] for trace in log for event in trace])
    )
    unique_activities = sorted(unique_activities)

    # 5. Создание словаря activity_dict
    activity_dict = {}
    for i, u in enumerate(unique_activities, start=1):
        activity_dict[u] = i
    activity_dict["[EOC]"] = len(unique_activities) + 1

    # print("ACTIVITY DICT: ", activity_dict)

    # 6. Максимальная длина трассы
    max_len = max(list(len(trace) for trace in log))

    # 7. Извлечение атрибутов (например, org:resource)
    attribute_count = {}
    for attribute in attributes:
        unique_attributes = list(
            set([str(event[attribute]) for trace in log for event in trace])
        )
        # This step is very important
        # The set structure has no order so when iterating it the id for the activities can be different in each execution
        unique_attributes = sorted(unique_attributes)
        # print("Unique list for ", attribute, " :", unique_attributes)
        attribute_dict[attribute] = {}
        for i, unique_attribute in enumerate(unique_attributes, start=1):
            attribute_dict[attribute][unique_attribute] = i

        attribute_count[attribute] = len(unique_attributes)

    return log, unique_activities, attribute_dict, max_len, activity_dict, attribute_count


def main():
    # Загрузка конфига

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    log_folder = "data/logs"
    log_name = config["dataset"]
    train_val_log = "train_val_" + log_name
    train_log = "train_" + log_name
    val_log = "val_" + log_name
    test_log = "test_" + log_name

    log_path = os.path.join(
        os.path.dirname(__file__),
        log_folder,
        log_name  # Пример: "Helpdesk.xes.gz"
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.stdout = DualLogger(f"/Users/ruslanageev/PycharmProjects/PPM/data/logger/{timestamp}_stdout.txt")
    print(f"▶️ Logging started at {timestamp}")

    log, unique_activities, unique_attributes, max_len, activity_dict, attribute_count = load_log_and_extract_params(
        log_path=log_path, log_name=log_name)
    # net, initial_marking, final_marking = preprocess(log_path=log_path)

    # t_v_l = xes_importer.apply(os.path.join(log_folder, train_val_log))

    mining_factory = MiningFactory(log, os.path.join(os.path.dirname(__file__), log_folder, train_val_log))
    net, initial_marking, final_marking = mining_factory.mine(config['mining_algorithm'])
    print("Model mined.")

    # print(f"___Статистика по датасету {log_name}:___")
    # print("Путь к главному логу: ", log_path)
    # print("Количество уникальных активитис: ", len(unique_activities))
    # print("Список всех активитис: ", *unique_activities)
    # print("Максимальная длина кейса", max_len)
    # print("Используемые аттрибуты: ", *unique_attributes.keys())

    log_folder = "../data/logs"
    # Создание векторайзера для получения доп фичей
    time_vectorizer = EnhancedLastStateVectorizer(
        net,
        initial_marking,
        final_marking,
        0,
        unique_activities,
        max_len,
        activity_dict,
    )
    t_v_log = time_vectorizer.load_log(os.path.join(log_folder, train_val_log))
    (
        bins_between_events,
        bins_since_start,
        normalization_between_events,
        normalization_since_start,
    ) = time_vectorizer.get_time_measures(t_v_log, 10, 10)

    # # Инициализация векторизатора
    vectorizer = (
        EmbeddingGraphVectorizerPostMarkingNoTransitionsMultihotAdditionalFeatures(
            net,
            initial_marking,
            final_marking,
            unique_attributes,
            unique_activities,
            max_len,
            activity_dict,
            bins_between_events,
            bins_since_start,
            normalization_between_events,
            normalization_since_start,
            is_inductive=False,
        )
    )

    all_y = []
    for trace in log:
        for idx in range(1, len(trace)):
            all_y.append(trace[idx]["concept:name"])
    from collections import Counter
    print("[DEBUG] Distribution of next events:", Counter(all_y))

    # vectorizer = EmbeddingGraphVectorizer(
    #     net=net,
    #     initial_marking=im,
    #     final_marking=fm,
    #     unique_attributes=unique_attributes,
    #     unique_activities=unique_activities,
    #     max_len=max_len,
    #     activity_dict=activity_dict,
    #     is_inductive=False
    # )

    train_file = os.path.join(log_folder, train_log)
    val_file = os.path.join(log_folder, val_log)
    test_file = os.path.join(log_folder, test_log)

    X_prefixes, y_next, y_suffix = vectorizer.get_prefixes_suffixes(
        os.path.join(log_folder, log_name)
    )
    X_train_prefixes, y_train_next, y_train_suffix = vectorizer.get_prefixes_suffixes(train_file)
    X_val_prefixes, y_val_next, y_val_suffix = vectorizer.get_prefixes_suffixes(val_file)
    X_test_prefixes, y_test_next, y_test_suffix = vectorizer.get_prefixes_suffixes(test_file)

    print("[DEBUG] Number of traces in train:", len(X_train_prefixes))
    print("[DEBUG] Number of traces in val:", len(X_val_prefixes))
    print("[DEBUG] First train trace:", [ev["concept:name"] for ev in X_train_prefixes[0]])
    print("[DEBUG] First val trace:", [ev["concept:name"] for ev in X_val_prefixes[0]])

    # print('X_train_prefixes: ', X_train_prefixes)
    # print('y_train_next', len(y_train_next))
    # print('X_val_prefixes: ', X_val_prefixes)
    # print('y_val_next', len(y_val_next))
    # print('X_test_prefixes: ', X_test_prefixes)
    # print('y_test_next', len(y_test_next))

    default_hyperparams = {
        "rnn_1": 256,
        "rnn_2": 256,
        "attribute_rnn": 256,
        "dropout_rnn_attribute": 0.3,
        "n_layer_attribute": 2,
        "embedding": 32,
        "lr": config['learning_rate'],
        "optimizer": torch.optim.Adam,
        "batch_size": config['batch_size'],
        "n_bucket_time_1": 10,
        "n_bucket_time_2": 10
    }

    # Векторизация данных (пример для одного батча)
    # X, y, adjacency_matrix, N, F, _ = vectorizer.vectorize_batch(log_file=f"data/{config['dataset']}",
    #                                                              partial_trace_batch=log[:10],
    #                                                              next_events=[trace[-1]["concept:name"] for trace in
    #                                                                           log[:10]])

    # Создание DataLoader
    # train_loader = create_dataloader(
    #     graph_data=X_prefixes,  # Убедитесь, что X соответствует формату Data из PyG
    #     sequences=X_prefixes,   # Замените на реальные последовательности
    #     labels=y_next,
    #     batch_size=default_hyperparams['batch_size']
    # )

    model_name = Utils.dict_to_str(default_hyperparams, "pyt_model")
    print(model_name)

    torch_generator = TorchGenerator(
        X_train_prefixes,
        y_train_next,
        vectorizer,
        vectorizer.adjacency_matrix,
        batch_size=default_hyperparams["batch_size"],
    )

    train_generator = torch.utils.data.DataLoader(
        dataset=torch_generator,
        batch_size=default_hyperparams["batch_size"],
        shuffle=False,
        collate_fn=torch_generator.collate_fn,
        num_workers=0,
    )
    val_generator = GeneratorMultioutput(
        X_val_prefixes,
        y_val_next,
        vectorizer,
        vectorizer.adjacency_matrix,
        batch_size=1,
    )
    test_generator = GeneratorMultioutput(
        X_test_prefixes,
        y_test_next,
        vectorizer,
        vectorizer.adjacency_matrix,
        batch_size=1,
    )

    out_dir = "data/xai_inputs"
    os.makedirs(out_dir, exist_ok=True)

    # --- Итерация по префиксам и сохранение ---
    for idx in range(len(torch_generator)):
        X, y, last_place, X_attr = torch_generator[idx]
        graph_A = torch.from_numpy(vectorizer.adjacency_matrix).float()
        place_act = torch.from_numpy(np.array(X)).float()  # shape (seq_len, N, F)
        attr_seq = torch.from_numpy(np.array(X_attr)).float()  # shape (seq_len, num_attrs)
        y_next = int(y)

        dump = {
            "graph_A": graph_A,
            "place_act": place_act,
            "attr_seq": attr_seq,
            "y_next": y_next,
        }
        fname = os.path.join(out_dir, f"prefix_{idx:05d}.pkl")
        with open(fname, "wb") as f:
            pickle.dump(dump, f)

    print(f"Сериализовано {len(torch_generator)} префиксов в {out_dir}")

    # Обучение

    # Инициализация модели
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # model = GGRNN(config['model'], num_classes=len(activity_dict)).to(device)
    model = MainModel(
        default_hyperparams,
        vectorizer.N,
        vectorizer.F,
        unique_activities,
        log_name,
        max_len,
        vectorizer,
        attribute_count,
        len(vectorizer.transitions) + 1,
        len(vectorizer.places) + 1,
        10,
        10,
        config['arcitechture'],
        is_search=False,
    )
    print("Model")
    print(model)
    print(
        "Number of parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad), )
    torch.save(model, "./data/torch_models/" + log_name
        + "_full_model.pt")

    #
    # ckpt = {
    #     "timestamp": str(dt.datetime.utcnow()),
    #     "graph_state": model.state_dict(),  # всё, кроме головы
    #     "head_state": model.learning_model.head.state_dict(),
    #     "hyperparams": default_hyperparams,  # словарь
    #     "vectorizer": vectorizer,  # объект с .N .F …
    #     "unique_acts": unique_activities,
    #     "attr_count": attribute_count,
    #     "max_len": max_len,
    #     "meta": {
    #         "log_name": log_name,
    #         "buckets_1": default_hyperparams['n_bucket_time_1'],
    #         "buckets_2": default_hyperparams['n_bucket_time_2'],
    #         "architecture": config["arcitechture"]
    #     }
    # }
    #
    # with gzip.open(f"./data/checkpoints/taco_bundle_best.pkl.gz", "wb") as f:
    #     pickle.dump(ckpt, f)

    trainer = Trainer(
        train_generator,
        val_generator,
        model,
        unique_attributes,
        log_name,
        model_name,
        len(X_train_prefixes),
        config['arcitechture'],
        test_generator=test_generator,
        optimization_mode="max",
        batch_size=default_hyperparams["batch_size"],
    )
    trainer.train()
    loaded_statedict = torch.load(
        "./data/torch_models/"
        + log_name
        + "_"
        + config['arcitechture']
        + "/"
        + model_name
    )
    model.load_state_dict(loaded_statedict["ggrnn"], strict=False)
    model.learning_model.head.load_state_dict(loaded_statedict["head"])

    # keys = model.load_state_dict(loaded_statedict)
    model.eval()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # criterion = torch.nn.CrossEntropyLoss()
    with open(
            "./data/results/" + log_name + "/" + config['arcitechture'] + "_results.txt", "w"
    ) as result_file:
        tester = Tester(vectorizer, test_generator, attribute_count)
        accuracy = tester.test_next_activity_timestamp(
            model,
            os.path.join("./data/logs", train_val_log),
            X_test_prefixes,
            y_test_suffix,
            activity_dict,
            config['arcitechture'],
            next_act=y_test_next,
            log_name=log_name,
        )
        result_file.write("Accuracy: " + str(accuracy) + "\n")






if __name__ == "__main__":
    main()
