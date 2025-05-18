import torch, pickle, gzip
from src.model.MainModel import MainModel

def load_taco_bundle(path, device="cpu"):
    with gzip.open(path, "rb") as f:
        bundle = pickle.load(f)

    model = MainModel(
        bundle["hyperparams"],
        bundle["vectorizer"].N,
        bundle["vectorizer"].F,
        bundle["unique_acts"],
        bundle["meta"]["log_name"],
        bundle["max_len"],
        bundle["vectorizer"],
        bundle["attr_count"],
        len(bundle["vectorizer"].transitions)+1,
        len(bundle["vectorizer"].places)+1,
        bundle["meta"]["buckets_1"],
        bundle["meta"]["buckets_2"],
        bundle["meta"]["architecture"],
        is_search=False).to(device)

    model.load_state_dict(bundle["graph_state"], strict=False)
    model.learning_model.head.load_state_dict(bundle["head_state"])
    model.eval()
    return model, bundle
