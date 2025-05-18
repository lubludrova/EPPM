import os
from pm4py.objects.petri_net.importer import importer as pnml_importer

class Utils:
    @staticmethod
    def setup_folders():
        if not os.path.exists("data/results"):
            Utils._make_dir_if_not_exists("data")
            Utils._make_dir_if_not_exists("data/logs")
            Utils._make_dir_if_not_exists("data/models")
            Utils._make_dir_if_not_exists("data/checkpoints")
            Utils._make_dir_if_not_exists("data/split_miner_best_models")
            Utils._make_dir_if_not_exists("data/split_miner_models")

    @staticmethod
    def _make_dir_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def import_process_model(model_folder, train_val_name):
        if os.path.exists(model_folder):
            import re

            regex = train_val_name + "_\\d.\\d_\\d.\\d.pnml"
            list_files = os.listdir(model_folder)
            pattern = re.compile(regex)
            print("Pattern: ", pattern)
            for file in list_files:
                if re.match(regex, file):
                    print("Found")
                    net, initial_marking, final_marking = pnml_importer.apply(os.path.join(model_folder, file))
                    return net, initial_marking, final_marking
        else:
            return None

    @staticmethod
    def import_base_process_model(model_folder, train_val_name):
        if os.path.exists("./data/models"):
            net, initial_marking, final_marking = pnml_importer(os.path.join(model_folder, "inductive_" + train_val_name + ".pnml"))
            return net, initial_marking, final_marking
        else:
            return None

    @staticmethod
    def dict_to_str(dict, base_str):
        concat_string = base_str
        for key in dict.keys():
            if key != "wandb":
                concat_string += "_" + key + "_" + str(dict[key]).replace(".","_")

        return concat_string


