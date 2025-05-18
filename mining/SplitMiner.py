import json
# from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
import gzip
import tqdm
from shutil import copyfile
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
# from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator


from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
# from pm4py.evaluation.replay_fitness import evaluator as replay_factory
import os
import re
from pathlib import Path

# from pm4py.objects.log.log import EventLog

from src.utils.utils import Utils


class SplitMiner:
    def __init__(self, log, output_folder, output_best_model):
        self.log = log
        self.output_folder = os.path.join(output_folder, Path(log).name)
        self.output_best_model = os.path.join(output_best_model, Path(log).name)

    def mine(self, store_results=None):
        if not os.path.exists(self.output_folder):
            Path(self.output_folder).mkdir(parents=True)
        if not os.path.exists(self.output_best_model):
            Path(self.output_best_model).mkdir(parents=True)

        if self.find_models_in_folder(self.output_best_model):
            print("Using cached model.")
            return Utils.import_process_model(self.output_best_model, Path(self.log).name)
        else:
            self.strip_log()
            os.system("java -jar ../lib/splitminer_cmd-1.0.0-all.jar -l " + self.log + " -b " + self.output_best_model + " -m " + self.output_folder + " -t " + "4")

            input_models = self.find_models_in_folder(self.output_folder)
            model_fitnesses, model_full_results = self.run_fitness_calculation(input_models)

            if store_results is not None:
                store_file = os.path.join(store_results, Path(self.log).name + "_split_miner_full_results.txt")
                Path(store_results).mkdir(parents=True, exist_ok=True)
                with open(store_file, "w") as safe_f:
                    safe_f.write(json.dumps(model_full_results, indent=2, separators=(",", ":")))

            return self.get_best_model(model_fitnesses)

    def find_models_in_folder(self, folder):
        log_file = Path(self.log).name
        model_regex = log_file + "_\d\.\d_\d\.\d\.pnml"
        files_to_process = []
        for file in os.listdir(folder):
            if re.match(model_regex, file):
                files_to_process.append(file)

        return files_to_process

    def strip_log(self):
        # Delete unnecesary fields of the log to avoid nans.
        imported_log = xes_importer.apply(self.log)
        df = log_converter.apply(imported_log, variant=log_converter.Variants.TO_DATA_FRAME)
        df = df[["case:concept:name", "concept:name", "time:timestamp"]]
        imported_striped_log = log_converter.apply(df)
        # For some reason compressing here does not work
        base_log_name = os.path.join("/tmp/", Path(self.log).name.replace(".gz", ""))
        xes_exporter.apply(imported_striped_log, base_log_name)
        with open(base_log_name, "rb") as f, gzip.open(base_log_name + ".gz", "wb") as f_out:
            f_out.writelines(f)
        self.log = os.path.join("/tmp/", Path(self.log).name)

    def process_file(self, file):
        # Import petri net and calculate fitness
        # This function throws a warning and it is unavoidable
        #print("Running: ", file)
        net, initial_marking, final_marking = pnml_importer.apply(os.path.join(self.output_folder, file))
        log = xes_importer.apply(self.log)
        fitness = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        precision_replay = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        try:
            fitness = fitness["averageFitness"]
        except:
            # Depending on the version is one or another
            fitness = fitness["average_trace_fitness"]
        return fitness, file, precision_replay

    def run_fitness_calculation(self, files_to_process):
        model_fitnesses = {}
        model_full_results = {}
        import multiprocessing as mp
        print("Calculating fitnesses. Please wait.")
        print("CPU count: ", os.cpu_count())
        if 8 * 4 <= os.cpu_count():
            max_workers = 8 * 6
        else:
            max_workers = 8
        pool = mp.Pool(processes=max_workers)
        for fitness, file, precision in tqdm.tqdm(pool.imap_unordered(self.process_file, files_to_process),
                                       total=len(files_to_process)):
            model_fitnesses[file] = fitness
            model_full_results[file] = {"fitness" : fitness, "precision" : precision}
        """
        for file_to_process in files_to_process:
            fitness, file, precision = self.process_file(file_to_process)
            model_fitnesses[file] = fitness
            model_full_results[file] = {"fitness" : fitness, "precision" : precision}
        """

        return model_fitnesses, model_full_results

    def get_best_model(self, model_fitnesses):
        sorted_fitnesses = {k: v for k, v in sorted(model_fitnesses.items(), key=lambda item: item[1], reverse=True)}
        best_model = next(iter(sorted_fitnesses))
        print("Sorted fitnesses: ", sorted_fitnesses)
        print("Best model: ", best_model)
        copyfile(
            os.path.join(self.output_folder, best_model),
            os.path.join(self.output_best_model, best_model)
        )
        net, initial_marking, final_marking = pnml_importer.apply(os.path.join(self.output_best_model, best_model))
        return net, initial_marking, final_marking
