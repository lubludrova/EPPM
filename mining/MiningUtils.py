class MiningUtils:
    @staticmethod
    def calculate_metrics(log, net, initial_marking, final_marking, save_file=None, model_name=""):
        # from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
        from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
        # from pm4py.evaluation.precision import evaluator as precision_evaluator
        from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

        fitness_replay = replay_fitness_evaluator.apply(log, net, initial_marking, final_marking, variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        precision_replay = precision_evaluator.apply(log, net, initial_marking, final_marking, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)

        if save_file is None:
            print(model_name + " Mined model fitness: " + fitness_replay)
            print(model_name + " Precision model mined: " + precision_replay)
        else:
            with open(save_file, "w") as f_save:
                f_save.write(model_name + " Mined model fitness: " + str(fitness_replay) + "\n")
                f_save.write(model_name + " Precision model mined: " + str(precision_replay) + "\n")





