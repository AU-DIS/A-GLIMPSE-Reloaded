import os
import time
from os import name, path
import experiment
import glimpseonline as g
import experiment as experiment
import sys
sys.path.append('..')


# IngredientsZZZZ:
# Gem i bestemte intervaller, navngiv banditterne det rigtige
# Gem grafen (TJEK)
# Gem experiment med prectictable navn
# Override experiments navn'
# Input tidsintervaller
# Sæt en long boi exp3m i gang


def run_on_graph(graph, experiment_name, delta, batch_size=1000, rf="kg", k_proportion=0.01):
    timed_bandit_dir = f"timed_bandits_{experiment_name}_{rf}_{delta}"

    exp = experiment.Experiment(comment="Timed trained bandit",
                                graph=graph, dir=timed_bandit_dir, name=f"Timed_bandit_{experiment_name}", batch_size=batch_size)

    regret_list_of_properties = ["round", "regret"]
    annotation = "regret"

    regret_id = exp.create_experiment(
        regret_list_of_properties, annotation, f"Timed experiment with delta {delta} and k = {k_proportion} and rewrd function {rf}")

    k = int(exp.kg().number_of_triples * k_proportion)
    exp.begin_experiment(regret_id)

    q = exp.batch()

    glimpse_online = g.Online_GLIMPSE(
        exp.kg(), k, initial_entities=None, gamma=0.07, bandit="exp3", reward_function=rf)

    end_times = time.time() + delta

    while time.time() <= end_times:
        for i in range(1000):
            summary = glimpse_online.construct_summary()
            regrets = glimpse_online.update_queries(q)
            exp.add_experiment_results(
                regret_id, [i+1, sum(regrets)/len(regrets)])
    exp.end_experiment(regret_id)
    glimpse_online.save_model(f"experiments_results/{timed_bandit_dir}/bandit")


if __name__ == "__main__":
    graph = "10pow3_edges"
    reward_functions = ["kg", "binary"]
    batch_size = 5000
    deltas = [10, 100, 1000, 7200, 18000, 36000]

    for reward_function in reward_functions:
        for delta in deltas:
            run_on_graph(graph, f"timed_run_{graph}",
                         delta, batch_size=5000, rf=reward_function)
