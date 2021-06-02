from main import plot_bandit_run
from plotting.plot_bandit_vs_glimpse import plot_combined, plot_combined_regret, plot_bandit_weights
from glimpse.src.glimpse import GLIMPSE
from bandits.efficient_bandits.exp3 import exp3_efficient_bandit
import os
import time
from os import name, path
import experiment
import glimpseonline as g
import experiment as experiment
import sys
import random
import numpy as np
sys.path.append('..')

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def run_bandit_configuration(graph="10pow3_edges", k_proportion=0.01,  batch_size=100, rf="binary", query_generator="proprietary"):
    k = 100
    weights_dir = "weights"
    regret_list_of_properties = ["regret"]
    annotation = f"graph{graph}_k{k_proportion}_bs{batch_size}"

    exp = experiment.Experiment(
        graph=graph, dir=weights_dir, query_generator="proprietary")

    glimpse_online = g.Online_GLIMPSE(
        exp.kg(), k, bandit="exp3", reward_function=rf, gamma=0.07/k)
    plot_bandit_weights(glimpse_online.bandit, 100,
                        f"experiments_results/{weights_dir}/banditprior")
    regret_id = exp.create_experiment(
        regret_list_of_properties, annotation, "")
    exp.batch(batch_size)
    for i in range(10):
        exp.batch(batch_size)
        glimpse_online.construct_summary()
        for _ in range(exp.kg().number_of_triples * 10):
            for r in glimpse_online.update_queries(exp.all_batches()):
                exp.add_experiment_results(regret_id, [r])

    exp.end_experiment(regret_id)
    plot_combined_regret(exp.files_[regret_id],
                         [exp.files_[regret_id]], f"Number of rounds {exp.kg().number_of_triples}")
    plot_bandit_weights(glimpse_online.bandit, 100,
                        f"experiments_results/{weights_dir}/bandit")


def compute_accuracy(kg, queries, summary):
    summary = set(summary)
    #print(f"Computing accuracy for summary size {len(summary)}")
    t2 = time.process_time()
    unique_entities = set()
    total_hits = 0
    total = 0
    unique_hits = set()
    for q in queries:
        total += 1
        unique_entities.add(q)
        if q in summary:
            unique_hits.add(q)
            total_hits += 1
    return len(unique_hits), len(unique_entities), total_hits, total, len(unique_hits)/len(unique_entities)


def glimpse_summary_to_list_of_entities(summary):
    res = []
    for e1, _, e2 in summary.triples():
        res.append(e1)
        res.append(e2)
    return res
