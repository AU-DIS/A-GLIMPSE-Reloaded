from glimpse.src.glimpse import GLIMPSE
from bandits.efficient_bandits.exp3 import exp3_efficient_bandit
import os
import time
from os import name, path
import experiment
import glimpseonline as g
import experiment as experiment
import sys
sys.path.append('..')

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def load_experiment(exp_path):
    with open(f"{exp_path}/experiment_picked", 'rb') as f:
        return pickle.load(f)


# UDREGN HVOR LANG TID DET TAGER AT LAVE ET GLIMPSE SUMMARY
# GIV EXP3 TID TIL AT TRÆNE PÅ NYE QUERIES PÅ SAMME MÆNGDE TID
def run_static_experiment(experiment_dir, graph="10pow3_edges", number_of_rounds=10, k_proportion=0.01):
    data_dir = "experiments_results"
    bandit_path = f"{data_dir}/{experiment_dir}/bandit.npy"

    exp = load_experiment(f"{data_dir}/{experiment_dir}")
    exp.update_graph(graph)
    k = int(exp.kg().number_of_triples * k_proportion)
    glimpse_online = g.Online_GLIMPSE(
        exp.kg(), k, bandit_path, bandit="exp3", reward_function="kg")

    annotation = f"k{k}rounds{number_of_rounds}_2delta"
    comment = f"Bandit versus GLIMPSE k = {k} rounds = {number_of_rounds}"

    exp.experiment_id = "glimpse"
    list_of_properties = [
        "round", "glimpse_unique_hits, glimpse_no_unique_entities, glimpse_total_hits, glimpse_total, glimpse_accuracy, glimpse_speed, bandit_unique_hits, bandit_no_unique_entities, bandit_total_hits, bandit_total, bandit_accuracy, bandit_speed"
    ]
    experiment_id = exp.create_experiment(
        list_of_properties, annotation, comment)

    exp.begin_experiment(experiment_id)
    q = exp.batch()
    t1 = time.process_time()
    glimpse_summary = GLIMPSE(exp.kg(), k, q)
    t2 = time.process_time()
    bandit_delta = t2 - t1

    # Bandit was trained on the first batch, so no need to pass it to it
    # //TODO: Give bandit delta time to iterate over the first batch to make it fair?
    for i in range(number_of_rounds):
        # Compute accuracy
        all_q = exp.all_batches()
        log = [i+1]
        log.extend(list(compute_accuracy(
            exp.kg(), all_q, glimpse_summary_to_list_of_entities(glimpse_summary))))
        log.append(t2 - t1)
        bandit_summary = glimpse_online.construct_summary()
        log.extend(list(
            compute_accuracy(
                exp.kg(), q, bandit_glimpse_summary_to_list_of_entities(bandit_summary, exp.kg()))
        ))

        delta = time.process_time() + (bandit_delta * 2)
        log.append(delta)
        while time.process_time() < delta:
            glimpse_online.construct_summary()
            glimpse_online.update_queries(all_q)

        exp.add_experiment_results(experiment_id, log)
        q = exp.batch()

    exp.end_experiment(experiment_id)


def compute_accuracy(kg, queries, summary):
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
    res = set()
    for e1, _, e2 in summary.triples():
        res.add(e1)
        res.add(e2)
    return res


def bandit_glimpse_summary_to_list_of_entities(summary, kg):
    res = set()
    for e1, _, e2 in summary.triples():
        res.add(kg.entity_to_id[e1])
        res.add(kg.entity_to_id[e2])
    return res
