from plotting.plot_bandit_vs_glimpse import plot_combined
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


def run_compare_experiment(graph="10pow3_edges", number_of_rounds=10, k_proportion=0.01, batch_size=10, rf="kg", query_generator="proprietary"):
    compare_bandits_dir = "comparison_results"
    exp = experiment.Experiment(
        graph=graph, dir=compare_bandits_dir, query_generator=query_generator)

    list_of_properties = [
        "round", "glimpse_unique_hits, glimpse_no_unique_entities, glimpse_total_hits, glimpse_total, glimpse_accuracy, glimpse_speed, bandit_unique_hits, bandit_no_unique_entities, bandit_total_hits, bandit_total, bandit_accuracy, bandit_speed, random_unique_hits, random_no_unique_entities, random_total_hits, random_total, random_accuracy, random_speed"
    ]

    experiment_files = []
    number_of_aggregations = 1

    for a in range(number_of_aggregations):
        annotation = f"graph{graph}_norounds{number_of_rounds}_bs{batch_size}_kprop{k_proportion}_rf{rf}_generator{query_generator}_{a}"

        experiment_id = exp.create_experiment(
            list_of_properties, annotation, comment=f"Bandit vs GLIMPSE graph={graph}, number of rounds = {number_of_rounds}, kproportion={k_proportion} query generator {query_generator}")

        exp.begin_experiment(experiment_id)

        k = int(k_proportion * exp.kg().number_of_entities)
        q = exp.batch(batch_size)
        t1 = time.process_time()
        glimpse_summary = GLIMPSE(exp.kg(), k, exp.all_batches())
        t2 = time.process_time()
        bandit_delta = t2 - t1
        print(bandit_delta)
        glimpse_online = g.Online_GLIMPSE(
            exp.kg(), k, bandit="exp3", reward_function=rf)

        random_summaries = np.random.choice(
            range(exp.kg().number_of_entities), int(k * number_of_rounds), replace=True)

        for i in range(number_of_rounds):
            log = [i+1]
            log.extend(list(compute_accuracy(
                exp.kg(), q, glimpse_summary_to_list_of_entities(glimpse_summary))))
            log.append(t2 - t1)
            bandit_summary = glimpse_online.construct_summary()
            log.extend(list(
                compute_accuracy(
                    exp.kg(), q, bandit_glimpse_summary_to_list_of_entities(bandit_summary, exp.kg()))
            ))
            delta = time.process_time() + (bandit_delta)
            log.append(delta)
            all_q = exp.all_batches()

            # for _ in range(exp.kg().number_of_triples):
            while time.process_time() < delta:
                glimpse_online.construct_summary()
                glimpse_online.update_queries(all_q)

            random_t1 = time.process_time()
            random_summary = random_summaries[i*k:(i+1)*k]

            log.extend(list(
                compute_accuracy(
                    exp.kg(), q, random_summary)
            ))
            random_t2 = time.process_time()
            log.append(random_t2 - random_t1)

            exp.add_experiment_results(experiment_id, log)
            q = exp.batch(batch_size)

        exp.end_experiment(experiment_id)
        experiment_files.append(exp.files_[experiment_id])

    plot_combined(exp.files_[experiment_id], experiment_files,
                  f"Number of rounds {number_of_rounds}\nk_proportion {k_proportion}\nbatch_size {batch_size}\nReward function {rf}\nSize of summary {k}\nQuery generator {query_generator}\nGraph triples: {exp.kg().number_of_triples} Graph entities: {exp.kg().number_of_entities}")


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
