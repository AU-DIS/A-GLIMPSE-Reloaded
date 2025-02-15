from os import write
from glimpse.src import glimpse
import util as util
from glimpse.src import user
import numpy as np
import logging
from glimpse.src.glimpse import GLIMPSE, Summary
import time
import glimpseonline as g
from importlib import reload
from random import random, sample, randint, shuffle
from queries import queries
import threading


def bandit_glimpse(k, rounds, experiment, gamma=0.07, bandit="exp3m", same_queries=False, binary_reward=False):
    regret_list_of_properties = ["round", "k", "regret"]
    list_of_properties = ["round", "unique_hits", "no_unique_entities",
                          "total_hits", "total", "accuracy", "speed_summary", "speed_feedback"]

    comment = f"k: {k}, rounds: {rounds}, gamma: {gamma}, len_queries_train: {experiment.batch_size()}, model_path: {None}, bandit={bandit}, same_queries={same_queries}"

    annotation_regret = "regret"
    annotation = "accuracy"

    regret_id = experiment.create_experiment(
        regret_list_of_properties, annotation_regret, str(rounds))

    # normal_id = experiment.create_experiment(
    #    list_of_properties, annotation, comment)

    experiment.begin_experiment(regret_id)
    # experiment.begin_experiment(normal_id)

    q = experiment.batch(k)

    # If we are going to be using the same queries
    if same_queries:
        q = experiment.batch(k)

    rf = "kg" if binary_reward is not True else "binary"

    glimpse_online = g.Online_GLIMPSE(
        experiment.kg(), k, initial_entities=None, gamma=gamma, bandit=bandit, reward_function=rf)

    for i in range(rounds):
        t1 = time.time()
        summary = glimpse_online.construct_summary()
        t2 = time.time()

        if same_queries:
            q = q
        else:
            q = experiment.batch(k)

        regrets = glimpse_online.update_queries(q)
        t3 = time.time()
        # t = threading.Thread(target=finish_bandit_glimpse,
        #                     args = (experiment, i, regret_id, regrets, summary, q, t1, t2, t3, normal_id))
        # t.start()
        for j, regret in enumerate(regrets):
            experiment.add_experiment_results(regret_id, [i+1, j+1, regret])

    experiment.end_experiment(regret_id)
    # experiment.end_experiment(normal_id)
    glimpse_online.save_model(f"models/{experiment.id}")


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
