from os import write
from glimpse.src import glimpse
import util as util
from glimpse.src import user
import topics as t
import numpy as np
import logging
from glimpse.src.glimpse import GLIMPSE, Summary
import time
import glimpseonline as g
from importlib import reload
from random import random, sample, randint, shuffle
from queries import queries


def generate_run_name():
    return time.time()


run_name = generate_run_name()


def run_static_glimpse(kg, k, rounds, e, queries):
    global run_name

    t1 = time.time()

    summary = GLIMPSE(
        kg, k, queries.batch(), e
    )
    t2 = time.time()

    write_buffer = []
    with open(f"experiments_results/{run_name}_static.csv", "w+") as f:
        f.write(
            f"# k: {k}, rounds: {rounds}, e: {e}, len_queries_train: {queries.batch_size}\n")
        f.write(
            "round,unique_hits,no_unique_entities,total_hits,total,accuracy,speed_summary\n")
        for i in range(rounds):

            unique_hits, no_unique, total_hits, total, accuracy = compute_accuracy(
                kg, queries.batch(), glimpse_summary_to_list_of_entities(summary))

            write_buffer.append(
                f"{i+1},{unique_hits},{no_unique},{total_hits},{total},{accuracy},{t2-t1}\n")

        for line in write_buffer:
            f.write(line)


def recompute_glimpse(kg, k, rounds, e, queries, n):
    global run_name

    t1 = time.time()
    summary = GLIMPSE(
        kg, k, queries.batch(), e
    )
    t2 = time.time()

    write_buffer = []
    with open(f"experiments_results/{run_name}_recompute.csv", "w+") as f:
        f.write(
            f"# k: {k}, rounds: {rounds}, e: {e}, len_queries_train: {queries.batch_size}, n: {n}\n")
        f.write(
            "round,unique_hits,no_unique_entities,total_hits,total,accuracy,speed_summary\n")

        for i in range(rounds):
            unique_hits, no_unique, total_hits, total, accuracy = compute_accuracy(
                kg, queries.batch(), glimpse_summary_to_list_of_entities(summary))

            write_buffer.append(
                f"{i+1},{unique_hits},{no_unique},{total_hits},{total},{accuracy},{t2-t1}\n")

            if (i+1) % n == 0:
                t1 = time.time()

                summary = GLIMPSE(
                    kg, k, queries.all_batches(), e
                )
                t2 = time.time()
                for line in write_buffer:
                    f.write(line)
                write_buffer = []

        for line in write_buffer:
            f.write(line)


def bandit_glimpse(kg, k, rounds, queries, model_path, gamma=0.07, bandit="exp3m", same_queries=False):

    with open(f"experiments_results/{run_name}_bandit_regret.csv", "w+") as regret_file:
        regret_file.write("round,k,regret\n")
        with open(f"experiments_results/{run_name}_bandit.csv", "w+") as f:
            f.write(
                f"# k: {k}, rounds: {rounds}, gamma: {gamma}, len_queries_train: {len(queries_train)}, model_path: {model_path}\n")
            f.write(
                "round,unique_hits,no_unique_entities,total_hits,total,accuracy,speed_summary,speed_feedback\n")

            write_buffer = []
            regret_buffer = []

            # If we are going to be using the same queries
            if same_queries:
                q = queries.batch()

            glimpse_online = g.Online_GLIMPSE(
                kg, k, initial_entities=None, gamma=gamma, bandit=bandit)

            for i in range(rounds):
                t1 = time.time()
                summary = glimpse_online.construct_summary()
                t2 = time.time()

                if same_queries:
                    q = q
                else:
                    q = queries.batch()

                regrets = glimpse_online.update_queries(q)
                for j, regret in enumerate(regrets):
                    regret_buffer.append(f"{i+1},{j+1},{regret}\n")

                t3 = time.time()

                unique_hits, no_unique, total_hits, total, accuracy = compute_accuracy(
                    kg, q, bandit_glimpse_summary_to_list_of_entities(summary, kg))

                write_buffer.append(
                    f"{i+1},{unique_hits},{no_unique},{total_hits},{total},{accuracy},{t2-t1},{t3-t2}\n")

                if i % 5000:
                    for line in write_buffer:
                        f.write(line)
                    for line in regret_buffer:
                        regret_file.write(line)
                    write_buffer = []
                    regret_buffer = []

            for line in write_buffer:
                f.write(line)
        for line in regret_buffer:
            regret_file.write(line)


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
        res.add(kg.entities_[e1])
        res.add(kg.entities_[e2])
    return res
