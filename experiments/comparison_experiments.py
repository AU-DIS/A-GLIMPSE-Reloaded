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


def generate_run_name():
    return time.time()


run_name = generate_run_name()


# //TODO: Make a split
def generate_queries(kg, number_of_queries):
    nt = randint(int(number_of_queries/5), int(number_of_queries))
    reload(g)
    reload(util)

    topics = sample(kg.entity_id_.keys(), kg.number_of_entities)

    queries = util.generate_queries(kg, topics, number_of_queries, nt)
    queries = [q for q in queries if len(q) > 0]
    shuffle(queries)
    return queries


def split_queries(queries, rounds, init_percentage=0.20):
    init_queries = queries[:int(len(queries)*init_percentage)]
    round_queries = queries[int(len(queries)*init_percentage):]
    splitted_queries = [round_queries[i*(len(round_queries)//rounds):(
        i+1)*(len(round_queries)//rounds)] for i in range(rounds)]
    return init_queries, splitted_queries

# //TODO: Make sure it works


def run_static_glimpse(kg, k, rounds, e, queries_train, queries_validation):
    global run_name

    print(len(kg.entities_))

    t1 = time.time()
    # //TODO: Make sure GLIMPSE works with new indices
    summary = GLIMPSE(
        kg, k, queries_train, e
    )
    t2 = time.time()

    write_buffer = []
    # //TODO: Fix the path
    with open(f"experiments_results/{run_name}_static.csv", "w+") as f:
        f.write(
            f"# k: {k}, rounds: {rounds}, e: {e}, len_queries_train: {len(queries_train)}\n")
        f.write(
            "round,unique_hits,no_unique_entities,total_hits,total,accuracy,speed_summary\n")
        round_queries = queries_train.copy()
        for i in range(rounds):
            round_queries.extend(queries_validation[i])
            unique_hits, no_unique, total_hits, total, accuracy = compute_accuracy(
                # //TODO: Is it :i?
                kg, round_queries, glimpse_summary_to_list_of_entities(summary))

            write_buffer.append(
                f"{i+1},{unique_hits},{no_unique},{total_hits},{total},{accuracy},{t2-t1}\n")

        for line in write_buffer:
            f.write(line)

# //TODO: Audit it


def recompute_glimpse(kg, k, rounds, e, queries_train, queries_validation, n):
    global run_name

    t1 = time.time()
    # //TODO: Make sure GLIMPSE works with new indices
    summary = GLIMPSE(
        kg, k, queries_train, e
    )
    t2 = time.time()

    write_buffer = []
    # //TODO: Fix the path
    with open(f"experiments_results/{run_name}_recompute.csv", "w+") as f:
        f.write(
            f"# k: {k}, rounds: {rounds}, e: {e}, len_queries_train: {len(queries_train)}, n: {n}\n")
        f.write(
            "round,unique_hits,no_unique_entities,total_hits,total,accuracy,speed_summary\n")

        round_queries = []
        queue_queries = queries_validation.copy()

        for i in range(rounds):
            round_queries.extend(queue_queries[i])

            unique_hits, no_unique, total_hits, total, accuracy = compute_accuracy(
                # //TODO: Is it :i?
                kg, round_queries, glimpse_summary_to_list_of_entities(summary))

            write_buffer.append(
                f"{i+1},{unique_hits},{no_unique},{total_hits},{total},{accuracy},{t2-t1}\n")

            if (i+1) % n == 0:
                t1 = time.time()
                summary = GLIMPSE(
                    kg, k, round_queries, e
                )
                t2 = time.time()
                for line in write_buffer:
                    f.write(line)
                write_buffer = []

        for line in write_buffer:
            f.write(line)


def bandit_glimpse(kg, k, rounds, queries_train, queries_validation, model_path, gamma=0.07):

    reload(g)

    with open(f"experiments_results/{run_name}_bandit_regret.csv", "w+") as regret_file:
        regret_file.write("round,k,regret\n")
        with open(f"experiments_results/{run_name}_bandit.csv", "w+") as f:
            f.write(
                f"# k: {k}, rounds: {rounds}, gamma: {gamma}, len_queries_train: {len(queries_train)}, model_path: {model_path}\n")
            f.write(
                "round,unique_hits,no_unique_entities,total_hits,total,accuracy,speed_summary,speed_feedback\n")

            write_buffer = []
            regret_buffer = []

            entities = set()
            for q in queries_train:
                for answer in q:
                    entities.add(answer)

            glimpse_online = g.Online_GLIMPSE(
                kg, k, initial_entities=entities, gamma=gamma)

            round_queries = queries_train.copy()
            for i in range(rounds):
                t1 = time.time()
                summary = glimpse_online.construct_summary()
                t2 = time.time()

                round_queries.extend(queries_validation[i])
                regrets = glimpse_online.update_queries(round_queries)
                for j, regret in enumerate(regrets):
                    regret_buffer.append(f"{i+1},{j+1},{regret}\n")

                t3 = time.time()

                unique_hits, no_unique, total_hits, total, accuracy = compute_accuracy(
                    kg, round_queries, bandit_glimpse_summary_to_list_of_entities(summary, kg))

                write_buffer.append(
                    f"{i+1},{unique_hits},{no_unique},{total_hits},{total},{accuracy},{t2-t1},{t3-t2}\n")

                if i % 1000:
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

# //TODO: Fix for GLIMPSE summaries


def compute_accuracy(kg, queries, summary):
    # //TODO: Get no unique entities from bandit trainer code
    unique_entities = set()
    total_hits = 0
    total = 0
    unique_hits = set()

    for q in queries:
        for answer in q:
            total += 1
            unique_entities.add(answer)
            if answer in summary:
                unique_hits.add(answer)
                total_hits = total_hits + 1

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
