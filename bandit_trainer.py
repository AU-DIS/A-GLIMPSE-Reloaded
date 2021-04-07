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
from random import sample, randint


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


no_unique_entities = 0
topics = t.topics


def trainer(kg, queries, k, rounds, model_path=None):
    global no_unique_entities
    reload(t)
    reload(g)
    reload(util)
    nt = randint(int(queries/1000), int(queries/5))

    queries = util.generate_queries(kg, topics, queries, nt)
    run_name = time.time()

    with open(f"runs/{run_name}.csv", "w+") as f:
        f.write(
            "round, unique_hits, no_unique_entities, total_hits, total, accuracy, speed_summary, speed_feedback\n")

        write_buffer = []

        unique_entities = set()
        for x in queries:
            for y in x:
                unique_entities.add(y)
        no_unique_entities = len(unique_entities)

        if model_path is None:
            glimpse_online = g.Online_GLIMPSE(
                kg, k, initial_entities=unique_entities)
        else:
            glimpse_online = g.Online_GLIMPSE(
                kg, k, model_path=model_path)
        unique_entities = set()

        for i in range(rounds):
            t1 = time.time()
            summary = glimpse_online.construct_summary()
            t2 = time.time()
            round_queries = queries
            glimpse_online.update_queries(round_queries)
            t3 = time.time()
            unique_hits, no_unique, total_hits, total, accuracy = compute_accuracy(
                kg, round_queries, summary)
            write_buffer.append(
                f"{i+1},{unique_hits},{no_unique},{total_hits},{total},{accuracy},{t2-t1},{t3-t2}\n")

            if i % (rounds//10) == 0:
                for line in write_buffer:
                    f.write(line)
                    write_buffer = []
                    glimpse_online.save_model(f"models/{run_name}")

        for line in write_buffer:
            f.write(line)
        glimpse_online.save_model(f"models/{run_name}")


def compute_accuracy(kg, queries, summary):
    global no_unique_entities
    total_hits = 0
    total = 0
    unique_hits = set()
    for q in queries:
        for answer in q:
            total += 1
            answer = kg.entity_to_id[answer]
            if summary.has_entity(answer):
                unique_hits.add(answer)
                total_hits = total_hits + 1

    return len(unique_hits), no_unique_entities, total_hits, total, len(unique_hits)/no_unique_entities
