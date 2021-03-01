from util import generate_queries, makeTrainingTestSplit, calculateAccuracyAndTotals
from glimpse.src.user import generate_queries_by_topic
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


def trainer(kg, queries, k, rounds):
    global no_unique_entities
    reload(t)
    reload(g)
    nt = randint(int(queries/1000), int(queries/5))

    queries = generate_queries(kg, topics, queries, nt)

    logging.info("KG entities: " + str(kg.number_of_entities()))
    logging.info("KG triples: " + str(kg.number_of_triples_))
    logging.info(f"k, nt = {k} {nt}")

    unique_entities = set()
    for x in queries:
        for y in x:
            unique_entities.add(y)
    no_unique_entities = len(unique_entities)
    unique_entities = set()

    glimpse_online = g.Online_GLIMPSE(kg, k)
    # glimpse_online.init_bandit_weights(queries)

    for i in range(rounds):
        logging.info(f"Round {i+1}/{rounds}")
        t1 = time.time()
        summary = glimpse_online.construct_summary()
        t2 = time.time()
        round_queries = queries

        mean_accuracy = compute_accuracy(round_queries, summary)

        logging.info("Giving feedback")
        glimpse_online.update_queries(round_queries, queries, 0)
        t3 = time.time()
        logging.info(f"Speed: Summary: {t2 - t1} Feedback: {t3 - t2}")


def compute_accuracy(queries, summary):
    global no_unique_entities
    total_hits = 0
    total = 0
    unique_hits = set()
    for q in queries:
        for answer in q:
            total = total + 1
            if summary.has_entity(answer):
                unique_hits.add(answer)
                total_hits = total_hits + 1

    logging.info(
        f"{bcolors.OKCYAN}Number of hits: {total_hits}/{total} unique: {len(unique_hits)}/{no_unique_entities}")
    logging.info(
        f"     Accuracy: {total_hits/total}, {len(unique_hits) / no_unique_entities}{bcolors.ENDC}")
    return len(unique_hits) / no_unique_entities
