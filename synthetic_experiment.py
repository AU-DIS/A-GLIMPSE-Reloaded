import os
from glimpse.src.experiment_base import DBPedia, KnowledgeGraph, Freebase
from glimpse.src.query import generate_query
from experiments import calculateAccuracyAndTotals
from glimpse.src.glimpse import GLIMPSE, Summary
from glimpse.src.user import generate_synthetic_queries_by_topic
from glimpse.src.query import answer_query
import json
import math
import sys
import time
from os import listdir
from os.path import isfile, join
import logging
import argparse
import numpy as np
import pandas as pd


def extract_answers_from_queries(kg, queries):
    return [
        answer_query(kg, q) for q in queries
    ]


def makeTrainingTestSplit(answers, kg):
    n = len(answers)
    split = int(0.7 * n)
    train_split = [[entity for entity in answer_list if kg.is_entity(
        entity)] for answer_list in answers[0:split]]
    test_split = [[entity for entity in answer_list if kg.is_entity(
        entity)] for answer_list in answers[split:]]

    return train_split, test_split


def runsyntheticGLIMPSEExperimentOnce(topics, kg, e=0.1, k_pct=0.01, number_of_users=3, no_per_user=200):
    user_answers = [
        extract_answers_from_queries(kg, generate_synthetic_queries_by_topic(kg, topics, no_per_user, no_per_user)) for _ in range(number_of_users)
    ]
    user_ids = [x for x in range(number_of_users)]

    # for file in user_log_answer_files:
    #    df = pd.read_csv(path + str(file))
    #    user_ids.append(file.split(".csv")[0])
    #    # list of lists of answers as iris
    #    user_answers.append(
    #        [["<" + iri + ">" for iri in f.split(" ")] for f in df['answers']])

    user_log_train = []
    user_log_test = []
    for i in range(number_of_users):
        train, test = makeTrainingTestSplit(user_answers[i], kg)
        user_log_train.append(train)
        user_log_test.append(test)

    k = k_pct*kg.number_of_triples()

    logging.info("KG entities: " + str(kg.number_of_entities()))
    logging.info("KG triples: " + str(kg.number_of_triples_))

    logging.info("Running for K=" + str(k) + ", e=" + str(e))
    rows = []
    for idx_u in range(number_of_users):
        kg.reset()
        # model user pref
        logging.info("  Running GLIMPSE on user: " + str(user_ids[idx_u]))
        t1 = time.time()
        summary = GLIMPSE(kg, k, user_log_train[idx_u], e)
        user_log_train[idx_u] = []
        logging.info("  Done")
        t2 = time.time()

        mean_accuracy, total_entities, total_count = calculateAccuracyAndTotals(
            user_log_test[idx_u], summary)
        summary = []

        logging.info("      Summary  accuracy " + str(mean_accuracy) + "%")
        rows.append({'match': total_count, 'total': total_entities,
                     '%': mean_accuracy, 'runtime': t2 - t1})

    pd.DataFrame(rows).to_csv("experiments_results/v" + "test" + "T#" + str(kg.number_of_triples()) + "_E#" + str(
        kg.number_of_entities()) + "K#" + str(int(k)) + "e#" + str(e) + ".csv")
