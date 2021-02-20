import os
from GLIMPSE_personalized_KGsummarization.src.experiment_base import DBPedia, KnowledgeGraph, Freebase
from GLIMPSE_personalized_KGsummarization.src.query import generate_query
from experiments import calculateAccuracyAndTotals
from GLIMPSE_personalized_KGsummarization.src.glimpse import GLIMPSE, Summary

kg = ""


def makeTrainingTestSplit(answers, kg, n):
    split = int(0.7 * n)
    train_split = [[entity for entity in answer_list if kg.is_entity(
        entity)] for answer_list in answers[0:split]]
    test_split = [[entity for entity in answer_list if kg.is_entity(
        entity)] for answer_list in answers[split:]]

    return train_split, test_split


def load_kg():
    global kg
    kg = DBPedia('DBPedia3.9/')
    kg.load()


def run_glimpse_on_queries(queries, kg, n):
    answers = [x['Parse']['Answers'] for x in queries]
    answer_entity_names = [[a_name['EntityName']
                            for a_name in answer] for answer in answers]
    train_split, test_split = makeTrainingTestSplit(answer_entity_names, kg)
    e = 0.1
    k = 0.01
    summary = GLIMPSE(kg, k, train_split, e)
    mean_accuracy, total_entities, total_count = calculateAccuracyAndTotals(
        test_split, summary)
    print(mean_accuracy, total_entities, total_count)

