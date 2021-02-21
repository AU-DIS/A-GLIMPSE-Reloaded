from glimpse.src.query import answer_query
from glimpse.src.user import generate_synthetic_queries_by_topic
import numpy as np


def generate_queries(kg, topics, number_of_queries, n_topic_mids, algorithm=generate_synthetic_queries_by_topic):
    return extract_answers_from_queries(kg, algorithm(kg, topics, number_of_queries, n_topic_mids))


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


def calculateAccuracyAndTotals(user_log_test_u, summary):
    accuracies = []
    total_count = 0
    total_entities = 0
    for answers_to_query in user_log_test_u:
        count = 0
        total_answers = len(answers_to_query)
        if total_answers == 0:
            continue
        else:
            total_entities += total_answers
            for iri in answers_to_query:
                if summary.has_entity(iri):
                    count += 1
                    total_count += 1
            accuracies.append(count / total_answers)

    return np.mean(np.array(accuracies)), total_entities, total_count
