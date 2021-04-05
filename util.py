import glimpse.src.query as query
import glimpse.src.user as user
import numpy as np
from importlib import reload

reload(user)
reload(query)


def generate_queries(kg, topics, number_of_queries, n_topic_mids, algorithm=user.generate_synthetic_queries_by_topic):
    queries = algorithm(kg, topics, number_of_queries, n_topic_mids)
    return extract_answers_from_queries(kg, queries)


def extract_answers_from_queries(kg, queries):
    return [
        query.answer_query(kg, q) for q in queries
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
