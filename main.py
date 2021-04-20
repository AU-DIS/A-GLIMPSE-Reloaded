import os
from time import sleep
from glimpse.src.experiment_base import DBPedia, KnowledgeGraph, Freebase
from glimpse.src.query import generate_query
import repl_assistant as repl
import numpy as np
from multiprocessing import Process


def makeTrainingTestSplit(answers, kg):
    train_split = [[entity for entity in answer_list if kg.is_entity(
        entity)] for answer_list in answers[0:140]]
    test_split = [[entity for entity in answer_list if kg.is_entity(
        entity)] for answer_list in answers[140:]]

    return train_split, test_split


def run_glimpse_once():
    K = [10**-i for i in range(1, 6)]
    E = [1e-2]

    answer_version = "2"
    version = "3"
    e = 0.1
    k = 0.01
    os.system('cd originalprojects && python3 experiments.py'
              + ' --method glimpse'
              + ' --percent-triples ' + str(k)
              + ' --version ' + version
              + ' --version-answers ' + answer_version
              + ' --epsilon ' + str(e)
              #                  + ' > ' + 'GLIMPSE' + version+'a' + answer_version + ' #K' + str(k) + '#E' + str(e) + '.out'
              #                  + ' &')
              )


def synthetic_experiment():
    kg = DBPedia('DBPedia3.9/')
    kg.load()
    number_of_topics = 200
    topics = kg.entity_names()
    topic_keys = [x for x in topics.keys()]
    topics = [topics[topic_keys[x]] for x in range(number_of_topics)]
    topic_keys = []
    queries = [generate_query(kg, topic) for topic in topics]
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


def parameters_experiment():
    import experiments.comparison_experiments as comp
    repl.load_kg()

    # Make sure to explicitly set the run_name every time
    comp.run_name = comp.generate_run_name()

    # We will use the same queries for all the experiments
    q = comp.generate_queries(repl.kg, 100000)
    init, rounds = comp.split_queries(q, 1000)

    # //TODO: Small hack in naming is necessary, fix
    initial_run_name = comp.run_name
    E = np.linspace(0.01, 1, 20, endpoint=False)

    for i in range(0, 20):
        comp.run_name = f"{initial_run_name}_epsilon_test_{i}"
        comp.run_static_glimpse(repl.kg, 10000, 1000,
                                E[i], init, rounds)

    for i in range(0, 20):
        comp.run_name = f"{initial_run_name}_epsilon_test_{i}"
        comp.recompute_glimpse(repl.kg, 10000, 1000,
                               E[i], init, rounds, 100)

    for i in range(0, 20):
        comp.run_name = f"{initial_run_name}_recompute_interval_test_{i}"
        comp.recompute_glimpse(repl.kg, 10000, 1000,
                               0.1, init, rounds, 10 * (i+1))

    num_entities = len(q)
    S = np.linspace(0.01 * num_entities, 10 * num_entities, 20)
    for i in range(0, 20):
        comp.run_name = f"{initial_run_name}_summary_size_{i}"
        comp.run_static_glimpse(repl.kg, S[i], 1000,
                                0.1, init, rounds)


def bandit_experiment(bandit, rounds):
    import experiments.comparison_experiments as comp
    repl.load_kg()

    # Make sure to explicitly set the run_name every time
    comp.run_name = comp.generate_run_name()

    # We will use the same queries for all the experiments
    q = comp.generate_queries(repl.kg, 10000)
    init, rounds = comp.split_queries(
        q, 2)

    # //TODO: Small hack in naming is necessary, fix
    initial_run_name = comp.run_name
    E = np.linspace(0.01, 1, 10, endpoint=False)

    for i in range(0, 10):
        comp.run_name = f"{initial_run_name}_gamma_test_{i}"
        comp.bandit_glimpse(repl.kg, 100000, 5000000, init,
                            rounds, None, gamma=E[i], bandit=bandit, same_queries=True)


if __name__ == "__main__":
    from multiprocessing import Process
    p1 = Process(target=bandit_experiment, args=('exp3m', 50000,))
    sleep(2)
    p2 = Process(target=bandit_experiment, args=('exp3', 5000000,))

    p1.start()
    p2.start()
