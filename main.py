from subgraphs import random_induced_subgraph
from experiments.subgraph_experiments import subgraph_experiments
import os
from time import sleep
from glimpse.src.experiment_base import DBPedia, KnowledgeGraph, Freebase, load_kg, save_kg
from glimpse.src.query import generate_query
import repl_assistant as repl
import numpy as np
from multiprocessing import Process
import experiment
from subgraphs import random_induced_subgraph
from theoretical.exp3_subgraph import plot_combined_theoretical
from experiments.timed_bandit_trainer import run_on_graph
import experiments.pretrained_bandit_versus_glimpse as pretrained


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

    exp = experiment.Experiment("Static", "Parameters experiment")

    E = np.linspace(0.01, 1, 20, endpoint=False)
    for i in range(0, 20):
        run_static_glimpse(10000, 30, E[i], exp)

    for i in range(0, 20):
        recompute_glimpse(10000, 100, E[i], 10, exp)

    for i in range(0, 20):
        recompute_glimpse(10000, 100, 0.1, 5 * (i+1), exp)


def exp3m_parameters():
    exp = experiment.Experiment(
        comment="Diverse experiment med bandit parametre")

    E = np.linspace(0.01, 1, 10, endpoint=False)

    for i in range(0, 10):
        bandit_glimpse(10000, 1000, exp, E[i], same_queries=True)

    for i in range(0, 10):
        bandit_glimpse(10000, 1000, exp, E[i], same_queries=False)


def exp3m_longrun():
    exp = experiment.Experiment(
        comment="exp3m longrun same queries")

    bandit_glimpse(10000, 3000, exp, 0.1, same_queries=True)


def exp_longrun():
    exp = experiment.Experiment(comment="exp3 longrun")

    bandit_glimpse(10000, 10800, exp, 0.1, "exp3", same_queries=True)


def exp3m_non_adversarial():
    exp = experiment.Experiment(
        comment="exp3m non adversarial queries",
        adversarial_degree=0.00001)

    bandit_glimpse(10000, 2000, exp, 0.1, same_queries=True)


def exp3_non_adversarial():
    exp = experiment.Experiment(
        comment="exp3 non adversarial queries",
        adversarial_degree=0.00001)

    bandit_glimpse(10000, 2000, exp, 0.1, same_queries=True)


def run_bandits_on_subgraph(subgraph, edge_budget, experiment_name):
    proportion = 0.01
    k = proportion * edge_budget

    rounds = [int((i * 100 * edge_budget)/(proportion * edge_budget))
              for i in range(1, 5)]

    processes = []
    exps = []
    for round in rounds:
        exp = experiment.Experiment(
            comment=f"Test of random induced subgraph with triples: {edge_budget} on graph {subgraph} with rounds {round}", graph=subgraph)
        p = Process(target=bandit_glimpse, args=(
            k, round, exp, 0.07, "exp3", False,))
        exps.append(exp)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    filenames = []
    for exp in exps:
        for expname in exp.files_.keys():
            filenames.append(exp.files_[expname])

    with open(f"experiments_results/{experiment_name}.txt", "w+") as f:

        f.writelines(filenames)

    labels = {filename: round for filename, round in zip(filenames, rounds)}
    plot_combined_theoretical(
        f"experiments_results/binary_new_queries_{edge_budget}", filenames, labels)


def find_regret_file(dir):
    for f in os.listdir(dir):
        if "regret" in f:
            return f


def extract_number_of_rounds(filename):
    with open(filename, 'r') as f:
        l = f.readline()
        return l.strip('#').strip('\n')


def plot_bandit_run(size, files):
    files = [f"experiments_results/{f}" for f in files]

    filenames = [f"{dir}/{find_regret_file(dir)}" for dir in files]
    labels = {filename: extract_number_of_rounds(
        filename) for filename in filenames}

    plot_combined_theoretical(
        f"experiments_results/{size}", filenames, labels)


def run_timed_training():
    graph = "10pow6_edges"
    reward_functions = ["kg", "binary"]
    batch_size = 5000
    deltas = [10, 100, 1000, 7200, 18000, 36000]

    for reward_function in reward_functions:
        for delta in deltas:
            p = Process(target=run_on_graph, args=(graph, f"timed_run_{graph}",
                                                   delta, 5000, reward_function, 0.01))
            p.start()


if __name__ == "__main__":
    pretrained.run_static_experiment()
