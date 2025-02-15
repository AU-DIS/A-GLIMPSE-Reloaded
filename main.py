from subgraphs import random_induced_subgraph
from experiments.subgraph_experiments import subgraph_experiments, run_bandits_on_subgraph
from experiments.bandit_glimpse_experiments import bandit_glimpse
import os
from time import process_time, sleep
from glimpse.src.experiment_base import DBPedia, KnowledgeGraph, Freebase, load_kg, save_kg
from glimpse.src.query import generate_query
import repl_assistant as repl
import numpy as np
from multiprocessing import Process, process
import experiment
from subgraphs import random_induced_subgraph
from theoretical.exp3_subgraph import plot_combined_theoretical
from experiments.timed_bandit_trainer import run_on_graph
import experiments.pretrained_bandit_versus_glimpse as pretrained
import experiments.bandit_weight as weight
from plotting.plot_bandit_vs_glimpse import plot_combined
from experiments.bandit_versus_glimpse import run_compare_experiment

import time
import cProfile


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


def run_bandits_on_subgraph(subgraph, k, rounds, edge_budget, experiment_name):
    proportion = 0.01

    #rounds = [int((i * 100 * edge_budget)/(proportion * edge_budget))
    #          for i in range(1, 5)]

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

    with open(f"replacement_results/{experiment_name}.txt", "w+") as f:

        f.writelines(filenames)

    labels = {filename: round for filename, round in zip(filenames, rounds)}
    plot_combined_theoretical(
        f"replacement_results/binary_new_queries_{edge_budget}", filenames, labels)


def find_regret_file(dir):
    for f in os.listdir(dir):
        if "regret" in f:
            return f


def extract_number_of_rounds(filename):
    with open(filename, 'r') as f:
        l = f.readline()
        return l.strip('#').strip('\n')


def plot_bandit_run(size, files):
    files = [f"replacement_results/{f}" for f in files]

    filenames = [f"{dir}/{find_regret_file(dir)}" for dir in files]
    labels = {filename: extract_number_of_rounds(
        filename) for filename in filenames}

    plot_combined_theoretical(
        f"replacement_results/{size}", filenames, labels)


def run_timed_training(graph, deltas):
    reward_functions = ["kg", "binary"]
    batch_size = 100

    for reward_function in reward_functions:
        for delta in deltas:
            p = Process(target=run_on_graph, args=(graph, f"timed_run_{graph}",
                                                   delta, reward_function, 0.01))
            p.start()


def run_pretrained_recompute_comparison(graph, deltas):
    reward_functions = ["kg", "binary"]
    recompute_n = [1, 3, 4, 5]
    for reward_function in reward_functions:
        for delta in deltas:
            for n in recompute_n:
                experiment_dir = f"timed_bandits_timed_run_{graph}_{reward_function}_{delta}"
                p = Process(target=pretrained.run_compare_function_experiment,
                            args=(experiment_dir, graph, 40, 0.1, f"recompute_{n}", n))

                p.start()


def run_pretrained_comparison(graph, deltas):
    reward_functions = ["kg", "binary"]
    for reward_function in reward_functions:
        for delta in deltas:
            experiment_dir = f"timed_bandits_timed_run_{graph}_{reward_function}_{delta}"
            p = Process(target=pretrained.run_compare_function_experiment,
                        args=(experiment_dir, graph, 40, 0.1, ""))

            p.start()


def plot_all_pretrained_comparison(graph, deltas):
    reward_functions = ["kg", "binary"]
    #filenames = ["k10000rounds40_.csv"]
    filenames = ["k100rounds40_.csv"]
    recompute_n = [1, 3, 4, 5]
    for n in recompute_n:
        # filenames.append(f"k10000rounds40_recompute_{n}.csv")
        filenames.append(f"k100rounds40_recompute_{n}.csv")

    for reward_function in reward_functions:
        for delta in deltas:
            for n in recompute_n:
                for filename in filenames:
                    filename = f"replacement_results/timed_bandits_timed_run_{graph}_{reward_function}_{delta}/{filename}"
                    p = Process(target=plot_combined,
                                args=(filename, filename))
                    p.start()


def run_compares(graph):
    #reward_functions = ["kg", "binary"]
    reward_functions = ["kg"]
    #ks = [0.01, 0.1, 0.2, 0.3]
    ks = [0.1]#, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # batch_sizes = [0.01 * graph_size, 0.1 * graph_size,
    #               0.2 * graph_size, 0.3 * graph_size]
    #batch_sizes = [100, 1000]
    batch_sizes = [50]

    n = 400
    #query_generators = ["proprietary", "reference"]
    query_generators = ["proprietary"]

    processes = []
    for rf in reward_functions:
        for k in ks:
            for bs in batch_sizes:
                for query_generator in query_generators:
                    run_compare_experiment(graph, n, k, bs, rf, query_generator)
                    #p = Process(target=run_compare_experiment,
                    #            args=(graph, n, k, bs, rf, query_generator))
                    #processes.append(p)

    #for p in processes:
        #Too much parralell if splitting this
        #p.start()
        #p.join()
     

if __name__ == "__main__":
    #graph = "top_triples_10pow3"
    #graph = "dbpedia39"
    #graph = "test_sub_graph"
    graph = 'bfs_graph1'
    #run_glimpse_once()
    #run_on_graph()
    #run_bandits_on_subgraph(graph, 100, [10], 1000, "This_is_test")
    
    #cProfile.runctx('run_compares(graph, graph_size)', globals(), locals())
    
    run_compares(graph)
    #un_pretrained_recompute_comparison(graph, [0.1])
    # run_pretrained_comparison(graph)
    # run_timed_training(graph)
    # plot_all_pretrained_comparison(graph)
