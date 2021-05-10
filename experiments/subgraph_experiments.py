from subgraphs import random_induced_subgraph
from experiments.bandit_glimpse_experiments import bandit_glimpse
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
from human_id import generate_id


def run_bandits_on_subgraph(subgraph, edge_budget):
    proportion = 0.01
    k = proportion * edge_budget

    rounds = int((100 * edge_budget * 4)/(proportion * edge_budget))

    filenames = []

    directory_names = [generate_id() for _ in range(0, 3)]
    filenames = [
        f"experiments_results/{name}/regret.csv" for name in directory_names]

    ordinary_exp = experiment.Experiment(
        comment=f"Test of random induced subgraph with triples: {edge_budget} on graph {subgraph} with rounds {rounds}", graph=subgraph, dir=directory_names[0])

    binary_exp = experiment.Experiment(
        comment=f"Test of random induced subgraph with triples: {edge_budget} on graph {subgraph} with rounds {rounds} and binary reward", graph=subgraph, dir=directory_names[1])

    p_ordinary = Process(target=bandit_glimpse, args=(
        k, rounds, ordinary_exp, 0.07, "exp3", True, False))

    p_binary = Process(target=bandit_glimpse, args=(
        k, rounds, binary_exp, 0.07, "exp3", True, True))

    p_ordinary.start()
    p_binary.start()
    p_binary.join()
    p_binary.join()

    plot_names = [f"ordinary_{edge_budget}", f"binary_{edge_budget}"]
    labels = {filename: rounds for filename in filenames}
    for filename, plot_name in zip(filenames, plot_names):
        plot_combined_theoretical(
            f"experiments_results/{plot_name}", [filename], labels)


def subgraph_experiments(create_subgraphs=False, existing_graph="ordinarykg"):
    subgraphs = ["10pow6_edges", "10pow5_edges",
                 "10pow4_edges", "10pow3_edges"]
    edge_budgets = [10**6, 10**5, 10**4, 10**3]

    if create_subgraphs:
        kg = DBPedia('dbpedia39')
        if existing_graph is None:
            kg.load()
        else:
            kg = load_kg(existing_graph)
        save_kg(kg, "main_graph")
        del kg

        processes = []
        for subgraph, budget in zip(subgraphs, edge_budgets):
            p = Process(target=random_induced_subgraph,
                        args=("main_graph", subgraph, 0, budget,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    for subgraph, edge_budget in zip(subgraphs, edge_budgets):
        p = Process(target=run_bandits_on_subgraph,
                    args=(subgraph, edge_budget))
        p.start()
