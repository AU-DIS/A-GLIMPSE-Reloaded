from plotting.plot_bandit_vs_glimpse import plot_combined, plot_combined_regret, plot_bandit_weights
from glimpse.src.glimpse import GLIMPSE
from bandits.efficient_bandits.exp3 import exp3_efficient_bandit
import os
import time
from os import name, path
import experiment
import glimpseonline as g
import experiment as experiment
import sys
import random
import numpy as np
from multiprocessing import Process, process
sys.path.append('..')


try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def run_compare_experiment(graph="10pow3_edges", number_of_rounds=10, k_proportion=0.01, batch_size=10, rf="binary", query_generator="proprietary"):
    d = f"{graph}revised_accuracy_test_{number_of_rounds}_static"
    if not path.exists(f"replacement_results/{d}"):
        os.mkdir(f"replacement_results/{d}")

    compare_bandits_dir = f"{d}/graph{graph}_norounds{number_of_rounds}_bs{batch_size}_kprop{k_proportion}_rf{rf}_generator{query_generator}"

    annotation_regret = f"regret_graph{graph}_norounds{number_of_rounds}_bs{batch_size}_kprop{k_proportion}_rf{rf}_generator{query_generator}"
    exp = experiment.Experiment(
        graph=graph, dir=compare_bandits_dir, query_generator=query_generator)

    list_of_properties = [
        "round", "glimpse_unique_hits, glimpse_no_unique_entities, glimpse_total_hits, glimpse_total, glimpse_accuracy, glimpse_speed, bandit_unique_hits, bandit_no_unique_entities, bandit_total_hits, bandit_total, bandit_accuracy, bandit_speed, random_unique_hits, random_no_unique_entities, random_total_hits, random_total, random_accuracy, random_speed"
    ]

    regret_list_of_properties = ["regret"]

    experiment_files = []
    regret_files = []
    number_of_aggregations = 10
    processes = []
    i = 0
    max_threads = 10
    for a in range(number_of_aggregations):
        annotation = f"accuracy_{a}"
        annotation_regret = f"bandit_{a}"

        experiment_id = exp.create_experiment(
            list_of_properties, annotation, comment=f"Bandit vs GLIMPSE graph={graph}, number of rounds = {number_of_rounds}, kproportion={k_proportion} query generator {query_generator}")

        regret_id = exp.create_experiment(
            regret_list_of_properties, annotation_regret, "")
        exp.begin_experiment(experiment_id)

        k = int(k_proportion * exp.kg().number_of_entities)
        gamma = 0.07
        glimpse_online = g.Online_GLIMPSE(
            exp.kg(), k, bandit="qbl", reward_function=rf, gamma=gamma)
        experiment_files.append(exp.files_[experiment_id])
        regret_files.append(exp.files_[regret_id])

        #p = Process(target=run_experiment, args=(exp, k, rf, batch_size,
        #            number_of_rounds, regret_id, experiment_id, glimpse_online, gamma, compare_bandits_dir, annotation, True if a == number_of_aggregations-1 else False))

        #processes.append(p)
        run_experiment(exp, k, rf, batch_size, number_of_rounds, regret_id, experiment_id, glimpse_online, gamma, compare_bandits_dir, annotation, True if a == number_of_aggregations-1 else False)

    #while len(processes) > 0:
    #    run = []
    #    for _ in range(max_threads):
    #        if len(processes) > 0:
    #            p = processes.pop()
    #            p.start()
    #            run.append(p)
    #    for p in run:
    #        p.join()

    plot_combined(f"{experiment_files[-1]}_{gamma}", experiment_files,
                  f"Number of rounds {number_of_rounds}\nk_proportion {k_proportion}\nbatch_size {batch_size}\nReward function {rf}\nSize of summary {k}\nQuery generator {query_generator}\nGraph triples: {exp.kg().number_of_triples} Graph entities: {exp.kg().number_of_entities}")

    plot_combined_regret(f"{regret_files[-1]}_regret", regret_files,
                        f"Number of rounds {number_of_rounds}\nk_proportion {k_proportion}\nbatch_size {batch_size}\nReward function {rf}\nSize of summary {k}\nQuery generator {query_generator}\nGraph triples: {exp.kg().number_of_triples} Graph entities: {exp.kg().number_of_entities}")


def run_experiment(exp, k, rf, batch_size, number_of_rounds, regret_id, experiment_id, glimpse_online, gamma, compare_bandits_dir, annotation, plot_bandit=False):
    #q = exp.all_batches()
    q = exp.batch(batch_size)
    
    
    t1 = time.process_time()
    glimpse_summary = GLIMPSE(exp.kg(), k, exp.all_batches())
    t2 = time.process_time()
    bandit_delta = t2 - t1
    print(f"Bandit delta: {bandit_delta}")

    for i in range(number_of_rounds):
        print(f"Round: {i+1} of {number_of_rounds}")
        log = [i+1]
        log.extend(list(compute_accuracy(
            exp.kg(), q, glimpse_summary_to_list_of_entities(glimpse_summary))))
        log.append(t2 - t1)
        bandit_summary = glimpse_online.construct_summary(True)
        #for r in glimpse_online.update_queries(q):
        #    exp.add_experiment_results(regret_id, [r])
        
        #Assumes all unique query entities can be in the summary
        exp.add_experiment_results(regret_id, [sum(glimpse_online.update_queries(q))])


        log.extend(list(
            compute_accuracy(
                exp.kg(), q, bandit_glimpse_summary_to_list_of_entities(bandit_summary, exp.kg()))
        ))
        delta = time.process_time() + (bandit_delta)
        log.append(delta)
        all_q = exp.all_batches()

        #for _ in range(int(exp.kg().number_of_triples * 1)):
            # while time.process_time() < delta:
        #glimpse_online.construct_summary()
        #for r in glimpse_online.update_queries(all_q):
        #    exp.add_experiment_results(regret_id, [r])
        #exp.add_experiment_results(regret_id, [sum(glimpse_online.update_queries(q))])

        random_t1 = time.process_time()
        random_triples = np.random.choice(
            range(exp.kg().number_of_triples), k, replace=False)
        random_summaries = []
        for i in random_triples:
            e1, _, e2 = exp.kg().id_to_triple[i]
            random_summaries.append(e1)
            random_summaries.append(e2)

        #print("Computing accuracy for random")
        log.extend(list(
            compute_accuracy(
                exp.kg(), q, random_summaries)
        ))
        #print("Finished computing accuracy for random")
        random_t2 = time.process_time()
        #print(f"Random time: {random_t2 - random_t1}")
        log.append(random_t2 - random_t1)

        exp.add_experiment_results(experiment_id, log)
        if i%50 == 0:
            q = exp.batch(batch_size)
            

    exp.end_experiment(experiment_id)
    exp.end_experiment(regret_id)
    if plot_bandit:
       plot_bandit_weights(glimpse_online.bandit, 100,
                            f"replacement_results/{compare_bandits_dir}/{annotation}_bandit")


def compute_accuracy(kg, queries, summary):
    summary = set(summary)
    #print(f"Computing accuracy for summary size {len(summary)}")
    t2 = time.process_time()
    unique_entities = set()
    total_hits = 0
    total = 0
    unique_hits = set()
    unique_summary = set()
    for q in queries:
        total += 1
        unique_entities.add(q)
        if q in summary:
            unique_hits.add(q)
            total_hits += 1
    for q in summary:
        unique_summary.add(q)
    #print(f"Accuracy time: {time.process_time() - t2}")
    return len(unique_hits), len(unique_summary), total_hits, total, total_hits/len(queries)


def glimpse_summary_to_list_of_entities(summary):
    res = []
    for e1, _, e2 in summary.triples():
        res.append(e1)
        res.append(e2)
    return res


def bandit_glimpse_summary_to_list_of_entities(summary, kg):
    res = []
    for e1, _, e2 in summary.triples():
        res.append(kg.entity_to_id[e1])
        res.append(kg.entity_to_id[e2])
    return res
