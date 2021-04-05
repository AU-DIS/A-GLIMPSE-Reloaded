from util import generate_queries, makeTrainingTestSplit, calculateAccuracyAndTotals
from glimpse.src.user import generate_synthetic_queries_by_topic
import topics as t
import numpy as np
import logging
from glimpse.src.glimpse import GLIMPSE, Summary
import time
import pandas as pd
import glimpseonline as g


# Set to t.topics by default (The big motherfucker list of topics)
# But this variable can be used to inject custom ones
topics = t.topics
# This experiment will use synthetic queries
# Set the algorithm to be used by this experiment
query_generation_algorithm = generate_synthetic_queries_by_topic
number_of_rounds = 300
# This variable determines if the GLIMPSE summary will be recomputed during the experiment
# Set to 0 in order to let the summary be static throughout the experiment
recompute_every_n = 0
# You don't have to set number of queries to a factor of number of rounds
# But it does more the code more clean
number_of_queries = number_of_rounds * 1000
# Number of unique topic entities.
n_topic_mids = int(number_of_queries/5)
# The k and epsilon parameters
k_pct = 0.01
e = 0.01
from importlib import reload


def round_experiment(kg):
    reload(g)

    print("Generating queries")
    queries = np.array_split(generate_queries(
        kg, topics, number_of_queries, n_topic_mids, algorithm=query_generation_algorithm), number_of_rounds)
    print("Finished with queries")

    k = k_pct*kg.number_of_triples()

    logging.info("KG entities: " + str(kg.number_of_entities()))
    logging.info("KG triples: " + str(kg.number_of_triples_))

    logging.info("Running for K=" + str(k) + ", e=" + str(e))

    rows = []
    # The first round will be hardcoded
    # //TODO: Add ability to recompute
    kg.reset()
    logging.info("  Running GLIMPSE on round 0")   
    print("Round 0")
    t1 = time.time()
    #summary = GLIMPSE(kg, k, queries[0], e)
    summary = Summary(kg)
    logging.info(" Done")
    t2 = time.time()
    mean_accuracy, total_entities, total_count = calculateAccuracyAndTotals(
        queries[0], summary)

    print("Initting glimpse online")
    glimpse_online = g.Online_GLIMPSE(kg, k)

    summary = glimpse_online.construct_summary()
    print("Constructed summary")
    mean_accuracy, total_entities, total_count = calculateAccuracyAndTotals(
        queries[0], summary)

    logging.info("      Summary  accuracy " + str(mean_accuracy))
    rows.append({'match': total_count, 'total': total_entities,
                 '%': mean_accuracy, 'runtime': t2 - t1})

    # The rest of the experiment will proceed in rounds, here the summary may or may not be recomputed
    for i in range(1, number_of_rounds):
        # Add together all queries from 0 to round i
        # Compute accuracy
        kg.reset()
        glimpse_online.update_queries(summary, queries[0], mean_accuracy)
        summary = glimpse_online.construct_summary()
    
        mean_accuracy, total_entities, total_count = calculateAccuracyAndTotals(
            np.concatenate(queries[0], axis=None), summary)
        
        print(f"Mean accuracy: {mean_accuracy} in round {i}")

        logging.info("      Summary  accuracy " + str(mean_accuracy))
        rows.append({'match': total_count, 'total': total_entities,
                     '%': mean_accuracy, 'runtime': t2 - t1})

    # //TODO: Add online version accuracy here

    pd.DataFrame(rows).to_csv("experiments_results/r" + "test" + "T#" + str(kg.number_of_triples()) + "_E#" + str(
        kg.number_of_entities()) + "K#" + str(int(k)) + "e#" + str(e) + ".csv")
