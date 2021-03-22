from util import generate_queries, makeTrainingTestSplit, calculateAccuracyAndTotals
from glimpse.src.user import generate_queries_by_topic
import topics as t
import numpy as np
import logging
from glimpse.src.glimpse import GLIMPSE, Summary
import time
import glimpseonline as g
from importlib import reload
from random import sample, randint
from bandits.efficient_bandits import exp3

glimpse_online = []


def run_heap_bandit(kg):
    reload(g)
    queries = generate_queries(kg, t.topics, 1000, int(1000/2))
    glimpse_online = g.Online_GLIMPSE(kg, 100)
    for i in range(4):
        summary = glimpse_online.construct_summary()
        print("Giving reward")
        glimpse_online.update_queries(summary, queries, 0)
