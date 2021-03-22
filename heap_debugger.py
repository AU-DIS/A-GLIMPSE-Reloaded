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




def run_heap_bandit(kg):
    

