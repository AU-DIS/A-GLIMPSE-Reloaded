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
import bandits.efficient_bandits.efficient_heap as heap
import math
import random

glimpse_online = []


def run_heap_bandit(kg):
    reload(g)
    queries = generate_queries(kg, t.topics, 1000, int(1000/2))
    glimpse_online = g.Online_GLIMPSE(kg, 100)
    for i in range(4):
        summary = glimpse_online.construct_summary()
        print("Giving reward")
        glimpse_online.update_queries(summary, queries, 0)


class test_bandit(object):
    def __init__(self, n):
        reload(heap)
        self.number_of_entities = n
        self.weights = np.full(self.number_of_entities,
                               1/self.number_of_entities)
        # np.random.uniform(0.01, 1, size=number_of_triples)
        self.reward_min = 0
        self.reward_max = 1
        self.round = 0
        self.distribution = heap.sumheap(self.weights)
        self.gamma = 0.07
        # heap.check(self.distribution, 1)

    def choose_k(self, k):
        entities = set()
        #logging.debug("Choosing triples")
        while len(entities) < k:
            c = heap.hsample(self.distribution)
            entities.add(c)

        return entities

    def give_reward(self, reward, i):
        global reward_max, reward_min

        scaled_reward = (reward - self.reward_min) / \
            (self.reward_max - self.reward_min)

        offset = len(self.distribution)//2

        estimated_reward = 1.0 * scaled_reward / \
            (self.distribution[offset + i])


        # If using negative, extract original value for proability updates
        self.weights[i] = self.weights[i] * math.exp(estimated_reward *
                                                     self.gamma / len(self.weights))

        heap.update(self.distribution, i, self.weights[i])


def run_bandit():
    for i in range(3, 10):

        b = test_bandit(10**i)
        ground_truth = set(random.sample(range(int(10**i)), int(0.1 * 10**i)))
        accuracies = []

        for _ in range(0, 10000):
            choices = set(b.choose_k((int(0.01 * 10**i))))
            for choice in choices:
                if choice in ground_truth:
                    b.give_reward(1, choice)
                else:
                    b.give_reward(0, choice)

        accu = len(ground_truth.intersection(choices))
        print(
            f"Round {i} with last accuracy of {accu/(0.01 * 10**i)} with size {(0.01 * 10**i)}")
        accuracies.append(accu)
    print(accuracies)
