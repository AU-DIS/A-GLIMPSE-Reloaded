import math
import random
import numpy as np
import random
from importlib import reload
import bandits.efficient_bandits.efficient_heap as heap
import logging

# Adapted from Jeremy Kuns implementation
# https://github.com/j2kun/exp3/blob/main/exp3.py


class exp3_efficient_bandit(object):
    def __init__(self, kg, model_path=None):
        reload(heap)
        self.number_of_triples = kg.number_of_triples_
        # np.random.uniform(0.01, 1, size=number_of_triples)
        self.reward_min = 0
        self.reward_max = 1
        self.round = 0
        if model_path is None:
            self.weights = np.full(self.number_of_triples,
                                   1/self.number_of_triples)
            self.distribution = heap.sumheap(self.weights)
        else:
            self.weights = np.load(model_path)
            self.distribution = heap.sumheap(self.weights)

        self.gamma = 0.07
        self.kg = kg
        # heap.check(self.distribution, 1)

    def save_model(self, model_path):
        np.save(model_path, self.weights)

    def choose_triple(self):
        self.choice = heap.hsample(self.distribution)
        # We return index of choice, the choice and the round
        return self.choice

    def choose_triples(self, k):
        triples = set()
        # print("Choosing triples")
        while len(triples) < k:
            c = heap.hsample(self.distribution)
            # c = random.sample(range(len(self.weights)), 1)[0]
            if c not in triples:
                triples.add(c)
        triples = set([(index, triple) for (index, triple)
                       in zip(triples, self.kg.get_triples(triples))])
        return triples

    def choose_k(self, k):
        entities = set()
        # logging.debug("Choosing triples")
        while len(entities) < k:
            c = heap.hsample(self.distribution)
            entities.add(c)

        return entities

    # IMPORTANT, WE GIVE A VECTOR OF REWARDS, WHERE EACH ENTRY EQUALS A REWARD FOR A CHOICE
    # WE CHOOSE K CHOICES IN A DIFFERENT FUNCTION, SO WE MUST GIVE K REWARDS

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
        # heap.check(self.distribution, 1)

    def create_rewards(self, queries, summary):
        queries_set = set()
        for qs in queries:
            for q in qs:
                queries_set.add(q)
        queries = queries_set

        for (e1, r, e2) in summary:
            choice_index = self.kg.triple_ids[e1][r][e2]
            reward = 0
            if e1 in queries:
                reward = 0.5
            if e2 in queries:
                reward = 0.5
            self.give_reward(reward, choice_index)

    def create_rewards_triples(self, queries, index_triple_set):
        # Substitute efficient lookup data structure here (For strings)
        rewards = []
        choice_indices = []
        # print(type(queries[0]))
        queries_set = set()
        for qs in queries:
            for q in qs:
                queries_set.add(q)
        queries = queries_set

        for (index, triple) in index_triple_set:
            acc_reward = 0
            (e1, _, e2) = triple
            if e1 in queries:
                acc_reward += 50
            if e2 in queries:
                acc_reward += 50
            rewards.append(acc_reward)
            choice_indices.append(index)
        return rewards, choice_indices

    def create_initialisation_rewards(self, queries, kg, k):
        queries = random.sample(queries, k)
        index_triple_set = set()
        for i, triple in enumerate(kg.triples()):
            (e1, _, e2) = triple
            if e1 or e2 in queries:
                index_triple_set.add((i, triple))

        print("Finished looping")
        rewards, choice_indices = self.create_rewards(
            queries, index_triple_set)
        self.give_reward(rewards, choice_indices)


# draw: [float] -> int
# pick an index from the given list of floats proportionally
# to the size of the entry (i.e. normalize to a probability
# distribution and draw according to the probabilities).
def draw(weights):
    choice = random.uniform(0, sum(weights))
    choiceIndex = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choiceIndex

        choiceIndex += 1


# distr: [float] -> (float)
# Normalize a list of floats to a probability distribution.  Gamma is an
# egalitarianism factor, which tempers the distribtuion toward being uniform as
# it grows from zero to one.
def distr(weights, gamma=0.0):
    theSum = float(sum(weights))
    return tuple((1.0 - gamma) * (w / theSum) + (gamma / len(weights)) for w in weights)
