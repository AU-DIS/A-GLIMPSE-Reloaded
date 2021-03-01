import math
import random
import numpy as np
import random
from .efficient_heap import hsample, sumheap, update

# Adapted from Jeremy Kuns implementation
# https://github.com/j2kun/exp3/blob/main/exp3.py


class exp3_efficient_bandit(object):
    def __init__(self, number_of_triples, kg):
        self.weights = [1.0] * number_of_triples
        #self.weights = np.random.uniform(0, 1, size=number_of_triples)
        self.reward_min = 0
        self.reward_max = 100
        self.round = 0
        self.distribution = sumheap(self.weights)
        self.gamma = 0.07
        self.kg = kg

    def choose_triple(self):
        self.choice = hsample(self.distribution)
        # We return index of choice, the choice and the round
        return self.choice

    def choose_triples(self, k):
        triples = set()
        while len(triples) <= k:
            c = self.choose_triple()
            if c not in triples:
                triples.add(c)
        triples = set([(index, triple) for (index, triple)
                       in zip(triples, self.kg.get_triples(triples))])
        return triples

    # IMPORTANT, WE GIVE A VECTOR OF REWARDS, WHERE EACH ENTRY EQUALS A REWARD FOR A CHOICE
    # WE CHOOSE K CHOICES IN A DIFFERENT FUNCTION, SO WE MUST GIVE K REWARDS
    def give_reward(self, rewards, choice_indices):
        global reward_max, reward_min
        for i in range(0, len(rewards)):
            scaled_reward = (rewards[i] - self.reward_min) / \
                (self.reward_max - self.reward_min)
            c = choice_indices[i]
            estimated_reward = 1.0 * scaled_reward / \
                (self.distribution[c]+0.0001)
            self.weights[c] *= math.exp(estimated_reward *
                                        self.gamma / len(self.weights))
            update(self.distribution, c, self.weights[c])

    def create_rewards(self, queries, index_triple_set):
        # Substitute efficient lookup data structure here (For strings)
        rewards = []
        choice_indices = []
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
