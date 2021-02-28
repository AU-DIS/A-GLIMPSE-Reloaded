import math
import random
import numpy as np
import random

# Adapted from Jeremy Kuns implementation
# https://github.com/j2kun/exp3/blob/main/exp3.py


class exp3_bandit(object):
    def __init__(self, choices):
        self.weights = [1.0] * len(choices)
        self.reward_min = 0
        self.reward_max = 1
        self.round = 0
        self.choices = choices
        self.gamma = 0.07

    def choose_triple(self):
        self.distribution = distr(self.weights)
        self.choice = draw(self.distribution)
        # We return index of choice, the choice and the round
        return self.choice, self.choices[self.choice], self.round

    def choose_triples(self, k):
        triples = set()
        while len(triples) <= k:
            i, c, _ = self.choose_triple()
            self.round = self.round + 1
            if c not in triples:
                triples.add((i, c))
        return triples

    # IMPORTANT, WE GIVE A VECTOR OF REWARDS, WHERE EACH ENTRY EQUALS A REWARD FOR A CHOICE
    # WE CHOOSE K CHOICES IN A DIFFERENT FUNCTION, SO WE MUST GIVE K REWARDS
    def give_reward(self, rewards, choice_indices):
        global reward_max, reward_min
        for i in range(len(rewards)):
            scaled_reward = (rewards[i] - self.reward_min) / \
                (self.reward_max - self.reward_min)

            estimated_reward = 1.0 * scaled_reward / \
                self.distribution[choice_indices[i]]
            self.weights[choice_indices[i]] *= math.exp(estimated_reward *
                                                        self.gamma / len(self.weights))


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
