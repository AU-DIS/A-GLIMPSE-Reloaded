import math
import random

# Adapted from Jeremy Kuns implementation
# https://github.com/j2kun/exp3/blob/main/exp3.py


class exp3_bandit(object):
    def __init__(self, number_of_triples, gamma=0.07, reward_min=0, reward_max=1):
        # //TODO: If k is large, we will be storing these variables which are the same
        # for all k instances an enourmous amount of times
        # a lot of these could be passed into the functions every single call in order to reduce
        # memory overhead.
        self.gamma = gamma
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.weights = [1.0] * number_of_triples
        self.k = number_of_triples
        self.round = 0
        self.distribution = []
        self.choice = 0

    def choose_triple(self):
        self.distribution = distr(self.weights, self.gamma)
        self.choice = draw(self.distribution)
        return self.choice, self.round

    # How much you get if you select this (adding) (difference in reward)
    # Marginal gain (phi is from eq 2)
    def give_reward(self, reward):
        scaled_reward = (reward - self.reward_min) / \
            (self.reward_max - self.reward_min)

        estimated_reward = 1.0 * scaled_reward / self.distribution[self.choice]
        self.weights[self.choice] *= math.exp(estimated_reward *
                                              self.gamma / self.k)


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
