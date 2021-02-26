import math
import random
import numpy as np
import random

# Adapted from Jeremy Kuns implementation
# https://github.com/j2kun/exp3/blob/main/exp3.py


gamma = 0.07
reward_min = 0
reward_max = 1
split_percent = 10
split_constant = 0
number_of_splits = 2


class exp3_bandit(object):
    def __init__(self, number_of_triples, choices):
        self.weights = [1.0] * number_of_triples
        self.k = number_of_triples
        self.reward_min = 0
        self.reward_max = 1
        self.round = 0
        self.distribution = [1.0] * number_of_triples
        self.choice = random.randint(0, number_of_triples)
        self.choices = choices
        self.gamma = 0.07
        self.leaf = False

    def choose_triple(self):
        self.distribution = distr(self.weights, self.gamma)
        self.choice = draw(self.distribution)
        return self.choice, self.choices[self.choice], self.round

    # How much you get if you select this (adding) (difference in reward)
    # Marginal gain (phi is from eq 2)
    def give_reward(self, reward):
        global reward_max, reward_min
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


class recursive_exp3(object):
    def __init__(self, choice_start, choice_end):
        global split_constant
        self.choice_start = choice_start
        self.choice_end = choice_end
        #split_constant = int((choice_end-choice_start)/split_percent)
        print(split_constant)
        split_constant = int(math.sqrt(choice_end - choice_start))

        self.recursive_bandits = []

        # Leaf decisions
        choices = np.array_split(
            np.arange(choice_start, choice_end, step=1), split_constant)

        print(len(choices[0]))
        for i in range(split_constant):
            self.recursive_bandits.append(
                exp3_bandit(len(choices[i]), choices[i])
            )

        self.bandit = exp3_bandit(split_constant, self.recursive_bandits)

    def make_choices(self, k):
        choices = set()
        while len(choices) <= k:
            _, b, _ = self.bandit.choose_triple()
            _, t, _ = b.choose_triple()
            choices.add(t)
        return choices

    def give_reward(self):
        pass