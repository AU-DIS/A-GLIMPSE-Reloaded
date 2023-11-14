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
    def __init__(self, kg, model_path=None, initial_entities=None, gamma=0.07):
        reload(heap)
        self.number_of_triples = kg.number_of_triples
        # np.random.uniform(0.01, 1, size=number_of_triples)
        self.reward_min = 0
        self.reward_max = 1
        self.round = 0
        self.debug_been_chosen = set()
        self.prob = np.full(self.number_of_triples,
                                   gamma/self.number_of_triples)
        if model_path is not None:
            self.weights = np.load(model_path)
            self.distribution = heap.sumheap(self.weights)
        elif initial_entities is None:
            self.weights = np.full(self.number_of_triples,
                                   1.0)
            self.distribution = heap.sumheap(self.weights)
        elif initial_entities is not None:
            priviliged_triples = set()
            for entity in initial_entities:
                if entity in kg.triples.keys():
                    for r in kg.triples[entity]:
                        for e2 in kg.triples[entity][r]:
                            priviliged_triples.add(
                                kg.triple_to_id[entity][r][e2])

            priviliged_weight = 1/len(priviliged_triples)
            self.weights = np.full(self.number_of_triples, (1 - priviliged_weight) /
                                   (self.number_of_triples - len(priviliged_triples)))

            # This extra step means we do not have to perform a divison every time
            # we update the weight of a priviliged triple
            priviliged_weight = priviliged_weight/len(priviliged_triples)
            for triple_id in priviliged_triples:
                self.weights[triple_id] = priviliged_weight
            self.distribution = heap.sumheap(self.weights)

        elif model_path is not None:
            self.weights = np.load(model_path)
            self.distribution = heap.sumheap(self.weights)

        self.gamma = gamma
        self.kg = kg
        # heap.check(self.distribution, 1)

    def save_model(self, model_path):
        np.save(model_path, self.weights)

    def choose_triple(self):
        self.choice = heap.hsample(self.distribution)
        # We return index of cqblhoice, the choice and the round
        return self.choice

    def choose_k(self, k, debug=False):
        entities = list()
        values = list()
        # logging.debug("Choosing triples")
        for _ in range(k):
            if random.uniform(0.0,1.0) < self.gamma:
                c = random.randrange(0, self.number_of_triples)
                v = self.weights[c]
            else:
                c, v = heap.hsample(self.distribution)
            self.prob[c] = ((1-self.gamma)*math.exp(self.distribution[self.number_of_triples + c]-self.distribution[1])+(self.gamma/self.number_of_triples))
            entities.append(c)
            values.append(v)
            heap.update(self.distribution, c, 0)
            if debug:
                self.debug_been_chosen.add(c)
        for e,v in zip(entities,values):
            heap.update(self.distribution, e, v)
        return list(entities)

    # IMPORTANT, WE GIVE A VECTOR OF REWARDS, WHERE EACH ENTRY EQUALS A REWARD FOR A CHOICE
    # WE CHOOSE K CHOICES IN A DIFFERENT FUNCTION, SO WE MUST GIVE K REWARDS

    def give_reward(self, reward, i):
        global reward_max, reward_min
        
        #scaled_reward = (reward - self.reward_min) / \
        #    (self.reward_max - self.reward_min)

        estimated_reward = reward / self.prob[i]

        # If using negative, extract original value for proability updates
        self.weights[i] = self.weights[i] + self.gamma*(estimated_reward*self.number_of_triples)

        heap.update(self.distribution, i, self.weights[i])
        # heap.check(self.distribution, 1)

    def create_rewards(self, queries, summary):
        queries_set = set()
        for q in queries:
            queries_set.add(q)
        queries = queries_set

        regrets = []

        for (e1, r, e2) in summary:
            e1 = self.kg.entity_to_id[e1]
            e2 = self.kg.entity_to_id[e2]
            r = self.kg.relationship_to_id[r]

            choice_index = self.kg.triple_to_id[(e1, r, e2)]
            reward = 0
            if e1 in queries:
                reward += 0.5
            if e2 in queries:
                reward += 0.5
            self.give_reward(reward, choice_index)
            regrets.append(reward)
        return regrets

    def create_binary_rewards(self, queries, summary):
        queries_set = set()
        for q in queries:
            queries_set.add(q)
        queries = queries_set

        regrets = []

        for (e1, r, e2) in summary:
            e1 = self.kg.entity_to_id[e1]
            e2 = self.kg.entity_to_id[e2]
            r = self.kg.relationship_to_id[r]

            choice_index = self.kg.triple_to_id[(e1, r, e2)]
            reward = 1 if (e1 in queries or e2 in queries) else 0
            self.give_reward(reward, choice_index)
            regrets.append(reward)
        return regrets

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
                acc_reward += 0.5
            if e2 in queries:
                acc_reward += 0.5
            rewards.append(acc_reward)
            choice_indices.append(index)
        return rewards, choice_indices


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
