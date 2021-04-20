import math
import random
import numpy as np
import random
from importlib import reload
import bandits.efficient_bandits.efficient_heap as heap
import logging


class exp3_m(object):
    def __init__(self, kg, model_path=None, initial_entities=None, gamma=0.07):
        self.number_of_triples = kg.number_of_triples_
        self.reward_min = 0
        self.reward_max = 1
        self.round = 0
        self.gamma = gamma
        self.S_0 = set()
        self.kg = kg
        self.probabilities = []

        if model_path is not None:
            self.weights = np.load(model_path)
        elif initial_entities is None:
            self.weights = np.ones(self.number_of_triples)
        else:
            self.weights = np.ones(self.number_of_triples)

    def depround(self, probabilities):
        one_probs = set()
        candidates = set(range(len(probabilities)))

        # We assume that all probabilities initally are 0 < p < 1
        while len(candidates) > 1:
            i = candidates.pop()
            j = candidates.pop()

            alpha = min(1 - probabilities[i], probabilities[j])
            beta = min(probabilities[i], 1 - probabilities[j])

            threshold = np.random.uniform(0, 1, 1)
            if threshold > (beta/(alpha+beta)):
                probabilities[i] = probabilities[i] + alpha
                probabilities[j] = probabilities[j] - alpha
            else:
                probabilities[i] = probabilities[i] - beta
                probabilities[j] = probabilities[j] + beta

            # Put back into pool or element has been chosen
            if probabilities[i] == 1:
                one_probs.add(i)
            elif probabilities[i] > 0:
                candidates.add(i)

            if probabilities[j] == 1:
                one_probs.add(j)
            elif probabilities[j] > 0:
                candidates.add(j)

        return np.array(list(one_probs))

    def choose_k(self, k):
        max_j = np.argmax(self.weights)
        K = self.number_of_triples
        self.S_0 = set()
        # Step 1
        sorted_weight_indices = np.argsort(self.weights)[::-1]
        if self.weights[max_j] >= (1/k - self.gamma/K) * (np.sum(self.weights)/(1-self.gamma)):
            rhs = (1/k - self.gamma/K)/(1 - self.gamma)
            alpha_t = 0
            # Find alpha_t
            for i, index in enumerate(sorted_weight_indices):
                x = i
                y = np.sum(self.weights[sorted_weight_indices[i:]])
                alpha_t_candidate = -(y * rhs)/(x*rhs - 1)
                self.S_0.add(index)
                if alpha_t_candidate == rhs:
                    alpha_t = alpha_t_candidate
                    break
        # Step 2
        W = set(sorted_weight_indices)
        weights_prime = np.zeros(K)
        for i in W.difference(self.S_0):
            weights_prime[i] = self.weights[i]
        for i in self.S_0:
            weights_prime[i] = alpha_t
        # Step 3
        w_prime_sum = np.sum(weights_prime)
        gamma_factor = (1 - self.gamma)
        gamma_term = self.gamma/K
        self.probabilities = np.array([
            k * (gamma_factor * w_i/w_prime_sum + gamma_term)
            for w_i in weights_prime
        ])

        # Step 4
        choices = self.depround(self.probabilities)
        # print(choices)
        return choices

    def create_rewards(self, queries, summary):
        queries_set = set()
        for qs in queries:
            for q in qs:
                queries_set.add(q)
        queries = queries_set

        k = len(summary)
        regrets = []

        for (e1, r, e2) in summary:
            e1 = self.kg.entities_[e1]
            e2 = self.kg.entities_[e2]
            r = self.kg.relationships_[r]

            choice_index = self.kg.triple_ids[e1][r][e2]
            reward = 0
            if e1 in queries:
                reward = 0.5
            if e2 in queries:
                reward = 0.5
            self.give_reward(reward, choice_index, k)
            regrets.append(self.reward_max-reward)
        return regrets

    def give_reward(self, reward, i, k):
        x_hat = reward/self.probabilities[i]
        if i not in self.S_0:
            self.weights[i] = self.weights[i] * \
                math.exp(k * self.gamma * x_hat/(self.number_of_triples))
