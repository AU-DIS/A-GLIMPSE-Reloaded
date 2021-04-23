import math
import random
import numpy as np
import random
from importlib import reload
import bandits.efficient_bandits.efficient_heap as heap
import logging
import time


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
        probs = probabilities.copy()
        # Calling np.random.uniform in the loop doubles execution time, allocate ahead of time
        randoms = set(np.random.uniform(0, 1, len(candidates)))
        # We assume that all probabilities initally are 0 < p < 1
        while len(candidates) > 1:
            i = candidates.pop()
            j = candidates.pop()

            alpha = min(1 - probs[i], probs[j])
            beta = min(probs[i], 1 - probs[j])

            threshold = randoms.pop()

            if threshold > (beta/(alpha+beta)):
                probs[i] = probs[i] + alpha
                probs[j] = probs[j] - alpha
            else:
                probs[i] = probs[i] - beta
                probs[j] = probs[j] + beta

            # Put back into pool or element has been chosen
            if probs[i] == 1:
                one_probs.add(i)
            elif probs[i] > 0:
                candidates.add(i)

            if probs[j] == 1:
                one_probs.add(j)
            elif probs[j] > 0:
                candidates.add(j)

        return np.array(list(one_probs))

    def choose_k(self, k):
        t1 = time.time()
        max_j = np.argmax(self.weights)
        K = self.number_of_triples
        self.S_0 = set()
        s_0_candidates = set()
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
                s_0_candidates.add(index)
                if alpha_t_candidate == rhs:
                    alpha_t = alpha_t_candidate
                    self.S_0 = s_0_candidates
                    break
        # Step 2
        t2 = time.time()
        W = set(sorted_weight_indices)
        weights_prime = np.zeros(K)
        diff_indices = list(W.difference(self.S_0))
        weights_prime[diff_indices] = self.weights[diff_indices]
        if len(self.S_0) > 0:
            S0_indices = list(self.S_0)
            weights_prime[S0_indices] = alpha_t

        t3 = time.time()
        # for i in W.difference(self.S_0):
        #    weights_prime[i]= self.weights[i]
        # for i in self.S_0:
        #    weights_prime[i]= alpha_t
        # Step 3
        w_prime_sum = np.sum(weights_prime)
        gamma_factor = (1 - self.gamma)
        gamma_term = self.gamma/K

        self.probabilities = 1/w_prime_sum * weights_prime * gamma_factor
        self.probabilities = self.probabilities + gamma_term
        self.probabilities = self.probabilities * k

        t4 = time.time()
        # Step 4
        choices = self.depround(self.probabilities)
        # print(choices)
        print(
            f"Done with alphas {t2 - t1}, manage weights {t3 - t2}, construct probabilities {t4 - t3}")
        return choices

    def create_rewards(self, queries, summary):
        queries_set = set()
        for q in queries:
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
