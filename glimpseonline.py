import logging

from numpy.core.numeric import indices
from glimpse.src.glimpse import Summary, SummaryMethod
import numpy as np
from importlib import reload
import gc as garbage
import bandits.efficient_bandits.exp3m as exp3m
import bandits.efficient_bandits.exp3 as exp3


class Online_GLIMPSE(object):
    def __init__(self, kg, K, model_path=None, initial_entities=None, gamma=0.07, bandit="exp3m", reward_function="kg"):
        garbage.collect()
        reload(exp3m)
        reload(exp3)
        self.KG = kg
        self.K = K
        self.number_of_triples = kg.number_of_triples
        self.reward_function = reward_function

        #self.bandit = e.exp3_efficient_bandit(kg, model_path, initial_entities, gamma)
        if bandit == "exp3m":
            self.bandit = exp3m.exp3_m(kg, model_path, initial_entities, gamma)
        else:
            self.bandit = exp3.exp3_efficient_bandit(
                kg, model_path, initial_entities, gamma)
        self.choices = list()

    def save_model(self, model_path):
        self.bandit.save_model(model_path)

    def construct_summary(self):
        s = Summary(self.KG)

        self.choices = self.bandit.choose_k(self.K)
        self.choices = self.indices_to_triples()
        s.fill(self.choices, self.K)
        return s

    def indices_to_triples(self):
        triple_choices = []
        for triple_index in self.choices:
            (e1_index, r_index,
             e2_index) = self.KG.id_to_triple[triple_index]
            triple_choices.append(
                (self.KG.id_to_entity[e1_index], self.KG.id_to_relationship[
                    r_index], self.KG.id_to_entity[e2_index]
                 ))
        return triple_choices

    def choose_entity_triples(self, entity_indices):
        triples = []
        for i in entity_indices:
            for r in self.KG[i]:
                for e2 in self.KG[i][r]:
                    if len(triples) < self.K:
                        triples.append((i, r, e2))
                    else:
                        return triples
        return triples

    def update_queries(self, queries):
        if self.reward_function == "kg":
            return self.bandit.create_rewards(queries, self.choices)
        else:
            return self.bandit.create_binary_rewards(queries, self.choices)
