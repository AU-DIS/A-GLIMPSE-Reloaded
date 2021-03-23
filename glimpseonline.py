import logging
from glimpse.src.glimpse import Summary, SummaryMethod
import numpy as np
from importlib import reload
import gc as garbage
import bandits.efficient_bandits.exp3 as e


class Online_GLIMPSE(object):
    def __init__(self, kg, K):
        garbage.collect()
        reload(e)
        # //TODO: I hate having to store this reference for memory overhead reasons
        # But the triples function is not deterministic due to KGs use of set to
        # Return them
        self.KG = kg
        self.K = K
        self.number_of_triples = kg.number_of_triples()
        self.bandit = e.exp3_efficient_bandit(kg)
        self.choices = set()

    def construct_summary(self):
        s = Summary(self.KG)

        # If we have fewer triples than K, we must select them all
        if self.number_of_triples <= self.K:
            s.fill(self.KG.triples(), self.K)

        else:
            self.choices = self.bandit.choose_k(self.K)
            s.fill((self.choose_entity_triples(self.choices)), self.K)

        return s

    def choose_entity_triples(self, entity_indices):
        triples = []
        for i in entity_indices:
            e1 = self.KG.entities_list_[i]
            for r in self.KG[e1]:
                for e2 in self.KG[e1][r]:
                    if len(triples) < self.K:
                        triples.append((e1, r, e2))
                    else:
                        return triples
        return triples

    def update_queries(self, queries):
        self.bandit.create_rewards(queries, self.choices)

        # rewards, choice_indices = self.bandit.create_rewards(
        # queries, self.choices)
        #self.bandit.give_reward(rewards, choice_indices)
        #self.choices = set()

    def init_bandit_weights(self, queries):
        self.bandit.create_initialisation_rewards(queries, self.KG)
