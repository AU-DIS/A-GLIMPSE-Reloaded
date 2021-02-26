import logging
from glimpse.src.glimpse import Summary, SummaryMethod
import exp3 as e
import numpy as np
from importlib import reload
import gc as garbage

class Online_GLIMPSE(object):
    def __init__(self, kg, K):
        reload(e)
        garbage.collect()
        # //TODO: I hate having to store this reference for memory overhead reasons
        # But the triples function is not deterministic due to KGs use of set to
        # Return them
        print("Initting")
        self.KG = kg
        self.K = K
        self.number_of_triples = kg.number_of_triples()
        self.bandit = e.recursive_exp3(0, self.number_of_triples)
        print("Finished making bandits")

    def construct_summary(self):
        s = Summary(self.KG)

        # If we have fewer triples than K, we must select them all
        if self.number_of_triples <= self.K:
            s.fill(self.KG.triples(), self.K)

        else:
            s.fill(self.KG.get_triples(self.bandit.make_choices(self.K)), self.K)

        return s

    def update_queries(self, summary, queries, acc):        
        self.bandit.bandit.give_reward(acc)
        for bandit in self.bandit.recursive_bandits:
            bandit.give_reward(acc)
