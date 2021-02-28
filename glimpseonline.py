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
        print("Initting")
        self.KG = kg
        self.K = K
        self.number_of_triples = kg.number_of_triples()
        self.bandit = e.exp3_efficient_bandit(self.number_of_triples, kg)
        self.choices = set()

    def construct_summary(self):
        s = Summary(self.KG)

        # If we have fewer triples than K, we must select them all
        if self.number_of_triples <= self.K:
            s.fill(self.KG.triples(), self.K)

        else:
            self.choices = self.bandit.choose_triples(self.K)
            s.fill(np.array([x[1] for x in self.choices]), self.K)
        print(f"Filled {self.K} triples")

        return s

    def update_queries(self, summary, queries, acc):
        rewards, choice_indices = self.bandit.create_rewards(
            queries, self.choices)
        self.bandit.give_reward(rewards, choice_indices)
        self.choices = set()
