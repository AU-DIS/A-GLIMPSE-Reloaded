import logging
from glimpse.src.glimpse import Summary, SummaryMethod
from exp3 import exp3_bandit


class Online_GLIMPSE(object):
    def __init__(self, kg, K):
        # //TODO: I hate having to store this reference for memory overhead reasons
        # But the triples function is not deterministic due to KGs use of set to
        # Return them
        print("Initting")
        self.KG = kg
        self.K = K
        self.number_of_triples = kg.number_of_triples()
        self.bandits = [
            exp3_bandit(self.number_of_triples) for _ in range(int(K))
        ]

    def construct_summary(self):
        s = Summary(self.KG)

        # If we have fewer triples than K, we must select them all
        if self.number_of_triples <= self.K:
            s.fill(self.KG.triples(), self.K)

        triples_selected = set()

        # We have created K bandits, so looping over them gives K triples
        for bandit in self.bandits:
            choice = bandit.choose_triple()
            # If another bandit has chosen that triple, we simply ask it to choose over and over
            # //TODO: THIS DEVIATES FROM SPECIFICATION OF CHOOSING RANDOMLY?
            while choice in triples_selected:
                choice = bandit.choose_triple()
            triples_selected.add(choice)

        s.fill(self.KG.get_triples(triples_selected), self.K)
        return s

    def update_queries(self, queries):
        # //TODO: Implement rewards
        for bandit in self.bandits:
            bandit.give_reward(queries)



