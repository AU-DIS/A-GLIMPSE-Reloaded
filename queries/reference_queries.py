from glimpse.src.query import generate_query, answer_query
import math
from random import shuffle
import numpy as np
from human_id import generate_id
import random


class Reference_Queries(object):
    def __init__(self, kg, mode='continuous', batch_size=1000, adversarial_degree=0.1):
        self.id = generate_id()
        self.mode = mode
        self.batch_size = batch_size
        self.adversarial_degree = adversarial_degree
        self.kg = kg

        self.has_yielded_set_ = set()

        self.base_topics = np.random.choice(
            range(kg.number_of_triples), int(adversarial_degree * kg.number_of_triples), replace=True)
        self.base_topics = [kg.id_to_triple[x] for x in self.base_topics]

        self.internal_entities_ = []
        for (e1, r, e2) in self.base_topics:
            self.internal_entities_.extend([e1, e2])

        self.base_topics = self.internal_entities_.copy()

        self.internal_entities_.extend(self.generate_queries(
            1000 * 10))

        shuffle(self.internal_entities_)

        self.iteration_count_ = 1

    def __contains__(self, entity):
        return entity in self.has_yielded_set_

    def all_batches(self):
        return self.internal_entities_[:self.iteration_count_]

    def batch(self, k):
        entities = []
        while len(entities) < k:
            if self.iteration_count_ >= len(self.internal_entities_):
                self.internal_entities_.extend(self.generate_queries(
                    self.batch_size * 10))
            entities.append(self.internal_entities_[self.iteration_count_])
            self.iteration_count_ += 1

        for e in entities:
            self.has_yielded_set_.add(e)

        return entities

    def reset(self):
        self.has_yielded_set_ = set()
        self.iteration_count_ = 0

    def __next__(self):
        if (self.iteration_count_ % self.batch_size) != 0:
            i = self.iteration_count_
            self.iteration_count_ += 1
            e = self.internal_entities_[i]
            print(e)
            self.has_yielded_set_.add(e)

            # Make 2 additional batches
            if self.iteration_count_ > len(self.internal_entities_):
                self.internal_entities_.extend(self.generate_queries(
                    1000 * 2, self.adversarial_degree))

            return e
        else:
            self.iteration_count_ += 1
            raise StopIteration()

    def __iter__(self):
        return self

    def generate_queries(self, number_of_queries):
        candidates = []

        while len(candidates) < number_of_queries:
            base_topics = np.random.choice(
                range(len(self.base_topics)), int((number_of_queries * (1 - self.adversarial_degree))), replace=True)

            base_topics = set([self.base_topics[i] for i in base_topics])

            adversarial_topics = set(np.random.choice(
                range(self.kg.number_of_entities), int(self.adversarial_degree * self.kg.number_of_triples), replace=True))

            topics = base_topics.union(adversarial_topics)

            for topic in topics:
                for a in answer_query(self.kg, generate_query(
                        self.kg, topic, entity_names=self.kg.id_to_entity, chain_len=10)):
                    candidates.append(a)

        return candidates
