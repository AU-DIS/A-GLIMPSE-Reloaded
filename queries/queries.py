import math
from random import shuffle
import numpy as np
from human_id import generate_id


class Queries(object):
    def __init__(self, kg, mode='continuous', batch_size=1000, adversarial_degree=0.1):
        self.id = generate_id()
        self.mode = mode
        self.batch_size = batch_size
        self.adversarial_degree = adversarial_degree
        self.kg = kg

        self.has_yielded_set_ = set()

        sampled_triples = np.random.choice(
            range(kg.number_of_triples), batch_size, replace=True)

        sampled_triples = [kg.id_to_triple[x] for x in sampled_triples]
        self.internal_entities_ = []
        for (e1, r, e2) in sampled_triples:
            self.internal_entities_.extend([e1, e2])

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

    def bfs(self, entities, breadth=5):
        has_seen = set()
        candidates = set(entities)
        for i in range(breadth):
            round_candidates = set()
            for e1 in candidates:
                has_seen.add(e1)
                if e1 in self.kg.triples:
                    for r in self.kg.triples[e1]:
                        for e2 in self.kg.triples[e1][r]:
                            round_candidates.add(e2)
            candidates = candidates.union(round_candidates)
            candidates = candidates.difference(has_seen)
        has_seen = has_seen.union(candidates)
        return list(has_seen)

    def generate_queries(self, number_of_queries):
        candidates = [self.internal_entities_[0]]
        while len(candidates) < number_of_queries:
            recycled = np.random.choice(
                range(len(self.internal_entities_)), int(number_of_queries * (1 - self.adversarial_degree)), replace=True)
            recycled = [self.internal_entities_[i] for i in recycled]
            bfs_of_recycled = self.bfs(recycled)
            candidates.extend(bfs_of_recycled)
            adversarial_entities = np.random.choice(
                range(self.kg.number_of_entities), int(self.adversarial_degree * number_of_queries))
            candidates.extend(adversarial_entities)

        return np.random.choice(candidates, number_of_queries, replace=True)
