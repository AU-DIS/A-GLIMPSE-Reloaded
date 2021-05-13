import math
import numpy as np
from human_id import generate_id


class Queries(object):
    def __init__(self, kg, mode='continuous', batch_size=1000, adversarial_degree=0.01):
        self.id = generate_id()
        self.mode = mode
        self.batch_size = batch_size
        self.adversarial_degree = adversarial_degree
        self.kg = kg

        number_to_sample = math.ceil(
            adversarial_degree * kg.number_of_entities)

        self.initial_entities = np.random.choice(
            list(self.kg.triples.keys()), number_to_sample)

        self.has_yielded_set_ = set()
        self.internal_entities_ = self.generate_queries(
            1000 * 10)

        self.iteration_count_ = 1

    def __contains__(self, entity):
        return entity in self.has_yielded_set_

    def all_batches(self):
        return self.internal_entities_[:self.iteration_count_]

    def batch(self):
        i = self.iteration_count_
        self.iteration_count_ += self.batch_size
        entities = self.internal_entities_[i:i+self.batch_size]
        for e in entities:
            self.has_yielded_set_.add(e)

        # Make 2 additional batches
        if self.iteration_count_ > len(self.internal_entities_):
            self.internal_entities_ = np.concatenate((self.internal_entities_, self.generate_queries(
                1000 * 10)))
            return self.batch()
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
                self.internal_entities_ = np.concatenate((self.internal_entities_, self.generate_queries(
                    1000 * 2, self.adversarial_degree)))

            return e
        else:
            self.iteration_count_ += 1
            raise StopIteration()

    def __iter__(self):
        return self

    def non_unique_bfs_(self, start, breadth=5):
        entities = np.array([start])

        for _ in range(breadth):
            for e1 in entities:
                if e1 in self.kg.triples.keys():
                    for r in self.kg.triples[e1]:
                        entities = np.append(
                            entities, list(self.kg.triples[e1][r]))

            #seen_entities = seen_entities.union(current_entities)
        return np.array(entities)

    def generate_queries(self, number_of_queries):
        n = self.kg.number_of_entities

        chosen_entities = np.array([])

        while len(chosen_entities) < number_of_queries:
            for e in self.initial_entities:
                chosen_entities = np.concatenate(
                    (self.non_unique_bfs_(e), chosen_entities))
        chosen_entities = chosen_entities.astype('int')

        chosen_entities = np.random.choice(chosen_entities, number_of_queries)
        return chosen_entities
