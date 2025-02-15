import os
import gzip
import re

import numpy as np
import logging

from collections import defaultdict
from scipy.sparse import csr_matrix
import bz2
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


from .algorithms import query_vector, random_walk_with_restart, query_vector_rdf

# TODO: Replace these data directories with your own paths
FREEBASE_DATA_DIR = '/x/tsafavi/data/WebQSDP/data/'
YAGO_DATA_DIR = '/x/tsafavi/data/yago3/'
DBPEDIA_DATA_DIR = '/home/kasper/Reps/A-GLIMPSE-Reloaded/dbpedia39'


def save_kg(kg, filename):
    with open(f"graphs/{filename}", 'wb') as f:
        pickle.dump(kg, f, -1)


def load_kg(filename):
    with open(f"graphs/{filename}", 'rb') as input:
        return pickle.load(input)


class KnowledgeGraph(object):

    def __init__(self):
        """A KG is a set of entities E, a set of relationships R,
        and a set of triples E x R x E."""
        self.number_of_relationships = 0
        self.relationship_to_id = {}
        self.id_to_relationship = {}

        self.number_of_entities = 0
        self.entity_to_id = {}
        self.id_to_entity = {}

        self.number_of_triples = 0
        self.triples = {}
        self.id_to_triple = {}
        self.triple_to_id = {}

    def name(self):
        return self.name_

    def compress_graph_indices(self):
        current_entity = 0
        current_relationship = 0
        current_triple = 0

        entity_to_new = {}
        relationship_to_new = {}
        triple_to_new = {}

        # create ids

        for entity in self.id_to_entity:
            entity_to_new[entity] = current_entity
            current_entity += 1

        for relationship in self.id_to_relationship:
            relationship_to_new[relationship] = current_relationship
            current_relationship += 1

        for triple in self.id_to_triple:
            triple_to_new[triple] = current_triple
            current_triple += 1

        self.number_of_triples = current_triple
        self.number_of_entities = current_entity
        self.number_of_relationships = current_relationship

        # Fix maps
        map = {}
        for relationship in self.relationship_to_id:
            map[relationship] = relationship_to_new[self.relationship_to_id[relationship]]
        self.relationship_to_id = map

        map = {}
        for relationship in self.id_to_relationship:
            map[relationship_to_new[relationship]
                ] = self.id_to_relationship[relationship]
        self.id_to_relationship = map

        for e in self.entity_to_id:
            self.entity_to_id[e] = entity_to_new[self.entity_to_id[e]]
        map = {}
        for e in self.id_to_entity:
            map[entity_to_new[e]] = self.id_to_entity[e]
        self.id_to_entity = map

        map = {}
        for t in self.triple_to_id:
            e1, r, e2 = t
            e1 = entity_to_new[e1]
            e2 = entity_to_new[e2]
            r = relationship_to_new[r]
            map[(e1, r, e2)] = triple_to_new[self.triple_to_id[t]]
        self.triple_to_id = map

        map = {}
        for t in self.id_to_triple:
            e1, r, e2 = self.id_to_triple[t]
            e1 = entity_to_new[e1]
            e2 = entity_to_new[e2]
            r = relationship_to_new[r]
            map[triple_to_new[t]] = (e1, r, e2)
        self.id_to_triple = map

    def entities(self):
        """
        :return entities: all entities in the KG
        """
        return self.id_to_entity.keys()

    def relationships(self):
        """
        :return relationships: all relations in the KG
        """
        return self.id_to_relationship.keys()

    def triples(self):
        """
        :return triples: {(e1, r, e2) triples}

        Note that this method is linear in the number of triples
        in the KG because it has to create a flat set of triples.
        """
        for e1 in self.triples:
            for r in self.triples[e1]:
                for e2 in self.triples[e1][r]:
                    yield (e1, r, e2)

    def number_of_entities(self):
        """
        :return n_entities: number of entities in the KG
        """
        return self.number_of_entities

    def number_of_relationships(self):
        """
        :return n_relations: number of relations in the KG
        """
        return self.number_of_relationships

    def has_entity(self, entity):
        """
        :param entity: str
        :return has_entity: True if KG contains this entity
        """
        return (entity in self.entity_to_id or entity in self.id_to_entity)

    def has_relationship(self, relationship):
        """
        :param relationship: str
        :return has_relationship: True if KG contains this relationship
        """
        return (relationship in self.id_to_relationship or relationship in self.relationship_to_id)

    def __getitem__(self, entity):
        """
        :param entity: str
        :return d: dict of set of {relation : entities}
        """
        return self.triples[entity]

    def __contains__(self, entity):
        """
        :param entity: str
        :return has_entity: True if KG contains this head entity
        """
        return entity in self.triples

    def has_triple(self, triple):
        """
        :param triple: (e1, r, e2)
        :return has_triple: True if KG contains this triple
        """
        e1, r, e2 = triple
        if e1 in self.entity_to_id and e2 in self.entity_to_id and r in self.relationship_to_id:
            e1_index = self.entity_to_id[e1]
            r_index = self.relationship_to_id[r]
            e2_index = self.entity_to_id[e2]
            return e1_index in self.triples and r_index in self.triples[e1_index] and e2_index == self.triples[e1_index][r_index]
        else:
            return False

    def add_triple(self, triple):
        e1, r, e2 = triple
        if not self.has_triple(triple):
            

            # Record new relationship
            if r not in self.relationship_to_id:
                self.relationship_to_id[r] = self.number_of_relationships
                self.id_to_relationship[self.number_of_relationships] = r
                self.number_of_relationships += 1

            # Record new entities
            for entity in [e1, e2]:
                if entity not in self.entity_to_id:
                    self.entity_to_id[entity] = self.number_of_entities
                    self.id_to_entity[self.number_of_entities] = entity
                    self.number_of_entities += 1

            r_index = self.relationship_to_id[r]
            e1_index = self.entity_to_id[e1]
            e2_index = self.entity_to_id[e2]

            if e1_index not in self.triples:
                self.triples[e1_index] = {}

            if r_index not in self.triples[e1_index]:
                self.triples[e1_index][r_index] = set()

            self.triples[e1_index][r_index].add(e2_index)

            self.id_to_triple[self.number_of_triples] = (
                e1_index, r_index, e2_index)
            self.triple_to_id[(e1_index, r_index, e2_index)
                              ] = self.number_of_triples
            self.number_of_triples += 1

    def entity_id(self, entity):
        """
        :param entity: str label
        :return eid: entity integer ID
        """
        return self.entity_id_[entity]

    def id_entity(self, eid):
        """
        :param eid: entity integer ID
        :return entity: str label
        """
        return self.id_to_entity[eid]

    def csr_matrix(self):
        """
        :return A: scipy sparse CSR adjacency matrix
        """
        row, col, data = [], [], []
        data = [0 for _ in range(self.number_of_entities)]
        col = [e for e in range(self.number_of_entities)]
        row = [e for e in range(self.number_of_entities)]
        for e1 in self.triples:
            for r in self.triples[e1]:
                for _ in self.triples[e1][r]:
                    data[e1] = 1

        n = self.number_of_entities
        m = csr_matrix(
            (np.array(data), (np.array(row), np.array(col))),
            shape=(n, n))
        return m

    def transition_matrix(self):
        """
        :return A: scipy CSR column-stochastic transition matrix
        """
        # Create the degree matrix
        row, col, data = [], [], []
        for e1 in self.triples:
            for r in self.triples[e1]:
                for _ in self.triples[e1][r]:
                    row.append(e1)
                    col.append(e1)
                    data.append(1)

        n = self.number_of_entities
        D = csr_matrix(
            (1 / np.array(data), (np.array(row), np.array(col))),
            shape=(n, n))
        return self.csr_matrix().transpose() * D

    def reset(self):
        """Sets all values to 0"""
        self.entity_value_ = defaultdict(float)
        self.triple_value_ = defaultdict(float)

    def entity_value(self, entity):
        """
        :param entity: str
        :return value: entity float value
        """
        return self.entity_value_[entity]

    def triple_value(self, triple):
        """
        :param triple: (e1, r, e2) triple
        :return value: triple float value
        """
        return self.triple_value_[triple]

    def model_user_pref(self, query_log, power=1, rdf_query_logs=False, include_relationship_prob=False):
        """
        :param query_log: list of queries as dicts
        :param power: number of terms in Taylor expansion
        """
        self.reset()

        # Perform random walk on the KG
        if rdf_query_logs:
            x, y = query_vector_rdf(self, query_log)
            y = y if include_relationship_prob else np.ones(
                self.number_of_relationships)
        else:
            x, y = query_vector(self, query_log), np.ones(
                self.number_of_relationships)

        M = self.transition_matrix()
        x = random_walk_with_restart(M, x, power=power)

        # Store entity and triple values
        for eid, val in enumerate(x):
            entity = eid
            self.entity_value_[entity] = np.log(val + 1)

        for e1 in self.triples:
            for r in self.triples[e1]:
                for e2 in self.triples[e1][r]:
                    triple = (e1, r, e2)
                    eid1, eid2 = e1, e2
                    r_id = r
                    # Y alway 1 in this setup
                    self.triple_value_[triple] = np.log(
                        x[eid1] * x[eid2] + 1)
                    #self.triple_value_[triple] = np.log(
                    #    x[eid1] * y[r_id] * x[eid2] + 1)

    def query_dir(self):
        raise NotImplementedError

    def topic_dir(self):
        raise NotImplementedError

    def mid_dir(self):
        raise NotImplementedError

    def topics(self):
        raise NotImplementedError

    def topic_mids(self):
        raise NotImplementedError

    def entity_names(self):
        raise NotImplementedError


class Freebase(KnowledgeGraph):

    def __init__(self, rdf_gz='webqsp-filtered-relations-freebase-rdfs.gz',
                 entity_names='all_entities.tsv', query_dir='queries/',
                 topic_dir='by-topic/', mid_dir='by-mid/'):
        """
        :param rdf_gz: filename of Freebase dump
        :param entity_names: mapping from MIDs to labels
        :param query_dir: directory where queries are saved as json
        :param topic_dir: directory where lists of query IDs by topic are stored
        :param mid_dir: directory where lists of query IDs by MID are stored
        """
        super().__init__()
        self.name_ = 'Freebase'

        self.rdf_gz_ = os.path.join(FREEBASE_DATA_DIR, rdf_gz)
        self.entity_names_ = os.path.join(FREEBASE_DATA_DIR, entity_names)
        self.query_dir_ = os.path.join(FREEBASE_DATA_DIR, query_dir)
        self.topic_dir_ = os.path.join(FREEBASE_DATA_DIR, topic_dir)
        self.mid_dir_ = os.path.join(FREEBASE_DATA_DIR, mid_dir)

    def has_fb_prefix(self, s):
        return s.startswith('<f_')

    def is_entity(self, s):
        return s.startswith('m.') or s.startswith('g.') or \
            s.startswith('<f_m.') or s.startswith('<f_g.')

    def strip_prefix(self, s):
        return s[3:-1]

    def query_dir(self):
        return self.query_dir_

    def topic_dir(self):
        return self.topic_dir_

    def mid_dir(self):
        return self.mid_dir_

    def topics(self):
        return [fname.split('.')[0] for fname in os.listdir(self.topic_dir_)]

    def topic_mids(self):
        return [fname[:-5] for fname in os.listdir(self.mid_dir_)]

    def entity_names(self):
        entity_names = {}
        with open(self.entity_names_, 'r') as f:
            next(f)
            for line in f:
                mid, name = line.rstrip().split('\t')
                entity_names[mid] = name
        return entity_names

    def load(self, head=None, strip=True):
        with gzip.open(self.rdf_gz_, 'rt') as f:
            for line in f:
                fact = tuple(line.rstrip().split('\t')[:-1])
                e1, r = fact[:2]
                e2 = ' '.join(fact[2:])

                if strip:
                    e1 = self.strip_prefix(
                        e1) if self.has_fb_prefix(e1) else e1
                    e2 = self.strip_prefix(
                        e2) if self.has_fb_prefix(e2) else e2
                    r = self.strip_prefix(r) if self.has_fb_prefix(r) else r

                triple = (e1, r, e2)
                self.add_triple(triple)

                if self.number_of_triples == head:
                    return


class YAGO(KnowledgeGraph):

    def __init__(self, rdf_gz='yagoFacts.gz', query_dir='queries/', mid_dir='by-mid/'):
        """
        :param rdf_gz: YAGO dump
        :param query_dir: directory where queries are saved as json
        :param mid_dir: directory where lists of query IDs by MID are stored
        """
        super().__init__()
        self.name_ = 'YAGO'

        self.rdf_gz_ = os.path.join(YAGO_DATA_DIR, rdf_gz)
        self.query_dir_ = os.path.join(YAGO_DATA_DIR, query_dir)
        self.mid_dir_ = os.path.join(YAGO_DATA_DIR, mid_dir)

    def is_entity(self, s):
        """Only use YAGO file with entities, no values"""
        return True

    def strip(self, s):
        return re.sub(r'([^\s\w]|)+', '', s)

    def query_dir(self):
        return self.query_dir_

    def mid_dir(self):
        return self.mid_dir_

    def topic_mids(self):
        return [fname[:-5] for fname in os.listdir(self.mid_dir_)]

    def entity_names(self):
        return {entity: entity for entity in self.entities()}

    def load(self, head=None, strip=True):
        with gzip.open(self.rdf_gz_, 'rt') as f:
            for line in f:
                fact = tuple(line.rstrip().split('\t')[:-1])
                e1, r = fact[:2]
                e2 = ' '.join(fact[2:])

                if strip:
                    e1 = self.strip(e1)
                    e2 = self.strip(e2)
                    r = self.strip(r)

                if not e1 or not e2:
                    continue

                triple = (e1, r, e2)
                self.add_triple(triple)

                if self.number_of_triples == head:
                    return


class DBPedia(KnowledgeGraph):

    def __init__(self, rdf_gz='./dbpedia39/rdf_gz', query_dir='query_dir/', mid_dir='by-mid/', include_properties=False):
        """
        :param rdf_gz: folder for data
        :param query_dir: directory where queries are saved as json
        :param mid_dir: directory where lists of query IDs by MID are stored
        """
        super().__init__()
        self.include_properties = include_properties
        self.name_ = 'DBPedia'

        self.rdf_gz_ = rdf_gz
        self.query_dir_ = os.path.join(DBPEDIA_DATA_DIR, query_dir)
        self.mid_dir_ = os.path.join(DBPEDIA_DATA_DIR, mid_dir)

    def is_entity(self, s):
        return s in self.entity_id_

    def query_dir(self):
        return self.query_dir_

    def mid_dir(self):
        return self.mid_dir_

    def topic_mids(self):
        return [fname[:-5] for fname in os.listdir(self.query_dir_)]

    def entity_names(self):
        return {entity: entity for entity in self.entities()}

    def get_triples(self, triple_indices):
        # //TODO: This is an ugly way of indexing triples
        t = []
        for i, triple in enumerate(self.triples):
            if i in triple_indices:
                t.append(triple)
        return t

    def load(self, head=None, strip=True):
        files = [f for f in os.listdir(self.rdf_gz_)]
        for f in files:
            with bz2.open(self.rdf_gz_ + "/" + f, 'rt') as f:
                for line in f:
                    if line.startswith("#"):
                        continue

                    fact = line.rstrip('\n')[:-2].split(' ')
                    e1, r = fact[:2]
                    e2 = ' '.join(fact[2:])

                    if (not e1 or not e2):
                        continue

                    # remove property values
                    if not self.include_properties:
                        if not (e1.startswith("<") and e1.endswith(">") and e2.startswith("<") and e2.endswith(">")):
                            continue

                    triple = (e1, r, e2)
                    self.add_triple(triple)

                    if self.number_of_triples == head:
                        return

                print("Loaded: " + str(f))
