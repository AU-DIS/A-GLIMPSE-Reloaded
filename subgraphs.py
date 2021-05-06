from glimpse.src.experiment_base import DBPedia, KnowledgeGraph, Freebase, save_kg, load_kg
import random
import numpy as np



#vertex_budget = int(1.4 * 10**5)
#edge_budget = int(10**6)


def random_induced_subgraph(input_graph, output_graph, vertex_budget, edge_budget):
    kg = load_kg(input_graph)

    triples = random.sample(kg.id_to_triple.keys(), edge_budget)
    triples = [kg.id_to_triple[triple] for triple in triples]

    triples_unindixed = [
        (kg.id_to_entity[e1], kg.id_to_relationship[r], kg.id_to_entity[e2]) for (e1, r, e2) in triples
    ]

    del kg
    kg = DBPedia('dbpedia39')
    for triple in triples_unindixed:
        kg.add_triple(triple)

    save_kg(kg, output_graph)

