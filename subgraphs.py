from glimpse.src.experiment_base import DBPedia, KnowledgeGraph, Freebase, save_kg, load_kg
import random
import numpy as np


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
    kg.compress_graph_indices()

    save_kg(kg, output_graph)


def random_induced_by_vertices(input_graph, output_graph, vertex_budget):
    kg = load_kg(input_graph)

    vertices = np.random.choice(
        range(kg.number_of_entities), vertex_budget, replace=False)

    del kg
    kg = DBPedia('dbpedia39')

    for e1 in vertices:
        for r in kg.triples[e1]:
            for e2 in kg.triples[e1][r]:
                kg.add_triple((e1, r, e2))
    kg.compress_graph_indices()
    save_kg(kg, output_graph)


def random_induced_by_size_and_ratio(input_graph, output_graph, number_vertices, number_edges):
    kg = load_kg(input_graph)
    pass
