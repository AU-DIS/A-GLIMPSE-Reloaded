from glimpse.src.experiment_base import DBPedia, KnowledgeGraph, Freebase, save_kg, load_kg
import random
import numpy as np


def random_induced_subgraph(input_graph, output_graph, vertex_budget, edge_budget):
    kg = load_kg(input_graph)
    print(len(kg.id_to_triple.keys()))
    triples = random.sample(list(kg.id_to_triple.keys()), edge_budget)
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


def random_induced_by_size_and_ratio(input_graph, output_graph, number_vertices):
    kg = load_kg(input_graph)

    top_indices = sorted(range(kg.number_of_entities), reverse=True,
                         key=lambda x: len(kg.triples[x].keys()) if x in kg.triples else 0)
    print(kg.number_of_triples, kg.number_of_entities)
    triples = []

    for e1 in top_indices[:number_vertices]:
        if e1 in kg.triples:
            for r in kg.triples[e1]:
                for e2 in kg.triples[e1][r]:
                    triples.append((e1, r, e2))
    del kg
    kg = DBPedia('dbpedia39')
    for triple in triples:
        kg.add_triple(triple)

    kg.compress_graph_indices()
    save_kg(kg, output_graph)


if __name__ == "__main__":
    random_induced_subgraph('dbpedia39', 'test_sub_graph', None, 1000)
