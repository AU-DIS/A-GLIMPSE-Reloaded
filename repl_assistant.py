import os
from glimpse.src.experiment_base import DBPedia, KnowledgeGraph, Freebase
from glimpse.src.query import generate_query
from glimpse.src.glimpse import GLIMPSE, Summary
from util import makeTrainingTestSplit

kg = ""

def load_kg():
    global kg
    kg = DBPedia('dbpedia39')
    kg.load()