import os
from GLIMPSE_personalized_KGsummarization.src.experiment_base import DBPedia, KnowledgeGraph, Freebase
from GLIMPSE_personalized_KGsummarization.src.query import generate_query

#kg = DBPedia('DBPedia3.9/')
#kg.load()

#print("Loaded KG")
#input()
K = [10**-i for i in range(1, 6)]
E = [1e-2]

answer_version = "2"
version = "3"
e = 0.1
k = 0.01

os.system('python3 experiments.py'
                  + ' --method glimpse'
                  + ' --percent-triples ' + str(k)
                  + ' --version ' + version
                  + ' --version-answers ' + answer_version
                  + ' --epsilon ' + str(e)
#                  + ' > ' + 'GLIMPSE' + version+'a' + answer_version + ' #K' + str(k) + '#E' + str(e) + '.out'
#                  + ' &')
)