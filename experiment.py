from os import path
from human_id import generate_id
import human_id
from queries.queries import Queries
from queries.reference_queries import Reference_Queries
from glimpse.src.experiment_base import DBPedia, load_kg
import os
import time
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class Experiment(object):
    def __init__(self, type_string="bandit", comment="", adversarial_degree=0.01, graph=None, dir=None, name=None, batch_size=100, query_generator="proprietary"):
        if dir is None:
            self.id = generate_id()
        else:
            self.id = dir
        if graph is None:
            self.kg_ = DBPedia('dbpedia39')
            self.kg_.load()
        else:
            self.kg_ = load_kg(graph)
        self.type_tring = type_string
        if query_generator == "proprietary":
            self.Q_ = Queries(
                self.kg_, adversarial_degree=adversarial_degree, batch_size=batch_size)
        elif query_generator == "reference":
            self.Q_ = Reference_Queries(
                self.kg_, adversarial_degree=adversarial_degree, batch_size=batch_size
            )

        self.path_ = f"replacement_results/{self.id}"
        self.write_buffers_ = {}
        self.write_every_ = 1000**2
        self.files_ = {}
        self.name = name

        # Write a new directory for all the results
        if not path.exists(self.path_):
            os.mkdir(self.path_)
        with open(f"{self.path_}/information.txt", 'w+') as f:
            f.write(f"""
                Time of initialisation: {time.localtime(time.time())}
                Type: {type_string}
                Comment: {comment}
            """)

    def update_graph(self, graph):
        self.kg_ = load_kg(graph)

    def reset_name(self, new_name=None):
        if new_name is None:
            self.name = generate_id()
        else:
            self.name = new_name

    def begin_experiment(self, experiment_id):
        self.Q_.reset()
        pass

    def end_experiment(self, experiment_id):
        self.empty_write_buffer(experiment_id)
        self.Q_.reset()
        with open(f"{self.path_}/information.txt", 'a') as f:
            f.write(f"Experiment {experiment_id} ended at {time.time()}\n")

        with open(f"{self.path_}/experiment_picked", 'wb') as f:
            pickle.dump(self, f, -1)

    def create_experiment(self, list_of_properties, file_annotation, comment):
        if self.name is None:
            experiment_id = generate_id()
        else:
            experiment_id = self.name

        file_name = f"{self.path_}/{file_annotation}.csv"

        with open(file_name, 'w+') as f:
            f.write(f"#{comment}\n")
            f.write(",".join(list_of_properties)+'\n')

        self.write_buffers_[experiment_id] = []
        self.files_[experiment_id] = file_name
        return experiment_id

    def empty_write_buffer(self, experiment_id):
        with open(self.files_[experiment_id], 'a') as f:
            for s in self.write_buffers_[experiment_id]:
                f.write(s+'\n')
        self.write_buffers_[experiment_id] = []

    def add_experiment_results(self, experiment_id, results):
        results = [str(x) for x in results]
        self.write_buffers_[experiment_id].append(','.join(results))
        if len(self.write_buffers_[experiment_id]) > self.write_every_:
            self.empty_write_buffer(experiment_id)

    def kg(self):
        return self.kg_

    def batch_size(self):
        return self.Q_.batch_size

    def batch(self, k):
        return self.Q_.batch(k)

    # Expose queries so experiments don't have to deal with an extra object
    def all_batches(self):
        return self.Q_.all_batches()
