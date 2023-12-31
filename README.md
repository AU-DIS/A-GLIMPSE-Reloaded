# A-GLIMPSE 
This repository contains experimental setups to compare bandit approaches against Glimpse for online knowledgegraph summarisation.

# Setup

1. Install Python packages from requiriements.txt
2. Download dbpedia39 
    
    2.1 run `downLoadDBPedia39()` from **`util.py`** to download the requiried rdf files into folder **`dbpedia/rdf_gz/`**

    2.3 outcomment `downLoadDBPedia39()` and uncomment the graph load and save instead. Run it. 
    
    2.2 run **`subgraphs.py`** to create a simple smaller test graph for quick experimentation. Its generation can freely be modified.

    2.3 if any step fail. You may have to update the folder path to **`dbpedia/rdf_gz`** in both  **`glimpse/src/experiment_base.py`** and **`glimpse/src/base.py`** for `class DBPedia(KnowledgeGraph):`.

4. run **`main.py`** and check if it works! 

5. Plots are saved in **`replacements_results/`**. Beware that a lot of naming is legacy stats that may or may not mean anything.

# How, where and what to modify
There is 4 things of interest to modify.
* The methods
* The parameters
* The queries
* The graphs

**The Methods**

Bandits are saved ins **`bandits/effecients_bandits/`**.
The Exp3 variants are older implementations that I may update.
Changing bandits is currently not that flexible and has to be done manually in `run_compare_experiment()` line ~60 by changing between `bandit="exp3"` and `bandit="qbl"`. To make a new bandit simply create a new file and implement the same methods. Then add it in **`glimpseonline.py`** like the other bandits. 
> **Note** All of this method swapping can be automated. 

**The Parameters**

Bandit parameters are currently set individually and very unorganised. 

Most Experiment parameters can be found in **`main.py`**.
You can execute multiple experiments at once multi threaded (See below). For now use only `run_compares()` as other methods contain legacy.

`graph`: name of graph to load.\
`graph_size`: custom defined size of the graph (LEGACY when using `run_compares()`)\
`n`: number of timesteps.\
`ks`: target size of summary graph proportional to size of the full graph.\
`batch_size`: number of queries provided in each timestep.\
`query_generator`: name of query generator to use.

`ks, batch_size, query_Generator` is wrapped in a list, allowing to execute experiments with multiple settings at once.

In **`experiments/bandit_versus_glimpse.py`** the variable `number_of_aggregations` can be set to make an average result of each experiment. `number_of_aggregations=1` executes each experiment once. `number_of_aggregations=10` executes each experiment 10 times and report the average.

**The graph**

All experiments are based on dbpedia or subgraphs thereof.
subgraphs can be generated. See examples of how in **`subgraphs.py`**. Things to consider: to generate some more meaningfull graphs, it would be smart to utilize graph traversal, otherwise they become very disconected. Though that is fine in some cases e.g. special query loads or just for simple code experimentation. 



# Multithreaded experiments

To run multiple experiments in parallelle, simply uncomment
```Python
                    #p = Process(target=run_compare_experiment,
                    #            args=(graph, n, k, bs, rf, query_generator))
                    #processes.append(p)

    #for p in processes:
        #Too much parralell if splitting this
        #p.start()
        #p.join()
```
in `run_compares()` and remove the old `run_compare_experiment()` line above it.

