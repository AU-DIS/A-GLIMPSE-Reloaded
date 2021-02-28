from efficient_bandits.exp3 import exp3_efficient_bandit
from reference_bandit import exp3_bandit
import numpy as np
from numpy.random import uniform
import time
import sys 

n = 47 * 100000
rounds = 100
k = 100000
reward = 1

data = np.round(uniform(0, 10, size=n), 1)
rewards = np.ones(k)

efficient = exp3_efficient_bandit(data)
reference = exp3_bandit(data)
    
bandit = efficient
print(sys.getsizeof(data))

avg_time_sample = 0
avg_time_update = 0
for i in range(1, rounds+1):
    t1 = time.time()
    triples = bandit.choose_triples(k)
    triples = np.array([x[0] for x in triples])
    t2 = time.time()
    avg_time_sample += t2 - t1
    t1 = time.time()
    bandit.give_reward(rewards, list(triples))
    t2 = time.time()
    avg_time_update += t2 - t1
    print(avg_time_sample * 1/i, avg_time_update * 1/i)


