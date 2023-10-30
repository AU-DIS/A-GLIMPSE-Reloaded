# Follows reference implementation by Tim Vieria
# http://timvieira.github.io/blog/post/2016/11/21/heaps-for-incremental-computation/


import numpy as np
import time
from numpy.random import uniform
import math


random_samples = list(uniform(0, 1, 100000))


def update(S, k, v):
    "Update value position `k` in time O(log n)."
    
    i = (len(S))//2 + k 
    #print(k, i, len(S))
    S[i] = v
    while i > 0:
        i //= 2
        if 2*i+1 == len(S):
            S[i] = S[2*i]
        elif S[2*i] > S[2*i+1]:
            S[i] = S[2*i]+math.log(math.exp(S[2*i+1]-S[2*i])+1)
        else:
            S[i] = S[2*i+1]+math.log(math.exp(S[2*i]-S[2*i+1])+1)


def sumheap(w):
    "Create sumheap from weights `w` in O(n) time."
    np.random.seed(int(time.time()))
    n = len(w)
    d = int(2**np.ceil(np.log2(n)))  # number of intermediates
    S = np.zeros(2*n+1)                # intermediates + leaves
   #print(n, d)
    for i in range(0, n):
        update(S, i, w[i])

    #S[d:d+n] = w                   # store `w` at leaves.
    #for i in reversed(range(1, d)):
    #    if S[2*i] > S[2*i+1]:
    #        S[i] = S[2*i]+math.log(math.exp(S[2*i+1]-S[2*i])+1)
    #    else:
    #        S[i] = S[2*i+1]+math.log(math.exp(S[2*i]-S[2*i+1])+1)
    return S


def check(S, i):
    "Check heap invariant."
    d = len(S)
    if i >= d//2:   # only checks internal nodes.
        return
    if S[i] != S[2*i] + S[2*i + 1]:
        S[i] = S[2*i] + S[2*i + 1]
    check(S, 2*i)
    check(S, 2*i + 1)


def dump(S):
    "Print heap for debugging."
    for i in range(int(np.ceil(np.log2(len(S))))):
        print('depth', i, S[2**i:2**(i+1)])


def sample(w, u):
    "Ordinary sampling method, O(n) to build heap, O(log n) per sample after that."
    c = w.cumsum()
    return c.searchsorted(u * c[-1])


def hsample(S):
    global random_samples
    "Sample from sumheap, O(log n) per sample."
    offset = len(S)//2  # number of internal nodes.
    # random probe
    p = random_samples.pop()
    if len(random_samples) == 0:
        random_samples = list(uniform(0, 1, 100000))

    # Use binary search to find the index of the largest CDF (represented as a
    # heap) value that is less than a random probe.
    i = 1
    while i < offset:
        # Determine if the value is in the left or right subtree.
        i *= 2
        left = S[i]
        if p > math.exp(left-S[1]):
            # Value is in right subtree. Subtract mass under left subtree.
            p -= math.exp(left-S[1])
            i += 1
    #print(i, offset, len(S))
    return i - offset, S[i]


def main():
    for n in np.random.choice(range(1, 100), size=10):
        print(n)
        w = np.round(uniform(0, 1, size=n), 1)
        S = sumheap(w)
        check(S, 1)
        for _ in range(100):
            # test uses same random number because the methods should be identical up to ties.
            u = uniform()
            p1 = sample(w, u)
            p2 = hsample(S)
            # change a random value in the weight array
            c = np.random.randint(n)
            v = uniform(10)
            w[c] = v
            update(S, c, v)
            check(S, 1)


if __name__ == '__main__':
    main()
