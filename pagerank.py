# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 20:58:24 2021

@author: pault
"""

import numpy as np
import random
import networkx as nx
from time import time

def calc_M(adj_m):
    '''
    calculates the pagerank matrix from a given adjacency matrix
    
    input: adjacency matrix
    output: pagerank matrix as csr-matrix
    '''
    M = np.array(adj_m, dtype = float).T
    for i in range(len(M)):
        summed = M[:,i].sum()
        if summed == 0: continue
        M[:,i] *= 1/M[:,i].sum()
    return M

def pagerank(beta, M, epsilon, r = 0):
    '''
    calculates pagerank values
    epsilon can also be used as "counter" if integer
    
    
    input:
        r: personalization vector 
            if integer: r is set to default values
        beta: teleportation probability
        M: pagerank matrix (has to be csr)
        epsilon:
            epsilon % 1 = 0: counter for loops
            else: allowed error
    output: pagerank values
    '''
    assert(epsilon >= 0)
    assert(M.shape[0] == M.shape[1])
    if(isinstance(r, int)):
        r = np.array([1/M.shape[0]] *M.shape[0], dtype = float).T
    r_old = r
    while True:
        r_new = (M * beta).dot(r_old)
        D = r_new.sum()
        r_new = r_new + np.array([(1-D)/M.shape[0]] * M.shape[0]).transpose()
        if(epsilon % 1 == 0):
            if(epsilon == 0):
                break
            else:
                epsilon -= 1
        elif(np.linalg.norm(r_new - r_old, ord = 1) < epsilon): 
            break
        r_old = r_new
    return r_new

def generate(size, add_edges = 0):
    '''
    generates the simplest strongly connected aperiodic adjacency matrix I could think of
    example: size = 3, add_edges = 0
        0 1 0
        1 0 1
        1 1 0
    example[2][0] == 1 to ensure aperiodicity
    
    add edges just adds random, non-loop edges (-> off diagonal) 
    '''
    adj_m = np.zeros((size,size))
    adj_m[0][1] = 1
    adj_m[-1][-2] = 1
    adj_m[-1][0] = 1
    for i in range(1, len(adj_m) - 1):
        adj_m[i][i+1] = 1
        adj_m[i][i-1] = 1
    for i in range(add_edges):
        rand_1 = random.randint(0,size - 1)
        rand_2 = random.randint(0, size - 1)
        while(rand_1 == i or rand_2 == i): #no edge to itself
            rand_1 = random.randint(0, size - 1)
            rand_2 = random.randint(0, size - 1)
        adj_m[rand_1][rand_2] = 1
    return adj_m

def power_it(google_matrix, epsilon):
    #breakpoint()
    r_old = np.array([1/google_matrix.shape[0]] * google_matrix.shape[0], dtype = float)
    while True:
        r_new = google_matrix.dot(r_old)
        #r_new = np.dot(google_matrix, r)
        if(epsilon % 1 == 0):
            if(epsilon == 0):
                break
            else:
                epsilon -= 1
        elif(abs(r_new - r_old).sum() < epsilon): 
            break
        r_old = r_new
        
    return r_new

print("example from lecture:\n")
adj_m = np.matrix([[0,1,0,0,0],[1,0,1,0,0],[0,0,0,1,1],[0,0,0,0,0],[0,1,0,0,0]], dtype = float)
pr_matrix = calc_M(adj_m)
for i in range(10):
    print("iterations: ", i + 1 ," \n ", pagerank(r = 0, beta = 0.85, M = pr_matrix, epsilon = i))
    
print("example from lecture with varying beta \n")
for i in np.arange(0,1.1,0.1):
    print("beta = ", i ," \n ",pagerank(r = 0, beta = i, M = pr_matrix, epsilon = 0.01))
    
#lecture example with new personalization vector
r = np.array([0.9,0.02,0.02,0.02,0.02], dtype = float)
it_ = 100
print("lecture example with new personalization vector")
print("iterations: ", it_)
print("personalization vector: ", r, "\n")
for i in np.arange(0,1.1,0.1):
    #beta = 0 => outgoing edges always chosen randomly
    print("beta: ",i, "\n ",pagerank(r = r, beta = i, M = pr_matrix, epsilon = it_)) 

#lecture with networkx
lecture_graph = nx.DiGraph(adj_m)
pr_val = nx.pagerank(lecture_graph)
print("lecture example with networks:", "\n ",pr_val)


print("lecture example with network x and verifying alpha")
for i in np.arange(0,1.1,0.1):
    print("alpha: ", i, "\n ",nx.pagerank(G = lecture_graph, alpha = i))
#same results as our implementation           
    

#power iteration test
size = 10000
additional_edges = 5000
iterations = 30
tol = 0.01

gen_start = time()
test_m = generate(size = size, add_edges = additional_edges) 
gen_end = time()

gm_start = time()
google_m = calc_M(test_m)
gm_end = time()

pii_start = time()
powerit = power_it(google_m, epsilon = 30)
pii_end = time()

pri_start = time()
pager = pagerank(beta = 0.85, M = google_m, epsilon = 30)
pri_end = time()

pit_start = time()
powerit2 = power_it(google_m, epsilon = 0.01)
pit_end = time()

prt_start = time()
pager2 = power_it(google_m, epsilon = 0.01)
prt_end = time()

nx_start = time()
nxp = nx.pagerank(nx.DiGraph(test_m))
nx_end = time()



print("n =", size, "iterations =", iterations, "epsilon =", tol, "added edges =", additional_edges)
print("generation: ", gen_end - gen_start)
print("google_matrix calculation: ", gm_end - gm_start)
print("power-iteration with max_iter: ", pii_end - pii_start)
print("pagerank with max_iter: ", pri_end - pri_start)
print("power-iteration with epsilon: ", pit_end - pit_start)
print("pagerank with epsilon: ", prt_end - prt_start)



