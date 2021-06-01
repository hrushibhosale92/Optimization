# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:26:43 2021

@author: hrushikesh.bhosale
"""

import numpy as np 


def RosenBrock(x1,x2):
   return ((1.0 - x1)**2 + (100.0 * (x2 - x1**2)**2))

def simple_funtion(x):
    return x**2

bit_size = 6
pop_size = 50
Range = (-5,5)
genration = 10

population = np.random.choice(2,size = (pop_size,bit_size))

def BinToDec(member):
    return int(''.join([str(i) for i in member]),2)
    

def DecodePop(p1):
    x = Range[0] + ((Range[1] -Range[0]) / (2**bit_size - 1)) * BinToDec(p1)
    return simple_funtion(x)

def tournament_sel(fitness,population):
    x = []
    for i in range(pop_size):
        a = np.random.randint(pop_size)
        b = np.random.randint(pop_size)
        if fitness[a] < fitness[b]:
            x.append(population[a])
        else:
            x.append(population[b])
    return x


def cross_over(population):
    x = []
    for i in range(pop_size//2):
        a = np.random.randint(pop_size)
        b = np.random.randint(pop_size) 
        
        if np.random.uniform() < 0.7:
            split_point = np.random.randint(1,bit_size)
            
            x.append(np.append(population[a][:split_point],population[b][split_point:]))
            x.append(np.append(population[b][:split_point],population[a][split_point:]))
        else:
            x.append(population[a])
            x.append(population[b])
    return x


def mutation(population):
    x= []   
    for i in range(pop_size):
        a = np.random.randint(pop_size)
        member = population[a]
        
        if np.random.uniform() > 0.1:
            x.append(member)
        else:
            mut_point = np.random.randint(bit_size)
            member[mut_point] = abs(member[mut_point] - 1)
            x.append(member)
    return x
        


for k in range(genration+1):
    
    fitness = [DecodePop(i) for i in population]
    print(k,min(fitness))
    if k >= genration :
        break
    tor_pop = tournament_sel(fitness,population)
    cross_pop = cross_over(tor_pop)
    mut_population = mutation(cross_pop)

    population = mut_population.copy()
    
optimal_ind = np.argmin(fitness)
optimal_solution = population[optimal_ind]

