from math import sin, pi
from matplotlib import pyplot as plt
from numpy.random import rand, randint
import numpy as np

# Genetic algorithm variables
bounds = [[-3, 12.1], [4.1, 5.8]]   # Lower and Upper bounds of x and y
generation = 50                     # Generation number (iterate main program 50 times)
chromosome_size = 5                 # number of chromosome_size for each variable
population_size = 40                # Size of Population
crossover_rate = 0.85               # Ratio of Crossover
mutation_rate = 0.01                # Ratio of Mutation

# Objective function has to be defined to solve with genetic algorithm.
# Objective function can be change if and only if it is an unconstrained problem.
# To solve different unconstrained problem you just need to update the objective function
# rest of the code can be kept the same as it is.


def objective_function(Input):
    x = Input[0]
    y = Input[1]
    Objective_min = 21.5 + x*sin(4*pi*x) + y*sin(20*pi*y)    # This fitness function can be changed.
    Objective_max = 1 / (1 + Objective_min)         # Convert the min to max problem
    # print("Objective_max", Objective_max)
    return Objective_max


def crossover(pop, crossover_rate):
    offspring = []
    for i in range(int(len(pop) / 2)):
        parent1 = pop[2 * i - 1].copy()     # parent 1
        # print("parent1", parent1)
        parent2 = pop[2 * i].copy()         # parent 2
        if rand() < crossover_rate:         # if random number is less than crossover rate (rand = 0 - 1)
            cut_point = randint(1, len(parent1) - 1, size=50)       # there will be two random cutting points
            while cut_point[0] == cut_point[1]:
                cut_point = randint(1, len(parent1) - 1, size=50)   # two random cutting points

            cut_point = sorted(cut_point)
            crossover1 = parent1[:cut_point[0]] + parent2[cut_point[0]:cut_point[1]] + parent1[cut_point[1]:]
            crossover2 = parent2[:cut_point[0]] + parent1[cut_point[0]:cut_point[1]] + parent2[cut_point[1]:]
            offspring.append(crossover1)
            offspring.append(crossover2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)
        # print("offspring", offspring)
        return offspring


def mutation(pop, mutation_rate):
    offspring = list()
    for i in range(int(len(pop))):
        parent1 = pop[i].copy()  # parent
        if rand() < mutation_rate:
            cut_point = randint(0, len(parent1))    # random gene
            crossover1 = parent1
            if crossover1[cut_point] == 1:
                crossover1[cut_point] = 0           # flip
            else:
                crossover1[cut_point] = 1
                offspring.append(crossover1)
        else:
            offspring.append(parent1)
    return offspring


# roulette wheel selection
def selection(pop, fitness, population_size):
    next_generation = list()
    elite = np.argmax(fitness)
    next_generation.append(pop[elite])  # keep the best
    Probability = [f / sum(fitness) for f in fitness]  # selection probability
    index = list(range(int(len(pop))))
    index_selected = np.random.choice(index, size=population_size - 1, replace=False, p=Probability)
    s = 0
    for ite in range(population_size - 1):
        next_generation.append(pop[index_selected[s]])
        s += 1
    return next_generation


def decoding(bounds, chromosome_size, chromosome):
    real_chromosome = list()
    for i in range(len(bounds)):
        st, en = i * chromosome_size, (i * chromosome_size) + chromosome_size    # extract the chromosome
        sub = chromosome[st:en]
        chars = ''.join([str(s) for s in sub])  # convert to chars
        integer = int(chars, 2)                 # convert to integer
        real_value = bounds[i][0] + (integer / (2 ** chromosome_size)) * (bounds[i][1] - bounds[i][0])
        real_chromosome.append(real_value)
        # print("real_value", real_value)
    return real_chromosome


# First population
pop = [randint(0, 2, chromosome_size * len(bounds)).tolist() for _ in range(population_size)]
print("First pop", pop)
# main program
best_fitness = []  # Define the best fitness as empty list
for gen in range(generation):
    offspring = crossover(pop, crossover_rate)      # Generate offspring as much as generation
    # print("Crossover offspring", offspring)
    offspring = mutation(offspring, mutation_rate)

    for j in offspring:
        pop.append(j)
        # print("Population", pop)
        real_chromosome = [decoding(bounds, chromosome_size, p) for p in pop]
        # print("Real Chromosome", real_chromosome)
        fitness = [objective_function(d) for d in real_chromosome]              # fitness value
        print("Fitness Value", fitness)
        index = np.argmax(fitness)
        current_best = pop[index]
        best_fitness.append(1 / min(fitness) - 1)
        pop = selection(pop, fitness, population_size)

print('Min objective function value:', min(best_fitness))
print("Optimal solution", decoding(bounds, chromosome_size, pop[0]))

fig = plt.figure()
plt.plot(best_fitness)
fig.suptitle('Evolution of the best chromosome')
plt.xlabel('Number of generation')
plt.ylabel('Objective function value')
plt.show()
