from matplotlib import pyplot as plt
from numpy.random import rand, randint
import numpy as np

# Genetic algorithm variables
bounds = [[-10, 10], [-10, 10]]
iteration = 25
bits = 10               # number of bits for each variable
population_size = 30    # Size of Population
crossover_rate = 0.85   # Ratio of Crossover
mutation_rate = 0.01    # Ratio of Mutation

# Objective function has to be defined to solve with genetic algorithm.
# Objective function can be change if and only if it is an unconstrained problem.
# To solve different unconstrained problem you just need to update the objective function
# rest of the code can be kept the same as it is.


def objective_function(Input):
    x = Input[0]
    y = Input[1]
    Objective_min = (x ** 2 + y ** 2) - 0.48 * x # This fitness function can be changed.
    Objective_max = 1 / (1 + Objective_min)  # Convert the min to max problem
    return Objective_max


def crossover(pop, crossover_rate):
    offspring = list()
    for i in range(int(len(pop) / 2)):
        parent1 = pop[2 * i - 1].copy()  # parent 1
        parent2 = pop[2 * i].copy()  # parent 2
        if rand() < crossover_rate:
            cutting_point = randint(1, len(parent1) - 1, size=50)  # two random cutting points
            while cutting_point[0] == cutting_point[1]:
                cutting_point = randint(1, len(parent1) - 1, size=50)  # two random cutting points

            cutting_point = sorted(cutting_point)
            c1 = parent1[:cutting_point[0]] + parent2[cutting_point[0]:cutting_point[1]] + parent1[cutting_point[1]:]
            c2 = parent2[:cutting_point[0]] + parent1[cutting_point[0]:cutting_point[1]] + parent2[cutting_point[1]:]
            offspring.append(c1)
            offspring.append(c2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

        return offspring


def mutation(pop, mutation_rate):
    offspring = list()
    for i in range(int(len(pop))):
        parent1 = pop[i].copy()  # parent
        if rand() < mutation_rate:
            cutting_point = randint(0, len(parent1))  # random gene
            c1 = parent1
            if c1[cutting_point] == 1:
                c1[cutting_point] = 0  # flip
            else:
                c1[cutting_point] = 1
                offspring.append(c1)
        else:
            offspring.append(parent1)

    return offspring


# roulette wheel selection
def selection(pop, fitness, population_size):
    next_generation = list()
    elite = np.argmax(fitness)
    next_generation.append(pop[elite])  # keep the best
    P = [f / sum(fitness) for f in fitness]  # selection probability
    index = list(range(int(len(pop))))
    index_selected = np.random.choice(index, size=population_size - 1, replace=False, p=P)
    s = 0
    for j in range(population_size - 1):
        next_generation.append(pop[index_selected[s]])
        s += 1
    return next_generation


def decoding(bounds, bits, chromosome):
    real_chromosome = list()
    for i in range(len(bounds)):
        st, en = i * bits, (i * bits) + bits  # extract the chromosome
        sub = chromosome[st:en]
        chars = ''.join([str(s) for s in sub])  # convert to chars
        integer = int(chars, 2)  # convert to integer
        real_value = bounds[i][0] + (integer / (2 ** bits)) * (bounds[i][1] - bounds[i][0])
        real_chromosome.append(real_value)
    return real_chromosome


# Initial population
pop = [randint(0, 2, bits * len(bounds)).tolist() for _ in range(population_size)]

# main program
best_fitness = []
for gen in range(iteration):
    offspring = crossover(pop, crossover_rate)
    offspring = mutation(offspring, mutation_rate)

    for s in offspring:
        pop.append(s)

        real_chromosome = [decoding(bounds, bits, p) for p in pop]
        fitness = [objective_function(d) for d in real_chromosome]  # fitness value

        index = np.argmax(fitness)
        current_best = pop[index]
        best_fitness.append(1 / min(fitness) - 1)
        pop = selection(pop, fitness, population_size)
print('Min objective function value:', min(best_fitness))
print("Optimal solution", decoding(bounds, bits, current_best))

fig = plt.figure()

plt.plot(best_fitness)
fig.suptitle('Evolution of the best chromosome')
plt.xlabel('Number of Iteration')
plt.ylabel('Objective function value')
plt.show()
