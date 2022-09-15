# Binary-Genetic-Algorithm
The genetic algorithm is a method for solving both constrained and unconstrained optimization problems that is based on natural selection, the process that drives biological evolution. The genetic algorithm repeatedly modifies a population of individual solutions. At each step, the genetic algorithm selects individuals from the current population to be parents and uses them to produce the children for the next generation. Over successive generations, the population "evolves" toward an optimal solution. You can apply the genetic algorithm to solve a variety of optimization problems that are not well suited for standard optimization algorithms, including problems in which the objective function is discontinuous, nondifferentiable, stochastic, or highly nonlinear. The genetic algorithm can address problems of mixed integer programming, where some components are restricted to be integer-valued.
![image](https://user-images.githubusercontent.com/80236488/190425290-0e31f786-4fd5-4766-ba40-be0ee0fcf25a.png)


The genetic algorithm uses three main types of rules at each step to create the next generation from the current population:

Selection rules select the individuals, called parents, that contribute to the population at the next generation. The selection is generally stochastic, and can depend on the individuals' scores.

Crossover rules combine two parents to form children for the next generation.

Mutation rules apply random changes to individual parents to form children.  Reference [https://www.mathworks.com/help/gads/what-is-the-genetic-algorithm.html]
