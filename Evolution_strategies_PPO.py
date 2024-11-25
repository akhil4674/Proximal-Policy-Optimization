import numpy as np

# The objective function we are trying to minimize.
# In this case, it's a simple quadratic function (sum of squares).
def objective_function(x):
    return np.sum(x**2)

# Function to mutate a solution by adding random noise.
# This helps the algorithm explore new potential solutions.
def mutate(parent, mutation_strength):
    # Adding random Gaussian noise to the parent solution to create a new solution.
    return parent + mutation_strength * np.random.randn(*parent.shape)

# Main Evolution Strategy algorithm (1+λ)-ES.
# This involves creating offspring from the best individual and selecting the best ones.
def evolution_strategy(objective_function, dim, population_size, mutation_strength, num_generations):
    # Initialize the population randomly. Each individual is a vector of `dim` values.
    population = np.random.randn(population_size, dim)
    
    for generation in range(num_generations):
        # Evaluate the fitness of each individual in the population.
        fitness = np.array([objective_function(ind) for ind in population])
        
        # Find the best individual (lowest fitness).
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        # Generate new candidates (offspring) by mutating the best individual.
        offspring = np.array([mutate(best_solution, mutation_strength) for _ in range(population_size)])
        
        # Evaluate the fitness of the offspring.
        offspring_fitness = np.array([objective_function(ind) for ind in offspring])
        
        # Combine the parents and offspring into one large pool.
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.hstack((fitness, offspring_fitness))
        
        # Sort the combined pool by fitness and select the best `population_size` individuals.
        best_indices = np.argsort(combined_fitness)[:population_size]
        population = combined_population[best_indices]

        # Print out the best fitness found so far in the current generation.
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")
    
    # After all generations, return the best solution found.
    return best_solution, best_fitness

# Parameters for the Evolution Strategy.
dim = 10  # The dimensionality of the solution space.
population_size = 50  # Number of individuals in each generation.
mutation_strength = 0.1  # How much variation to introduce during mutation.
num_generations = 100  # How many generations to run the algorithm for.

# Run the Evolution Strategy with the parameters.
best_solution, best_fitness = evolution_strategy(objective_function, dim, population_size, mutation_strength, num_generations)

# Print the best solution and its fitness at the end of the optimization process.
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)

