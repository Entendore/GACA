# 2d_ca_ga.py
import numpy as np
import random

# ==========================================
# 1. 2D Cellular Automata Engine
# ==========================================

def get_neighbor_sum(grid):
    """Calculates the sum of the 8 Moore neighbors for every cell in the 2D grid."""
    neighbor_sum = (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) +
        np.roll(grid, (1, 1), axis=(0, 1)) + np.roll(grid, (-1, -1), axis=(0, 1)) +
        np.roll(grid, (1, -1), axis=(0, 1)) + np.roll(grid, (-1, 1), axis=(0, 1))
    )
    return neighbor_sum

def step_ca_2d(grid, rule_bits):
    """Executes one step of a 2D Outer Totalistic CA."""
    neighbor_sum = get_neighbor_sum(grid)
    new_grid = np.zeros_like(grid)
    
    # Apply Birth rules (for currently dead cells)
    dead_mask = (grid == 0)
    for n in range(9):
        new_grid[(dead_mask) & (neighbor_sum == n)] = rule_bits[n]
        
    # Apply Survival rules (for currently live cells)
    live_mask = (grid == 1)
    for n in range(9):
        new_grid[(live_mask) & (neighbor_sum == n)] = rule_bits[9 + n]
        
    return new_grid

def run_ca_2d(rule_bits, initial_grid, steps):
    """Runs the 2D CA for a given number of steps."""
    grid = initial_grid.copy()
    for _ in range(steps):
        grid = step_ca_2d(grid, rule_bits)
    return grid

# ==========================================
# 2. Genetic Algorithm Components
# ==========================================

def calculate_fitness_2d(rule_bits, initial_grid, target_grid, steps):
    """Fitness is the number of matching cells between the CA's final state and the target."""
    final_grid = run_ca_2d(rule_bits, initial_grid, steps)
    matches = np.sum(final_grid == target_grid)
    return matches

def tournament_selection(population, fitnesses, k=3):
    """Selects the best individual from a random tournament of size k."""
    tournament_indices = random.sample(range(len(population)), k)
    best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
    return population[best_idx]

def crossover(parent1, parent2):
    """Single-point crossover between two 18-bit rulesets."""
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(bits, mutation_rate):
    """Bit-flip mutation."""
    for i in range(len(bits)):
        if random.random() < mutation_rate:
            bits[i] = 1 - bits[i]
    return bits

def rule_to_string(rule_bits):
    """Converts the 18-bit genome into a readable B/S (Birth/Survival) string like B3/S23"""
    birth = [str(i) for i in range(9) if rule_bits[i] == 1]
    survive = [str(i) for i in range(9) if rule_bits[i+9] == 1]
    return f"B{''.join(birth)}/S{''.join(survive)}"

def grid_to_ascii(grid):
    """Converts a 2D numpy grid to an ASCII string for console printing."""
    return "\n".join("".join('█' if cell else ' ' for cell in row) for row in grid)

# ==========================================
# 3. Main Evolution Loop
# ==========================================

def evolve_ca_2d():
    # GA Parameters
    POP_SIZE = 50
    GENERATIONS = 50
    MUTATION_RATE = 0.05
    
    # CA Parameters
    GRID_SIZE = 15       # 15x15 grid
    CA_STEPS = 10        # Number of generations the CA runs before evaluation
    
    # Create a random initial state
    initial_grid = np.random.randint(0, 2, (GRID_SIZE, GRID_SIZE))
    
    # Target: A solid block of alive cells in the center
    target_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    target_grid[3:12, 3:12] = 1 
    
    # Initialize Population (random 18-bit rules)
    population = [[random.randint(0, 1) for _ in range(18)] for _ in range(POP_SIZE)]
    
    best_fitness_overall = 0
    best_rule_overall = None
    max_fitness = GRID_SIZE * GRID_SIZE
    
    print("--- 2D Cellular Automata Genetic Algorithm ---")
    print(f"Target Grid Size: {GRID_SIZE}x{GRID_SIZE} ({max_fitness} cells)")
    print(f"Target: A solid square block in the center.\n")
    
    for generation in range(GENERATIONS):
        # 1. Evaluate Fitness
        fitnesses = []
        for rule_bits in population:
            fit = calculate_fitness_2d(rule_bits, initial_grid, target_grid, CA_STEPS)
            fitnesses.append(fit)
            
        # Track the best
        current_best_fit = max(fitnesses)
        current_best_idx = fitnesses.index(current_best_fit)
        if current_best_fit > best_fitness_overall:
            best_fitness_overall = current_best_fit
            best_rule_overall = population[current_best_idx].copy()
            
        # Print progress
        if generation % 5 == 0 or best_fitness_overall == max_fitness:
            rule_str = rule_to_string(best_rule_overall)
            print(f"Gen {generation:3d} | Best Fitness: {best_fitness_overall:3d}/{max_fitness} | Rule: {rule_str}")
            
        # Stop early if perfect solution is found
        if best_fitness_overall == max_fitness:
            print("\nPerfect solution found!")
            break
            
        # 2. Create Next Generation
        new_population = []
        new_population.append(population[current_best_idx]) # Elitism
        
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, MUTATION_RATE))
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(child2, MUTATION_RATE))
                
        population = new_population

    # --- Final Visualization ---
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"Best Rule String: {rule_to_string(best_rule_overall)}")
    
    final_grid = run_ca_2d(best_rule_overall, initial_grid, CA_STEPS)
    
    print("\nInitial Grid (Gen 0):")
    print(grid_to_ascii(initial_grid))
    
    print(f"\nFinal Grid (After {CA_STEPS} CA steps):")
    print(grid_to_ascii(final_grid))
    
    print("\nTarget Grid:")
    print(grid_to_ascii(target_grid))

if __name__ == "__main__":
    evolve_ca_2d()