# 1d_ca_ga.py
import numpy as np
import random

# ==========================================
# 1. 1D Cellular Automata Engine
# ==========================================

def dec_to_binary(n):
    """Converts an integer to an 8-bit list."""
    return [int(x) for x in format(n, '08b')]

def binary_to_dec(bits):
    """Converts an 8-bit list to an integer."""
    return int("".join(str(x) for x in bits), 2)

def step_ca_1d(state, rule_bits):
    """Executes one step of a 1D Cellular Automaton."""
    new_state = np.zeros_like(state)
    n = len(state)
    for i in range(n):
        # Get left, center, right neighbors (with wrap-around/periodic boundaries)
        left = state[(i-1) % n]
        center = state[i]
        right = state[(i+1) % n]
        
        # The 3-bit neighborhood determines the index in the ruleset
        index = (left << 2) | (center << 1) | right
        
        # Lookup the rule for this index
        new_state[i] = rule_bits[7 - index] 
    return new_state

def run_ca_1d(rule_bits, initial_state, steps):
    """Runs the 1D CA for a given number of steps and returns the final state."""
    state = initial_state.copy()
    for _ in range(steps):
        state = step_ca_1d(state, rule_bits)
    return state

# ==========================================
# 2. Genetic Algorithm Components
# ==========================================

def calculate_fitness_1d(rule_bits, initial_state, target_state, steps):
    """Fitness is the number of matching cells between the CA's final state and the target."""
    final_state = run_ca_1d(rule_bits, initial_state, steps)
    matches = np.sum(final_state == target_state)
    return matches

def tournament_selection(population, fitnesses, k=3):
    """Selects the best individual from a random tournament of size k."""
    tournament_indices = random.sample(range(len(population)), k)
    best_idx = max(tournament_indices, key=lambda i: fitnesses[i])
    return population[best_idx]

def crossover(parent1, parent2):
    """Single-point crossover between two 8-bit rulesets."""
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

# ==========================================
# 3. Main Evolution Loop
# ==========================================

def evolve_ca_1d():
    # GA Parameters
    POP_SIZE = 50
    GENERATIONS = 100
    MUTATION_RATE = 0.05
    
    # CA Parameters
    CA_WIDTH = 50
    CA_STEPS = 20
    
    # Create a random initial state
    initial_state = np.random.randint(0, 2, CA_WIDTH)
    
    # Target: A perfect checkerboard pattern [1,0,1,0,1,0...]
    target_state = np.array([1 if i % 2 == 0 else 0 for i in range(CA_WIDTH)])
    
    # Initialize Population (random 8-bit rules)
    population = [dec_to_binary(random.randint(0, 255)) for _ in range(POP_SIZE)]
    
    best_fitness_overall = 0
    best_rule_overall = None
    
    print("--- 1D Cellular Automata Genetic Algorithm ---")
    print(f"Target Pattern: {''.join(map(str, target_state))}")
    print(f"Initial State:  {''.join(map(str, initial_state))}\n")
    
    for generation in range(GENERATIONS):
        # 1. Evaluate Fitness
        fitnesses = []
        for rule_bits in population:
            fit = calculate_fitness_1d(rule_bits, initial_state, target_state, CA_STEPS)
            fitnesses.append(fit)
            
        # Track the best
        current_best_fit = max(fitnesses)
        current_best_idx = fitnesses.index(current_best_fit)
        if current_best_fit > best_fitness_overall:
            best_fitness_overall = current_best_fit
            best_rule_overall = population[current_best_idx]
            
        # Print progress
        if generation % 10 == 0 or best_fitness_overall == CA_WIDTH:
            rule_num = binary_to_dec(best_rule_overall)
            print(f"Gen {generation:3d} | Best Fitness: {best_fitness_overall}/{CA_WIDTH} | Rule: {rule_num} ({best_rule_overall})")
            
        # Stop early if perfect solution is found
        if best_fitness_overall == CA_WIDTH:
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

    # Final Visualization
    print("\n--- Final Results ---")
    print(f"Best Rule Number: {binary_to_dec(best_rule_overall)}")
    final_state = run_ca_1d(best_rule_overall, initial_state, CA_STEPS)
    print(f"Target State:  {''.join(map(str, target_state))}")
    print(f"Initial State: {''.join(map(str, initial_state))}")
    print(f"Final State:   {''.join(map(str, final_state.astype(int)))}")

if __name__ == "__main__":
    evolve_ca_1d()