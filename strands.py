import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
from copy import deepcopy

# https://www.nytimes.com/games-assets/strands/2024-06-20.json
THEME = "They're good for a laugh"
NUM_WORDS = 7
STARTING_BOARD = [
    "LLASMO",
    "IETTEC",
    "TETERS",
    "OUECAP",
    "ALAGMO",
    "XYETNO",
    "ANASTE",
    "LPDIOR",
]

PREFIXES = json.load(open("prefixes.json"))


def create_graph(board):
    G = nx.grid_2d_graph(8, 6)

    # Create diagonal edges
    edges = (
        (node, (node[0] + dx, node[1] + dy))
        for node in G.nodes
        for dx, dy in [(1, 1), (-1, -1), (-1, 1), (1, -1)]
        if 0 <= node[0] + dx < 8 and 0 <= node[1] + dy < 6
    )

    G.add_edges_from(edges)

    attrs = dict()
    for node in G.nodes:
        attrs[node] = board[node[0]][node[1]].lower()
    nx.set_node_attributes(G, attrs, "letter")

    nx.draw(G, labels=nx.get_node_attributes(G, "letter"))
    plt.savefig("graph.png")
    return G


def search_words(graph: nx.Graph, length=18):
    all_paths = set()

    def dfs(curr_node, path=[], word=""):
        path.append(curr_node)
        word += graph.nodes[curr_node]["letter"]

        if word in PREFIXES:
            if 4 <= len(path) <= length and PREFIXES[word] == 0:
                all_paths.add(tuple(path))
            if len(path) != length:
                for neighbor in graph.neighbors(curr_node):
                    if neighbor not in path:
                        dfs(neighbor, path, word)
        # Backtrack: Remove the current node from the path
        path.pop()

    for node in graph.nodes():
        dfs(node)

    return all_paths


# words = dict()
# Note: there may be more than one math for the same word
# for path in paths:
#     words[("".join(g.nodes[n]["letter"] for n in path))] = path
# for k, v in dict(sorted(words.items())).items():
#     print(k, v)


g = create_graph(STARTING_BOARD)
paths = search_words(g)

# Initialize Y and X dictionaries
Y = dict()
X = dict()

for path in paths:
    Y[path] = list()
for node in g.nodes():
    for path in paths:
        if node in path:
            Y[path].append(node)

for node in g.nodes():
    X[node] = set()
for path in paths:
    for node in path:
        X[node].add(path)

solution = []


# https://www.cs.mcgill.ca/~aassaf9/python/algorithm_x.html
def solve():
    if not X:  # Solution found
        return True
    # Choose node with smallest number of overlapping paths
    c = min(X, key=lambda k: len(X[k]))
    if not X[c] or len(solution) > NUM_WORDS:  # No solution
        return False

    for r in sorted(X[c], key=len, reverse=True):
        solution.append(r)
        cols = select(r)
        if solve():
            return True
        # Backtrack: remove the path from the solution add covered paths and nodes back
        deselect(r, cols)
        solution.pop()


def select(r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


solve()
test = []
for s in solution:
    for n in s:
        test.append(n)
    print("".join([g.nodes[n]["letter"] for n in s]))
print(test)

# Encode each solution as a binary vector of length n
# def initialize_population(pop_size, num_sets):
#     return np.random.randint(2, size=(pop_size, num_sets))


# # Takes 25 ms to compute... too slow
# def fitness(solution, H, K):
#     selected_sets = [H[i] for i in range(len(solution)) if solution[i] == 1]
#     union_of_selected = set.union(set(selected_sets)) if selected_sets else set()

#     # Penalize if not covering all elements of K
#     fitness_score = len(union_of_selected & K) - len(union_of_selected - K)

#     # Penalize for non-disjoint sets
#     for i in range(len(selected_sets)):
#         for j in range(i + 1, len(selected_sets)):
#             if not set(selected_sets[i]).isdisjoint(selected_sets[j]):
#                 fitness_score -= 1

#     return fitness_score


# def selection(population, fitnesses):
#     # Select based on fitness proportionate selection (roulette wheel)
#     total_fitness = sum(fitnesses)
#     probs = [f / total_fitness for f in fitnesses]
#     indices = np.random.choice(range(len(population)), size=len(population), p=probs)
#     return np.array(population)[indices]


# def crossover(parent1, parent2):
#     crossover_point = np.random.randint(1, len(parent1) - 1)
#     child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
#     child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
#     return child1, child2


# def mutate(solution, mutation_rate=0.01):
#     for i in range(len(solution)):
#         if np.random.rand() < mutation_rate:
#             solution[i] = 1 - solution[i]  # Flip the bit
#     return solution


# def genetic_algorithm(H, K, pop_size=10000, num_generations=100, mutation_rate=0.1):
#     num_sets = len(H)
#     population = initialize_population(pop_size, num_sets)

#     for generation in range(num_generations):
#         fitnesses = [fitness(individual, H, K) for individual in population]

#         # Selection
#         selected_population = selection(population, fitnesses)

#         # Crossover
#         next_population = []
#         for i in range(0, len(selected_population), 2):
#             parent1, parent2 = selected_population[i], selected_population[i + 1]
#             child1, child2 = crossover(parent1, parent2)
#             next_population.extend([child1, child2])

#         # Mutation
#         population = [
#             mutate(individual, mutation_rate) for individual in next_population
#         ]

#         # Print the best fitness score in the current generation
#         best_fitness = max(fitnesses)
#         print(f"Generation {generation}: Best Fitness = {best_fitness}")

#         # Early stopping if we find an optimal solution
#         if best_fitness == len(K):
#             break

#     # Return the best solution found
#     best_index = np.argmax(fitnesses)
#     return population[best_index]


# # Example usage
# H = list(paths)
# K = set(g.nodes)
# best_solution = genetic_algorithm(H, K)
# print(best_solution)


# def isLegal(value, maxVal):
#     return value < maxVal and value >= 0

# def find_most_probable_path(matrix, board, startRow, startCol):
#     directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
#     in_progress_heap = [(startRow,startCol)]
#     final_heap = []
#     row = startRow
#     col = startCol
#     while len(in_progress_heap) > 0:
#         for dir in directions:
#             if isLegal(row+dir[0],8) and isLegal(col+dir[1],6):

# matrix = train.train_most_probable_matrix()
# for row in range(len(starting_board)):
#     for col in range(len(starting_board[row])):
#         currHeap = find_most_probable_path(matrix, starting_board, row, col)


# https://networkx.org/documentation/stable/reference/classes/graph.html
# https://networkx.org/documentation/stable/reference/algorithms/traversal.html
# print(start)
# # Start random walk
# directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]
# path = [start]
# for _ in range(5):
#     random_dir = random.choice(directions)
#     print(random_dir)
#     while not (
#         0 <= start[0] + random_dir[0] < 6 and 0 <= start[1] + random_dir[1] < 8
#     ):
#         random.choice(directions)
#         path.append((start[0] + random_dir[0], start[1] + random_dir[1]))
# for x, y in path:
#     print(starting_board[x][y])
