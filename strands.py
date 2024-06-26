import networkx as nx
from train import abt, train_most_probable_matrix, find_string_prob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import heapq

# https://www.nytimes.com/games-assets/strands/2024-06-20.json
# Theme: They're good for a laugh 
# No. words: 7
starting_board = [
    "CMUNIP",
    "OMYTEE",
    "SITCOV",
    "URDEMS",
    "BCSHCT",
    "SFEWIR",
    "IRBSSE",
    "ENDCHE"
  ]


def create_word_graph(board):
    G = nx.grid_2d_graph(8, 6)

    # Diagonal edges
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


def get_prob_matrix():
    try:
        matrix = np.load("trained_matrix.npy")
    except IOError:
        matrix = train_most_probable_matrix()
        np.save("trained_matrix", matrix)
    finally:
        return matrix


def search_words(start_node, matrix, word_graph: nx.Graph, path=[], paths=set(), max_length=4):
    path = deepcopy(path)
    if len(path) == max_length:
        print("".join([word_graph.nodes[n]["letter"] for n in path]))
        paths.add(tuple(path))
        return
    path.append(start_node)
    neighbors = list(word_graph.neighbors(start_node))
    neighbor_indices = [abt(word_graph.nodes[n]["letter"]) for n in neighbors]
    neighbor_probs = [matrix[len(path)][abt(word_graph.nodes[start_node]["letter"])][i] for i in neighbor_indices]
    #print(neighbor_probs)
    normalized_probs = np.divide(neighbor_probs, np.sum(neighbor_probs))
    #print(normalized_probs)
    neighborDict = dict(zip(neighbors, normalized_probs))

    # Keep exploring neighbors if prob > 0
    for n, p in neighborDict.items():
        if p > 0 and (n not in path):
            search_words(n, matrix, word_graph, path)
    return paths


word_graph = create_word_graph(starting_board)
matrix = get_prob_matrix()
print(search_words((0,0), matrix, word_graph))
print(find_string_prob("cmit", get_prob_matrix()))


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


