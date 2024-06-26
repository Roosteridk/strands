import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import heapq
from copy import deepcopy


# https://www.nytimes.com/games-assets/strands/2024-06-20.json
THEME = "They're good for a laugh"
NO_WORDS = 7
STARTING_BOARD = [
    "CMUNIP",
    "OMYTEE",
    "SITCOV",
    "URDEMS",
    "BCSHCT",
    "SFEWIR",
    "IRBSSE",
    "ENDCHE",
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
    def dfs(curr_node, path, all_paths, length, word):
        path.append(curr_node)
        word += graph.nodes[curr_node]["letter"]

        if word in PREFIXES:
            if 4 <= len(path) <= length and PREFIXES[word] == 0:
                all_paths.add(tuple(path))
            if len(path) != length:
                for neighbor in graph.neighbors(curr_node):
                    if neighbor not in path:
                        dfs(neighbor, path, all_paths, length, word)
        # Backtrack: Remove the current node from the path
        path.pop()

    all_paths = set()
    for node in graph.nodes():
        dfs(node, [], all_paths, length, "")
    return all_paths


g = create_graph(STARTING_BOARD)
paths = search_words(g)
words = dict()
# Note: there may be more than one math for the same word
for path in paths:
    words[("".join(g.nodes[n]["letter"] for n in path))] = path
for k, v in dict(sorted(words.items())).items():
    print(k, v)

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
