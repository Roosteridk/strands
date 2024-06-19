import networkx as nx
import train
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Theme: Home work helpers
starting_board = [
    "ROMEMO",
    "TOTNRU",
    "INRISE",
    "DEPTOP",
    "PWSKAD",
    "ESEOBR",
    "SACBMY",
    "REKAKE",
]


def create_word_graph(board):
    G = nx.grid_2d_graph(8, 6)

    edges = (
        (node, (node[0] + dx, node[1] + dy))
        for node in G.nodes
        for dx, dy in [(1, 1), (-1, -1), (-1, 1), (1, -1)]
        if 0 <= node[0] + dx < 8 and 0 <= node[1] + dy < 6
    )

    G.add_edges_from(edges)

    attrs = dict()
    for node in G.nodes:
        print(node[0], node[1])
        attrs[node] = board[node[0]][node[1]]
    nx.set_node_attributes(G, attrs, "letter")

    nx.draw(G, labels=nx.get_node_attributes(G, "letter"))
    plt.savefig("graph.png")
    return G


def get_prob_matrix():
    try:
        matrix = np.load("trained_matrix.npy")
    except IOError:
        matrix = train.train_most_probable_matrix()
        np.save("trained_matrix", matrix)
    finally:
        return matrix


def search(node, matrix, word_graph):
    neighbors = word_graph.neighbors(node)
    marked = []
    word_heap = []
    marked.append(node)
    for neighbor in neighbors:
        return
        search_helper(neighbor, matrix, word_graph, marked)


def search_helper(node, matrix, word_graph, marked, permutations):
    neighbors = word_graph.neighbors(node)
    marked.add(node)
    permutations.add(node)
    for neighbor in neighbors:
        search_helper(neighbor, matrix, word_graph, marked)
    marked.remove(node)


word_graph = create_word_graph(starting_board)
print(word_graph)
matrix = get_prob_matrix()
for node in word_graph:
    search(node, matrix, word_graph)
    break


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


# search()
print(train.find_string_prob("will", get_prob_matrix()))
