import networkx as nx
import train
import numpy as np


def create_word_graph(matrix_to_board):

    board_matrix = []

    for row in starting_board:
        board_matrix.append([row])

    G = nx.grid_2d_graph(6, 8)

    edges = (
        (node, (node[0] + dx, node[1] + dy))
        for node in G.nodes
        for dx, dy in [(1, 1), (-1, -1), (-1, 1), (1, -1)]
        if 0 <= node[0] + dx < 6 and 0 <= node[1] + dy < 8
    )
    G.add_edges_from(edges)

    return G


starting_board = [
    "SRETNU",
    "LPACEO",
    "ANCHFC",
    "UTRYLN",
    "SNEEIE",
    "PDORGT",
    "LHWHOI",
    "ASNTRB",
]

word_board = create_word_graph(starting_board)

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


def get_prob_matrix():
    try:
        matrix = np.load("trained_matrix.npy")
    except IOError:
        matrix = train.train_most_probable_matrix()
        np.save("trained_matrix", matrix)
    finally:
        return matrix


print(train.find_string_prob("test", get_prob_matrix()))
