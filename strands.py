import networkx as nx

def create_word_graph(matrix_to_board):

    board_matrix = []

    for row in starting_board:
        board_matrix.append([row])

    for lst in board_matrix:
        print(lst)

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



print(word_board.edges)
print(word_board.number_of_edges())

# Find hamiltonian paths from spanning trees
paths = []
for tree in nx.all_simple_paths(word_board, (0, 0), (5, 7)):
    paths.append(tree)
    print(len(paths))
