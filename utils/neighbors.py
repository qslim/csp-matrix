import random


def build_neighbors(num_vars, density):
    is_cons = [[False for _ in range(num_vars)] for _ in range(num_vars)]
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            if random.random() < density:
                is_cons[i][j] = True
                is_cons[j][i] = True
    neighbors = [[] for _ in range(num_vars)]
    for i in range(num_vars):
        for j in range(num_vars):
            if is_cons[i][j]:
                neighbors[i].append(j)
    return neighbors
