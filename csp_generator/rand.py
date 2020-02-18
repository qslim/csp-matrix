import torch
import random
from sparse_dom import SparseDom


def rand_generator():
    # build vars_map
    max_dom = 20
    num_vars = 20
    vars_map = []
    for _ in range(num_vars):
        line = SparseDom(max_dom)
        vars_map.append(line)

    # build rels_map
    rels_map = []
    rels_map_r = []
    for r1 in range(num_vars):
        rel_map = [[0 for _ in range(max_dom)] for _ in range(max_dom)]
        rel_map_r = [[0 for _ in range(max_dom)] for _ in range(max_dom)]
        for i1 in range(max_dom):
            for i2 in range(max_dom):
                val = random.randint(0, 7)
                if val > 0:
                    val = 1
                rel_map[i1][i2] = val
                rel_map_r[i2][i1] = val
        for r2 in range(r1):
            rels_map.append(rel_map)
            rels_map_r.append(rel_map_r)
    # print(np.array(rels_map).shape)
    # print(np.array(rels_map[0]))

    # build cons_map
    cons_map = []
    for i in range(num_vars):
        cube = []
        for j in range(num_vars):
            if i == j:
                cube.append(torch.eye(max_dom).tolist())
            elif j < i:
                cube.append(rels_map.pop())
            else:
                cube.append(rels_map_r.pop())
        cons_map.append(cube)

    return num_vars, max_dom, vars_map, cons_map


# parser("/home/ymq/Downloads/rand-2-30-15/rand-2-30-15-306-230-10_ext.xml")
# parser("/home/ymq/csp_benchmark/run_dir/rand-2-30-15-fcd/rand-2-30-15-306-230-fcd-17_ext.xml")

