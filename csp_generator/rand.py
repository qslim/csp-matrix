import torch
import random


def rand_generator():
    # build vars_map
    max_dom = 50
    num_vars = 50
    vars_map = []
    for _ in range(num_vars):
        line = []
        for _ in range(max_dom):
            line.append(1)
        vars_map.append(line)
    vars_map = torch.tensor(vars_map)
    # print(vars_map)
    # print(vars_map.shape)

    # build rels_map
    rels_map = []
    rels_map_r = []
    for r1 in range(num_vars):
        for r2 in range(r1):
            rel_map = []
            rel_map_r = []
            for i1 in range(max_dom):
                for i2 in range(max_dom):
                    val = random.randint(0, 9) % 2
                    rel_map[i1][i2] = val
                    rel_map_r[i2][i1] = val
            rels_map.append(rel_map)
            rels_map_r.append(rel_map_r)
    for rel_map_r_ in rels_map_r:
        rels_map.append(rel_map_r_)
    # print(np.array(rels_map).shape)
    # print(np.array(rels_map[0]))

    # build cons_map
    mid_val = (num_vars * (num_vars - 1)) // 2
    cons_map = []
    for i in range(num_vars):
        cube = []
        for j in range(num_vars):
            position = num_vars * i + j
            if i == j:
                cube.append(torch.eye(max_dom).tolist())
            elif j < i:
                cube.append(rels_map[position])
            else:
                cube.append(rels_map[position + mid_val])
        cons_map.append(cube)
    cons_map = torch.tensor(cons_map)
    # print(cons_map.shape)
    # print(cons_map[0][3])
    # print(cons_map[3][0])

    # return num_vars, max_dom, vars_map, cons_map
    return num_vars, max_dom, vars_map.type(torch.float), cons_map.type(torch.float)


# parser("/home/ymq/Downloads/rand-2-30-15/rand-2-30-15-306-230-10_ext.xml")
# parser("/home/ymq/csp_benchmark/run_dir/rand-2-30-15-fcd/rand-2-30-15-306-230-fcd-17_ext.xml")

