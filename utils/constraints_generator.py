import numpy as np
import random
import pickle
import sys
import time


def constraints_generator(max_dom, num_vars, density):
    # build rels_map
    rels_map = []
    rels_map_r = []
    num_cons = num_vars * (num_vars - 1) // 2
    for r1 in range(num_cons):
        rel_map = [[0 for _ in range(max_dom)] for _ in range(max_dom)]
        rel_map_r = [[0 for _ in range(max_dom)] for _ in range(max_dom)]
        for i1 in range(max_dom):
            for i2 in range(max_dom):
                if random.random() < density:
                    rel_map[i1][i2] = 1
                    rel_map_r[i2][i1] = 1
        rels_map.append(rel_map)
        rels_map_r.append(rel_map_r)

    # build cons_map
    cons_mark = [[-1 for _ in range(num_vars)] for _ in range(num_vars)]
    cnt = 0
    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                continue
            if i < j:
                cons_mark[i][j] = cnt
                cons_mark[j][i] = cnt
                cnt += 1

    cons_map = []
    for i in range(num_vars):
        cube = []
        for j in range(num_vars):
            if i == j:
                cube.append(np.eye(max_dom, dtype=int).tolist())
            elif i < j:
                cube.append(rels_map[cons_mark[i][j]])
            else:
                cube.append(rels_map_r[cons_mark[j][i]])
        cons_map.append(cube)

    return cons_map


max_domain = int(sys.argv[1])
num_variables = int(sys.argv[2])
con_density = float(sys.argv[3])
seed = sys.argv[4]
random.seed(seed)
print('max_domain:', max_domain, 'num_variables:', num_variables, 'con_density:', con_density, 'seed:', seed)
benchmark_dir = '../rand_benchmark/var' + str(max_domain) + '/'
bm_name = '../rand_benchmark/' \
          + 'dom' + str(max_domain) \
          + '-var' + str(num_variables) \
          + '-den' + str(con_density) \
          + '-seed' + str(seed) \
          + '-ts' + str(int(time.time())) \
          + '.dump'

constraints_map = constraints_generator(max_domain, num_variables, con_density)
f = open(bm_name, 'wb')
pickle.dump(constraints_map, f)
f.close()



