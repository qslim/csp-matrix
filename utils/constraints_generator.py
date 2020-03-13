import numpy as np
import random
import sys
import time
import json


def constraints_generator(max_dom, num_vars, density, Diffy):
    # build rels_map
    rels_map = []
    rels_map_r = []
    if Diffy == 1:
        num_cons = num_vars * (num_vars - 1) // 2
    else:
        num_cons = num_vars
    for r1 in range(num_cons):
        rel_map = [[0 for _ in range(max_dom)] for _ in range(max_dom)]
        rel_map_r = [[0 for _ in range(max_dom)] for _ in range(max_dom)]
        for i1 in range(max_dom):
            for i2 in range(max_dom):
                val = random.randint(0, density)
                if val > 0:
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
                cons_mark[i][j] = cnt % num_cons
                cons_mark[j][i] = cnt % num_cons
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


num_variables = int(sys.argv[1])
max_domain = int(sys.argv[2])
con_density = int(sys.argv[3])
differenty = int(sys.argv[4])
print('num_variables:', num_variables, 'max_domain:', max_domain,
      'con_density:', con_density, 'differenty:', differenty)
bm_name = '../csp-benchmark/conmap-' \
          + str(num_variables) \
          + '-' + str(max_domain) \
          + '-' + str(con_density) \
          + '-' + str(differenty) \
          + '-' + str(int(time.time())) \
          + '.json'

constraints_map = constraints_generator(max_domain, num_variables, con_density, differenty)
json_string = json.dumps(constraints_map)
# print(json_string)
# f = open(bm_name, 'w')
# f.write(json_string)
# f.close()

with open(bm_name, 'w') as f:
    json.dump(json_string, f)



