import jax.numpy as jnp
import time
import sys
import random
import numpy as np


class ACEnforcer:
    def __init__(self, cons_map, n_vars, n_dom):
        self.cons_map = cons_map
        self.n_mask0 = jnp.zeros(n_vars)
        self.nnd_mask1 = jnp.ones((n_vars, n_vars, n_dom))
        self.nd_mask0 = jnp.zeros((n_vars, n_dom))
        self.ac_count = 0

    def ac_enforcer(self, vars_map, changed_idx):
        n_idx = changed_idx.shape[0]
        vars_map_pre = vars_map.sum(1)
        while n_idx != 0:
            self.ac_count += 1

            nkd = jnp.matmul(self.cons_map[:, changed_idx, :, :], jnp.expand_dims(vars_map[changed_idx, :], axis=2)).squeeze(-1)

            nd = jnp.where(nkd > 1, self.nnd_mask1[:, : n_idx, :], nkd).sum(1)

            vars_map = jnp.where(nd != n_idx, self.nd_mask0, vars_map)

            vars_map_sum = vars_map.sum(1)
            if (vars_map_sum == self.n_mask0).any():
                return None
            changed_idx = (vars_map_sum != vars_map_pre).nonzero()[0]
            n_idx = changed_idx.shape[0]
            vars_map_pre = vars_map_sum

        return vars_map


class BackTrackSearcher:
    def __init__(self, rel_, n_vars, n_dom):
        self.acer = ACEnforcer(rel_, n_vars, n_dom)
        self.n_dom = n_dom
        self.n_vars = n_vars
        self.assign_mask = jnp.eye(n_vars)
        self.n_mask10000 = (jnp.ones(n_vars) * 10000)
        self.count = 0
        self.answer = None

    def assignment(self, var_index, val_index, vars_pre):
        # self.count += 1
        assign_mask = self.assign_mask.at[var_index, var_index].set(0)
        vars_re = jnp.matmul(assign_mask, vars_pre)
        return vars_re.at[var_index, val_index].set(1)

    def var_heuristics(self, vars_map):
        vars_size = vars_map.sum(1)
        vars_size = jnp.where(vars_size == 1, self.n_mask10000, vars_size)
        min_idx = vars_size.argmin().item()
        if vars_size[min_idx].item() >= 10000:
            return -1
        return min_idx

    def dfs(self, level, vars_map):
        self.count += 1
        if self.count % 100 == 0:
            print(level, self.count)
            if self.count >= cutoff:
                return True
        if level == self.n_vars:
            self.answer = vars_map
            return True

        if vars_map is None:
            return False

        # var_idx = self.var_heuristics(vars_map)
        var_idx = level
        if var_idx == -1:
            self.answer = vars_map
            return True

        var = vars_map[var_idx]
        for i in range(self.n_dom):
            if var[i] == 0:
                continue
            _vars_map = self.assignment(var_idx, i, vars_map)
            _vars_map = self.acer.ac_enforcer(_vars_map, changed_idx=jnp.array([var_idx]))
            if self.dfs(level + 1, _vars_map):
                return True

        return False


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


random.seed(0)
# device = torch.device('cuda')
cutoff = 5000

constraints_map = constraints_generator(10, 100, 8)
max_domain = len(constraints_map[0][0])
num_variables = len(constraints_map)

constraints_map = jnp.array(constraints_map)

# build vars_map
variables_map = []
for _ in range(num_variables):
    line = []
    for _ in range(max_domain):
        line.append(1)
    variables_map.append(line)
variables_map = jnp.array(variables_map)

print("cons shape:", constraints_map.shape, " vars shape:", variables_map.shape)

bs = BackTrackSearcher(constraints_map, num_variables, max_domain)

ticks = time.time()

variables_map = bs.acer.ac_enforcer(variables_map, changed_idx=jnp.array([i for i in range(num_variables)]))
satisfied = bs.dfs(0, variables_map)
duration = time.time() - ticks
count = bs.count
ac_per = bs.acer.ac_count / bs.count
print("Duration =", duration)
print(count)
print(ac_per)
print(satisfied)