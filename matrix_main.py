import jax.numpy as jnp
import jax
import time
import pickle
import sys
import csv
import random


class ACEnforcer:
    def __init__(self, cons_map, n_vars, n_dom):
        # self.cons_map = cons_map
        self.n_mask0 = jnp.zeros(n_vars)
        self.nnd_mask1 = jnp.ones((n_vars, n_vars, n_dom))
        self.nd_mask0 = jnp.zeros((n_vars, n_dom))
        self.ac_count = 0

        self.cons_map = cons_map

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
        self.assign_mask.at[var_index, var_index].set(0)
        vars_re = jnp.matmul(self.assign_mask, vars_pre)
        self.assign_mask.at[var_index, var_index].set(1)
        vars_re.at[var_index, val_index].set(1)
        return vars_re

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


random.seed(0)
chosen_device = 'cuda'
device = jax.default_device(chosen_device)
bm_name = None
cutoff = -1
bm_cut = [
    ('dom10-var500-den10-ts1661607900.dump', 5000)
]
csvheader = ['name', 'duration', 'count', 'ac_per', 'satisfied']
with open('cuda_results.csv', 'w', encoding='UTF8', newline='') as mycsv:
    writer = csv.writer(mycsv)
    writer.writerow(csvheader)

    for bc in bm_cut:
        bm_name = bc[0]
        cutoff = bc[1]

        f = open('rand_benchmark/' + bm_name, 'rb')
        constraints_map = pickle.load(f)
        max_domain = len(constraints_map[0][0])
        num_variables = len(constraints_map)
        f.close()

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
        # print(constraints_map.type(), " ", variables_map.type())

        # variables_map = variables_map.to(device)
        # constraints_map = constraints_map.to(device)

        bs = BackTrackSearcher(constraints_map, num_variables, max_domain)

        ticks = time.time()

        # if bs.dfs(0, variables_map):
        #     print("got answer...")
        #     print(bs.answer.squeeze())
        # else:
        #     print("no answer...")
        variables_map = bs.acer.ac_enforcer(variables_map, changed_idx=jnp.array([i for i in range(num_variables)]))
        satisfied = bs.dfs(0, variables_map)
        duration = time.time() - ticks
        count = bs.count
        ac_per = bs.acer.ac_count / bs.count
        print("Duration =", duration)
        print(count)
        print(ac_per)
        print(satisfied)

        csv_data = [
            bm_name, duration, count, ac_per, satisfied
        ]
        writer.writerow(csv_data)
        mycsv.flush()

        # print("Node =", bs.count)
        # print("Iterations =", bs.acer.count)
