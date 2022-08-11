import torch
import time
from utils.build_matrix import parser
import pickle
import sys


class ACEnforcer:
    def __init__(self, cons_map, N, D):
        self.N = N
        self.cons_map = cons_map
        self.n1_mask0 = torch.zeros(N).to(device)
        self.nnd_mask1 = torch.ones((N, N, D)).to(device)
        self.nd_mask1 = torch.ones((N, D)).to(device)
        self.nd_mask0 = torch.zeros((N, D)).to(device)
        self.iteration_count = 0
        # self.n_true = torch.tensor([True for _ in range(N)]).to(device)
        # self.n_false = torch.tensor([False for _ in range(N)]).to(device)
        self.idx_mask = torch.tensor([True for _ in range(N)]).to(device)

    def ac_enforcer(self, vars_map):
        idx = self.idx_mask
        num_idx = self.N
        while num_idx != 0:
            self.iteration_count += 1
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")
            vars_map_pre = vars_map.sum(1)
            nkd = torch.matmul(self.cons_map[:, idx, :, :], vars_map[idx, :].unsqueeze(2)).squeeze()
            nd = torch.where(nkd > 1, self.nnd_mask1[:, idx, :], nkd).sum(1)
            vars_map_reduced = torch.where(nd != num_idx, self.nd_mask0, vars_map)
            if (vars_map_reduced.sum(1) == self.n1_mask0).any():
                return None
            vars_map = vars_map_reduced
            idx = vars_map.sum(1) != vars_map_pre
            num_idx = torch.count_nonzero(idx)
        return vars_map

    # def ac_enforcer(self, vars_map):
    #     vars_map_pre = self.n1_mask0
    #     vars_map_sum = vars_map.sum(1)
    #     while not torch.equal(vars_map_sum, vars_map_pre):
    #         self.iteration_count += 1
    #         # print("~~~~~~~~~~~~~~~~~~~~~~~~")
    #         idx = vars_map_sum != vars_map_pre
    #         # idx = torch.where(vars_map_sum != vars_map_pre, self.n_true, self.n_false)
    #         # print(idx)
    #         num_idx = torch.count_nonzero(idx)
    #         # print(num_idx)
    #         # print(num_idx == torch.where(vars_map.sum(1) != vars_map_pre.sum(1), torch.ones(self.N).to(device), torch.zeros(self.N).to(device)).sum(0))
    #         vars_map_pre = vars_map.sum(1)
    #         nkd = torch.matmul(self.cons_map[:, idx, :, :], vars_map[idx, :].unsqueeze(2)).squeeze()
    #         nd = torch.where(nkd > 1, self.nnd_mask1[:, idx, :], nkd).sum(1)
    #         vars_map_reduced = torch.where(nd != num_idx, self.nd_mask0, vars_map)
    #         if (vars_map_reduced.sum(1) == self.n1_mask0).any():
    #             return None
    #         vars_map = vars_map_reduced
    #         vars_map_sum = vars_map.sum(1)
    #     return vars_map

    # def ac_enforcer(self, vars_map):
    #     vars_map_pre = self.n1_mask0
    #     while not torch.equal(vars_map.sum(1), vars_map_pre):
    #         self.iteration_count += 1
    #         # print("~~~~~~~~~~~~~~~~~~~~~~~~")
    #         vars_map_pre = vars_map.sum(1)
    #         nnd = torch.matmul(self.cons_map, vars_map.unsqueeze(2)).squeeze()
    #         nnd[nnd > 1] = 1
    #         vars_map[nnd.sum(1) != self.N] = 0
    #         if (vars_map.sum(1) == self.n1_mask0).any():
    #             return None
    #     return vars_map


class BackTrackSearcher:
    def __init__(self, rel_, N, D):
        self.acer = ACEnforcer(rel_, N, D)
        self.D = D
        self.N = N
        self.assign_mask = torch.eye(N).to(device)
        self.n_mask10000 = (torch.ones(N) * 10000).to(device)
        self.count = 0
        self.answer = None

    def assignment(self, var_index, val_index, vars_pre):
        # self.count += 1
        self.assign_mask[var_index][var_index] = 0
        vars_re = torch.matmul(self.assign_mask, vars_pre)
        self.assign_mask[var_index][var_index] = 1
        vars_re[var_index][val_index] = 1
        return vars_re

    def var_heuristics(self, vars_):
        dom = vars_.sum(1)
        dom = torch.where(dom == 1, self.n_mask10000, dom)
        min_index = dom.argmin().item()
        if dom[min_index].item() >= 10000:
            return -1
        return min_index

    def dfs(self, level, vars_pre):
        self.count += 1
        if self.count % 100 == 0:
            print(level, self.count)
            if self.count >= cutoff:
                return True
        if level == self.N:
            self.answer = vars_pre
            return True

        vars_pre = self.acer.ac_enforcer(vars_pre)
        if vars_pre is None:
            return False

        var_index = self.var_heuristics(vars_pre)
        if var_index == -1:
            self.answer = vars_pre
            return True

        var = vars_pre[var_index]
        for i in range(self.D):
            if var[i] == 0:
                continue
            vars_re = self.assignment(var_index, i, vars_pre)
            if self.dfs(level + 1, vars_re):
                return True

        return False


chosen_device = sys.argv[1]
device = torch.device(chosen_device)
bm_name = sys.argv[2]
cutoff = int(sys.argv[3])
f = open(bm_name, 'rb')
constraints_map = pickle.load(f)
max_domain = len(constraints_map[0][0])
num_variables = len(constraints_map)
f.close()

constraints_map = torch.tensor(constraints_map).type(torch.float)

# build vars_map
variables_map = []
for _ in range(num_variables):
    line = []
    for _ in range(max_domain):
        line.append(1)
    variables_map.append(line)
variables_map = torch.tensor(variables_map).type(torch.float)

print("cons shape:", constraints_map.shape, " vars shape:", variables_map.shape)
print(constraints_map.type(), " ", variables_map.type())

variables_map = variables_map.to(device)
constraints_map = constraints_map.to(device)

bs = BackTrackSearcher(constraints_map, num_variables, max_domain)

ticks = time.time()

# if bs.dfs(0, variables_map):
#     print("got answer...")
#     print(bs.answer.squeeze())
# else:
#     print("no answer...")
satisfied = bs.dfs(0, variables_map)
print("Lasts =", time.time() - ticks)
print(bs.count)
print(bs.acer.iteration_count / bs.count)
print(satisfied)

# print("Node =", bs.count)
# print("Iterations =", bs.acer.count)
