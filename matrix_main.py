import torch
import time
from utils.build_matrix import parser
import pickle
import sys


class ACEnforcer:
    def __init__(self, cons_map, N, D):
        self.N = N
        self.cons_map = cons_map
        self.n_mask0 = torch.zeros(N).to(device)
        self.nnd_mask1 = torch.ones((N, N, D)).to(device)
        self.nd_mask0 = torch.zeros((N, D)).to(device)
        self.iteration_count = 0
        self.idx_mask = torch.tensor([i for i in range(N)]).to(device)

        self.nd_mask1 = torch.ones((N, D)).to(device)

    def ac_enforcer(self, vars_map):
        idx = self.idx_mask
        vars_map_pre = self.nd_mask0
        self.iteration_count = 0
        while not torch.equal(vars_map, vars_map_pre):
            self.iteration_count += 1
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")
            vars_map_pre = vars_map

            nnd1 = torch.matmul(self.cons_map[:, :, :, :], vars_map[:, :].unsqueeze(2)).squeeze()
            nd1 = torch.where(nnd1 > 1, self.nnd_mask1[:, :, :], nnd1).sum(1)
            # nd1_map = torch.where(nd1 == self.N, self.nd_mask1, self.nd_mask0)
            vars_map_reduced1 = torch.where(nd1 != self.N, self.nd_mask0, vars_map)

            nnd2 = torch.matmul(self.cons_map[:, idx, :, :], vars_map[idx, :].unsqueeze(2)).squeeze()
            nd2 = torch.where(nnd2 > 1, self.nnd_mask1[:, : idx.shape[0], :], nnd2).sum(1)
            # nd2_map = torch.where(nd2 == idx.shape[0], self.nd_mask1, self.nd_mask0)
            vars_map_reduced2 = torch.where(nd2 != idx.shape[0], self.nd_mask0, vars_map)

            # print(torch.equal(vars_map_reduced1, vars_map_reduced2), self.iteration_count)
            if not torch.equal(vars_map_reduced1, vars_map_reduced2):
                print(vars_map_reduced1)
                print("--------")
                print(vars_map_reduced2)

            vars_map_reduced = torch.where(nd1 != self.N, self.nd_mask0, vars_map)
            # if (vars_map_reduced.sum(1) == self.n_mask0).any():
            #     return None
            vars_map = vars_map_reduced
            idx = (vars_map.sum(1) != vars_map_pre.sum(1)).nonzero(as_tuple=True)[0]
        return vars_map


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
