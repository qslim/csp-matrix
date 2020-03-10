import torch
import time
from utils.build_matrix import parser
import pickle
import sys


device = torch.device("cpu")


class SACEnforcer:
    def __init__(self, cons_map, N, D):
        self.N = N
        self.D = D
        self.cons_map = cons_map
        self.n1_mask0 = torch.zeros((N, 1)).to(device)
        self.ndnnd_mask1 = torch.ones((N, D, N, N, D)).to(device)
        self.ndnd_mask1 = torch.ones((N, D, N, D)).to(device)
        self.ndnd_mask0 = torch.zeros((N, D, N, D)).to(device)
        self.ndn_mask1 = torch.ones((N, D, N)).to(device)
        self.nd_mask1 = torch.ones((N, D)).to(device)
        self.nd_mask0 = torch.zeros((N, D)).to(device)
        self.assigh_mask = self.build_assign_mask(N, D).to(device)

    @staticmethod
    def build_assign_mask(N, D):
        assign_mask = torch.ones((N, D, N, D))
        for i in range(N):
            for j in range(D):
                nd = assign_mask[i][j]
                for k in range(D):
                    if j != k:
                        nd[i][k] = 0
        return assign_mask

    def ac_enforcer(self, vars_map):
        ndnd = vars_map.unsqueeze(0)
        ndnd = ndnd.unsqueeze(0)
        ndnd = ndnd.expand(self.N, self.D, self.N, self.D)
        ndnd = torch.where(self.assigh_mask == 0, self.assigh_mask, ndnd)
        vars_map_pre = self.ndnd_mask0
        while not torch.equal(ndnd, vars_map_pre):
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")
            vars_map_pre = ndnd
            nd1nd = ndnd.unsqueeze(2)
            ndnnd = torch.matmul(self.cons_map, nd1nd.unsqueeze(5)).squeeze()
            ndnnd_reduce = torch.where(ndnnd > 1, self.ndnnd_mask1, ndnnd)
            ndnd = ndnnd_reduce.sum(3)
            ndnd = torch.where(ndnd == self.N, self.ndnd_mask1, self.ndnd_mask0)

            ndn = ndnd.sum(3)
            ndn = torch.where(ndn > 1, self.ndn_mask1, ndn)
            nd = ndn.sum(2)
            vars_map = torch.where(nd == self.N, self.nd_mask1, self.nd_mask0)
            if (vars_map.sum(1) == self.n1_mask0).any():
                return None

        return vars_map


class BackTrackSearcher:
    def __init__(self, rel_, N, D):
        self.acer = SACEnforcer(rel_, N, D)
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
        # print(level)
        self.count += 1
        print(level, self.count)
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


bm_name = sys.argv[1]
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

if bs.dfs(0, variables_map):
    print("got answer...")
    print(bs.answer.squeeze())
else:
    print("no answer...")
print(bs.count)

print("Lasts =", time.time() - ticks)

# print("Node =", bs.count)
# print("Iterations =", bs.acer.count)
