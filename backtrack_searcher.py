import torch
from sac1_enforcer import SAC1Enforcer
from build_matrix import parser
import time


device = torch.device("cuda")


class BackTrackSearcher:
    def __init__(self, rel_, N, D):
        self.acer = SAC1Enforcer(rel_, N, D)
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
        if dom[min_index] == 100000:
            return -1
        return min_index

    def dfs(self, level, vars_pre):
        # print(level)
        self.count += 1
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


# N, D, vars_map, cons_map = parser("/home/ymq/csp_benchmark/rand-2-26/rand-26-26-325-155-58021_ext.xml")
N, D, vars_map, cons_map = parser("/home/ymq/csp_benchmark/rand-2-23/rand-23-23-253-131-55021_ext.xml")
# N, D, vars_map, cons_map = parser("/home/ymq/csp_benchmark/rand-2-30-15-fcd/rand-2-30-15-306-230-fcd-22_ext.xml")
print("cons shape:", cons_map.shape, " vars shape:", vars_map.shape)
# print(N.type(), " ", D.type(), " ", cons_map.type(), " ", vars_map.type())

vars_map = vars_map.to(device)
cons_map = cons_map.to(device)

bs = BackTrackSearcher(cons_map, N, D)

ticks = time.time()

if bs.dfs(0, vars_map):
    print("got answer...")
    # print(bs.answer.squeeze())
else:
    print("no answer...")
print(bs.count)

print("Lasts =", time.time() - ticks)

# print("Node =", bs.count)
# print("Iterations =", bs.acer.count)
