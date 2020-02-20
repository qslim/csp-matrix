import torch
from ac_enforcer import ACEnforcer
from constraints_generator import constraints_generator
import time
import pickle


device = torch.device("cuda")


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
        if dom[min_index] == 100000:
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


# max_dom = 50
# num_vars = 100
# cons_map = constraints_generator(max_dom, num_vars)
# f = open('constraints.dump', 'wb')
# pickle.dump(cons_map, f)
# f.close()

f = open('constraints.dump', 'rb')
cons_map = pickle.load(f)
max_dom = len(cons_map[0][0])
num_vars = len(cons_map)
f.close()

cons_map = torch.tensor(cons_map).type(torch.float)

# build vars_map
vars_map = []
for _ in range(num_vars):
    line = []
    for _ in range(max_dom):
        line.append(1)
    vars_map.append(line)
vars_map = torch.tensor(vars_map).type(torch.float)

print("cons shape:", cons_map.shape, " vars shape:", vars_map.shape)
print(cons_map.type(), " ", vars_map.type())

vars_map = vars_map.to(device)
cons_map = cons_map.to(device)

bs = BackTrackSearcher(cons_map, num_vars, max_dom)

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
