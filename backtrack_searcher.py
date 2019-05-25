import torch
from ac_enforcer import ACEnforcer
from build_matrix import parser
import time


# device = torch.device("cuda")


class BackTrackSearcher:
    def __init__(self, rel_, N, D):
        self.acer = ACEnforcer(rel_, N, D)
        self.D = D
        self.N = N
        self.d1_mask1 = torch.ones((D, 1))
        self.assign_mask = torch.eye(N)
        self.n_mask10000 = (torch.ones(N) * 10000)
        # self.d1_mask1 = torch.ones((D, 1)).to(device)
        # self.assign_mask = torch.eye(N).to(device)
        # self.n_mask10000 = (torch.ones(N) * 10000).to(device)
        # self.count = 0
        self.answer = None

    def assignment(self, var_index, val_index, vars_pre):
        # print(var_index, " ", val_index)
        # self.count += 1
        self.assign_mask[var_index][var_index] = 0
        # print(self.assign_mask)
        vars_re = torch.matmul(self.assign_mask, vars_pre.squeeze())
        self.assign_mask[var_index][var_index] = 1
        vars_re[var_index][val_index] = 1
        # print(vars_re)
        vars_re = vars_re.unsqueeze(1)
        return vars_re

    def var_heuristics(self, vars_):
        dom = torch.matmul(vars_, self.d1_mask1).squeeze()
        dom = torch.where(dom == 1, self.n_mask10000, dom)
        min_index = dom.argmin()
        if dom[min_index] == 100000:
            return torch.tensor(-1)
        # print(dom.argmin())
        return min_index

    def dfs(self, level, vars_pre):
        # print(level)
        if level == self.N:
            self.answer = vars_pre
            return True

        vars_pre = self.acer.ac_enforcer(vars_pre)
        if vars_pre is None:
            return False

        var_index = self.var_heuristics(vars_pre)

        # print(var_index)

        if var_index == torch.tensor(-1):
            self.answer = vars_pre
            return True

        var = vars_pre[var_index][0]
        for i in range(self.D):
            i = torch.tensor(i)
            if var[i] == 0:
                continue
            vars_re = self.assignment(var_index, i, vars_pre)
            if self.dfs(level + 1, vars_re):
                return True

        return False


# N, D, vars_map, cons_map = parser("/home/ymq/csp_benchmark/run_dir/rand-2-26/rand-26-26-325-155-53021_ext.xml")
N, D, vars_map, cons_map = parser("/home/ymq/csp_benchmark/rand-2-30-15-fcd/rand-2-30-15-306-230-fcd-16_ext.xml")
print("cons shape:", cons_map.shape, " vars shape:", vars_map.shape)

# vars_map = vars_map.to(device)
# cons_map = cons_map.to(device)

bs = BackTrackSearcher(cons_map, N, D)

ticks = time.time()

if bs.dfs(0, vars_map):
    print("got answer...")
    # print(bs.answer.squeeze())
else:
    print("no answer...")

print("Lasts =", time.time() - ticks)

# print("Node =", bs.count)
# print("Iterations =", bs.acer.count)
