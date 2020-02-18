import torch
from ac_enforcer import ACEnforcer
from csp_generator.rand import rand_generator
import time


class BackTrackSearcher:
    def __init__(self, rel_, N):
        self.acer = ACEnforcer(rel_, N)
        self.N = N
        self.count = 0
        self.answer = None
        self.backup_vars = []
        for i in range(self.N):
            vars_l = [N - 1 for _ in range(N)]
            self.backup_vars.append(vars_l)

    def backup(self, vars_map_cur, level):
        for i in range(self.N):
            self.backup_vars[level][i] = vars_map_cur[i].pointer

    def restore(self, vars_map_cur, level):
        for i in range(self.N):
            vars_map_cur[i].pointer = self.backup_vars[level][i]
        return vars_map_cur

    def var_heuristics(self, vars_):
        min_dom = 99999
        min_index = -1
        for i in range(1, self.N):
            if vars_[i].pointer > 0:
                if min_dom > vars_[i].pointer:
                    min_index = i
                    min_dom = vars_[i].pointer
        return min_index

    def dfs(self, level, vars_pre, var_index):
        # print(level)
        self.count += 1
        print(level, self.count)
        if level == self.N:
            self.answer = vars_pre
            return True

        vars_pre = self.acer.ac_enforcer(vars_pre, var_index)
        if vars_pre is None:
            return False

        var_index = self.var_heuristics(vars_pre)
        if var_index == -1:
            self.answer = vars_pre
            return True

        self.backup(vars_pre, level)
        for i in range(vars_pre[var_index].pointer + 1):
            val_index = vars_pre[var_index].get(i)
            vars_pre[var_index].assign(val_index)
            if self.dfs(level + 1, vars_pre, var_index):
                return True
            vars_pre = self.restore(vars_pre, level)
        return False


N, D, vars_map, cons_map = rand_generator()
#N, D, vars_map, cons_map = parser("./tightness0.65/rand-2-40-40-135-650-71_ext.xml")
# print("cons shape:", cons_map.shape, " vars shape:", vars_map.shape)
# print(cons_map.type(), " ", vars_map.type())


bs = BackTrackSearcher(cons_map, N)

ticks = time.time()

if bs.dfs(0, vars_map, None):
    print("got answer...")
    # print(bs.answer.squeeze())
else:
    print("no answer...")
print(bs.count)

print("Lasts =", time.time() - ticks)

# print("Node =", bs.count)
# print("Iterations =", bs.acer.count)
