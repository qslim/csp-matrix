import copy
import numpy as np
from ac_enforcer import ACEnforcer
from build_matrix import parser
import time

class BackTrackSearcher:
    def __init__(self, rel_, N, D):
        # self.vars = [[1, 2, 3], [1, 3, 4], [3, 6, 7]]
        self.acer = ACEnforcer(rel_, N, D)
        # self.vars_backup = []
        self.D = D
        self.N = N
        self.var_mask = np.ones([D, 1])
        self.count = 0
        self.answer = None

    # def pre_processing(self, vars_):
    #     if self.acer.ac_enforcer(vars_):
    #         print("network inconsistency.")
    #         return False
    #     return True

    def assignment(self, var_index, val_index, vars_pre):
        self.count += 1
        vars_re = copy.deepcopy(vars_pre)
        var = vars_re[var_index]
        for p in range(self.D):
            var[p] = 0
        var[val_index] = 1
        return vars_re

    # @staticmethod
    # def arrive_answer(vars_):
    #     print("got an answer...")
    #     print(vars_)

    def var_heuristics(self, vars_):
        dom = np.matmul(np.squeeze(vars_), self.var_mask).squeeze()
        dom = np.where(dom <= 1, 100000, dom)
        min_val = dom.min()
        if min_val == 100000:
            return -1
        # print(dom.argmin())
        return dom.argmin()

    def dfs(self, level, vars_pre):
        # print(level)
        if level == self.N:
            print("NNNNNNNN")
            self.answer = vars_pre
            return True

        # tmp = vars_pre.copy()
        vars_pre = self.acer.ac_enforcer(vars_pre)
        if vars_pre is None:
            # print((tmp == vars_pre).all())
            return False
        # print((tmp == vars_pre).all())

        var_index = self.var_heuristics(vars_pre)

        if var_index == -1:
            print("IIIIIIII")
            self.answer = vars_pre
            return True

        var = vars_pre[var_index]
        for i in range(self.D):
            if var[i] == 0:
                continue

            vars_re = self.assignment(var_index, i, vars_pre)
            # if self.acer.ac_enforcer(vars_re):
            #     continue

            if self.dfs(level + 1, vars_re):
                return True

        return False


N, D, vars_map, cons_map = parser("/home/ymq/csp_benchmark/run_dir/rand-2-26/rand-26-26-325-155-53021_ext.xml")
print("DONE LOADING")

# print(D)
# print(vars_map.shape)
# print(cons_map.shape)
# print(cons_map[0][2])
# print(vars_map)


bs = BackTrackSearcher(cons_map, N, D)

ticks = time.time()

if bs.dfs(0, vars_map):
    print(bs.answer.squeeze())
else:
    print("no answer...")

print("Lasts: ", time.time() - ticks)

print("Node =", bs.count)

# if bs.pre_processing(vars_map):
#     bs.dfs(0, vars_map)
    # pass