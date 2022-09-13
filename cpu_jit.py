import time
from utils.build_matrix import parser
import pickle
import sys
import csv
import numpy as np
from numba.experimental import jitclass
from numba import int32, types, typed

spec = {
    'pointer': int32,
    'dom': int32[:],
    'idx': int32[:]
}


@jitclass(spec)
class SparseDom(object):
    def __init__(self, D):
        self.pointer = D - 1
        self.dom = np.arange(D, dtype=np.int32)
        self.idx = np.arange(D, dtype=np.int32)

    def delete(self, index):
        val = self.dom[index]
        tail_val = self.dom[self.pointer]
        self.dom[index] = tail_val
        self.idx[tail_val] = index
        self.dom[self.pointer] = val
        self.idx[val] = self.pointer
        self.pointer -= 1

    def assign(self, val):
        idx_v = self.idx[val]
        head_val = self.dom[0]
        self.dom[0] = val
        self.idx[val] = 0
        self.dom[idx_v] = head_val
        self.idx[head_val] = idx_v

        self.pointer = 0


spec2 = {
    'cons_map': int32[:, :, :, :],
    'vars_map': types.ListType(SparseDom.class_type.instance_type),
    'N': int32,
    'count': int32,
    # 'answer': types.ListType(SparseDom.class_type.instance_type),

    'heapSize': int32,
    'heapList': int32[:],
    'heapMap': int32[:],

    'ts_v': int32[:],
    'ts_c': int32[:, :],
    'ts_global': int32,

    'assign_map': int32[:],
    'revise_count': int32,

    'vars_stack': int32[:, :],
}


@jitclass(spec2)
class BackTrackSearcher(object):
    def __init__(self, rel_, vars_, N):
        self.cons_map = rel_
        self.vars_map = vars_
        self.N = N
        self.count = 0
        # self.answer = None

        self.heapSize = 0
        self.heapList = np.full(N, -1, dtype=np.int32)
        self.heapMap = np.full(N, -1, dtype=np.int32)

        self.ts_v = np.ones(N, dtype=np.int32)
        self.ts_c = np.zeros((N, N), dtype=np.int32)
        self.ts_global = 2

        self.assign_map = np.zeros(N, dtype=np.int32)

        self.revise_count = 0

        self.vars_stack = np.zeros((N, N), dtype=np.int32)

    def push(self, x):
        pos = self.heapSize
        qos = pos

        self.heapSize += 1

        while pos != 0:
            pos = (pos - 1) // 2
            a = self.heapList[pos]
            if self.vars_map[a].pointer < self.vars_map[x].pointer:
                break
            self.heapList[qos] = a
            self.heapMap[a] = qos
            qos = pos
        self.heapList[qos] = x
        self.heapMap[x] = qos

    def pop(self):
        a = self.heapList[0]
        self.heapSize -= 1
        self.heapList[0] = self.heapList[self.heapSize]
        self.heapMap[self.heapList[self.heapSize]] = 0
        self.heap_down(0)
        self.heapMap[a] = -1
        return a

    def heap_down(self, pos):
        x = self.heapList[pos]
        qos = pos * 2 + 1
        while qos < self.heapSize - 1:
            b = self.heapList[qos + 1]
            a = self.heapList[qos]
            if self.vars_map[b].pointer < self.vars_map[a].pointer:
                qos += 1
                a = b
            if self.vars_map[a].pointer > self.vars_map[x].pointer:
                self.heapList[pos] = x
                self.heapMap[x] = pos
                break
            self.heapList[pos] = a
            self.heapMap[a] = pos
            pos = qos
            qos = pos * 2 + 1
        if qos > self.heapSize - 1:
            self.heapList[pos] = x
            self.heapMap[x] = pos
        elif qos == self.heapSize - 1:
            a = self.heapList[qos]
            if self.vars_map[a].pointer > self.vars_map[x].pointer:
                self.heapList[pos] = x
                self.heapMap[x] = pos
            else:
                self.heapList[pos] = a
                self.heapMap[a] = pos
                self.heapList[qos] = x
                self.heapMap[x] = qos

    def heap_up(self, x):
        pos = self.heapMap[x]
        qos = pos
        while pos > 0:
            pos = (pos - 1) // 2
            a = self.heapList[pos]
            if self.vars_map[a].pointer < self.vars_map[x].pointer:
                break
            self.heapList[qos] = a
            self.heapMap[a] = qos
            qos = pos
        self.heapList[qos] = x
        self.heapMap[x] = qos

    def heap_clear(self):
        for i in range(self.heapSize):
            x = self.heapList[i]
            self.heapMap[x] = -1
        self.heapSize = 0

    def var_heuristics(self):
        min_dom = 10000
        min_index = -1
        for i in range(self.N):
            if self.vars_map[i].pointer > 0:
                if self.vars_map[i].pointer < min_dom:
                    min_index = i
                    min_dom = self.vars_map[i].pointer
        return min_index

    def backup_vars(self, level):
        for i in range(self.N):
            self.vars_stack[level][i] = self.vars_map[i].pointer

    def restore_vars(self, level):
        for i in range(self.N):
            self.vars_map[i].pointer = self.vars_stack[level][i]
            self.ts_v[i] = 0

    def revise(self, x, y):
        self.revise_count += 1
        con_map = self.cons_map[x][y]
        x_pre = self.vars_map[x].pointer
        for i in range(self.vars_map[x].pointer, -1, -1):
            val_x = self.vars_map[x].dom[i]
            find_sup = False
            for j in range(self.vars_map[y].pointer + 1):
                val_y = self.vars_map[y].dom[j]
                if con_map[val_x][val_y] == 1:
                    find_sup = True
                    break
            if not find_sup:
                self.vars_map[x].delete(i)
        if self.vars_map[x].pointer != x_pre:
            if self.vars_map[x].pointer < 0:
                return True
            self.ts_v[x] = self.ts_global
            self.ts_global += 1
            if self.heapMap[x] == -1:
                self.push(x)
            else:
                self.heap_up(x)
        return False

    def ac_enforcer(self, var_ids):
        for i in var_ids:
            self.push(i)

        # print(self.queue.qsize())

        while self.heapSize > 0:
            var = self.pop()
            for i in range(self.N):
                if var != i and self.ts_v[var] > self.ts_c[var][i]:
                    if self.revise(var, i) or (self.assign_map[i] == 0 and self.revise(i, var)):
                        self.heap_clear()
                        return False
                    self.ts_c[var][i] = self.ts_global
                    self.ts_c[i][var] = self.ts_global
                    self.ts_global += 1
        return True

    def dfs(self, level, var_ids):
        self.count += 1
        if self.count % 100 == 0:
            print(level, self.count)
            if self.count >= cutoff:
                return True
        if level == self.N:
            # self.answer = self.vars_map
            return True

        if not self.ac_enforcer(var_ids):
            return False

        # var_index = self.var_heuristics()
        var_index = level
        if var_index == -1:
            # self.answer = self.vars_map
            return True

        # backup
        backup_vars = [self.vars_map[i].pointer for i in range(self.N)]

        # sorted_dom = [self.vars_map[var_index].dom[i] for i in range(self.vars_map[var_index].pointer + 1)]
        # sorted_dom.sort()
        # self.assign_map[var_index] = 1
        # for i in sorted_dom:
        self.assign_map[var_index] = 1
        for ptr in range(self.vars_map[var_index].pointer + 1):
            i = self.vars_map[var_index].dom[ptr]

            self.vars_map[var_index].assign(i)
            self.ts_v[var_index] = self.ts_global
            self.ts_global += 1
            if self.dfs(level + 1, np.array([var_index])):
                return True
            for j in range(self.N):
                self.vars_map[j].pointer = backup_vars[j]
                self.ts_v[j] = 0
        self.assign_map[var_index] = 0
        return False

    def main_search2(self):
        if not self.ac_enforcer(np.arange(num_variables, dtype=np.int32)):
            return False
        level = 0
        while level >= 0:
            self.backup_vars(level)
            self.vars_map[level].assign(self.vars_map[level].dom[0])
            self.count += 1
            if self.count % 100 == 0:
                print(level, self.count)
                if self.count >= cutoff:
                    return True
            self.ts_v[level] = self.ts_global
            self.ts_global += 1
            if not self.ac_enforcer([level]):
                self.restore_vars(level)
                self.vars_map[level].delete(0)
                while self.vars_map[level].pointer < 0:
                    level = level - 1
                    self.restore_vars(level)
                    self.vars_map[level].delete(0)
                    if level < 0:
                        return False
            else:
                level = level + 1
                if level == self.N - 1:
                    return True
        return False


bm_name = None
cutoff = -1
bm_cut = [
    ('dom10-var100-den10-seed0-ts1662958482.dump', 20000)
]
csvheader = ['name', 'duration', 'count', 'ac_per', 'satisfied']
with open('trad_results.csv', 'w', encoding='UTF8', newline='') as mycsv:
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

        # build vars_map
        variables_map = []
        for _ in range(num_variables):
            line = SparseDom(max_domain)
            variables_map.append(line)

        bs = BackTrackSearcher(np.array(constraints_map, dtype=np.int32), typed.List(variables_map), num_variables)

        ticks = time.time()

        # if bs.dfs(0, None):
        #     print("got answer...")
        #     for ii in bs.answer:
        #         print(ii.dom)
        # else:
        #     print("no answer...")
        satisfied = bs.main_search2()
        duration = time.time() - ticks
        count = bs.count
        ac_per = bs.revise_count / bs.count
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
