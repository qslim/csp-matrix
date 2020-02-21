from constraints_generator import constraints_generator
import time
import pickle
import queue


class SparseDom:
    def __init__(self, D):
        self.pointer = D - 1
        self.dom = [1 for _ in range(D)]


class BackTrackSearcher:
    def __init__(self, rel_, vars_, N, D):
        self.cons_map = rel_
        self.vars_map = vars_
        self.N = N
        self.D = D
        self.count = 0
        self.answer = None

        # self.heapSize = 0
        # self.heapList = [-1 for _ in range(N)]
        # self.heapMap = [-1 for _ in range(N)]
        self.queue = queue.Queue()

    def var_heuristics(self):
        min_dom = 99999
        min_index = -1
        for i in range(1, self.N):
            if self.vars_map[i].pointer > 0:
                if min_dom > self.vars_map[i].pointer:
                    min_index = i
                    min_dom = self.vars_map[i].pointer
        return min_index

    def revise(self, x, y):
        con_map = self.cons_map[x][y]
        x_pre = self.vars_map[x].pointer
        for i in range(self.D):
            if self.vars_map[x].dom[i] == 0:
                continue
            find_sup = False
            for j in range(self.D):
                if self.vars_map[y].dom[j] == 0:
                    continue
                if con_map[i][j] == 1:
                    find_sup = True
                    break
            if not find_sup:
                self.vars_map[x].dom[i] = 0
                self.vars_map[x].pointer -= 1
        if self.vars_map[x].pointer != x_pre:
            self.queue.put(x)
            if self.vars_map[x].pointer < 0:
                return True
        return False

    def ac_enforcer(self, var_id=None):
        if var_id is None:
            for i in range(self.N):
                self.queue.put(i)
        else:
            self.queue.put(var_id)

        # print(self.queue.qsize())

        while not self.queue.empty() > 0:
            var = self.queue.get()
            for i in range(0, self.N):
                if var != i and (self.revise(var, i) or
                                 self.revise(i, var)):
                    while not self.queue.empty():
                        self.queue.get()
                    return False
        return True

    def dfs(self, level, var_index):
        # print(level)
        self.count += 1
        print(level, self.count)
        if level == self.N:
            self.answer = self.vars_map
            return True

        if not self.ac_enforcer(var_index):
            return False

        var_index = self.var_heuristics()
        if var_index == -1:
            self.answer = self.vars_map
            return True

        # backup
        backup_vars = []
        backup_n = []
        for i in range(self.N):
            tmp = [self.vars_map[i].dom[j] for j in range(self.D)]
            backup_vars.append(tmp)
            backup_n.append(self.vars_map[i].pointer)

        for i in range(self.D):
            if self.vars_map[var_index].dom[i] == 0:
                continue

            # assign
            for j in range(self.D):
                self.vars_map[var_index].dom[j] = 0
            self.vars_map[var_index].dom[i] = 1
            self.vars_map[var_index].pointer = 0

            if self.dfs(level + 1, var_index):
                return True

            # restore
            for ii in range(self.N):
                self.vars_map[ii].pointer = backup_n[ii]
                for jj in range(self.D):
                    self.vars_map[ii].dom[jj] = backup_vars[ii][jj]
        return False


# max_dom = 50
# num_vars = 50
# cons_map_ = constraints_generator(max_dom, num_vars)
#
# f = open('constraints.dump', 'wb')
# pickle.dump(cons_map_, f)
# f.close()


f = open('constraints.dump', 'rb')
cons_map_ = pickle.load(f)
max_dom = len(cons_map_[0][0])
num_vars = len(cons_map_)
f.close()

# build vars_map
vars_map_cpu = []
for _ in range(num_vars):
    line_cpu = SparseDom(max_dom)
    vars_map_cpu.append(line_cpu)

# print(cons_map.type(), " ", vars_map.type())

bs = BackTrackSearcher(cons_map_,  vars_map_cpu, num_vars, max_dom)

ticks = time.time()

if bs.dfs(0, None):
    print("got answer...")
    # print(bs.answer.squeeze())
else:
    print("no answer...")
print(bs.count)

print("Lasts =", time.time() - ticks)

# print("Node =", bs.count)
# print("Iterations =", bs.acer.count)
