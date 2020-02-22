from constraints_generator import constraints_generator
import time
import pickle
from build_matrix import parser


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

        self.heapSize = 0
        self.heapList = [-1 for _ in range(N)]
        self.heapMap = [-1 for _ in range(N)]

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
            if self.vars_map[x].pointer < 0:
                return True
            if self.heapMap[x] == -1:
                self.push(x)
            else:
                self.heap_up(x)
        return False

    def ac_enforcer(self, var_id=None):
        if var_id is None:
            for i in range(self.N):
                self.push(i)
        else:
            self.push(var_id)

        # print(self.queue.qsize())

        while self.heapSize > 0:
            var = self.pop()
            for i in range(0, self.N):
                if var != i and (self.revise(var, i) or
                                 self.revise(i, var)):
                    self.heap_clear()
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


# num_vars, max_dom, vars_map_cpu, cons_map_ = \
#     parser("./tightness0.1/rand-2-40-8-753-100-66_ext.xml")


max_dom = 3
num_vars = 4
cons_map_ = constraints_generator(max_dom, num_vars)

print(cons_map_)

f = open('constraints.dump', 'wb')
pickle.dump(cons_map_, f)
f.close()


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
    for i in bs.answer:
        print(i.dom)
else:
    print("no answer...")
print(bs.count)

print("Lasts =", time.time() - ticks)

# print("Node =", bs.count)
# print("Iterations =", bs.acer.count)
