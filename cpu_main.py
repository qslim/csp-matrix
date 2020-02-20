from constraints_generator import constraints_generator
from sparse_dom import SparseDom
import time
import pickle


class BackTrackSearcher:
    def __init__(self, rel_, vars_, N):
        self.cons_map = rel_
        self.vars_map = vars_
        self.N = N
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
        for i in range(self.vars_map[x].pointer + 1):
            val_x = self.vars_map[x].get(i)
            find_sup = False
            for j in range(self.vars_map[y].pointer + 1):
                val_y = self.vars_map[y].get(j)
                if con_map[val_x][val_y] == 1:
                    find_sup = True
                    break
            if not find_sup:
                self.vars_map[x].delete(i)
        if self.vars_map[x].pointer != x_pre:
            if self.heapMap[x] == -1:
                self.push(x)
            else:
                self.heap_up(x)
            if self.vars_map[x].pointer < 0:
                return True
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

        backup_vars = [self.vars_map[i].pointer for i in range(self.N)]
        for i in range(self.vars_map[var_index].pointer + 1):
            for j in range(self.N):
                self.vars_map[j].pointer = backup_vars[j]
            val_index = self.vars_map[var_index].get(i)
            self.vars_map[var_index].assign(val_index)
            if self.dfs(level + 1, var_index):
                return True
        return False


max_dom = 50
num_vars = 100
cons_map_ = constraints_generator(max_dom, num_vars)

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

bs = BackTrackSearcher(cons_map_,  vars_map_cpu, num_vars)

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
