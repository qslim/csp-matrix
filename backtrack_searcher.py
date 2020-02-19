from csp_generator.rand import rand_generator
import time


class BackTrackSearcher:
    def __init__(self, rel_, N):
        self.cons_map = rel_
        self.vars_map = vars_map_
        self.N = N
        self.count = 0
        self.answer = None
        self.backup_vars = []
        for i in range(self.N):
            vars_l = [N - 1 for _ in range(N)]
            self.backup_vars.append(vars_l)

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

    def revise(self, x, y, vars_map):
        con_map = self.cons_map[x][y]
        x_pre = vars_map[x].pointer
        for i in range(vars_map[x].pointer + 1):
            val_x = vars_map[x].get(i)
            find_sup = False
            for j in range(vars_map[y].pointer + 1):
                val_y = vars_map[y].get(j)
                if con_map[val_x][val_y] == 1:
                    find_sup = True
                    break
            if not find_sup:
                vars_map[x].delete(i)
        if vars_map[x].pointer != x_pre:
            if self.heapMap[x] == -1:
                self.push(x)
            else:
                self.heap_up(x)
            if vars_map[x].pointer < 0:
                return True
        return False

    def ac_enforcer(self, vars_map, var_id=None):
        if var_id is None:
            for i in range(self.N):
                self.push(i)
        else:
            self.push(var_id)

        # print(self.queue.qsize())

        while self.heapSize > 0:
            var = self.pop()
            for i in range(0, self.N):
                if var != i and (self.revise(var, i, vars_map) or
                                 self.revise(i, var, vars_map)):
                    self.heap_clear()
                    return None
        return vars_map

    def dfs(self, level, vars_pre, var_index):
        # print(level)
        self.count += 1
        print(level, self.count)
        if level == self.N:
            self.answer = vars_pre
            return True

        vars_pre = self.ac_enforcer(vars_pre, var_index)
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


N, D, vars_map_, cons_map_ = rand_generator()
#N, D, vars_map, cons_map = parser("./tightness0.65/rand-2-40-40-135-650-71_ext.xml")
# print("cons shape:", cons_map.shape, " vars shape:", vars_map.shape)
# print(cons_map.type(), " ", vars_map.type())


bs = BackTrackSearcher(cons_map_, N)

ticks = time.time()

if bs.dfs(0, vars_map_, None):
    print("got answer...")
    # print(bs.answer.squeeze())
else:
    print("no answer...")
print(bs.count)

print("Lasts =", time.time() - ticks)

# print("Node =", bs.count)
# print("Iterations =", bs.acer.count)
