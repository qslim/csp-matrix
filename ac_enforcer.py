from priority_queue import PriorityQueue


class ACEnforcer:
    def __init__(self, cons_map, N):
        self.cons_map = cons_map
        self.pQueue = PriorityQueue(N)
        self.N = N

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
            self.pQueue.push(vars_map[x].pointer, x)
            if vars_map[x].pointer < 0:
                return True
        return False

    def ac_enforcer(self, vars_map, var_id=None):
        if var_id is None:
            for i in range(self.N):
                self.pQueue.push(vars_map[i].pointer, i)
        else:
            self.pQueue.push(vars_map[var_id].pointer, var_id)

        # print(self.queue.qsize())

        while not self.pQueue.empty():
            var = self.pQueue.pop()
            for i in range(0, self.N):
                if var != i and (self.revise(var, i, vars_map) or
                                 self.revise(i, var, vars_map)):
                    self.pQueue.clear()
                    return None
        return vars_map





