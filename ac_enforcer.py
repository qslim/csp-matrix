import numpy as np


class ACEnforcer:
    def __init__(self, cons_map, N, D):
        self.N = N
        self.cons_map = cons_map
        self.var_mask = np.ones([N, N, 1])
        self.dom_mask = np.ones([D, 1])
        self.dummy = np.zeros([N, 1])

    def ac_enforcer(self, vars_map):
        vars_map_pre = None
        while (vars_map_pre != vars_map).any():
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")
            vars_map_pre = vars_map.copy()

            NND = np.matmul(self.cons_map, vars_map).squeeze()

            NND = np.minimum(NND, 1)

            NDN = NND.transpose((0, 2, 1))

            ND1 = np.matmul(NDN, self.var_mask)

            vars_map = np.where(ND1 == self.N, 1, 0)

            if (np.matmul(vars_map.squeeze(), self.dom_mask) == self.dummy).any():
                return None

        return vars_map
