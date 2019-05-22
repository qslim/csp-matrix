import numpy as np


class ACEnforcer:
    def __init__(self, cons_map, N, D):
        self.N = N
        self.cons_map = cons_map
        self.var_mask = np.ones([N, 1, N])
        self.dom_mask = np.ones([D, 1])
        self.dummy = np.zeros([N, 1])

    def ac_enforcer(self, vars_map):
        vars_map_pre = None
        while (vars_map_pre != vars_map).any():
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")
            vars_map_pre = vars_map.copy()

            NN1D = np.matmul(vars_map, self.cons_map)

            NND = np.minimum(NN1D, 1).squeeze()

            # NDN = NND.transpose((0, 2, 1))

            N1D = np.matmul(self.var_mask, NND)

            vars_map = np.where(N1D == self.N, 1, 0)

            if (np.matmul(vars_map.squeeze(), self.dom_mask) == self.dummy).any():
                return None

        return vars_map
