import numpy as np


class ACEnforcer:
    def __init__(self, cons_map, N, D):

        self.cons_map = cons_map
        # self.tmp = np.arange(N * D * 1).reshape(N, D, 1)
        self.var_mask = np.ones([N, N, 1])
        self.dom_mask = np.ones([D, 1])
        # print(var_mask.shape)
        self.sup_mask = (N - 1) * np.ones([N, D, 1])
        # print(var_mask_)
        # var_map = np.zeros([N, D, 1])

        self.dummy = np.zeros([N, 1])

    def ac_enforcer(self, vars_map):
        # print(type(vars_map))
        vars_map_pre = None
        while (vars_map_pre != vars_map).any():
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")
            # print((vars_map_pre != vars_map).any())
            vars_map_pre = vars_map.copy()

            st1 = np.matmul(self.cons_map, vars_map)
            # print(st1.shape)

            st2 = st1.squeeze()
            # print(st2.shape)

            st3 = np.minimum(st2, 1)

            st4 = st3.transpose((0, 2, 1))
            # print(st4.shape)

            st5 = np.matmul(st4, self.var_mask)
            # print(st5.shape)

            vars_map = np.maximum(st5 - self.sup_mask, 0)
            # print(vars_map.shape)

            if (np.matmul(vars_map.squeeze(), self.dom_mask) == self.dummy).any():
                # print("throw inconsistency")
                return None

        return vars_map
