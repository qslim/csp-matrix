# import numpy as np
import torch


class ACEnforcer:
    def __init__(self, cons_map, N, D):
        device = torch.device("cuda")
        self.N = N
        self.cons_map = cons_map
        self.n1_mask0 = torch.zeros((N, 1)).to(device)
        self.nnd_mask1 = torch.ones((N, N, D)).to(device)
        self.nd_mask1 = torch.ones((N, D)).to(device)
        self.nd_mask0 = torch.zeros((N, D)).to(device)
        # self.count = 0

    def ac_enforcer(self, vars_map):
        # print(vars_map.type())
        vars_map_pre = self.nd_mask0
        while not torch.equal(vars_map, vars_map_pre):
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")

            vars_map_pre = vars_map

            nnd = torch.matmul(self.cons_map, vars_map.unsqueeze(2)).squeeze()

            nnd_reduce = torch.where(nnd > 1, self.nnd_mask1, nnd)

            nd = nnd_reduce.sum(1, keepdim=True).squeeze()

            vars_map = torch.where(nd == self.N, self.nd_mask1, self.nd_mask0)

            if (vars_map.sum(1) == self.n1_mask0).any():
                return None

        return vars_map
