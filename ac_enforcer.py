# import numpy as np
import torch


class ACEnforcer:
    def __init__(self, cons_map, N, D):
        # device = torch.device("cuda")
        self.N = N
        self.cons_map = cons_map

        self.n1n_mask1 = torch.ones((N, 1, N))
        self.d1_mask1 = torch.ones((D, 1))
        self.n1_mask0 = torch.zeros((N, 1))
        self.nnd_mask1 = torch.ones((N, N, D))
        self.n1d_mask1 = torch.ones((N, 1, D))
        self.n1d_mask0 = torch.zeros((N, 1, D))

        # self.n1n_mask1 = torch.ones((N, 1, N)).to(device)
        # self.d1_mask1 = torch.ones((D, 1)).to(device)
        # self.n1_mask0 = torch.zeros((N, 1)).to(device)
        # self.nnd_mask1 = torch.ones((N, N, D)).to(device)
        # self.n1d_mask1 = torch.ones((N, 1, D)).to(device)
        # self.n1d_mask0 = torch.zeros((N, 1, D)).to(device)
        # self.count = 0

    def ac_enforcer(self, vars_map):
        # print(vars_map.type())
        vars_map = vars_map.unsqueeze(1)
        vars_map_pre = self.n1d_mask0
        while torch.equal(vars_map, vars_map_pre) is False:
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")

            # self.count += 1

            vars_map_pre = vars_map

            nnd = torch.matmul(vars_map, self.cons_map).squeeze()

            nnd_reduce = torch.where(nnd > 1, self.nnd_mask1, nnd)

            n1d = torch.matmul(self.n1n_mask1, nnd_reduce)

            vars_map = torch.where(n1d == self.N, self.n1d_mask1, self.n1d_mask0)

            if (torch.matmul(vars_map.squeeze(), self.d1_mask1) == self.n1_mask0).any():
                return None

        return vars_map.squeeze()
