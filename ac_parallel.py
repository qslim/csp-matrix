import torch


class ACParallel:
    def __init__(self, cons_map, N, D):
        device = torch.device("cuda")
        self.N = N
        self.cons_map = cons_map
        self._1n_mask1 = torch.ones((1, N)).to(device)
        self.n1_mask0 = torch.zeros((N, 1)).to(device)
        self.nnndd_mask1 = torch.ones((N, N, N, D, D)).to(device)
        self.nndd_mask1 = torch.ones((N, N, D, D)).to(device)
        self.nndd_mask0 = torch.zeros((N, N, D, D)).to(device)
        # self.count = 0
        self.nndt1_mask1 = torch.ones((N, N, D, 1)).to(device)
        self.nnd_mask1 = torch.ones((N, N, D)).to(device)
        self.nd_mask1 = torch.ones((N, D)).to(device)
        self.nd_mask0 = torch.zeros((N, D)).to(device)

    def ac_enforcer(self, vars_map):
        # print(vars_map.type())
        n1d = vars_map.unsqueeze(1)
        nndd = torch.matmul(self.nndt1_mask1, n1d)
        vars_map_pre = self.nndd_mask0
        while torch.equal(nndd, vars_map_pre) is False:
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")

            vars_map_pre = nndd
            nn1dd = nndd.unsqueeze(1)
            nnndd = torch.matmul(nn1dd, self.cons_map)
            nnndd_reduce = torch.where(nnndd > 1, self.nnndd_mask1, nnndd)
            nndnd = nnndd_reduce.transpose(2, 3)
            nndd_ = torch.matmul(self._1n_mask1, nndnd).squeeze()
            nndd = torch.where(nndd_ == self.N, self.nndd_mask1, self.nndd_mask0)

            nnd = nndd.sum(3)
            nnd = torch.where(nnd > 1, self.nnd_mask1, nnd)
            nd = nnd.sum(1)
            vars_map = torch.where(nd == self.N, self.nd_mask1, self.nd_mask0)
            if (vars_map.sum(1) == self.n1_mask0).any():
                return None

        return vars_map
