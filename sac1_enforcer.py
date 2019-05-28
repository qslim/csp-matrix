import torch


class SAC1Enforcer:
    def __init__(self, cons_map, N, D):
        device = torch.device("cuda")
        self.N = N
        self.cons_map = cons_map
        self._1n_mask1 = torch.ones((1, N)).to(device)
        self.n1_mask0 = torch.zeros((N, 1)).to(device)
        self.nnndd_mask1 = torch.ones((N, N, N, D, D)).to(device)
        self.nndd_mask1 = torch.ones((N, N, D, D)).to(device)
        self.nndd_mask0 = torch.zeros((N, N, D, D)).to(device)
        self.n1d1_mask1 = torch.ones((N, 1, D, 1)).to(device)
        self.nnd_mask1 = torch.ones((N, N, D)).to(device)
        self.nd_mask1 = torch.ones((N, D)).to(device)
        self.nd_mask0 = torch.zeros((N, D)).to(device)
        self.assigh_mask = self.build_assign_mask(N, D).to(device)
        # self.count = 0

    @staticmethod
    def build_assign_mask(N, D):
        assign_mask = torch.ones((N, N, D, D))
        for i in range(N):
            ndd = assign_mask[i]
            for j in range(D):
                for k in range(D):
                    if j != k:
                        ndd[i][j][k] = 0
        return assign_mask

    def ac_enforcer(self, vars_map):
        # print(vars_map.type())
        nndd = torch.matmul(self.n1d1_mask1, vars_map.unsqueeze(1))
        nndd = torch.where(self.assigh_mask == 1, nndd, self.assigh_mask)
        vars_map_pre = self.nndd_mask0
        while torch.equal(nndd, vars_map_pre) is False:
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")
            vars_map_pre = nndd

            n1ndd = nndd.unsqueeze(1)
            nnndd = torch.matmul(n1ndd, self.cons_map)
            nnndd_reduce = torch.where(nnndd > 1, self.nnndd_mask1, nnndd)
            nndd = nnndd_reduce.sum(2)
            nndd = torch.where(nndd == self.N, self.nndd_mask1, self.nndd_mask0)

            _nnd = nndd.sum(3)
            _nnd = torch.where(_nnd > 1, self.nnd_mask1, _nnd)
            _nd = _nnd.sum(1)
            vars_map = torch.where(_nd == self.N, self.nd_mask1, self.nd_mask0)
            if (vars_map.sum(1) == self.n1_mask0).any():
                return None

        return vars_map
