import torch


class ACParallel:
    def __init__(self, cons_map, N, D):
        device = torch.device("cuda")
        self.N = N
        self.cons_map = cons_map
        self._1n_mask1 = torch.ones((1, N)).to(device)
        self.n1_mask0 = torch.zeros((N, 1)).to(device)
        self.nndd_mask1 = torch.ones((N, N, D, D)).to(device)
        self.ndd_mask1 = torch.ones((N, D, D)).to(device)
        self.ndd_mask0 = torch.zeros((N, D, D)).to(device)
        # self.count = 0

    def ac_enforcer(self, vars_map):
        # print(vars_map.type())
        vars_map_pre = self.ndd_mask0
        while torch.equal(vars_map, vars_map_pre) is False:
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")

            vars_map_pre = vars_map

            nndd = torch.matmul(vars_map, self.cons_map)

            nndd_reduce = torch.where(nndd > 1, self.nndd_mask1, nndd)

            ndnd = nndd_reduce.transpose(1, 2)

            ndd = torch.matmul(self._1n_mask1, ndnd).squeeze()

            vars_map = torch.where(ndd == self.N, self.ndd_mask1, self.ndd_mask0)

            if (vars_map.sum((1, 2)) == self.n1_mask0).any():
                return None

        return vars_map
