import torch


class SACEnforcer:
    def __init__(self, cons_map, N, D):
        device = torch.device("cuda")
        self.N = N
        self.cons_map = cons_map
        self._1n_mask1 = torch.ones((1, N)).to(device)
        self.n1_mask0 = torch.zeros((N, 1)).to(device)
        self.ndnnd_mask1 = torch.ones((N, D, N, N, D)).to(device)
        self.ndn1d_mask1 = torch.ones((N, D, N, 1, D)).to(device)
        self.ndn1d_mask0 = torch.zeros((N, D, N, 1, D)).to(device)
        self.ndn11_mask1 = torch.ones((N, D, N, 1, 1)).to(device)
        self.ndn_mask1 = torch.ones((N, D, N)).to(device)
        self.nd_mask1 = torch.ones((N, D)).to(device)
        self.nd_mask0 = torch.zeros((N, D)).to(device)
        self.assigh_mask = self.build_assign_mask(N, D).to(device)
        # self.count = 0

    @staticmethod
    def build_assign_mask(N, D):
        assign_mask = torch.ones((N, D, N, 1, D))
        for i in range(N):
            dn1d = assign_mask[i]
            for j in range(D):
                for k in range(D):
                    if j != k:
                        dn1d[j][i][0][k] = 0
        return assign_mask

    def ac_enforcer(self, vars_map):
        # print(vars_map.type())
        ndn1d = torch.matmul(self.ndn11_mask1, vars_map.unsqueeze(1))
        ndn1d = torch.where(self.assigh_mask == 1, ndn1d, self.assigh_mask)
        vars_map_pre = self.ndn1d_mask0
        while torch.equal(ndn1d, vars_map_pre) is False:
            # print("~~~~~~~~~~~~~~~~~~~~~~~~")

            vars_map_pre = ndn1d
            nd1n1d = ndn1d.unsqueeze(2)
            ndnnd = torch.matmul(nd1n1d, self.cons_map).squeeze()
            ndnnd_reduce = torch.where(ndnnd > 1, self.ndnnd_mask1, ndnnd)
            ndn1d_ = torch.matmul(self._1n_mask1, ndnnd_reduce)
            ndn1d = torch.where(ndn1d_ == self.N, self.ndn1d_mask1, self.ndn1d_mask0)

            _ndnd = ndn1d.squeeze()
            _ndn = _ndnd.sum(3)
            _ndn = torch.where(_ndn > 1, self.ndn_mask1, _ndn)
            _nd = _ndn.sum(2)
            vars_map = torch.where(_nd == self.N, self.nd_mask1, self.nd_mask0)
            if (vars_map.sum(1) == self.n1_mask0).any():
                return None

        return vars_map
