import torch


def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (
        x0,
        dx0,
        -2 * dx0 / tf - dxf / tf - 3 * x0 / tf**2 + 3 * xf / tf**2,
        dx0 / tf**2 + dxf / tf**2 + 2 * x0 / tf**3 - 2 * xf / tf**3,
    )


def compute_spline_xyvaqrt(v0, dv0, vf, tf, path, N, offset):
    t = torch.arange(N + 1).to(v0.device) * tf / N
    tp = t[..., None] ** torch.arange(4).to(v0.device)
    dtp = t[..., None] ** torch.tensor([0, 0, 1, 2]).to(v0.device) * torch.arange(4).to(
        v0.device
    )

    coefficients = cubic_spline_coefficients(v0, dv0, vf, 0, tf)
    coefficients = torch.stack(coefficients).unsqueeze(-1)

    v = tp @ coefficients
    a = dtp @ coefficients
    s = torch.cumsum(v * tf / N, dim=0)
    s = torch.cat((torch.zeros(1, 1).to(v0.device), s[:-1]), dim=0)
    s += offset
    i = (s / 0.1).long()

    if i[-1] > path.shape[0] - 1:
        return

    x = path[i, 0]
    y = path[i, 1]
    yaw = path[i, 2]
    r = path[i, 3]

    return torch.cat((x, y, yaw, v, a, r, t.unsqueeze(-1)), -1).squeeze(0)


class SplinePlanner:
    def __init__(self, first_stage_horizion, horizon):
        self.spline_order = 3
        self.max_curve = 0.3
        self.max_lat_acc = 3.0
        self.acce_bound = [-5, 3]
        self.vbound = [0, 15.0]
        self.first_stage_horizion = first_stage_horizion
        self.horizon = horizon

    def calc_trajectory(self, v0, a0, vf, tf, path, N_seg, offset=0):
        traj = compute_spline_xyvaqrt(v0, a0, vf, tf, path, N_seg, offset)

        return traj

    def gen_short_term_trajs(self, x0, tf, paths, dyn_filter):
        xf_set = []
        trajs = []

        # generate speed profile and trajectories
        for path in paths:
            path = torch.from_numpy(path).to(x0.device).type(torch.float)
            for v in self.v_grid:
                traj = self.calc_trajectory(
                    x0[3], x0[4], v, tf, path, self.first_stage_horizion * 10
                )  # [x, y, yaw, v, a, r, t]
                if traj is None:
                    continue

                xf = traj[-1, :2]
                if (
                    xf_set
                    and torch.cdist(xf.unsqueeze(0), torch.stack(xf_set)).min() < 0.5
                ):
                    continue
                else:
                    xf_set.append(xf)
                    trajs.append(traj)

        trajs = torch.stack(trajs)

        # remove trajectories that are not feasible
        if dyn_filter:
            feas_flag = self.feasible_flag(trajs)
            trajs = trajs[feas_flag]

        return trajs

    def gen_long_term_trajs(self, x0, tf, paths, dyn_filter):
        xf_set = []
        trajs = []

        # generate speed profile and trajectories
        for path in paths:
            path = torch.from_numpy(path).to(x0.device).type(torch.float)
            dist = torch.norm(path[:, :2] - x0[:2], dim=1)
            if dist.min() > 0.1:
                continue

            offset = torch.argmin(dist) * 0.1

            for v in self.v_grid:
                traj = self.calc_trajectory(
                    x0[3],
                    x0[4],
                    v,
                    tf,
                    path,
                    (self.horizon - self.first_stage_horizion) * 10,
                    offset,
                )  # [x, y, yaw, v, a, r, t]
                if traj is None:
                    continue

                xf = traj[-1, :2]

                if (
                    xf_set
                    and torch.cdist(xf.unsqueeze(0), torch.stack(xf_set)).min() < 0.5
                ):
                    continue
                else:
                    xf_set.append(xf)
                    trajs.append(traj)

        if len(trajs) == 0:
            return
        else:
            trajs = torch.stack(trajs)

        # remove trajectories that are not feasible
        if dyn_filter:
            feas_flag = self.feasible_flag(trajs)
            trajs = trajs[feas_flag]

        return trajs

    def feasible_flag(self, trajs):
        feas_flag = (
            (trajs[:, 1:, 3] >= self.vbound[0])
            & (trajs[:, 1:, 3] <= self.vbound[1])
            & (trajs[:, 1:, 4] >= self.acce_bound[0])
            & (trajs[:, 1:, 4] <= self.acce_bound[1])
            & (trajs[:, 1:, 5].abs() * trajs[:, 1:, 3] ** 2 <= self.max_lat_acc)
            & (trajs[:, 1:, 5].abs() <= self.max_curve)
        ).all(1)

        if feas_flag.sum() == 0:
            print("No feasible trajectory")
            feas_flag = torch.ones(trajs.shape[0], dtype=torch.bool).to(trajs.device)

        return feas_flag

    def gen_trajectories(self, x0, tf, paths, speed_limit, is_root):
        # generate trajectories
        v0 = x0[3]

        if is_root:
            v_min = max(v0 - 4.0 * tf, 0.0)
            v_max = min(v0 + 2.4 * tf, speed_limit)
            self.v_grid = torch.linspace(v_min, v_max, 10).to(x0.device)
            trajs = self.gen_short_term_trajs(x0, tf, paths, dyn_filter=False)
        else:
            v_min = max(v0 - tf, 0.0)
            v_max = min(v0 + tf, speed_limit)
            self.v_grid = torch.linspace(v_min, v_max, 5).to(x0.device)
            trajs = self.gen_long_term_trajs(x0, tf, paths, dyn_filter=False)

        # adjust timestep
        if not is_root:
            trajs[:, :, -1] += self.horizon - self.first_stage_horizion

        # remove the first time step
        trajs = trajs[:, 1:]

        return trajs
