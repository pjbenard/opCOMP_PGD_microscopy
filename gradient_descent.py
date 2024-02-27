import numpy as np
import numpy.linalg as npl
from tqdm import tqdm

from descent_utilities import project_theta_eps, clip_domain
from g_utils import gradient_g, g

def norm_residue(y, linop, a, t):
    return g(np.concatenate((t, a[:, None]), axis=-1), y, linop)**.5

def gradient_descent(
    y,
    linop,
    a_init,
    t_init,
    eps_proj=0.05,
    project=True,
    tau={"min": -5, "max": 0},
    nit=500,
    disable_tqdm_gd=False,
    clip=True,
):

    k, d = np.shape(t_init)
    k_int = k

    a_est = np.copy(a_init)
    t_est = np.copy(t_init)

    traj_a = np.zeros((nit + 1, k))
    traj_t = np.zeros((nit + 1, k, d))
    
    traj_a[0, :k_int] = a_init
    traj_t[0, :k_int] = t_init

    errors = []

    norm_res_old = np.inf

    for it in tqdm(range(nit), desc=f"Nb spikes = {k}", disable=disable_tqdm_gd):
        norm_res = norm_residue(y, linop, a_est, t_est)
        errors.append(norm_res)

        d_X_est = gradient_g(np.concatenate((t_est, a_est[:, None]), axis=-1), y, linop)
        dt = d_X_est[:, :-1]
        da = d_X_est[:, -1]

        for etau in range(tau["max"], tau["min"], -1):
            step = 10**etau

            a_est_int = a_est - step * da
            t_est_int = t_est - step * dt

            if clip:
                clip_domain(linop, t_est_int)

            norm_res_int = norm_residue(y, linop, a_est_int, t_est_int)

            if norm_res_int < norm_res:
                a_est = np.copy(a_est_int)
                t_est = np.copy(t_est_int)

                norm_res = norm_res_int
                best_etau = etau

        if project and (it % 2 == 0):
            a_est, t_est = project_theta_eps(a_est, t_est, eps_proj)
            k_int = np.shape(t_est)[0]

        traj_a[it + 1, :k_int] = np.copy(a_est)
        traj_t[it + 1, :k_int] = np.copy(t_est)

        norm_res_old = norm_res

    errors.append(norm_res)

    return a_est, t_est, traj_a, traj_t, np.array(errors)