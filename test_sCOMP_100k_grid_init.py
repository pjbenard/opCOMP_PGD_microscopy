import numpy as np
from tqdm import tqdm

from functools import partial

from linop_MA_TIRF import MA_TIRF

from g_utils import gradient_g, g

from sCOMP_init import sCOMP
from descent_utilities import clip_domain, project_theta_eps

from FISTA_restart_descent import FISTA_restart
from init_utils import FISTA_restart_single_spike, init_position_max_val
from data_utils import create_batches, semi_gridded_init

from analyse_utils import pair_GT_estimation, compute_metrics, RMSE

from scipy.spatial import distance_matrix


def projection(X, eps_proj, cut_off=1e-2):
    a, t = X[:, -1], X[:, :-1]
    a, t = project_theta_eps(a, t, eps_proj, cut_off)
    return np.concatenate((t, a[:, None]), axis=-1)


seed = 100
np.random.seed(seed)

N = 100
k_min = N
k_max = N + 35
nb_batch = 1

single_batch = nb_batch == 1

lambda_l = 0.66
detail_coef = 1
N1, N2 = 64 * detail_coef, 64 * detail_coef
K_angles = 4 * detail_coef


if __name__ == "__main__":
    linop = MA_TIRF(lambda_l=lambda_l, N1=N1, N2=N2, K=K_angles)
    # batches = create_batches(nbatch=N, plafrim_path=True)

    T0 = np.zeros([0, 3])
    A0 = np.zeros([0])

    T_INIT = np.zeros([0, 3])
    A_INIT = np.zeros([0])

    T_ESTI = np.zeros([0, 3])
    A_ESTI = np.zeros([0])

    for _ in tqdm(range(nb_batch), disable=single_batch):
        t0 = semi_gridded_init()
        T0 = np.concatenate((T0, t0), axis=0)
        N = t0.shape[0]

        a0 = np.random.uniform(1, 1.5, N)
        A0 = np.concatenate((A0, a0))

        min_dist = np.min(distance_matrix(t0, t0) + np.eye(N))
        eps_proj = min_dist * 0.75

        y = linop.Ax(a0, t0)

        clip = partial(
            clip_domain,
            linop=linop
        )

        a_init, t_init, errors, r = sCOMP(
            y,
            linop=linop,
            step=.001,
            nb_tests=1,
            descent_nit=1000,
            max_iter=k_min,
            min_iter=k_min + 1,
            clip=clip,
            disable_tqdm_init=False
        )

        T_INIT = np.concatenate((T_INIT, t_init), axis=0)
        A_INIT = np.concatenate((A_INIT, a_init))

    t_true_paired, t_true_not_paired, t_esti_paired, t_esti_not_paired = pair_GT_estimation(T0, T_INIT)

    metrics = compute_metrics(t_true_paired, t_true_not_paired, t_esti_paired, t_esti_not_paired)
    rmse = np.array(RMSE(t_true_paired, t_esti_paired)) # * 1000 # micro_m to nm

    print(f"{metrics = }")
    print(f"{rmse = }")

    np.savez_compressed('data_large_sCOMP_k100_grid.npz',
                        T0=T0, T_INIT=T_INIT,
                        A0=A0, A_INIT=A_INIT,
                        error_sCOMP=errors)
