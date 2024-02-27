import numpy as np
import numpy.linalg as npl

from tqdm import tqdm
from functools import partial

from init_utils import *

from g_utils import g, gradient_g

from FISTA_restart_descent import FISTA_restart


def opCOMP(
    y,
    linop,
    step,
    nb_tests=1,
    max_iter=None,
    min_iter=None,
    descent_nit=500,
    clip=None,
    init_position=init_position_max_val,
    disable_tqdm_init=False,
    sliding=True,
):
    d = linop.d
    bounds_min = linop.bounds["min"]
    bounds_max = linop.bounds["max"]

    norm_y = npl.norm(y)

    r = np.copy(y)
    T = np.array([]).reshape(0, d)
    a_best = np.array([])

    errors_criterion = [1]
    errors = [npl.norm(y) ** 2]
    # errors.append(npl.norm(r))

    matrix_condition = []

    descent = partial(
        FISTA_restart_single_spike,
        linop=linop,
        step=step,
        nit=descent_nit,
        alpha=3,
        convergence_criteria_abs=1e-8,
        convergence_criteria_rel=1e-8,
        clip=clip,
    )

    grad_sliding = partial(gradient_g, y=y, linop=linop)
    func_sliding = partial(g, y=y, linop=linop)

    for i in tqdm(range(max_iter), disable=disable_tqdm_init):
        t_gen = init_position(r, linop)
        t_best = descent(t_gen, residue=r)
        r_best = npl.norm(r - linop.Adelta(t_best))

        # Tries multiple random initialization to find the best
        for test in range(1, nb_tests):
            t_gen = init_position(r, linop)
            t_int = descent(t_gen, residue=r)
            r_int = npl.norm(r - linop.Adelta(t_int))
            if r_int < r_best:
                t_best = np.copy(t_int)
                r_best = r_int

        T = np.concatenate((T, t_best), axis=0)

        M = linop.Adelta(T).T

        a_best = npl.lstsq(M, y, rcond=None)[0]

        if sliding:
            X_best = np.concatenate((T, a_best[:, None]), axis=1)

            X_sliding, _ = FISTA_restart(
                X_best,
                step=step,
                gradient_functional=grad_sliding,
                nit=10,
                alpha=3,
                functional=func_sliding,
                # convergence_criteria_abs=1e-10,
                # convergence_criteria_rel=1e-10,
                project=None,
                disable_tqdm=True,
                restart=True
            )
            a_best = X_sliding[:, -1]
            T = X_sliding[:, :-1]

        else:
            X_sliding = np.concatenate((T, a_best[:, None]), axis=1)

        r = y - linop.Ax(a_best, T)

        error_criterion = (g(X_sliding, y, linop) ** .5) / norm_y
        errors_criterion.append(error_criterion)
        error = g(X_sliding, y, linop)
        errors.append(error)

        # if errors[-1] > errors[-2]:
        #     # print(i, X_sliding)
        #     a_best = a_best[:-1]
        #     T = T[:-1]

        if i >= min_iter - 1:
            if errors_criterion[-1] <= 0.03:
                break

    a_init = np.copy(a_best)
    t_init = np.copy(T)

    return a_init, t_init, np.array(errors), r
