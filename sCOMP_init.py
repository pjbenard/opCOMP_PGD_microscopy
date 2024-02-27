import numpy as np
import numpy.linalg as npl

from tqdm import tqdm
from functools import partial

from init_utils import *

from g_utils import g, gradient_g

from FISTA_restart_descent import FISTA_restart


def sCOMP(
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
):
    d = linop.d
    bounds_min = linop.bounds["min"]
    bounds_max = linop.bounds["max"]

    norm_y = npl.norm(y)

    r = np.copy(y)
    T = np.array([]).reshape(0, d)
    a_best = np.array([])

    errors = []
    errors.append(1)

    matrix_condition = []

    gradient = partial(
        gradient_Adeltat_dot_residue,
        linop=linop)

    func = partial(
        Adeltat_dot_residue,
        linop=linop
    )

    descent = partial(
        FISTA_restart_single_spike,
        linop=linop,
        step=step,
        nit=descent_nit,
        alpha=3,
        convergence_criteria_abs=1e-8,
        clip=clip,
    )

    grad_sliding = partial(gradient_g, y=y, linop=linop)
    func_sliding = partial(g, y=y, linop=linop)

    for i in tqdm(range(max_iter), disable=disable_tqdm_init):
        t_gen = init_position(r, linop)
        # print(t_gen)
        # print(t_gen)
        t_best = descent(t_gen, residue=r)

        # Tries multiple random initialization to find the best
        r_best = npl.norm(r - linop.Adelta(t_best))
        for test in range(1, nb_tests):
            t_gen = init_position(r, linop)
            t_int = descent(t_gen, residue=r)
            r_int = npl.norm(r - linop.Adelta(t_int))
            if r_int < r_best:
                t_best = np.copy(t_int)
                r_best = r_int

        T = np.concatenate((T, t_best), axis=0)
        a_prev = np.concatenate((a_best, [0]))

        M = linop.Adelta(T).T

        a_best = npl.lstsq(M, y, rcond=None)[0]

        X_best = np.concatenate((T, a_best[:, None]), axis=1)
        # print('Before GD')
        # print(X_best)
        X_sliding, _ = FISTA_restart(
            X_best,
            step=step,
            gradient_functional=grad_sliding,
            nit=descent_nit,
            alpha=3,
            functional=func_sliding,
            convergence_criteria_abs=1e-10,
            convergence_criteria_rel=1e-10,
            project=None,
            disable_tqdm=True,
            restart=True
        )
        # print('After GD')
        # print(_[0][-10:])
        # print(X_sliding)

        a_best = X_sliding[:, -1]
        T = X_sliding[:, :-1]

        M = linop.Adelta(T).T

        r = y - np.dot(M, a_best)

        y_x = linop.Ax(a_best, T)
        error = npl.norm(y_x - y) / norm_y
        errors.append(error)

        if i >= min_iter - 1:
            if errors[-1] <= 0.05:
                break

    a_init = np.copy(a_best)
    t_init = np.copy(T)

    return a_init, t_init, np.array(errors), r
