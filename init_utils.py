import numpy as np
import numpy.linalg as npl

from typing import Callable


def init_position_random(d, lower_bounds, upper_bounds):
    t = np.random.rand(1, d) * (upper_bounds - lower_bounds) + lower_bounds
    return t


def init_position_max_val(y, linop, repeats=9):
    argmax = np.argmax(y)
    index = np.unravel_index(argmax, (linop.N1, linop.N2, linop.K))
    t = index / np.array([linop.N1 - 1, linop.N2 - 1, linop.K - 1]) * linop.bounds['max']
    t[[0, 1]] = t[[1, 0]]
    t = t[None, :]
    z_pos = np.linspace(linop.bounds['min'][2], linop.bounds['max'][2],
                        num=repeats, endpoint=True)
    t_temp = np.repeat(t, repeats=repeats, axis=0)
    t_temp[:, 2] = z_pos
    # print(t_temp)
    norms = np.linalg.norm(y - linop.Adelta(t_temp), axis=-1)
    # print(norms, np.argmin(norms))
    t = t_temp[np.argmin(norms)]

    return t[None, :]


def gradient_Adeltat_dot_residue(t, residue, linop):
    df_dt = np.real(np.inner(linop.Adeltap(t), np.conj(residue))).T
    return df_dt


def Adeltat_dot_residue(t, residue, linop):
    f = np.inner(linop.Adelta(t), np.conj(residue))
    return f


def FISTA_restart_single_spike(
        X: np.ndarray,
        residue: np.ndarray,
        linop,
        step: float,
        nit: int,
        alpha: float = 3,
        convergence_criteria_abs: float = 1e-5,
        convergence_criteria_rel: float = 1e-5,
        clip: Callable = None
) -> np.ndarray:
    X0 = X
    Y0 = X0
    fX0 = Adeltat_dot_residue(X0, residue, linop)

    it_fista = 0

    for it in range(nit):
        gradY = gradient_Adeltat_dot_residue(X0, residue, linop)
        # print(gradY)
        X1 = Y0 - step * gradY
        fista_acc = it_fista / (it_fista + alpha)
        Y1 = X1 + fista_acc * (X1 - X0)

        if clip is not None:
            X1 = clip(X1)

        if npl.norm(X1 - X0) < convergence_criteria_abs:
            # print("Reached Absolute Tolerance")
            break
        if (npl.norm(X1 - X0) / npl.norm(X0)) <= convergence_criteria_rel:
            break

        # Restart
        # fX1 = Adeltat_dot_residue(X1, residue, linop)
        # if fX1 > fX0:
        #     it_fista = 0
        # else:
        #     fX0 = fX1
        #     X0, Y0 = np.copy(X1), Y1
        #     it_fista += 1

        X0, Y0 = np.copy(X1), Y1
        it_fista += 1

    # print(it)

    return X1
