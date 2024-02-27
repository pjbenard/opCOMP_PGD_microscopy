import numpy as np
from tqdm import tqdm

from typing import Callable, Tuple


def FISTA(
        X: np.ndarray,
        step: float,
        gradient_functional: Callable,
        nit: int,
        alpha: float = 3,
        functional: Callable = None,
        disable_tqdm: bool = False,
        convergence_criteria_abs: float = None,
        convergence_criteria_rel: float = None,
        project: Callable = None,
        clip: Callable = None
) -> Tuple[np.ndarray, list]:
    X0 = X
    Y = X0
    if functional is not None:
        functional_values = np.zeros(nit + 1)
        functional_values[0] = functional(X0)

    it_fista = 0

    for it in tqdm(range(nit), disable=disable_tqdm):
        X1 = Y - step * gradient_functional(Y)
        Y = X1 + (it_fista / (it_fista + alpha)) * (X1 - X0)

        if clip is not None:
            X1[:, :-1] = clip(X1[:, :-1])

        # X0 = np.copy(X1)
        it_fista += 1

        if project is not None:
            X1 = project(X1)
            if X1.shape[0] < X0.shape[0]:
                it_fista = 0
                Y = X1

        if functional is not None:
            functional_values[it + 1] = functional(X1)
            if convergence_criteria_abs is not None:
                if functional_values[it + 1] < convergence_criteria_abs:
                    # print("Reached Absolute Tolerance")
                    break
            if convergence_criteria_rel is not None:
                relative_difference = abs(functional_values[it + 1] - functional_values[it]) / abs(
                    functional_values[it + 1])
                if relative_difference < convergence_criteria_rel:
                    # print("Reached Relati*ve Tolerance")
                    break

        X0 = np.copy(X1)

    returns = []
    if functional is not None:
        returns += [functional_values]

    return X1, returns
