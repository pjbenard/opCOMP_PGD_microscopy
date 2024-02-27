import numpy as np
from tqdm import tqdm

from typing import Callable, Tuple

def FISTA_restart(
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
        clip: Callable = None,
        restart=True
) -> Tuple[np.ndarray, list]:
    X0 = X
    Y = X0

    traj_X = np.zeros((nit+1, X.shape[0], X.shape[1]))
    traj_X[0] = np.copy(X0)

    # Numpy array
    functional_values = np.zeros(nit + 1)
    functional_values[0] = functional(X0)
    # Pyton list
    # functional_values = [functional(X0)]

    it_fista = 0

    for it in tqdm(range(nit), disable=disable_tqdm):
        # print(f"{it = }")
        projected = False

        it_fista += 1

        # print(f"{Y = }")
        # print("gradient", gradient_functional(Y))
        X1 = Y - step * gradient_functional(Y)

        if clip is not None:
            X1[:, :-1] = clip(X1[:, :-1])

        # X0 = np.copy(X1)

        # if it > 200:
        #     print(f"{it = }")
        #     print("before proj", X1)

        if (project is not None) and (it > 100):
            X1 = project(X1)
            projected = X1.shape[0] < X0.shape[0]

        # if it > 200:
        #     print("after proj", X1)

        functional_values[it + 1] = functional(X1)

        if convergence_criteria_abs is not None:
            if functional_values[it + 1] < convergence_criteria_abs:
                # print("Reached Absolute Tolerance")
                break
        if convergence_criteria_rel is not None:
            a, b = functional_values[it + 1], functional_values[it]
            if (abs(a - b) / abs(b)) <= convergence_criteria_rel:
                # print(a, b)
                break

        # Restart
        if projected:
            it_fista = 1
            X0 = np.copy(X1)
        elif restart and (functional_values[it + 1] > functional_values[it]):
            it_fista = 1

        fista_acc = (it_fista - 1) / (it_fista - 1 + alpha)
        # print(f"{X0 = }")
        # print(f"{X1 = }")
        # print(f"{it_fista = }, {fista_acc = }")
        Y = X1 + fista_acc * (X1 - X0)
        X0 = np.copy(X1)
        traj_X[it + 1, :X1.shape[0], :] = np.copy(X1)

    returns = []
    returns += [np.array(functional_values)]
    returns += [traj_X]

    return X1, returns


# if convergence_criteria_abs is not None:
#     if functional_values[it + 1] < convergence_criteria_abs:
#         # if functional_values[-1] < convergence_criteria_abs:
#         #     print("Reached Absolute Tolerance")
#         break
# if convergence_criteria_rel is not None:
#     relative_difference = abs(functional_values[it + 1] - functional_values[it]) / abs(
#         functional_values[it + 1])
#     # relative_difference = abs(functional_values[-1] - functional_values[-2]) / abs(
#     #     functional_values[-1])
#     if 0 < relative_difference < convergence_criteria_rel:
#         # print("Reached Relative Tolerance")
#         break
