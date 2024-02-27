import numpy as np


def barycenter(a, t):
    abs_a = np.abs(a)
    return (np.sum(abs_a[:, None] * t, axis=0)) / (sum(abs_a))


def project_theta_eps(a, t, eps_proj, cut_off=1e-2):
    k, d = np.shape(t)
    a_temp = np.copy(a)
    t_temp = np.copy(t)
    argsort = np.argsort(a_temp)[::-1]
    a_temp = a_temp[argsort]
    # print(a_temp)
    t_temp = t_temp[argsort]

    for i in range(0, k):
        if a_temp[i] == 0:
            continue

        elif abs(a_temp[i]) < cut_off:
            a_temp[i] = 0
            t_temp[i] = 0
            continue

        dist_i_remainder = np.sqrt(np.sum((t_temp[i:] - t_temp[i]) ** 2, axis=-1))
        idx_close = dist_i_remainder < eps_proj

        if not any(idx_close[1:]):
            continue

        a_temp_close = a_temp[i:][idx_close]
        t_temp_close = t_temp[i:][idx_close]

        t_barycenter = barycenter(a_temp_close, t_temp_close)
        a_barycenter = np.sum(a_temp_close)

        t_temp[i:][idx_close] = 0
        a_temp[i:][idx_close] = 0

        t_temp[i] = t_barycenter
        a_temp[i] = a_barycenter

        # print('before proj', a_temp_close, t_temp_close)
        # print('after proj', a_barycenter, t_barycenter)

        break

    if all(a_temp != 0):
        return a, t

    a_out = a_temp[a_temp != 0]
    t_out = t_temp[a_temp != 0]

    return a_out, t_out


def clip_domain(t, linop):
    return np.clip(t, linop.bounds["min"], linop.bounds["max"])
