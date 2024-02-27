import numpy as np


def gradient_g(X, y, linop):
    a = X[:, -1]
    t = X[:, :-1]
    gradX = np.empty_like(X)

    conj_residue = np.conj(linop.Ax(a, t) - y)
    gradX[:, -1] = 2 * np.real(np.dot(linop.Adelta(t), conj_residue))
    gradX[:, :-1] = -2 * a[:, None] * np.real(np.dot(
        linop.Adeltap(t),
        conj_residue
    )).T
    return gradX


# def g(X, y, linop):
#     residue = linop.Ax(X[:, -1], X[:, :-1]) - y
#     return np.inner(residue, np.conj(residue)).real

def g(X, y, linop):
    residue = linop.Ax(X[:, -1], X[:, :-1]) - y
    return np.linalg.norm(residue) ** 2
