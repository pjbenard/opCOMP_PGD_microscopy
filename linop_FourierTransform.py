import numpy as np


class FourierTransform:
    def __init__(self, **parameters):
        super().__init__()

        default_params = {"m": 200, "d": 2, "lamb": 0.1}
        for key, value in default_params.items():
            setattr(self, key, value)

        for key, value in parameters.items():
            setattr(self, key, value)

        if "bounds" not in parameters.keys():
            self.bounds = {
                "min": np.array((0,) * self.d),
                "max": np.array((1,) * self.d),
            }

        self.w = np.random.randn(self.m, self.d) / self.lamb

    def Adelta(self, t):
        """
(k, m)
        """
        expval = np.exp(-1j * np.dot(t, self.w.T))

        return expval / self.m**0.5

    def Adeltap(self, t):
        """
        (d, k, m)
        """
        dA_dt = 1j * self.w.T[:, None, :] * self.Adelta(t)[None, :, :]

        return dA_dt

    def Adeltapp(self, t):
        """
        (d, d, k, m)
        """
        ddA_ddt = (
            -self.w.T[None, :, None, :]
            * self.w.T[:, None, None, :]
            * self.Adelta(t)[None, None, :, :]
        )

        return ddA_ddt

    def Ax(self, a, t):
        """
        (m)
        """
        expval = self.Adelta(t)

        return np.dot(a, expval)

    def image(self, grid, y=None, a=None, t=None):
        if y is None:
            y = self.Ax(a, t)

        m = np.size(y)
        dims_size = np.shape(grid)[:-1]

        expval = 1 / self.Adelta(grid.reshape((-1, self.d)))

        z = np.dot(expval, y).reshape(dims_size)

        return np.abs(z) / m  # Modulus of complex numbers

    def hermitian_prod(self, u, v):
        return np.sum(u * np.conj(v))

    def Hessian(self, a, t, y):
        k, d = t.shape

        H = np.zeros((k * (d + 1), k * (d + 1)), dtype=complex)

        Adelta_t = self.Adelta(t)  # (k, m)
        Adelta_t_p = self.Adeltap(t)  # (d, k, m)
        Adelta_t_pp = self.Adeltapp(t)  # (d, d, k, m)

        Aphi_theta = self.Ax(a, t)
        residue = Aphi_theta - y

        # H_1 = G_1 + F_1 (k, k)
        G_1 = np.zeros((k, k), dtype=complex)
        for i in range(k):
            for j in range(k):
                G_1[i, j] = self.hermitian_prod(Adelta_t[i], Adelta_t[j])

        F_1 = np.zeros((k, k), dtype=complex)

        H_1 = G_1 + F_1

        # H_2 = G_2 + F_2 (k * d, k * d)
        G_2 = np.zeros((k * d, k * d), dtype=complex)
        for i in range(k):
            for j in range(k):
                for di in range(d):
                    for dj in range(d):
                        G_2[d * i + di, d * j + dj] = (
                            a[i]
                            * a[j]
                            * self.hermitian_prod(Adelta_t_p[di, i], Adelta_t_p[dj, j])
                        )

        F_2 = np.zeros((k * d, k * d), dtype=complex)
        for i in range(k):
            for di in range(d):
                for dj in range(d):
                    F_2[d * i + di, d * i + dj] = a[i] * self.hermitian_prod(
                        Adelta_t_pp[dj, di, i], residue
                    )

        H_2 = G_2 + F_2

        # H_12 = G_12 + F_12 (k, k * d)
        G_12 = np.zeros((k, k * d), dtype=complex)
        for i in range(k):
            for j in range(k):
                for dj in range(d):
                    G_12[i, d * j + dj] = -a[i] * self.hermitian_prod(
                        Adelta_t[i], Adelta_t_p[dj, j]
                    )

        F_12 = np.zeros((k, k * d), dtype=complex)
        for i in range(k):
            for di in range(d):
                F_12[i, d * i + di] = -self.hermitian_prod(Adelta_t_p[di, i], residue)

        H_12 = G_12 + F_12

        # Assembling into H
        H[:k, :k] = H_1
        H[k:, k:] = H_2
        H[:k, k:] = H_12
        H[k:, :k] = H_12.T

        return 2 * np.real(H)
