import numpy as np

import scipy.special as spsp


class MA_TIRF:
    def __init_GRID(self):
        X_range = np.linspace(
            self.half_len_pixel, self.b1 - self.half_len_pixel, num=self.N1
        )
        Y_range = np.linspace(
            self.half_len_pixel, self.b2 - self.half_len_pixel, num=self.N2
        )
        self.GRID = np.meshgrid(X_range, Y_range)

    def __init__(self, **parameters):
        super().__init__()

        default_params = {
            "K": 4,
            "b1": 6.4,
            "b2": 6.4,
            "b3": 0.8,
            "d": 3,
            "N1": 64,
            "N2": 64,
            "NA": 1.49,
            "ni": 1.515,
            "nt": 1.333,
            "lambda_l": 0.66,
            "n_photon": 1000,
            "sigma": 1e-4,
        }
        for key, value in default_params.items():
            setattr(self, key, value)

        for key, value in parameters.items():
            setattr(self, key, value)

        self.bounds = {
            "min": np.array((0,) * self.d),
            "max": np.array((self.b1, self.b2, self.b3)),
        }

        self.sigma1 = self.sigma2 = 0.42 * self.lambda_l / self.NA
        self.alpha_max = np.arcsin(self.NA / self.ni)
        self.alpha_crit = np.arcsin(self.nt / self.ni)
        self.alpha_k = self.alpha_crit + np.arange(0, self.K) * (
            self.alpha_max - self.alpha_crit
        ) / (self.K - 1)

        self.s_k = (
            (np.sin(self.alpha_k) ** 2 - np.sin(self.alpha_crit) ** 2)
            * (4 * np.pi * self.ni)
            / self.lambda_l
        )

        self.len_pixel = self.b1 / self.N1
        self.half_len_pixel = self.len_pixel / 2

        self._den_sigma1 = 2**0.5 * self.sigma1
        self._den_sigma2 = 2**0.5 * self.sigma2

        self.__init_GRID()

    def _xi(self, z):
        return np.sum(np.exp(-2 * self.s_k[None, :] * z[:, None]), axis=-1) ** -0.5

    def _vmx(self, X):
        return (
            X[:, 0][:, None, None, None]
            - self.half_len_pixel
            - self.GRID[0][None, :, :, None]
        ) / self._den_sigma1

    def _vpx(self, X):
        return (
            X[:, 0][:, None, None, None]
            + self.half_len_pixel
            - self.GRID[0][None, :, :, None]
        ) / self._den_sigma1

    def _vmy(self, X):
        return (
            X[:, 1][:, None, None, None]
            - self.half_len_pixel
            - self.GRID[1][None, :, :, None]
        ) / self._den_sigma2

    def _vpy(self, X):
        return (
            X[:, 1][:, None, None, None]
            + self.half_len_pixel
            - self.GRID[1][None, :, :, None]
        ) / self._den_sigma2

    def Adelta(self, X):
        fx = spsp.erf(self._vpx(X)) - spsp.erf(self._vmx(X))
        fy = spsp.erf(self._vpy(X)) - spsp.erf(self._vmy(X))

        incidence_coef = self._xi(X[:, 2])[:, None] * np.exp(
            -self.s_k[None, :] * X[:, 2][:, None]
        )

        phi = incidence_coef[:, None, None, :] * (fx * fy) / 4
        phi_reshaped = np.reshape(phi, (phi.shape[0], -1))

        return phi_reshaped

    def Adeltap(self, X):
        vpx = self._vpx(X)
        vmx = self._vmx(X)
        vpy = self._vpy(X)
        vmy = self._vmy(X)

        exp_diff_vx = np.exp(-(vpx**2)) - np.exp(-(vmx**2))
        exp_diff_vy = np.exp(-(vpy**2)) - np.exp(-(vmy**2))

        fx = spsp.erf(vpx) - spsp.erf(vmx)
        fy = spsp.erf(vpy) - spsp.erf(vmy)

        # print(f'{self.s_k = }, {X[:, :2] = }, xi = {self._xi(X[:, 2])}')
        incidence_coef = self._xi(X[:, 2])[:, None] * np.exp(
            -self.s_k[None, :] * X[:, 2][:, None]
        )

        dA_dt1 = (
            incidence_coef[:, None, None, :]
            * (exp_diff_vx * fy)
            / (self._den_sigma1 * (4 * np.pi) ** 0.5)
        )
        dA_dt2 = (
            incidence_coef[:, None, None, :]
            * (fx * exp_diff_vy)
            / (self._den_sigma2 * (4 * np.pi) ** 0.5)
        )

        dA_dt3 = (
            (
                incidence_coef
                * (
                    np.sum(self.s_k[None, :] * incidence_coef**2, axis=-1)[:, None]
                    - self.s_k[None, :]
                )
            )[:, None, None, :]
            * fx
            * fy
            / 4
        )

        dA_dt = np.stack((dA_dt1, dA_dt2, dA_dt3), axis=0)
        dA_dt_reshaped = np.reshape(dA_dt, (*dA_dt.shape[:2], -1))

        # add minus sign
        return -dA_dt_reshaped

    def Ax(self, a, X):
        phi = self.Adelta(X)

        return np.dot(phi.T, a)
        # return np.dot(phi, a)

    def image(self, y):
        return np.reshape(y, (self.N1, self.N2, self.K))

    def add_noise(self, y0):
        y0_reshaped = self.image(y0)
        y0_max = np.max(np.sum(y0_reshaped, axis=-1))
        y0_normalized = self.n_photon * y0 / y0_max

        y0_poisson = np.random.poisson(y0_normalized)

        y = y0_poisson + self.sigma * np.random.randn(*y0_poisson.shape)
        return y
