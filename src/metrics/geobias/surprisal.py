"""Analytical surprisal of Moran's I under random relabeling.

Ported from GeoBS (MIT): ``gbs/ssi/surprisal.py``, with the non-negativity
guards from ``baselines/gbsloss.py`` (they only matter when a variance term
goes negative, where the original returns NaN).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


class AnalyticalSurprisal:
    def fit(self, cs: np.ndarray, ns: np.ndarray, w_map: np.ndarray, ignores: np.ndarray) -> None:
        self.d = w_map.shape[0]
        self.N, self.W = np.sum(ns), np.sum(w_map)
        self.mean, self.mean_dict, self.mean_coef_dict = self.compute_mean(cs, ns)
        self.std = self.compute_std(cs, ns, ignores)

    @staticmethod
    def moran_i_upper(Z: np.ndarray, w_map: np.ndarray) -> float:
        x = Z.ravel() - np.mean(Z)
        return float(x @ (w_map @ x))

    def get_probability(self, Z: np.ndarray, w_map: np.ndarray) -> float:
        """Two-sided tail probability of the observed (unnormalized) Moran's I."""
        m = self.moran_i_upper(Z, w_map)
        low = min(m, 2 * self.mean - m)
        return float(2 * norm.cdf(low, loc=self.mean, scale=self.std))

    @staticmethod
    def compute_mean(cs, ns):
        N = np.sum(ns)
        xmean = np.sum(cs * ns) / N
        mu, mu_dict, mu_coef_dict = 0.0, {}, {}
        for i, (ci, ni) in enumerate(zip(cs, ns)):
            for j, (cj, nj) in enumerate(zip(cs, ns)):
                coef = (ci - xmean) * (cj - xmean)
                if ci != cj:
                    num = min(ni, nj) * 4 * max(ni, nj) / N
                else:
                    num = ((ni - 1) * 4 * ni / N) - 1
                mu += coef * num
                mu_dict[(i, j)] = num
                mu_coef_dict[(i, j)] = coef
        return mu, mu_dict, mu_coef_dict

    @staticmethod
    def compute_std(cs, ns, ignores):
        N = np.sum(ns)
        r_max = np.flatnonzero(ignores == 0)[0]
        xmean = np.sum(cs * ns) / N
        var = 0.0
        for ci, ni, igi in zip(cs, ns, ignores):
            for cj, nj, igj in zip(cs, ns, ignores):
                if ci != cj:
                    num = min(ni, nj) * (4 * max(ni, nj) / N) * (1 - 4 * max(ni, nj) / N)
                    coef = (
                        (ci - xmean) * (cj - xmean)
                        - 2 * (ci - xmean) * (cs[r_max] - xmean)
                        + (cs[r_max] - xmean) ** 2
                    ) ** 2
                else:
                    num = 2 * (ni - 1) * 4 * ni / N * (1 - (4 * (2 * ni - 1)) / (3 * N))
                    coef = (ci - cs[r_max]) ** 4
                var += coef * num * igi * igj
        return np.sqrt(max(0.0, var)) + 1.0
