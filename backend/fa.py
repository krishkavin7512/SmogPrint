"""Factor analysis and PCA, written out in NumPy.

Model: each day's standardised pollutant vector x = W z + noise, with z ~ N(0, I) and
noise ~ N(0, diag(psi)). So cov(x) = W W^T + Psi. EM handles missing values exactly: every
day only contributes the pollutants its monitor actually reported.
"""
import numpy as np


def _patterns(Z):
    """Group rows by which columns are observed, so each group shares one matrix inverse."""
    mask = ~np.isnan(Z)
    codes = (mask * (1 << np.arange(Z.shape[1]))).sum(1)
    return [(np.where(codes == c)[0], mask[codes == c][0]) for c in np.unique(codes)]


def pairwise_corr(Z):
    """Correlation using every pair of readings available, ignoring gaps."""
    d = Z.shape[1]
    R = np.eye(d)
    for i in range(d):
        for j in range(i + 1, d):
            ok = ~np.isnan(Z[:, i]) & ~np.isnan(Z[:, j])
            R[i, j] = R[j, i] = np.corrcoef(Z[ok, i], Z[ok, j])[0, 1]
    return R


class FactorAnalysis:
    def __init__(self, k, n_iter=80, psi_floor=1e-3):
        self.k, self.n_iter, self.psi_floor = k, n_iter, psi_floor

    def fit(self, Z):
        n, d = Z.shape
        k = self.k
        groups = _patterns(Z)
        counts = (~np.isnan(Z)).sum(0)
        w, v = np.linalg.eigh(pairwise_corr(Z))
        self.W = v[:, -k:] * np.sqrt(np.maximum(w[-k:], 1e-6))    # start from the principal axes
        self.psi = np.maximum(1 - (self.W ** 2).sum(1), 0.1)
        self.history = []
        for _ in range(self.n_iter):
            A, b, s = np.zeros((d, k, k)), np.zeros((d, k)), np.zeros(d)
            ll = 0.0
            for idx, m in groups:
                if not m.any():
                    continue
                Wo, Po = self.W[m], self.psi[m]
                x = Z[idx][:, m]
                M = np.eye(k) + (Wo.T / Po) @ Wo
                Mi = np.linalg.inv(M)
                Ez = x @ ((Wo.T / Po).T @ Mi)
                Ezz = len(idx) * Mi + Ez.T @ Ez
                ll += self._group_ll(x, Wo, Po, M, Mi)
                jj = np.where(m)[0]
                A[jj] += Ezz
                b[jj] += x.T @ Ez
                s[jj] += (x ** 2).sum(0)
            self.history.append(ll)
            for j in range(d):
                self.W[j] = np.linalg.solve(A[j] + 1e-8 * np.eye(k), b[j])
                self.psi[j] = max((s[j] - self.W[j] @ b[j]) / counts[j], self.psi_floor)
        return self

    @staticmethod
    def _group_ll(x, Wo, Po, M, Mi):
        """Log N(x | 0, Wo Wo^T + Po) for one missing-value pattern, by the Woodbury identity."""
        y = (x / Po) @ Wo
        quad = (x ** 2 / Po).sum(1) - (y @ Mi * y).sum(1)
        return -0.5 * (quad.sum() + len(x) * (x.shape[1] * np.log(2 * np.pi) + np.log(Po).sum() + np.linalg.slogdet(M)[1]))

    def loglik(self, Z):
        ll = 0.0
        for idx, m in _patterns(Z):
            if not m.any():
                continue
            Wo, Po = self.W[m], self.psi[m]
            M = np.eye(self.k) + (Wo.T / Po) @ Wo
            ll += self._group_ll(Z[idx][:, m], Wo, Po, M, np.linalg.inv(M))
        return float(ll)

    def scores(self, Z):
        """Posterior mean of z for each row, from whichever pollutants that row has. NaN if none."""
        out = np.full((len(Z), self.k), np.nan)
        for idx, m in _patterns(Z):
            if not m.any():
                continue
            Wo, Po = self.W[m], self.psi[m]
            Mi = np.linalg.inv(np.eye(self.k) + (Wo.T / Po) @ Wo)
            out[idx] = Z[idx][:, m] @ ((Wo.T / Po).T @ Mi)
        return out

    def rotate(self, R):
        """Replace W by W R. For an orthogonal R the fitted distribution does not change."""
        self.W = self.W @ R
        return self

    def covariance(self):
        return self.W @ self.W.T + np.diag(self.psi)

    def communalities(self):
        return (self.W ** 2).sum(1)

    def n_params(self):
        d, k = self.W.shape
        return d * k + d - k * (k - 1) / 2


def varimax(L, n_sweeps=100, tol=1e-8):
    """Rotate loadings so each factor is high on a few variables and near zero on the rest.

    Kaiser's pairwise method: each pair of factors has a closed-form best angle, so sweep over the
    pairs until no angle is worth turning. Loadings straight from PCA can start near a saddle point,
    where the shorter SVD iteration crawls; this does not. Returns (rotated loadings, rotation matrix)."""
    p, k = L.shape
    h = np.sqrt((L ** 2).sum(1, keepdims=True))
    A = L / h                                       # Kaiser normalisation: every pollutant counts equally
    R = np.eye(k)
    for _ in range(n_sweeps):
        biggest = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                u = A[:, i] ** 2 - A[:, j] ** 2
                v = 2 * A[:, i] * A[:, j]
                phi = 0.25 * np.arctan2(2 * (u @ v) - 2 * u.sum() * v.sum() / p, (u @ u - v @ v) - (u.sum() ** 2 - v.sum() ** 2) / p)
                if abs(phi) > tol:
                    c, s = np.cos(phi), np.sin(phi)
                    turn = np.array([[c, -s], [s, c]])
                    A[:, [i, j]] = A[:, [i, j]] @ turn
                    R[:, [i, j]] = R[:, [i, j]] @ turn
                    biggest = max(biggest, abs(phi))
        if biggest < tol:
            break
    return A * h, R


def varimax_score(L):
    """The quantity varimax maximises: variance of squared loadings, summed over factors."""
    return float((L ** 2).var(0).sum())


def pca(R):
    """Eigen-decomposition of a correlation matrix, largest first."""
    w, v = np.linalg.eigh(R)
    return w[::-1], v[:, ::-1]


def parallel_analysis(Z, n_rep=20, seed=0, q=0.95):
    """How large would eigenvalues be if the pollutants were unrelated? Shuffle each column and see."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        S = Z.copy()
        for j in range(S.shape[1]):
            S[:, j] = rng.permutation(S[:, j])
        out.append(np.linalg.eigvalsh(pairwise_corr(S))[::-1])
    return np.quantile(out, q, axis=0)


def kmo(R):
    """Kaiser-Meyer-Olkin: are correlations shared, or just pairwise? Above 0.6 is usually acceptable."""
    Ri = np.linalg.inv(R)
    D = np.sqrt(np.outer(np.diag(Ri), np.diag(Ri)))
    P = -Ri / D
    off = ~np.eye(len(R), dtype=bool)
    r2, p2 = (R ** 2) * off, (P ** 2) * off
    return float(r2.sum() / (r2.sum() + p2.sum())), (r2.sum(0) / (r2.sum(0) + p2.sum(0))).tolist()


def bartlett(R, n):
    """Bartlett's sphericity test statistic: is the correlation matrix different from the identity?"""
    d = len(R)
    return float(-(n - 1 - (2 * d + 5) / 6) * np.linalg.slogdet(R)[1]), d * (d - 1) // 2


def principal_axis(R, k, n_iter=100):
    """Principal axis factoring: eigenvectors of R with communalities on the diagonal, iterated."""
    Ri = np.linalg.inv(R)
    h = 1 - 1 / np.diag(Ri)
    for _ in range(n_iter):
        Rh = R.copy()
        np.fill_diagonal(Rh, h)
        w, v = np.linalg.eigh(Rh)
        L = v[:, -k:] * np.sqrt(np.maximum(w[-k:], 0))
        h = (L ** 2).sum(1)
    return L
