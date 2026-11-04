"""K-means, Gaussian mixtures and hierarchical clustering, written out in NumPy."""
import numpy as np


# ------------------------------------------------------------------ k-means
class KMeans:
    def __init__(self, k, n_init=5, n_iter=100, seed=0):
        self.k, self.n_init, self.n_iter, self.seed = k, n_init, n_iter, seed

    def _seed(self, X, rng):
        """k-means++: each new centre is drawn far from the ones already chosen."""
        c = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d = ((X[:, None] - np.array(c)) ** 2).sum(-1).min(1)
            c.append(X[rng.choice(len(X), p=d / d.sum())])
        return np.array(c)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            c = self._seed(X, rng)
            hist = []
            for _ in range(self.n_iter):
                d = ((X[:, None] - c) ** 2).sum(-1)
                lab = d.argmin(1)
                hist.append(float(d[np.arange(len(X)), lab].sum()))
                new = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(self.k)])
                if np.allclose(new, c):
                    c = new
                    break
                c = new
            if best is None or hist[-1] < best[0]:
                best = (hist[-1], c, hist)
        self.inertia, self.centers, self.history = best
        return self

    def predict(self, X):
        return ((X[:, None] - self.centers) ** 2).sum(-1).argmin(1)


def silhouette(X, lab, sample=2000, seed=0):
    """Mean silhouette on a sample: (distance to nearest other cluster - own cluster) / larger of the two."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs, ls = X[idx], lab[idx]
    D = np.sqrt(((Xs[:, None] - Xs) ** 2).sum(-1))
    labels = np.unique(ls)
    out = np.zeros(len(Xs))
    for i in range(len(Xs)):
        same = ls == ls[i]
        same[i] = False
        if not same.any():
            continue
        a = D[i, same].mean()
        b = min(D[i, ls == l].mean() for l in labels if l != ls[i])
        out[i] = (b - a) / max(a, b)
    return float(out.mean())


def adjusted_rand(a, b):
    """Agreement of two clusterings, corrected for chance. 1 is identical, about 0 is random."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub)))
    np.add.at(C, (ia, ib), 1)
    comb = lambda v: v * (v - 1) / 2
    sum_ij, sum_a, sum_b = comb(C).sum(), comb(C.sum(1)).sum(), comb(C.sum(0)).sum()
    exp = sum_a * sum_b / comb(C.sum())
    return float((sum_ij - exp) / (0.5 * (sum_a + sum_b) - exp))


# ------------------------------------------------------------------ Gaussian mixture
class GaussianMixture:
    def __init__(self, k, n_iter=200, tol=1e-4, reg=1e-4, seed=0):
        self.k, self.n_iter, self.tol, self.reg, self.seed = k, n_iter, tol, reg, seed

    def _log_prob(self, X):
        d = X.shape[1]
        out = np.zeros((len(X), self.k))
        for j in range(self.k):
            L = np.linalg.cholesky(self.cov[j])
            y = np.linalg.solve(L, (X - self.mu[j]).T).T
            out[:, j] = np.log(self.pi[j]) - 0.5 * (d * np.log(2 * np.pi) + (y ** 2).sum(1)) - np.log(np.diag(L)).sum()
        return out

    def fit(self, X):
        n, d = X.shape
        km = KMeans(self.k, n_init=3, seed=self.seed).fit(X)
        lab = km.predict(X)
        self.pi = np.array([(lab == j).mean() for j in range(self.k)])
        self.mu = km.centers.copy()
        self.cov = np.array([np.cov(X[lab == j].T) + self.reg * np.eye(d) if (lab == j).sum() > d else np.eye(d) for j in range(self.k)])
        self.history = []
        for _ in range(self.n_iter):
            lp = self._log_prob(X)
            m = lp.max(1, keepdims=True)
            s = np.exp(lp - m).sum(1, keepdims=True)
            self.history.append(float((m + np.log(s)).sum()))
            r = np.exp(lp - m) / s                                 # E-step: responsibilities
            Nk = r.sum(0) + 1e-9                                   # M-step
            self.pi = Nk / n
            self.mu = (r.T @ X) / Nk[:, None]
            for j in range(self.k):
                dx = X - self.mu[j]
                self.cov[j] = (r[:, j, None] * dx).T @ dx / Nk[j] + self.reg * np.eye(d)
            if len(self.history) > 1 and self.history[-1] - self.history[-2] < self.tol * abs(self.history[-1]) * 1e-3:
                break
        return self

    def proba(self, X):
        lp = self._log_prob(X)
        m = lp.max(1, keepdims=True)
        p = np.exp(lp - m)
        return p / p.sum(1, keepdims=True)

    def loglik(self, X):
        lp = self._log_prob(X)
        m = lp.max(1, keepdims=True)
        return float((m + np.log(np.exp(lp - m).sum(1, keepdims=True))).sum())

    def n_params(self):
        d = self.mu.shape[1]
        return self.k - 1 + self.k * d + self.k * d * (d + 1) / 2

    def bic(self, X):
        return -2 * self.loglik(X) + self.n_params() * np.log(len(X))


# ------------------------------------------------------------------ hierarchical
LINKAGES = ("ward", "average", "complete", "single")


def agglomerative(X, method="ward"):
    """Bottom-up clustering by Lance-Williams updates. Returns a SciPy-style linkage matrix."""
    n = len(X)
    D = np.sqrt(((X[:, None] - X) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    active = list(range(n))
    ids = list(range(n))
    size = np.ones(n)
    Z = []
    for step in range(n - 1):
        sub = D[np.ix_(active, active)]
        a_, b_ = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = active[a_], active[b_]
        h = D[a, b]
        Z.append([min(ids[a], ids[b]), max(ids[a], ids[b]), h, size[a] + size[b]])
        for c in active:
            if c in (a, b):
                continue
            na, nb, nc = size[a], size[b], size[c]
            if method == "single":
                v = min(D[a, c], D[b, c])
            elif method == "complete":
                v = max(D[a, c], D[b, c])
            elif method == "average":
                v = (na * D[a, c] + nb * D[b, c]) / (na + nb)
            else:
                v = np.sqrt(((na + nc) * D[a, c] ** 2 + (nb + nc) * D[b, c] ** 2 - nc * h ** 2) / (na + nb + nc))
            D[a, c] = D[c, a] = v
        size[a] += size[b]
        ids[a] = n + step
        active.remove(b)
    return np.array(Z)


def cut_tree(Z, k):
    """Cluster labels after keeping only the last k-1 merges undone."""
    n = len(Z) + 1
    parent = list(range(2 * n - 1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(Z[: n - k]):
        parent[int(row[0])] = n + i
        parent[int(row[1])] = n + i
    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)])


def leaf_order(Z):
    n = len(Z) + 1

    def walk(i):
        if i < n:
            return [i]
        a, b = int(Z[i - n, 0]), int(Z[i - n, 1])
        return walk(a) + walk(b)
    return walk(2 * n - 2)


def dendrogram_segments(Z):
    """Drawing coordinates for the tree. Returns (leaf order, list of [x1, y1, x2, y2])."""
    n = len(Z) + 1
    order = leaf_order(Z)
    pos = {leaf: i for i, leaf in enumerate(order)}
    xy = {leaf: (pos[leaf], 0.0) for leaf in range(n)}
    segs = []
    for i, (a, b, h, _) in enumerate(Z):
        (xa, ya), (xb, yb) = xy[int(a)], xy[int(b)]
        segs += [[xa, ya, xa, h], [xb, yb, xb, h], [xa, h, xb, h]]
        xy[n + i] = ((xa + xb) / 2, h)
    return order, segs


def cophenetic_corr(X, Z):
    """How faithfully the tree keeps the original distances. 1 is perfect."""
    n = len(X)
    orig = np.sqrt(((X[:, None] - X) ** 2).sum(-1))
    coph = np.zeros((n, n))
    members = {i: [i] for i in range(n)}
    for i, (a, b, h, _) in enumerate(Z):
        A, B = members.pop(int(a)), members.pop(int(b))
        for p in A:
            coph[p, B] = h
            coph[B, p] = h
        members[n + i] = A + B
    tri = np.triu_indices(n, 1)
    return float(np.corrcoef(orig[tri], coph[tri])[0, 1])
