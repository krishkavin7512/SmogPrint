"""Fit every model, run every outside check, write artifacts/.  python -m backend.train"""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, evidence
from .cluster import LINKAGES, GaussianMixture, KMeans, adjusted_rand, agglomerative, cophenetic_corr, cut_tree, dendrogram_segments, silhouette
from .data import describe, enough, fit_scale, load_day, load_hour
from .fa import FactorAnalysis, bartlett, kmo, parallel_analysis, pairwise_corr, pca, principal_axis, varimax, varimax_score


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def choose_k(Z, dates):
    """Fit on data before 2019, score on 2019. The answer is never looked at when choosing."""
    tr = Z[(dates < config.SELECT_TRAIN_END).to_numpy()]
    va = Z[((dates >= config.SELECT_TRAIN_END) & (dates < config.FIT_END)).to_numpy()]
    rows = []
    for k in config.K_RANGE:
        m = FactorAnalysis(k, 60).fit(tr)
        rows.append({"k": k, "heldout_ll_per_row": m.loglik(va) / len(va), "train_ll_per_row": m.history[-1] / len(tr), "params": m.n_params(),
                     "bic": -2 * m.history[-1] + m.n_params() * np.log(len(tr)), "monotone": bool(np.all(np.diff(m.history) > -1e-3))})
    gains = np.diff([r["heldout_ll_per_row"] for r in rows])
    chosen = config.K_RANGE[-1]
    for i in range(1, len(gains)):
        if gains[i] < config.ELBOW_RULE * gains[i - 1] and gains[i - 1] > 0 and gains[i - 1] >= max(gains[:i]) * config.ELBOW_RULE:
            chosen = config.K_RANGE[i]
            break
    return chosen, rows, gains


def name_factors(L):
    """Label each factor by the group holding its largest loading. Ties go to the larger variance."""
    out = []
    for j in range(L.shape[1]):
        top = config.POLLUTANTS[int(np.argmax(L[:, j]))]
        name, what = next((n, w) for g, n, w in config.FACTOR_NAMES if top in g)
        out.append({"index": j, "letter": "ABCDEFGH"[j], "name": name, "what": what, "top": top})
    return out


def types_of_days(S, n_days_sample=12000):
    """Cluster days by their factor scores with k-means and a Gaussian mixture."""
    rng = np.random.default_rng(config.SEED)
    X = S[rng.choice(len(S), min(n_days_sample, len(S)), replace=False)]
    km_rows, gm_rows = [], []
    for k in range(2, 11):
        km = KMeans(k, n_init=3, seed=config.SEED).fit(X)
        lab = km.predict(X)
        km_rows.append({"k": k, "inertia": km.inertia, "silhouette": silhouette(X, lab)})
        gm = GaussianMixture(k, seed=config.SEED).fit(X)
        gm_rows.append({"k": k, "bic": gm.bic(X), "ll_per_row": gm.history[-1] / len(X), "iters": len(gm.history), "monotone": bool(np.all(np.diff(gm.history) > -1e-2))})
        log(f"  k={k}: k-means silhouette {km_rows[-1]['silhouette']:.3f}, GMM BIC {gm_rows[-1]['bic']:.0f}")
    return X, km_rows, gm_rows


def main():
    config.ARTIFACTS.mkdir(exist_ok=True)
    df = load_day()
    scale = fit_scale(df)
    Z_all = scale.apply(df)
    usable = enough(Z_all)
    fit_mask = (df.Date < config.FIT_END).to_numpy() & usable
    Zf = Z_all[fit_mask]
    rep = {"data": describe(df), "pollutants": config.POLLUTANTS, "fit_end": config.FIT_END, "usable_rows": int(usable.sum()), "fit_rows_used": int(fit_mask.sum())}
    log(f"rows {len(df)}, usable {usable.sum()}, fit {fit_mask.sum()}")

    # Is there shared structure at all, and how many dimensions?
    R = pairwise_corr(Zf)
    ev, evec = pca(R)
    pa = parallel_analysis(Zf, seed=config.SEED)
    kmo_all, kmo_each = kmo(R)
    chi2, dof = bartlett(R, len(Zf))
    rep["pca"] = {"eigenvalues": ev, "explained": ev / ev.sum(), "cumulative": np.cumsum(ev) / ev.sum(), "parallel_95": pa, "kaiser_k": int((ev > 1).sum()), "parallel_k": int((ev > pa).sum()), "loadings": evec[:, :4] * np.sqrt(ev[:4]), "correlation": R}
    rep["adequacy"] = {"kmo": kmo_all, "kmo_each": dict(zip(config.POLLUTANTS, kmo_each)), "bartlett_chi2": chi2, "bartlett_dof": dof}
    log(f"eigenvalues {np.round(ev, 2)}; Kaiser {rep['pca']['kaiser_k']}, parallel {rep['pca']['parallel_k']}; KMO {kmo_all:.3f}; Bartlett chi2 {chi2:.0f} on {dof} df")

    k, grid, gains = choose_k(Z_all[usable], df.Date[usable])
    rep["k_grid"], rep["k_gains"], rep["k"], rep["chain_rule"] = grid, gains, k, config.ELBOW_RULE
    log("k grid: " + ", ".join(f"{g['k']}:{g['heldout_ll_per_row']:.3f}" for g in grid) + f" -> chosen {k}")

    model = FactorAnalysis(k, 100).fit(Zf)
    unrotated = model.W.copy()
    L, Rot = varimax(unrotated)
    flip = np.sign(L.sum(0))
    L, Rot = L * flip, Rot * flip
    order = np.argsort(-(L ** 2).sum(0))             # largest factor first
    L, Rot = L[:, order], Rot[:, order]
    model.W = L
    ll_before = model.history[-1]
    ll_rotated = model.loglik(Zf)
    rng = np.random.default_rng(config.SEED)
    Q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    twist = FactorAnalysis(k)
    twist.W, twist.psi = model.W @ Q, model.psi
    rep["fa"] = {"k": k, "loadings": L, "unrotated": unrotated, "psi": model.psi, "communalities": (L ** 2).sum(1), "history": model.history,
                 "variance_share": (L ** 2).sum(0) / len(config.POLLUTANTS), "varimax_before": varimax_score(unrotated), "varimax_after": varimax_score(L),
                 "ll_any_rotation": {"fitted": ll_rotated, "random_rotation": twist.loglik(Zf)}, "implied_corr": model.covariance(), "corr_gap": float(np.abs((model.covariance() - np.diag(np.diag(model.covariance())) - (R - np.eye(len(R))))).mean()),
                 "paf_loadings": varimax(principal_axis(R, k))[0], "ll_last_em": ll_before}
    rep["factors"] = name_factors(L)
    cols = [f["letter"] for f in rep["factors"]]
    print(pd.DataFrame(L, index=config.POLLUTANTS, columns=[f"{f['letter']} {f['name']}" for f in rep["factors"]]).round(2).to_string())

    # Score every usable day, including 2020 which the model never saw
    S = pd.DataFrame(model.scores(Z_all), columns=cols)
    S[~usable] = np.nan
    s = pd.concat([df[["City", "Date", "AQI", "AQI_Bucket"]].reset_index(drop=True), S], axis=1)

    # Outside checks
    rep["lockdown"] = evidence.lockdown(s, cols)
    rep["lockdown_raw"] = evidence.raw_lockdown(df)
    rep["trace"] = evidence.daily_trace(s, cols)
    log(f"Lockdown 2020 effect {rep['lockdown'][2020]['effect']} t {rep['lockdown'][2020]['t']}")
    log(f"Placebo 2019 effect {rep['lockdown'][2019]['effect']} t {rep['lockdown'][2019]['t']}")
    log(f"Placebo 2018 effect {rep['lockdown'][2018]['effect']} t {rep['lockdown'][2018]['t']}")
    rep["diwali"] = evidence.diwali(s, cols)
    log(f"Diwali day-after {rep['diwali']['day_after']}; placebo sd {rep['diwali']['placebo_sd']}; p {rep['diwali']['p_value']}")
    rep["weekly"] = evidence.weekly(s, cols)
    log(f"Sunday t {rep['weekly']['sunday_t']}")
    rep["seasons"] = evidence.seasons(s, cols)
    prof, rank, n_days = evidence.cities(s, cols)
    rep["city_rank"], rep["city_days"] = rank, n_days
    rep["city_profiles"] = {c: prof.loc[c].round(3).to_dict() for c in prof.index}
    rep["city_months"] = evidence.city_months(s, cols)
    for f in rep["factors"]:
        log(f"  {f['letter']} {f['name']}: top {list(rank[f['letter']].items())[:5]}")

    hour = load_hour()
    Zh = scale.apply(hour)
    Sh = pd.DataFrame(model.scores(Zh), columns=cols)
    Sh[~enough(Zh)] = np.nan
    Sh["Datetime"] = hour.Datetime.values
    rep["hours"] = evidence.by_hour(Sh, cols)
    log(f"Peak hour by factor {rep['hours']['peak_hour']}, trough {rep['hours']['trough_hour']}")

    # Does AQI (never used) line up with the factors?
    has = s[(s.Date < config.FIT_END)].dropna(subset=cols + ["AQI"])
    rep["aqi_corr"] = {c: float(np.corrcoef(has[c], has.AQI)[0, 1]) for c in cols}
    A = np.c_[np.ones(len(has)), has[cols].to_numpy()]
    beta, *_ = np.linalg.lstsq(A, has.AQI.to_numpy(), rcond=None)
    pred = A @ beta
    rep["aqi_r2"] = float(1 - ((has.AQI - pred) ** 2).sum() / ((has.AQI - has.AQI.mean()) ** 2).sum())
    log(f"AQI R2 from factors {rep['aqi_r2']:.3f}; corr {rep['aqi_corr']}")

    # Types of day
    fit_scores = s[s.Date < config.FIT_END].dropna(subset=cols)
    X_all = fit_scores[cols].to_numpy()
    X, km_rows, gm_rows = types_of_days(X_all)
    rep["kmeans_grid"], rep["gmm_grid"] = km_rows, gm_rows
    inertia = [r["inertia"] for r in km_rows]
    drops = -np.diff(inertia)
    bic = [r["bic"] for r in gm_rows]
    gm_gain = -np.diff(bic)
    K_types = next((km_rows[i]["k"] for i in range(1, len(gm_gain)) if gm_gain[i] < config.ELBOW_RULE * gm_gain[i - 1]), gm_rows[-1]["k"])
    km = KMeans(K_types, n_init=8, seed=config.SEED).fit(X_all)
    gm = GaussianMixture(K_types, seed=config.SEED).fit(X_all)
    # Stable order: by how much AQI each type carries
    resp = gm.proba(X_all)
    hard = resp.argmax(1)
    aqi_by = np.array([np.nanmean(fit_scores.AQI.to_numpy()[hard == j]) for j in range(K_types)])
    order = np.argsort(aqi_by)
    gm.pi, gm.mu, gm.cov = gm.pi[order], gm.mu[order], gm.cov[order]
    resp = gm.proba(X_all)
    hard = resp.argmax(1)
    km_lab = km.predict(X_all)
    months = fit_scores.Date.dt.month.to_numpy()
    types = []
    for j in range(K_types):
        sel = hard == j
        types.append({"id": j, "share": float(sel.mean()), "aqi_mean": float(np.nanmean(fit_scores.AQI.to_numpy()[sel])), "center": gm.mu[j],
                      "month_share": [float((months[sel] == m).mean()) for m in range(1, 13)], "certain": float(resp[sel, j].mean()),
                      "cities": pd.Series(fit_scores.City.to_numpy()[sel]).value_counts(normalize=True).head(4).round(3).to_dict()})
    rep["types"] = {"k": K_types, "rows": types, "ari_kmeans_gmm": adjusted_rand(km_lab, hard), "silhouette_k": km_rows[K_types - 2]["silhouette"],
                    "uncertain_share": float((resp.max(1) < 0.6).mean()), "history": gm.history, "pi": gm.pi, "mu": gm.mu, "cov": gm.cov}
    log(f"Types of day k={K_types}; ARI k-means vs GMM {rep['types']['ari_kmeans_gmm']:.3f}; uncertain {rep['types']['uncertain_share']:.2f}")
    for t in types:
        log(f"  type {t['id']}: share {t['share']:.2f}, AQI {t['aqi_mean']:.0f}, centre {np.round(t['center'], 2)}, cities {t['cities']}")

    per_city = pd.DataFrame({"City": fit_scores.City.to_numpy(), "t": hard})
    rep["city_types"] = {c: g.t.value_counts(normalize=True).reindex(range(K_types), fill_value=0).round(3).tolist() for c, g in per_city.groupby("City")}

    # Real days for the "score a day" panel. Each is an actual record from the file.
    rep["examples"] = []
    for label, city, date in (("Delhi, the day after Diwali 2018", "Delhi", "2018-11-08"), ("Delhi, lockdown, 15 April 2020", "Delhi", "2020-04-15"),
                              ("Talcher, 15 January 2019", "Talcher", "2019-01-15"), ("Thiruvananthapuram, monsoon, 15 July 2019", "Thiruvananthapuram", "2019-07-15")):
        row = df[(df.City == city) & (df.Date == date)]
        if len(row):
            rep["examples"].append({"label": label, "readings": {k: (None if pd.isna(v) else float(v)) for k, v in row.iloc[0][config.POLLUTANTS].items()}})
    # Family tree of cities
    C = prof.to_numpy()
    names = prof.index.tolist()
    tree = {}
    for method in LINKAGES:
        Zl = agglomerative(C, method)
        tree[method] = {"cophenetic": cophenetic_corr(C, Zl), "linkage": Zl}
    Zw = tree["ward"]["linkage"]
    leaf, segs = dendrogram_segments(Zw)
    heights = Zw[:, 2]
    tree_groups = {}
    for kc in (2, 3, 4, 5, 6):
        lab = cut_tree(Zw, kc)
        tree_groups[kc] = {n: int(l) for n, l in zip(names, lab)}
    rep["tree"] = {"names": names, "order": [names[i] for i in leaf], "segments": segs, "heights": heights, "groups": tree_groups,
                   "cophenetic": {m: tree[m]["cophenetic"] for m in LINKAGES}, "linkage": Zw}
    log(f"Tree cophenetic {rep['tree']['cophenetic']}; groups at 4: {pd.Series(tree_groups[4]).groupby(pd.Series(tree_groups[4])).apply(lambda v: list(v.index)).to_dict()}")

    rep["trained_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    state = {"pollutants": config.POLLUTANTS, "mean": scale.mean.to_dict(), "sd": scale.sd.to_dict(), "loadings": L, "psi": model.psi, "factors": rep["factors"],
             "gmm": {"pi": gm.pi, "mu": gm.mu, "cov": gm.cov}, "kmeans_centers": km.centers}
    (config.ARTIFACTS / "model.json").write_text(json.dumps(clean(state)))
    (config.ARTIFACTS / "report.json").write_text(json.dumps(clean(rep)))
    s.dropna(subset=cols).assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))[["City", "Date", *cols]].round(3).to_csv(config.ARTIFACTS / "scores.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    from .charts import build_graphs
    graphs = build_graphs(dict(rep=clean(rep), df=df, s=s, Zf=Zf, X=X_all, cols=cols, resp=resp, hard=hard, fit_scores=fit_scores, km_lab=km_lab))
    (config.ARTIFACTS / "graphs.json").write_text(json.dumps(clean(graphs)))
    log(f"Done. {len(graphs)} graphs.")


if __name__ == "__main__":
    main()
