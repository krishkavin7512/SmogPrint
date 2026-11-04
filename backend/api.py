"""FastAPI backend for SmogPrint. Start with `python run.py`; API docs at http://127.0.0.1:8003/docs"""
import json

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .cluster import GaussianMixture
from .fa import FactorAnalysis

app = FastAPI(title="SmogPrint API", version="1.0",
              description="Find the hidden sources behind a city's air with factor analysis and Gaussian mixtures written from scratch.")


def _load(name):
    path = config.ARTIFACTS / name
    if not path.exists():
        raise RuntimeError(f"{path} is missing. Run `python -m backend.train` first.")
    return json.loads(path.read_text())


REPORT = _load("report.json")
GRAPHS = _load("graphs.json")
MODEL = _load("model.json")
LETTERS = [f["letter"] for f in MODEL["factors"]]
POL = MODEL["pollutants"]
FA = FactorAnalysis(len(LETTERS))
FA.W, FA.psi = np.array(MODEL["loadings"]), np.array(MODEL["psi"])
GM = GaussianMixture(len(MODEL["gmm"]["pi"]))
GM.pi, GM.mu, GM.cov = (np.array(MODEL["gmm"][k]) for k in ("pi", "mu", "cov"))
PROFILES = REPORT["city_profiles"]
MEAN, SD = np.array([MODEL["mean"][p] for p in POL]), np.array([MODEL["sd"][p] for p in POL])


def score_vector(readings):
    x = np.array([np.nan if readings.get(p) is None else float(readings[p]) for p in POL])
    z = ((np.log1p(np.clip(x, 0, None)) - MEAN) / SD)[None]
    s = FA.scores(z)
    return s, int((~np.isnan(z)).sum())


@app.get("/api/overview")
def overview():
    R = REPORT
    keep = ("data", "factors", "k", "lockdown", "lockdown_raw", "diwali", "weekly", "hours", "seasons", "city_rank", "trace", "adequacy", "aqi_r2", "aqi_corr", "examples", "trained_at", "fit_end", "pollutants", "usable_rows", "fit_rows_used")
    out = {k: R[k] for k in keep}
    out["fa"] = {k: R["fa"][k] for k in ("loadings", "psi", "communalities", "variance_share", "varimax_before", "varimax_after", "ll_any_rotation")}
    out["pca"] = {k: R["pca"][k] for k in ("kaiser_k", "parallel_k", "cumulative")}
    out["k_grid"] = [{"k": r["k"], "heldout": r["heldout_ll_per_row"]} for r in R["k_grid"]]
    out["types"] = {k: R["types"][k] for k in ("k", "rows", "ari_kmeans_gmm", "silhouette_k", "uncertain_share")}
    out["tree"] = {k: R["tree"][k] for k in ("names", "order", "segments", "groups", "cophenetic")}
    out["cities"] = sorted(PROFILES)
    out["sources"] = config.SOURCES
    out["coal_belt"] = config.COAL_BELT
    out["n_graphs"] = len(GRAPHS)
    return out


@app.get("/api/city/{name}")
def city(name: str):
    """One city's fingerprint: its factor scores, ranks, months, day types and nearest neighbours."""
    if name not in PROFILES:
        raise HTTPException(404, f"{name} has too few days. Choose one of {sorted(PROFILES)}")
    me = PROFILES[name]
    v = np.array([me[l] for l in LETTERS])
    dist = sorted(((float(np.linalg.norm(v - np.array([PROFILES[c][l] for l in LETTERS]))), c) for c in PROFILES if c != name))
    ranks = {l: list(REPORT["city_rank"][l]).index(name) + 1 for l in LETTERS}
    return {"name": name, "profile": me, "ranks": ranks, "n_cities": len(PROFILES), "months": REPORT["city_months"][name], "types": REPORT["city_types"][name],
            "days": REPORT["city_days"][name], "nearest": [{"city": c, "distance": d} for d, c in dist[:3]], "farthest": {"city": dist[-1][1], "distance": dist[-1][0]}}


class Readings(BaseModel):
    readings: dict[str, float | None] = Field(..., description="Pollutant name -> daily mean. Leave out or null anything not measured.")


@app.post("/api/score")
def score(body: Readings):
    """Score any day from whichever pollutants are available."""
    unknown = [k for k in body.readings if k not in POL]
    if unknown:
        raise HTTPException(400, f"unknown pollutants: {unknown}. Use {POL}")
    s, n = score_vector(body.readings)
    if n < config.MIN_READINGS:
        raise HTTPException(400, f"need at least {config.MIN_READINGS} pollutants, got {n}")
    return {"used": n, "scores": dict(zip(LETTERS, s[0].round(3).tolist())), "types": GM.proba(s)[0].round(4).tolist()}


@app.get("/api/graphs")
def graphs():
    return {"graphs": GRAPHS}


@app.get("/api/graphs/{gid}")
def graph(gid: str):
    for g in GRAPHS:
        if g["id"] == gid:
            return g
    raise HTTPException(404, "no such graph")


@app.middleware("http")
async def revalidate_assets(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/assets") or request.url.path in ("/", "/graphs"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/assets", StaticFiles(directory=config.FRONTEND / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.FRONTEND / "index.html")


@app.get("/graphs", include_in_schema=False)
def graphs_page():
    return FileResponse(config.FRONTEND / "graphs.html")
