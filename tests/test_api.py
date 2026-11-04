"""The running app: every route, on real trained artifacts."""
import pytest
from fastapi.testclient import TestClient

from backend.api import POL, PROFILES, REPORT, app

client = TestClient(app)


def test_pages_load():
    assert client.get("/").status_code == 200
    assert client.get("/graphs").status_code == 200


def test_overview():
    r = client.get("/api/overview").json()
    assert len(r["factors"]) == r["k"]
    assert len({f["name"] for f in r["factors"]}) == r["k"]
    assert r["n_graphs"] >= 30


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_city_page(name):
    r = client.get(f"/api/city/{name}").json()
    assert r["name"] == name and len(r["nearest"]) == 3 and name not in [n["city"] for n in r["nearest"]]
    assert abs(sum(r["types"]) - 1) < 0.01


def test_unknown_city():
    assert client.get("/api/city/Atlantis").status_code == 404


def test_score_handles_gaps():
    full = client.post("/api/score", json={"readings": {p: 50.0 for p in POL}}).json()
    assert full["used"] == len(POL) and abs(sum(full["types"]) - 1) < 0.01
    some = client.post("/api/score", json={"readings": {"PM2.5": 200, "PM10": 300, "NO2": 40, "SO2": 20}}).json()
    assert some["used"] == 4


def test_score_rejects_bad_input():
    assert client.post("/api/score", json={"readings": {"PM2.5": 200}}).status_code == 400
    assert client.post("/api/score", json={"readings": {"Plutonium": 5, "PM2.5": 1, "PM10": 1, "CO": 1}}).status_code == 400


def test_higher_particles_raise_the_dust_factor():
    dust = next(f["letter"] for f in REPORT["factors"] if f["name"] == "Dust and smoke")
    base = {"NO2": 30, "SO2": 15, "O3": 30, "CO": 1}
    low = client.post("/api/score", json={"readings": {**base, "PM2.5": 20, "PM10": 40}}).json()["scores"][dust]
    high = client.post("/api/score", json={"readings": {**base, "PM2.5": 300, "PM10": 500}}).json()["scores"][dust]
    assert high > low


def test_every_graph_is_explained():
    graphs = client.get("/api/graphs").json()["graphs"]
    assert len(graphs) == len({g["id"] for g in graphs})
    for g in graphs:
        assert g["figure"]["data"] and g["insight"]
        assert all(g["explain"][k] for k in ("what", "how", "read", "where"))
