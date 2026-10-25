#!/usr/bin/env python3
"""
Calibration RÉTROSPECTIVE de l'indice d'inversion, à partir des seules observations des stations.

Idée : pour chaque nuit passée, les minima Tn des stations sont décrits par
        Tn_i = a + g * altitude_i + alpha * motif_i + bruit
  - a    : niveau général de la nuit (situation synoptique) ;
  - g    : gradient vertical de la nuit, libre (positif lors d'une inversion) ;
  - alpha: intensité avec laquelle le motif ECOSTRESS (data/motif_150m_l93.tif) se retrouve dans les mesures.
On estime a, g et alpha par moindres carrés pour chaque nuit, puis on cherche comment alpha dépend du vent
nocturne observé : la règle a priori (model.py) est remplacée par une règle ajustée sur les données.

Limites à garder en tête
  - Seulement ~18 stations, surtout en altitude : alpha est estimé avec une incertitude (colonne « se »).
  - alpha « observé » est un alpha EFFECTIF : il intègre l'écart entre température de surface (motif) et
    température sous abri, et la représentativité des stations. S'il plafonne vers 0,6, c'est le facteur
    d'amplitude à appliquer au motif.
  - La nébulosité n'est pas observée par ces stations : seule la dépendance au vent est calibrée.

Lancement : workflow GitHub « Calibrer l'indice d'inversion » (INFOCLIMAT_TOKEN requis), ou
    python calibrate.py --start 2026-04-01 --end 2026-10-24
Sorties : history/alpha_obs.csv (une ligne par nuit), history/alpha_obs.png
Dépendances : numpy scipy rasterio pyproj requests matplotlib
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import obs

HERE = Path(__file__).resolve().parent
TZ = ZoneInfo("Europe/Paris")
UTC = dt.timezone.utc
OUT_CSV = HERE / "history" / "alpha_obs.csv"
OUT_PNG = HERE / "history" / "alpha_obs.png"
MIN_STATIONS = 10        # stations minimales pour estimer une nuit
MAX_SE = 0.5             # incertitude maximale sur alpha pour entrer dans l'ajustement de la règle
GRID_M = 150.0


# ------------------------------------------------------------------ statistiques (testables sans réseau)
def estimate_night(tn, z, m, min_stations=MIN_STATIONS):
    """Tn_i = a + g*z_i + alpha*m_i par moindres carrés. Renvoie un dict ou None si trop peu de stations."""
    tn, z, m = (np.asarray(v, dtype=float) for v in (tn, z, m))
    ok = np.isfinite(tn) & np.isfinite(z) & np.isfinite(m)
    n = int(ok.sum())
    if n < min_stations:
        return None
    zk = (z[ok] - z[ok].mean()) / 1000.0
    A = np.column_stack([np.ones(n), zk, m[ok]])
    coef, _, rank, _ = np.linalg.lstsq(A, tn[ok], rcond=None)
    if rank < 3:
        return None
    resid = tn[ok] - A @ coef
    s2 = float(resid @ resid) / (n - 3)
    cov = s2 * np.linalg.inv(A.T @ A)
    tot = float(((tn[ok] - tn[ok].mean()) ** 2).sum())
    return {"n": n, "a": float(coef[0]), "gamma_per_km": float(coef[1]), "alpha": float(coef[2]),
            "se": float(np.sqrt(cov[2, 2])), "r2": 1 - float(resid @ resid) / tot if tot > 0 else float("nan"),
            "resid_sd": float(np.sqrt(s2))}


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def rule(w, w0, w1):
    """1 si vent <= w0, 0 si vent >= w1, progressif entre les deux (même forme que model.py)."""
    return 1.0 - smoothstep((np.asarray(w, dtype=float) - w0) / (w1 - w0))


def fit_rule(alpha, se, wind):
    """
    Ajuste alpha ≈ A * rule(vent; w0, w1) par moindres carrés pondérés (poids 1/se²).
    Renvoie {w0, w1, A, gain} ; gain = part de variance expliquée par rapport à un alpha constant.
    """
    alpha, se, wind = (np.asarray(v, dtype=float) for v in (alpha, se, wind))
    w = 1.0 / np.maximum(se, 0.1) ** 2
    mean = np.sum(w * alpha) / np.sum(w)
    err_const = float(np.sum(w * (alpha - mean) ** 2))
    best = None
    for w0 in np.arange(0.5, 4.01, 0.25):
        for w1 in np.arange(w0 + 0.5, 9.01, 0.25):
            f = rule(wind, w0, w1)
            den = float(np.sum(w * f * f))
            if den <= 0:
                continue
            A = float(np.sum(w * f * alpha)) / den
            err = float(np.sum(w * (alpha - A * f) ** 2))
            if best is None or err < best["err"]:
                best = {"w0": float(w0), "w1": float(w1), "A": A, "err": err}
    best["gain"] = 1.0 - best["err"] / err_const if err_const > 0 else float("nan")
    return best


# ------------------------------------------------------------------ données
def load_stations():
    st = []
    for fname, src in (("stations_infoclimat.json", "infoclimat"), ("stations_datacake.json", "datacake")):
        p = HERE / "data" / fname
        if p.exists():
            for s in json.loads(p.read_text(encoding="utf-8")):
                s = dict(s)
                s["source"] = src
                st.append(s)
    return st


def station_motif(stations):
    """Motif d'inversion (°C) à chaque station ; NaN si la station est hors de la grille de la Lozère."""
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import map_coordinates
    with rasterio.open(HERE / "data" / "motif_150m_l93.tif") as src:
        motif = np.nan_to_num(src.read(2).astype("float64"))
        tr = src.transform
    h, w = motif.shape
    xs, ys = Transformer.from_crs(4326, 2154, always_xy=True).transform(
        [float(s["lon"]) for s in stations], [float(s["lat"]) for s in stations])
    cols = (np.asarray(xs) - tr.c) / GRID_M - 0.5
    rows = (tr.f - np.asarray(ys)) / GRID_M - 0.5
    m = map_coordinates(motif, [rows, cols], order=1, mode="nearest")
    outside = (cols < 0) | (rows < 0) | (cols > w - 1) | (rows > h - 1)
    m[outside] = np.nan
    return m


def fetch_history(ids, d0, d1, token, span_days=15):
    series = {}
    d = d0 - dt.timedelta(days=1)
    while d <= d1 + dt.timedelta(days=1):
        e = d + dt.timedelta(days=span_days)
        part = obs.fetch_infoclimat(ids, d.isoformat(), e.isoformat(), token, chunk=6)
        for sid, rows in part.items():
            series[sid] = obs.dedupe(series.get(sid, []) + rows)
        print(f"  observations {d} -> {e} : {sum(len(v) for v in part.values())} relevés")
        d = e
    return series


def night_wind(stations, series, tn0):
    """Vent moyen (m/s) de la nuit, moyenné sur les stations qui le mesurent (au moins 5)."""
    from prepare_motif import solar_elevation
    vals = []
    for s in stations:
        ser = series.get(s["id"])
        if not ser:
            continue
        v = [p["wind_ms"] for p in ser
             if p.get("wind_ms") is not None and tn0 <= p["t"] <= tn0 + dt.timedelta(hours=15)
             and solar_elevation(float(s["lat"]), float(s["lon"]), p["t"]) < -0.5]
        if len(v) >= 6:
            vals.append(sum(v) / len(v))
    return (float(np.mean(vals)), len(vals)) if len(vals) >= 5 else (None, len(vals))


def analyse(stations, series, motif, d0, d1):
    z = np.array([float(s["elevation"]) if s.get("elevation") is not None else np.nan for s in stations])
    rows = []
    day = d0
    while day <= d1:
        t0 = dt.datetime.combine(day, dt.time(18), TZ)
        t1 = t0 + dt.timedelta(days=1)
        tn = np.array([
            (lambda r: np.nan if r is None else r["value"])(
                obs.window_extreme(series.get(s["id"], []), t0.astimezone(UTC), t1.astimezone(UTC), "min"))
            for s in stations])
        est = estimate_night(tn, z, motif)
        if est is not None:
            wind, n_w = night_wind(stations, series, t0.astimezone(UTC))
            rows.append({"night": day.isoformat(), "month": day.month, **est,
                         "wind_obs": wind, "n_wind": n_w})
        day += dt.timedelta(days=1)
    return rows


def write_outputs(rows, fit):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["night", "month", "n", "alpha", "se", "gamma_per_km", "a", "r2", "resid_sd", "wind_obs", "n_wind"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (round(r[k], 3) if isinstance(r[k], float) else r[k]))
                        for k in fields})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use = [r for r in rows if r["wind_obs"] is not None]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].errorbar([r["wind_obs"] for r in use], [r["alpha"] for r in use], yerr=[r["se"] for r in use],
                   fmt="o", ms=3, alpha=0.5, elinewidth=0.6)
    if fit:
        ww = np.linspace(0, 9, 200)
        ax[0].plot(ww, fit["A"] * rule(ww, fit["w0"], fit["w1"]), "r", lw=2,
                   label=f"règle ajustée : A={fit['A']:.2f}, 1 jusqu'à {fit['w0']:.1f} m/s, 0 dès {fit['w1']:.1f} m/s")
        ax[0].legend()
    ax[0].axhline(0, c="k", lw=0.5)
    ax[0].set_xlabel("vent nocturne moyen observé (m/s)")
    ax[0].set_ylabel("alpha observé (amplitude du motif)")
    ax[0].set_title("Intensité du motif selon le vent")
    ax[1].scatter([r["wind_obs"] for r in use], [r["gamma_per_km"] for r in use], s=8, alpha=0.5)
    ax[1].axhline(-6.5, c="r", lw=0.8, ls="--", label="gradient standard")
    ax[1].axhline(0, c="k", lw=0.5)
    ax[1].set_xlabel("vent nocturne moyen observé (m/s)")
    ax[1].set_ylabel("gradient vertical de la nuit (°C/km)")
    ax[1].legend()
    ax[1].set_title("Gradient de Tn entre stations")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=80)


def main():
    today = dt.datetime.now(UTC).date()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=(today - dt.timedelta(days=150)).isoformat())
    ap.add_argument("--end", default=(today - dt.timedelta(days=1)).isoformat())
    args = ap.parse_args()
    d0, d1 = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    token = os.environ.get("INFOCLIMAT_TOKEN")
    if not token:
        sys.exit("INFOCLIMAT_TOKEN absent (secret du dépôt requis).")
    stations = [s for s in load_stations() if s["source"] == "infoclimat"]
    if not stations:
        sys.exit("Aucune station : lancer d'abord « Lister les stations Infoclimat ».")
    motif = station_motif(stations)
    keep = [i for i in range(len(stations)) if np.isfinite(motif[i])]
    print(f"{len(stations)} stations, {len(keep)} dans la grille du motif ; période {d0} -> {d1}")
    stations = [stations[i] for i in keep]
    motif = motif[keep]
    series = fetch_history([s["id"] for s in stations], d0, d1, token)
    rows = analyse(stations, series, motif, d0, d1)
    print(f"{len(rows)} nuits exploitables sur {(d1 - d0).days + 1} (au moins {MIN_STATIONS} stations)")
    if len(rows) < 15:
        print("Trop peu de nuits pour ajuster une règle.")
        write_outputs(rows, None)
        return

    al = np.array([r["alpha"] for r in rows])
    se = np.array([r["se"] for r in rows])
    print(f"alpha observé : médiane {np.median(al):.2f}, 10e-90e centiles {np.percentile(al, 10):.2f} / "
          f"{np.percentile(al, 90):.2f} ; incertitude typique {np.median(se):.2f}")
    sel = [r for r in rows if r["wind_obs"] is not None and r["se"] <= MAX_SE]
    print(f"{len(sel)} nuits avec vent observé et alpha assez précis (incertitude <= {MAX_SE})")
    fit = None
    if len(sel) >= 15:
        fit = fit_rule([r["alpha"] for r in sel], [r["se"] for r in sel], [r["wind_obs"] for r in sel])
        print(f"Règle ajustée : alpha = {fit['A']:.2f} jusqu'à {fit['w0']:.2f} m/s, nul à partir de {fit['w1']:.2f} m/s "
              f"(variance expliquée par rapport à un alpha constant : {100 * fit['gain']:.0f} %)")
        print("  A reporter dans model.py : WIND_FULL, WIND_ZERO et un facteur d'amplitude A sur le motif.")
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 20)):
            b = [r["alpha"] for r in sel if lo <= r["wind_obs"] < hi]
            if b:
                print(f"  vent {lo}-{hi} m/s : {len(b)} nuits, alpha moyen {np.mean(b):+.2f}")
    write_outputs(rows, fit)
    print(f"Écrit : {OUT_CSV.name}, {OUT_PNG.name}")


if __name__ == "__main__":
    main()
