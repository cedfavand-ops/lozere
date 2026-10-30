#!/usr/bin/env python3
"""
Construit la page statique (dossier site/) : Tn et Tx du lendemain sur la Lozère,
à partir du run ICON-CH1 de 12 UTC (repli sur le run de 9 UTC si le 12Z n'est pas sorti
après --wait-minutes). Rien n'est coloré hors du département.

  python build_site.py                       # télécharge le run 12Z du jour et construit site/
  python build_site.py --npz archive_icon_ch1/20261024T12Z_T_2M.npz   # utilise un run déjà archivé

Fenêtres (heure locale Europe/Paris, jour J = jour du run) :
  Tn : J 18h -> J+1 18h        Tx : J+1 6h -> J+2 6h (tronquée à 33 h d'échéance)

Le modèle de correction 150 m se branche dans apply_model() ; en attendant,
la page affiche l'interpolation brute d'ICON sur la grille 150 m.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

TZ = ZoneInfo("Europe/Paris")
GRID_M = 150            # maille de travail (Lambert-93)
FILL_UPSAMPLE = 2       # suréchantillonnage cubique de l'image de couleurs : dégradés doux, sans pixels visibles
SMOOTH_SIGMA = 0.8      # lissage léger (en mailles de 150 m) : bords nets mais arrondis
EXTRA_VARS = ["U_10M", "V_10M", "CLCT"]   # vent à 10 m et nébulosité, pour l'indice d'inversion
NODATA_INT = -32768     # valeur "vide" dans les .bin (température x10 en int16)
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# 1. Tn / Tx sur les fenêtres locales
# --------------------------------------------------------------------------
def tn_tx(values_k, lead_hours, ref_utc):
    """values_k : (n_échéances, n_mailles) en kelvins ; ref_utc : datetime UTC du run."""
    day = ref_utc.astimezone(TZ).date()
    t0 = dt.datetime.combine(day, dt.time(18), TZ)
    tn_win = (t0, t0 + dt.timedelta(days=1))                                  # J 18h -> J+1 18h
    tx_win = (t0 + dt.timedelta(hours=12), t0 + dt.timedelta(days=1, hours=12))  # J+1 6h -> J+2 6h

    valid = [ref_utc + dt.timedelta(hours=int(h)) for h in lead_hours]
    in_tn = np.array([tn_win[0] <= v <= tn_win[1] for v in valid])
    in_tx = np.array([tx_win[0] <= v <= tx_win[1] for v in valid])

    if valid[-1] < tn_win[1]:
        raise RuntimeError(f"Run incomplet : la fenêtre Tn n'est pas couverte (dernière échéance {valid[-1]}).")
    if valid[-1] < tx_win[0] + dt.timedelta(hours=12):
        raise RuntimeError("Run trop court pour estimer Tx (il faut au moins jusqu'à 18h locale J+1).")

    tn = values_k[in_tn].min(axis=0) - 273.15
    tx = values_k[in_tx].max(axis=0) - 273.15
    info = {
        "tn_window_utc": [tn_win[0].astimezone(dt.timezone.utc).isoformat(),
                          tn_win[1].astimezone(dt.timezone.utc).isoformat()],
        "tx_window_utc": [tx_win[0].astimezone(dt.timezone.utc).isoformat(),
                          tx_win[1].astimezone(dt.timezone.utc).isoformat()],
        "tx_truncated_at_utc": valid[-1].isoformat() if valid[-1] < tx_win[1] else None,
    }
    return tn.astype("float32"), tx.astype("float32"), info


# --------------------------------------------------------------------------
# 2. Géométrie : département, grille 150 m, masque
# --------------------------------------------------------------------------
def load_geometry(path):
    from shapely.geometry import shape
    from shapely.ops import unary_union
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    if gj["type"] == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in gj["features"]]
    elif gj["type"] == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    return unary_union(geoms)


def target_grid(geom_l93):
    from rasterio.transform import from_origin
    x0, y0, x1, y1 = geom_l93.bounds
    xmin, ymin = np.floor(x0 / GRID_M) * GRID_M, np.floor(y0 / GRID_M) * GRID_M
    xmax, ymax = np.ceil(x1 / GRID_M) * GRID_M, np.ceil(y1 / GRID_M) * GRID_M
    w, h = int((xmax - xmin) / GRID_M), int((ymax - ymin) / GRID_M)
    transform = from_origin(xmin, ymax, GRID_M, GRID_M)
    xs = xmin + (np.arange(w) + 0.5) * GRID_M
    ys = ymax - (np.arange(h) + 0.5) * GRID_M
    X, Y = np.meshgrid(xs, ys)
    return transform, X, Y


def load_layers(shape):
    """MNT 150 m et motif d'inversion, sur la grille de la carte ; (None, None) si absents ou incompatibles."""
    dem_p, motif_p = HERE / "data" / "dem_150m_l93.tif", HERE / "data" / "motif_150m_l93.tif"
    if not (dem_p.exists() and motif_p.exists()):
        return None, None
    import rasterio
    with rasterio.open(dem_p) as d:
        dem = d.read(1).astype("float32")
    with rasterio.open(motif_p) as m:
        motif = m.read(2).astype("float32")
    if dem.shape != shape or motif.shape != shape:
        print(f"Couches MNT/motif de taille {dem.shape}/{motif.shape} != grille {shape} : relancer 'Préparer les données'.")
        return None, None
    return dem, np.nan_to_num(motif, nan=0.0)


def apply_model(tn, tx, X, Y, ctx):
    """
    Correction 150 m (voir model.py). tn, tx : champs ICON interpolés sur la grille 150 m, en °C.
    ctx : pts (mailles ICON en Lambert-93), lat, lon, lead_hours, run_utc, extra (U_10M, V_10M, CLCT),
          hsurf (altitude du relief ICON), inside (masque du département).
    Renvoie (tn, tx, info) ; sans MNT/motif, renvoie ICON brut.
    """
    import model
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter

    info = {"model": "icon_brut"}
    dem, motif = load_layers(X.shape)
    if dem is None:
        print("MNT ou motif absent : carte = ICON brut interpolé (sans correction).")
        return tn, tx, info

    pts, inside = ctx["pts"], ctx["inside"]

    # Altitude du relief vu par ICON : champ 'h' du fichier de grille s'il est cohérent avec le MNT,
    # sinon MNT lissé à l'échelle d'une maille ICON.
    z_icon, z_src = None, "mnt_lisse"
    hs = ctx.get("hsurf")
    if hs is not None and len(hs) == len(pts):
        zg = griddata(pts, hs, (X, Y), method="linear")
        ok, r, mad = model.hsurf_ok(zg, dem)
        print(f"Relief ICON (champ h) vs MNT : corrélation {r:.2f}, écart moyen {mad:.0f} m -> "
              f"{'utilisé' if ok else 'REJETÉ (incohérent)'}")
        if ok:
            z_icon, z_src = zg, "icon_h"
    if z_icon is None:
        z_icon = gaussian_filter(dem, 500.0 / GRID_M, mode="nearest")
        print("Relief ICON : MNT lissé sur ~1 km (champ h indisponible ou rejeté).")
    z_icon = np.where(np.isfinite(z_icon), z_icon, gaussian_filter(dem, 500.0 / GRID_M, mode="nearest"))

    # Indice d'inversion alpha à partir des prévisions ICON de la nuit
    extra = ctx.get("extra", {})
    u, v, c = extra.get("U_10M"), extra.get("V_10M"), extra.get("CLCT")
    wind = np.hypot(u, v) if (u is not None and v is not None) else None
    for name, arr in (("vent (U_10M, V_10M)", wind), ("nébulosité (CLCT)", c)):
        if arr is None:
            print(f"ATTENTION : {name} indisponible -> facteur neutre (0,5).")
    night = model.night_leads(ctx["lead_hours"], ctx["run_utc"], float(np.mean(ctx["lat"])), float(np.mean(ctx["lon"])))
    alpha_pts, w_m, c_m = model.alpha_mesh(wind, c, night, len(pts))
    alpha = griddata(pts, alpha_pts, (X, Y), method="linear")
    alpha = np.where(np.isfinite(alpha), alpha, np.nanmean(alpha_pts))
    alpha_reg = model.regional_alpha(alpha_pts)                   # le quart le plus calme du domaine
    alpha = np.maximum(alpha, alpha_reg)
    print(f"Indice radiatif régional (quart le plus calme) : {alpha_reg:.2f} ; indice retenu en moyenne : "
          f"{float(alpha[ctx['inside']].mean()):.2f}")
    if w_m is not None:
        print(f"Vent nocturne à 10 m : moyenne {np.mean(w_m):.1f} m/s (mailles de {np.min(w_m):.1f} à {np.max(w_m):.1f})")
    if c_m is not None:
        print(f"Nébulosité nocturne : moyenne {100 * np.mean(c_m):.0f} % (mailles de {100 * np.min(c_m):.0f} à {100 * np.max(c_m):.0f} %)")
    forced = ctx.get("force_alpha")
    if forced is not None:
        alpha = np.full_like(alpha, float(forced))
        print(f"DÉMONSTRATION : indice d'inversion forcé à {float(forced):.2f} (ce n'est PAS la prévision réelle).")

    state = model.load_state(HERE / "history" / model.STATE_NAME)
    print(f"Correction radiative : gain sur le motif froid x{state['gain_neg']:.2f}, biais {state['bias']:+.2f} °C "
          f"({state['n_updates']} ajustements sur les stations)")
    # Ancrage sur les stations : chaque station a appris sa propre amplitude (update_state.py)
    S = None
    anchored = []
    if state.get("stations") and ctx.get("to_l93") is not None and ctx.get("transform") is not None:
        for st in load_stations():
            if st["id"] in state["stations"]:
                x, y = ctx["to_l93"].transform(float(st["lon"]), float(st["lat"]))
                anchored.append({"id": st["id"], "x": x, "y": y,
                                 "motif": float(sample_grid(motif, ctx["transform"], [x], [y])[0]),
                                 "dem": float(sample_grid(dem, ctx["transform"], [x], [y])[0])})
        S = model.station_field(X, Y, motif, dem, anchored, state)
        print(f"Ancrage aux stations : {len(anchored)} stations, terme de -{abs(S.min()):.1f} à {S.max():+.1f} °C par unité d'indice")
    tn2, tx2 = model.apply_correction(tn, tx, dem, z_icon, alpha, motif, state, S)
    ctx["layers"] = {"alpha": alpha, "motif": motif, "dz": np.clip(dem - z_icon, -model.MAX_DZ, model.MAX_DZ),
                     "term": model.radiative_term(alpha, motif, state),
                     "station_term": np.clip(alpha, 0, 1) * (S if S is not None else 0.0) * np.ones(X.shape)}
    d_tn, d_tx = (tn2 - tn)[inside], (tx2 - tx)[inside]
    print(f"Nuit retenue : {int(night.sum())} échéances ; alpha moyen {alpha[inside].mean():.2f} "
          f"(min {alpha[inside].min():.2f}, max {alpha[inside].max():.2f})")
    print(f"Correction Tn : médiane {np.median(d_tn):+.2f} °C, de {d_tn.min():+.1f} à {d_tn.max():+.1f} ; "
          f"correction Tx : médiane {np.median(d_tx):+.2f} °C, de {d_tx.min():+.1f} à {d_tx.max():+.1f}")
    if max(np.abs(d_tn).max(), np.abs(d_tx).max()) > 8:
        print("ATTENTION : correction supérieure à 8 °C en certains points, à vérifier.")

    info = {
        "model": "icon+motif",
        "alpha_mean": float(alpha[inside].mean()),
        "alpha_min": float(alpha[inside].min()),
        "alpha_max": float(alpha[inside].max()),
        "night_hours": int(night.sum()),
        "wind_night_mean": None if w_m is None else float(np.mean(w_m)),
        "cloud_night_mean": None if c_m is None else float(np.mean(c_m)),
        "z_icon_source": z_src,
        "alpha_regional": float(alpha_reg),
        "stations_anchored": len(anchored),
        "gain_neg": state["gain_neg"],
        "bias": state["bias"],
        "alpha_forced": None if forced is None else float(forced),
        "missing_inputs": [n for n, a_ in (("vent", wind), ("nebulosite", c)) if a_ is None],
    }
    return tn2, tx2, info


def to_mercator(arr, transform):
    """Reprojette la grille L93 vers Web Mercator (nécessaire pour L.imageOverlay)."""
    import rasterio
    from rasterio.transform import array_bounds, from_origin
    from rasterio.warp import Resampling, reproject, transform_bounds
    h, w = arr.shape
    west, south, east, north = array_bounds(h, w, transform)
    mx0, my0, mx1, my1 = transform_bounds("EPSG:2154", "EPSG:3857", west, south, east, north)
    res = GRID_M / np.cos(np.radians(44.5))      # ~ 150 m au sol à cette latitude
    dw, dh = int(np.ceil((mx1 - mx0) / res)), int(np.ceil((my1 - my0) / res))
    dst_tr = from_origin(mx0, my1, res, res)
    dst = np.full((dh, dw), np.nan, dtype="float32")
    reproject(arr, dst, src_transform=transform, src_crs="EPSG:2154",
              dst_transform=dst_tr, dst_crs="EPSG:3857",
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    bounds = (mx0, my1 - dh * res, mx0 + dw * res, my1)    # xmin, ymin, xmax, ymax effectifs
    return dst, bounds


# --------------------------------------------------------------------------
# 3. Rendu : PNG transparent + binaire pour la lecture au clic
# --------------------------------------------------------------------------
def color_scale(arr):
    vmin = float(np.floor(np.nanpercentile(arr, 0.5)))
    vmax = float(np.ceil(np.nanpercentile(arr, 99.5)))
    return (vmin, vmax) if vmax > vmin else (vmin, vmin + 1.0)


def get_cmap(name):
    """
    « lozere » : du plus froid au plus chaud, rose -> mauve -> bleu foncé -> bleu -> cyan -> vert -> jaune
    -> orange -> rouge -> rouge sombre (la fin est la palette « turbo »). Tout autre nom : palette matplotlib.
    """
    from matplotlib import colormaps
    if name != "lozere":
        return colormaps[name]
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    turbo = colormaps["turbo"]
    cold = [(0.00, "#ffc2e9"),    # rose clair : température la plus froide
            (0.05, "#ff6fcf"),    # rose vif
            (0.10, "#c34fe0"),    # mauve
            (0.15, "#7a37d4")]    # violet
    n = 16
    warm = [(0.20 + 0.80 * i / n, to_hex(turbo(0.06 + 0.94 * i / n))) for i in range(n + 1)]   # turbo : bleu foncé -> rouge sombre
    return LinearSegmentedColormap.from_list("lozere", cold + warm, N=512)


COLD_PINK, COLD_VIOLET = "#ffc2e9", "#8e3fd6"
T_PINK_MAX, T_VIOLET_MAX, T_EDGE = 0.5, 1.0, 0.15   # °C : rose jusqu'à +0,5 ; violet de +0,5 à +1 ; transitions de +-0,15 °C


def lozere_anchors(vmin, vmax, n=16):
    """
    Palette « lozere » ancrée sur la température minimale de la carte.

    - Rose pâle : températures comprises entre vmin et vmin + 0,5 °C.
    - Violet : températures comprises entre vmin + 0,5 °C et vmin + 1 °C.
    - Au-delà de vmin + 1 °C : dégradé bleu foncé, bleu, cyan,
      vert, jaune, orange, rouge et rouge sombre.

    Les seuils évoluent automatiquement selon le minimum de la carte.
    Renvoie une liste de points (température, couleur hexadécimale).
    """
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    turbo = colormaps["turbo"]

    # Seuils calculés à partir de la température minimale réelle
    t_pink_max = vmin + 0.5
    t_violet_max = vmin + 1.0

    # Début du dégradé coloré classique
    t0 = t_violet_max
    t1 = max(vmax, t0 + 1.0)

    # Zones rose pâle et violette
    pts = [
        (vmin - 100.0, COLD_PINK),
        (vmin, COLD_PINK),
        (t_pink_max, COLD_PINK),
        (t_violet_max, COLD_VIOLET),
    ]

    # Dégradé classique au-dessus de vmin + 1 °C
    pts += [
        (
            t0 + (t1 - t0) * k / n,
            to_hex(turbo(0.06 + 0.94 * k / n))
        )
        for k in range(n + 1)
    ]

    return pts



def lozere_rgb(values, vmin, vmax):
    from matplotlib.colors import to_rgb
    pts = lozere_anchors(vmin, vmax)
    T = np.array([p[0] for p in pts])
    cols = np.array([to_rgb(p[1]) for p in pts])
    v = np.asarray(values, dtype=float)
    return np.stack([np.interp(v, T, cols[:, j]) for j in range(3)], axis=-1)


def palette_stops(vmin, vmax, cmap_name="lozere"):
    """Couleurs de la légende : liste [position entre 0 et 1, couleur], positions croissantes."""
    from matplotlib.colors import to_hex
    if cmap_name != "lozere":
        pal = palette_hex(cmap_name)
        return [[i / (len(pal) - 1), c] for i, c in enumerate(pal)]
    pts = lozere_anchors(vmin, vmax)
    span = float(vmax - vmin)
    stops = [[0.0, to_hex(lozere_rgb(vmin, vmin, vmax))]]
    for t, c in pts:
        pos = (t - vmin) / span
        if 1e-6 < pos < 1 - 1e-6:
            stops.append([round(pos, 5), c])
    stops.append([1.0, to_hex(lozere_rgb(vmax, vmin, vmax))])
    return stops


def colorize(arr, vmin, vmax, cmap_name="lozere"):
    from PIL import Image
    if cmap_name == "lozere":
        rgb = lozere_rgb(np.nan_to_num(arr, nan=vmin), vmin, vmax)
        rgba = np.concatenate([(rgb * 255).round().astype("uint8"), np.zeros(rgb.shape[:-1] + (1,), "uint8")], axis=-1)
        rgba[..., 3] = np.where(np.isnan(arr), 0, 255)
        return Image.fromarray(rgba, "RGBA")
    norm = np.clip((np.nan_to_num(arr, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    rgba = (get_cmap(cmap_name)(norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isnan(arr), 0, 255)       # transparent hors Lozère
    return Image.fromarray(rgba, "RGBA")


def palette_hex(cmap_name, n=17):
    from matplotlib import colors
    cm = get_cmap(cmap_name)
    return [colors.to_hex(cm(i / (n - 1))) for i in range(n)]


def encode_int16(arr):
    q = np.rint(np.nan_to_num(arr, nan=0) * 10).astype("<i2")
    q[np.isnan(arr)] = NODATA_INT
    return q.tobytes()


# --------------------------------------------------------------------------
# 4. Construction du site
# --------------------------------------------------------------------------
def upsample_fill(a, factor=FILL_UPSAMPLE):
    """Champ complet -> image plus fine par spline cubique (les éventuelles mailles vides sont comblées d'abord)."""
    from scipy.ndimage import distance_transform_edt, zoom
    bad = ~np.isfinite(a)
    if bad.any():
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        a = a[tuple(idx)]
    return zoom(a, factor, order=3, mode="nearest").astype("float32")


def smooth_field(a, sigma=SMOOTH_SIGMA):
    """Lissage gaussien léger ; comble d'abord les éventuelles mailles sans valeur par le plus proche voisin."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    bad = ~np.isfinite(a)
    if bad.any():
        idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
        a = a[tuple(idx)]
    return gaussian_filter(a, sigma, mode="nearest").astype("float32")


def load_stations():
    """Stations à suivre : data/stations_infoclimat.json, stations_meteofrance.json et stations_datacake.json."""
    stations = []
    for fname, source in (("stations_infoclimat.json", "infoclimat"), ("stations_meteofrance.json", "meteofrance"),
                          ("stations_datacake.json", "datacake")):
        p = HERE / "data" / fname
        if p.exists():
            for st in json.loads(p.read_text(encoding="utf-8")):
                st = dict(st)
                st["source"] = source
                stations.append(st)
    return stations


def sample_grid(arr, transform, xs, ys):
    """Valeur d'une grille 150 m (Lambert-93) aux points (xs, ys), par interpolation bilinéaire."""
    from scipy.ndimage import map_coordinates
    cols = (np.asarray(xs, dtype=float) - transform.c) / GRID_M - 0.5
    rows = (transform.f - np.asarray(ys, dtype=float)) / GRID_M - 0.5
    return map_coordinates(arr, [rows, cols], order=1, mode="nearest")


def record_history(tn, tx, tn_pre, tx_pre, layers, transform, to_l93, run_utc, win_info, model_info):
    """
    Enregistre dans history/forecast_AAAA-MM-JJ.json les valeurs prévues (corrigées et ICON seul) aux stations,
    avec les fenêtres de prévision : verify.py les comparera le lendemain aux observations.
    """
    stations = load_stations()
    if not stations:
        print("Aucune station configurée (data/stations_*.json) : historique non enregistré.")
        return
    lon = [float(st["lon"]) for st in stations]
    lat = [float(st["lat"]) for st in stations]
    xs, ys = to_l93.transform(lon, lat)
    cols = {"tn": sample_grid(tn, transform, xs, ys), "tx": sample_grid(tx, transform, xs, ys),
            "tn_icon": sample_grid(tn_pre, transform, xs, ys), "tx_icon": sample_grid(tx_pre, transform, xs, ys)}
    if layers:
        for k in ("alpha", "motif", "dz", "term", "station_term"):
            if k in layers:
                cols[k] = sample_grid(layers[k], transform, xs, ys)
        cols["motif_term"] = cols.pop("term") if "term" in cols else None
        if cols["motif_term"] is None:
            del cols["motif_term"]
    rows = []
    for i, st in enumerate(stations):
        row = {"id": st["id"], "name": st.get("name"), "source": st["source"],
               "lat": st["lat"], "lon": st["lon"], "z_station": st.get("elevation")}
        row.update({k: round(float(v[i]), 2) for k, v in cols.items()})
        rows.append(row)
    day = run_utc.astimezone(TZ).date().isoformat()
    rec = {"day": day, "run_utc": run_utc.isoformat(),
           "tn_window_utc": win_info["tn_window_utc"], "tx_window_utc": win_info["tx_window_utc"],
           "tx_truncated_at_utc": win_info.get("tx_truncated_at_utc"), "fallback": win_info.get("fallback"),
           "model": model_info.get("model"), "alpha_mean": model_info.get("alpha_mean"),
           "wind_night_mean": model_info.get("wind_night_mean"), "cloud_night_mean": model_info.get("cloud_night_mean"),
           "gain_neg": model_info.get("gain_neg"), "bias": model_info.get("bias"),
           "stations": rows}
    path = HERE / "history" / f"forecast_{day}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Historique : prévision aux {len(rows)} stations enregistrée dans {path.name}")


def build_site(lat, lon, tn_pts, tx_pts, run_utc, win_info, geojson, out_dir, cmap="lozere", ctx=None):
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_bounds
    from scipy.interpolate import griddata
    from shapely.ops import transform as shp_transform

    to_l93 = Transformer.from_crs(4326, 2154, always_xy=True)
    geom = load_geometry(geojson)
    geom_l93 = shp_transform(to_l93.transform, geom)
    transform, X, Y = target_grid(geom_l93)
    h, w = X.shape

    px, py = to_l93.transform(lon, lat)
    pts = np.column_stack([px, py])
    tn = griddata(pts, tn_pts, (X, Y), method="linear").astype("float32")
    tx = griddata(pts, tx_pts, (X, Y), method="linear").astype("float32")
    tn_pre, tx_pre = tn.copy(), tx.copy()                # ICON seul, avant correction (pour l'historique)

    inside = geometry_mask([geom_l93], out_shape=(h, w), transform=transform, invert=True)
    ctx = dict(ctx or {}, run_utc=run_utc, pts=pts, lat=lat, lon=lon, inside=inside, to_l93=to_l93, transform=transform)
    tn, tx, model_info = apply_model(tn, tx, X, Y, ctx)
    tn, tx = smooth_field(tn), smooth_field(tx)          # lissage léger : bords nets mais arrondis
    tn_full, tx_full = tn.copy(), tx.copy()              # champs sur toute l'emprise (pour les contours)
    if ctx.get("record", True) and ctx.get("force_alpha") is None:    # jamais pour la page de démonstration
        try:
            record_history(tn_full, tx_full, tn_pre, tx_pre, ctx.get("layers"), transform, to_l93,
                           run_utc, win_info, model_info)
        except Exception as e:
            print(f"Historique non enregistré ({type(e).__name__}: {e})")
    tn[~inside] = np.nan
    tx[~inside] = np.nan

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_utc": run_utc.isoformat(),
        "grid_m": GRID_M,
        "palette": palette_hex(cmap),
        **win_info,
        **model_info,
    }
    for name, arr, arr_full in (("tn", tn, tn_full), ("tx", tx, tx_full)):
        merc, bounds = to_mercator(arr, transform)
        vmin, vmax = color_scale(merc)
        merc_full, _ = to_mercator(arr_full, transform)
        # Image de couleurs sur toute l'emprise (le voile blanc de la page la découpe au contour exact du département)
        colorize(upsample_fill(merc_full), vmin, vmax, cmap).convert("RGB").save(out / f"{name}.png", optimize=True)
        (out / f"{name}.bin").write_bytes(encode_int16(merc_full))   # champ complet : isolignes et lecture au survol
        meta[name] = {"vmin": vmin, "vmax": vmax,
                      "min": float(np.nanmin(merc)), "max": float(np.nanmax(merc)),
                      "stops": palette_stops(vmin, vmax, cmap)}
        meta["width"], meta["height"] = int(merc.shape[1]), int(merc.shape[0])
        meta["bounds_3857"] = [float(b) for b in bounds]
    w_, s_, e_, n_ = transform_bounds("EPSG:3857", "EPSG:4326", *meta["bounds_3857"])
    meta["bounds_latlng"] = [[s_, w_], [n_, e_]]

    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    shutil.copy(HERE / "web" / "index.html", out / "index.html")
    shutil.copy(geojson, out / "lozere.geojson")
    (out / ".nojekyll").write_text("")
    print(f"Site écrit dans {out}/ ({meta['width']}x{meta['height']} px).")


# --------------------------------------------------------------------------
# 5. Récupération du run 12Z
# --------------------------------------------------------------------------
def default_ref_12z():
    now = dt.datetime.now(dt.timezone.utc)
    ref = now.replace(hour=12, minute=0, second=0, microsecond=0)
    return ref if now >= ref else ref - dt.timedelta(days=1)


def fetch_run(refs, wait_minutes):
    """
    refs : runs par ordre de préférence (ex. [12Z, 9Z]).
    On attend le premier jusqu'à wait_minutes ; sinon on prend le premier repli disponible.
    Le run doit être complet (échéance +33 h présente).
    """
    import archive_icon_ch1 as icon
    first, others = refs[0], refs[1:]
    deadline = time.time() + wait_minutes * 60
    chosen = None
    while True:
        if icon.find_item(first, "T_2M", icon.MAX_HORIZON_H):
            chosen = first
            break
        if time.time() > deadline:
            break
        print(f"Run {first:%H} UTC pas encore complet, nouvelle tentative dans 2 min...")
        time.sleep(120)
    if chosen is None:
        for ref in others:
            if icon.find_item(ref, "T_2M", icon.MAX_HORIZON_H):
                chosen = ref
                print(f"REPLI : run {first:%H} UTC indisponible, utilisation du run {ref:%H} UTC.")
                break
    if chosen is None:
        raise RuntimeError("Aucun des runs demandés n'est complet : "
                           + ", ".join(f"{r:%d/%m %H} UTC" for r in refs))
    icon.archive_run(chosen, ["T_2M"] + EXTRA_VARS)
    return icon.OUT_DIR / f"{chosen:%Y%m%dT%H}Z_T_2M.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default=str(HERE / "data" / "lozere.geojson"))
    ap.add_argument("--out", default="site")
    ap.add_argument("--npz", help="Run déjà archivé (format de archive_icon_ch1.py)")
    ap.add_argument("--ref", help="Run UTC, ex. 2026-10-24T12 (défaut : dernier 12Z)")
    ap.add_argument("--wait-minutes", type=int, default=30,
                    help="attente maximale du run 12Z avant repli sur le 9Z")
    ap.add_argument("--no-fallback", action="store_true", help="n'utiliser que le run demandé (pas de repli 9Z)")
    ap.add_argument("--cmap", default="lozere")
    ap.add_argument("--no-history", action="store_true", help="ne pas enregistrer la prévision aux stations")
    ap.add_argument("--force-alpha", type=float, default=None,
                    help="DÉMO : impose l'indice d'inversion (0 à 1) au lieu de la règle météo")
    args = ap.parse_args()

    if args.npz:
        npz = Path(args.npz)
    else:
        ref = (dt.datetime.fromisoformat(args.ref).replace(tzinfo=dt.timezone.utc)
               if args.ref else default_ref_12z())
        refs = [ref] if args.no_fallback else [ref, ref - dt.timedelta(hours=3)]   # 12Z puis 9Z
        npz = fetch_run(refs, args.wait_minutes)

    d = np.load(npz)
    run_utc = dt.datetime.fromisoformat(str(d["ref_utc"]).replace("Z", "+00:00"))
    if run_utc.hour not in (9, 12):
        sys.exit(f"Ce pipeline n'utilise que les runs 12Z (ou 9Z en repli) ; reçu : {run_utc:%H} UTC.")
    tn_pts, tx_pts, info = tn_tx(d["values"], d["lead_hours"], run_utc)
    info["fallback"] = run_utc.hour != 12

    extra = {}
    for var in EXTRA_VARS:
        pv = npz.with_name(npz.name.replace("_T_2M", f"_{var}"))
        if pv.exists():
            dv = np.load(pv)
            if dv["values"].shape == d["values"].shape:
                extra[var] = dv["values"]
                continue
        print(f"Variable {var} absente ou incomplète pour ce run.")
    hsurf = None
    try:
        import archive_icon_ch1 as icon
        hsurf = icon.load_hsurf()
    except Exception as e:   # réseau absent en test local, etc.
        print(f"Altitude du relief ICON indisponible ({type(e).__name__}) : repli sur le MNT lissé.")
    ctx = {"lead_hours": d["lead_hours"], "extra": extra, "hsurf": hsurf, "force_alpha": args.force_alpha,
           "record": not args.no_history}
    build_site(d["lat"], d["lon"], tn_pts, tx_pts, run_utc, info,
               args.geojson, args.out, args.cmap, ctx=ctx)
    for name in ("index.html", "meta.json", "tn.png", "tx.png", "tn.bin", "tx.bin", "lozere.geojson"):
        if not (Path(args.out) / name).exists():
            sys.exit(f"Construction incomplète : {name} manquant dans {args.out}/")


if __name__ == "__main__":
    main()
    # Sortie immédiate, une fois le site vérifié : sur GitHub Actions, l'arrêt normal de l'interpréteur
    # provoquait « double free or corruption » (conflit entre les bibliothèques natives d'ecCodes et de
    # GDAL/PROJ chargées dans le même processus). Tout est déjà écrit sur le disque à ce stade.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
