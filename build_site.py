#!/usr/bin/env python3
"""
Construit la page statique (dossier site/) : Tn et Tx du lendemain sur la Lozère,
à partir du run ICON-CH1 de 12 UTC. Rien n'est coloré hors du département.

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
import shutil
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

TZ = ZoneInfo("Europe/Paris")
GRID_M = 150            # maille de travail (Lambert-93)
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


def apply_model(tn, tx, X, Y, ctx):
    """
    POINT D'ENTRÉE DU MODÈLE 150 m.
    tn, tx : champs ICON interpolés sur la grille 150 m (Lambert-93), en °C.
    X, Y   : coordonnées Lambert-93 du centre de chaque maille.
    ctx    : dict (date, run, etc.) ; y ajouter ensuite MNT, motif ECOSTRESS, coefficient alpha...
    Doit renvoyer (tn, tx) corrigés. Pour l'instant : identité.
    """
    return tn, tx


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


def colorize(arr, vmin, vmax, cmap_name="turbo"):
    from matplotlib import colormaps
    from PIL import Image
    norm = np.clip((np.nan_to_num(arr, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    rgba = (colormaps[cmap_name](norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isnan(arr), 0, 255)       # transparent hors Lozère
    return Image.fromarray(rgba, "RGBA")


def palette_hex(cmap_name, n=9):
    from matplotlib import colormaps, colors
    cm = colormaps[cmap_name]
    return [colors.to_hex(cm(i / (n - 1))) for i in range(n)]


def encode_int16(arr):
    q = np.rint(np.nan_to_num(arr, nan=0) * 10).astype("<i2")
    q[np.isnan(arr)] = NODATA_INT
    return q.tobytes()


# --------------------------------------------------------------------------
# 4. Construction du site
# --------------------------------------------------------------------------
def build_site(lat, lon, tn_pts, tx_pts, run_utc, win_info, geojson, out_dir, cmap="turbo"):
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

    tn, tx = apply_model(tn, tx, X, Y, {"run_utc": run_utc})

    inside = geometry_mask([geom_l93], out_shape=(h, w), transform=transform, invert=True)
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
    }
    for name, arr in (("tn", tn), ("tx", tx)):
        merc, bounds = to_mercator(arr, transform)
        vmin, vmax = color_scale(merc)
        colorize(merc, vmin, vmax, cmap).save(out / f"{name}.png", optimize=True)
        (out / f"{name}.bin").write_bytes(encode_int16(merc))
        meta[name] = {"vmin": vmin, "vmax": vmax,
                      "min": float(np.nanmin(merc)), "max": float(np.nanmax(merc))}
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


def fetch_run(ref, wait_minutes):
    import archive_icon_ch1 as icon
    deadline = time.time() + wait_minutes * 60
    while not icon.find_item(ref, "T_2M", icon.MAX_HORIZON_H):
        if time.time() > deadline:
            raise RuntimeError(f"Run {ref:%Y-%m-%d %H} UTC toujours indisponible après {wait_minutes} min.")
        print("Run pas encore complet, nouvelle tentative dans 3 min...")
        time.sleep(180)
    icon.archive_run(ref, ["T_2M"])
    return icon.OUT_DIR / f"{ref:%Y%m%dT%H}Z_T_2M.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", default=str(HERE / "data" / "lozere.geojson"))
    ap.add_argument("--out", default="site")
    ap.add_argument("--npz", help="Run déjà archivé (format de archive_icon_ch1.py)")
    ap.add_argument("--ref", help="Run UTC, ex. 2026-10-24T12 (défaut : dernier 12Z)")
    ap.add_argument("--wait-minutes", type=int, default=45)
    ap.add_argument("--cmap", default="turbo")
    args = ap.parse_args()

    if args.npz:
        npz = Path(args.npz)
    else:
        ref = (dt.datetime.fromisoformat(args.ref).replace(tzinfo=dt.timezone.utc)
               if args.ref else default_ref_12z())
        npz = fetch_run(ref, args.wait_minutes)

    d = np.load(npz)
    run_utc = dt.datetime.fromisoformat(str(d["ref_utc"]).replace("Z", "+00:00"))
    if run_utc.hour != 12:
        sys.exit(f"Ce pipeline n'utilise que le run 12Z (reçu : {run_utc:%H} UTC).")
    tn_pts, tx_pts, info = tn_tx(d["values"], d["lead_hours"], run_utc)
    build_site(d["lat"], d["lon"], tn_pts, tx_pts, run_utc, info,
               args.geojson, args.out, args.cmap)


if __name__ == "__main__":
    main()
