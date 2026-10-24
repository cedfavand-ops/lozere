#!/usr/bin/env python3
"""
Construit le MOTIF D'INVERSION à partir des thermographies ECOSTRESS (dalles GeoTIFF dans data/ecostress/).

Principe
  1. Chaque dalle (UTM 31N, 70 m, kelvins) est ramenée sur la grille de la carte (150 m, Lambert-93)
     par moyenne des pixels sources ; là où des dalles se recouvrent, on fait la moyenne.
  2. LST_150 = température de surface à 150 m (°C), prise telle quelle (choix du projet : pas de
     conversion surface -> abri).
  3. Le motif = LST_150 moins sa version lissée (filtre gaussien, échelle de coupure --sigma-m).
     On ne garde ainsi que la structure FINE, que le modèle ICON (grosses mailles) ne résout pas ;
     la structure à grande échelle reste celle d'ICON, pour ne pas la compter deux fois.
  4. À l'exécution du soir : T(x) = T_ICON(x) + alpha * motif(x), alpha dépendant de la situation météo.

Sorties
  data/motif_150m_l93.tif   bande 1 : LST 150 m (°C) ; bande 2 : motif (°C) ; bande 3 : 1 si donnée valide
  data/motif_apercu.png     aperçu à regarder avant de brancher le motif dans la prévision

Lancement : workflow GitHub "Préparer les données", ou  python prepare_motif.py [--sigma-m 2000]
Dépendances : numpy scipy rasterio pyproj shapely matplotlib
"""
import argparse
import datetime as dt
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np

import build_site as bs   # même grille 150 m / même contour que la carte

HERE = Path(__file__).resolve().parent
GEOJSON = HERE / "data" / "lozere.geojson"
ECO_DIR = HERE / "data" / "ecostress"
DEM_PATH = HERE / "data" / "dem_150m_l93.tif"
OUT_TIF = HERE / "data" / "motif_150m_l93.tif"
OUT_PNG = HERE / "data" / "motif_apercu.png"


def solar_elevation(lat, lon, t_utc):
    """Hauteur du soleil en degrés (algorithme NOAA simplifié, sans réfraction)."""
    t = t_utc.astimezone(dt.timezone.utc)
    n = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (n - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return 90 - math.degrees(math.acos(max(-1, min(1, cz))))


def acquisition_time(path: Path):
    m = re.search(r"_(\d{8}T\d{6})_", path.name)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=dt.timezone.utc)


def remove_seams(a, thr=2.0, min_votes=250, half_width_px=1.5, outer_px=5.0, outer_thr=1.2):
    """
    Détecte les liserés rectilignes (artefacts de scan ECOSTRESS, perpendiculaires ou parallèles à la fauchée)
    et met leurs pixels à NaN. On repère les pixels très différents de la médiane locale 7x7, puis on cherche
    des droites longues parmi eux (transformée de Hough). Le liseré réel n'est pas parfaitement droit : on masque
    entièrement une bande étroite autour de la droite (half_width_px) et, jusqu'à outer_px, seulement les pixels
    encore aberrants (> outer_thr). Les structures non rectilignes (lacs, vallées) sont laissées intactes.
    Renvoie (tableau corrigé, nombre de pixels masqués).
    """
    try:
        from skimage.transform import hough_line, hough_line_peaks
    except ImportError:
        print("  (scikit-image absent : détection des liserés ignorée)")
        return a, 0
    from scipy.ndimage import median_filter
    valid = np.isfinite(a)
    if valid.sum() < 1000:
        return a, 0
    filled = np.where(valid, a, np.nanmedian(a))
    dev = filled - median_filter(filled, size=7)
    sel = valid & (np.abs(dev) > thr)
    if sel.sum() < min_votes:
        return a, 0
    angles = np.deg2rad(np.linspace(-90, 90, 720, endpoint=False))
    hspace, theta, dist = hough_line(sel, theta=angles)
    acc, ang, dst = hough_line_peaks(hspace, theta, dist, num_peaks=3, min_distance=30,
                                     min_angle=20, threshold=min_votes)
    if len(acc) == 0:
        return a, 0
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    out = a.copy()
    for votes, an, di in zip(acc, ang, dst):
        dist_l = np.abs(xx * np.cos(an) + yy * np.sin(an) - di)
        core = dist_l <= half_width_px
        halo = (dist_l <= outer_px) & ~core & valid & (np.abs(dev) > outer_thr)
        print(f"  liseré détecté : angle {np.degrees(an):.1f}°, {int(votes)} votes, "
              f"{int(core.sum())} pixels masqués au coeur + {int(halo.sum())} aberrants alentour")
        out[core | halo] = np.nan
    return out, int(valid.sum() - np.isfinite(out).sum())


def smooth_masked(arr, valid, sigma_px):
    """Lissage gaussien qui ignore les mailles invalides (convolution normalisée)."""
    from scipy.ndimage import gaussian_filter
    num = gaussian_filter(np.where(valid, arr, 0.0), sigma_px, mode="nearest")
    den = gaussian_filter(valid.astype("float64"), sigma_px, mode="nearest")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-m", type=float, default=2000.0,
                    help="échelle de coupure du lissage (m) : en dessous, c'est le motif")
    args = ap.parse_args()

    import rasterio
    from pyproj import Transformer
    from rasterio.features import geometry_mask
    from rasterio.warp import Resampling, reproject
    from shapely.ops import transform as shp_transform

    files = sorted(ECO_DIR.glob("*.tif"))
    if not files:
        sys.exit(f"Aucun .tif dans {ECO_DIR}. Déposez-y les dalles ECOSTRESS.")

    to_l93 = Transformer.from_crs(4326, 2154, always_xy=True)
    to_wgs = Transformer.from_crs(2154, 4326, always_xy=True)
    geom = bs.load_geometry(GEOJSON)
    geom_l93 = shp_transform(to_l93.transform, geom)
    transform, X, Y = bs.target_grid(geom_l93)
    h, w = X.shape
    lon_c, lat_c = to_wgs.transform(float(X.mean()), float(Y.mean()))

    # --- 1. Dalles -> grille 150 m
    layers = []
    for f in files:
        t = acquisition_time(f)
        elev = solar_elevation(lat_c, lon_c, t) if t else float("nan")
        print(f"{f.name[:60]}  acquisition {t:%Y-%m-%d %H:%M:%S} UTC  soleil à {elev:+.1f}°")
        if t and elev > 3:
            print("  ATTENTION : soleil déjà levé de plus de 3° -> réchauffement des versants éclairés possible.")
        with rasterio.open(f) as src:
            a = src.read(1).astype("float32")
            a[~np.isfinite(a) | (a < 150) | (a > 350)] = np.nan
            a, n_seam = remove_seams(a)
            dst = np.full((h, w), np.nan, dtype="float32")
            reproject(a, dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs="EPSG:2154",
                      resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
        layers.append(dst)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lst_k = np.nanmean(np.stack(layers), axis=0)
    lst = (lst_k - 273.15).astype("float32")

    inside = geometry_mask([geom_l93], out_shape=(h, w), transform=transform, invert=True)
    valid = np.isfinite(lst)
    print(f"Grille {w}x{h} ; mailles valides dans la Lozère : {100 * valid[inside].mean():.1f} %")

    # --- 2. Motif = LST - LST lissée
    sigma_px = args.sigma_m / bs.GRID_M
    smooth = smooth_masked(lst, valid, sigma_px)
    motif = np.where(valid, lst - smooth, 0.0).astype("float32")      # 0 = pas de correction là où pas de donnée
    mi = motif[inside & valid]
    print(f"LST 150 m dans la Lozère : {np.nanmin(lst[inside]):.1f} à {np.nanmax(lst[inside]):.1f} °C "
          f"(médiane {np.nanmedian(lst[inside]):.1f})")
    print(f"Motif (échelle de coupure {args.sigma_m:.0f} m) : écart-type {mi.std():.2f} °C ; "
          f"5e-95e centiles {np.percentile(mi, 5):+.2f} / {np.percentile(mi, 95):+.2f} °C")

    # --- 3. Lien avec le relief (si le MNT est là)
    if DEM_PATH.exists():
        from scipy.ndimage import gaussian_filter
        with rasterio.open(DEM_PATH) as d:
            dem = d.read(1).astype("float32")
        if dem.shape == (h, w):
            tpi = dem - gaussian_filter(dem, 1000 / bs.GRID_M, mode="nearest")
            m = inside & valid
            print(f"Corrélation motif / TPI(1 km) : {np.corrcoef(mi, tpi[m])[0, 1]:+.2f} ; "
                  f"motif / altitude : {np.corrcoef(mi, dem[m])[0, 1]:+.2f}")
        else:
            print("MNT présent mais de taille différente de la grille : relancer prepare_dem.py.")
    else:
        print("(data/dem_150m_l93.tif absent : statistiques de relief ignorées)")

    # --- 4. Écriture
    OUT_TIF.parent.mkdir(parents=True, exist_ok=True)
    bands = [np.where(valid, lst, np.nan).astype("float32"), motif, valid.astype("float32")]
    names = ["LST_150m_C", "motif_C", "valide"]
    with rasterio.open(OUT_TIF, "w", driver="GTiff", height=h, width=w, count=3, dtype="float32",
                       crs="EPSG:2154", transform=transform, nodata=np.nan,
                       compress="deflate", predictor=3) as dst:
        for i, (b, n) in enumerate(zip(bands, names), start=1):
            dst.write(b, i)
            dst.set_band_description(i, n)
        dst.update_tags(sigma_m=str(args.sigma_m), tuiles=str(len(files)))
    print(f"Écrit : {OUT_TIF} ({OUT_TIF.stat().st_size / 1e6:.2f} Mo)")

    # --- 5. Aperçu
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ext = [X.min() - bs.GRID_M / 2, X.max() + bs.GRID_M / 2, Y.min() - bs.GRID_M / 2, Y.max() + bs.GRID_M / 2]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    pl = np.where(inside, lst, np.nan)
    im0 = ax[0].imshow(pl, cmap="turbo", extent=ext)
    ax[0].set_title("Température de surface ECOSTRESS à 150 m (°C)")
    plt.colorbar(im0, ax=ax[0], shrink=0.75)
    lim = float(max(abs(np.percentile(mi, 2)), abs(np.percentile(mi, 98))))
    pm = np.where(inside & valid, motif, np.nan)
    im1 = ax[1].imshow(pm, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
    ax[1].set_title(f"Motif : écart à la LST lissée à {args.sigma_m / 1000:g} km (°C)")
    plt.colorbar(im1, ax=ax[1], shrink=0.75)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=80)
    print(f"Écrit : {OUT_PNG}")


if __name__ == "__main__":
    main()
