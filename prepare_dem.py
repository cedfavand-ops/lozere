#!/usr/bin/env python3
"""
Prépare le MNT de la Lozère, une fois pour toutes, sur la MÊME grille que la carte (150 m, Lambert-93).

Source, par ordre de priorité :
  1. un MNT fourni (par défaut data/mnt_25m_lozere.tif s'il existe, ou --src fichier.tif),
     en GeoTIFF quelconque (lon/lat ou projeté) ; la valeur 0 est traitée comme "donnée manquante" ;
  2. sinon Copernicus DEM GLO-30 (30 m, libre d'accès, bucket public AWS "copernicus-dem-30m").

Sortie : data/dem_150m_l93.tif  (altitude en mètres, moyenne des mailles sources, trous comblés)

Lancement : via le workflow GitHub "Préparer le MNT" (bouton Run workflow), ou
    python prepare_dem.py [--src data/mon_mnt.tif]
Dépendances : numpy rasterio pyproj shapely requests
"""
import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests

import build_site as bs   # réutilise load_geometry() et target_grid() : même grille que la carte

HERE = Path(__file__).resolve().parent
GEOJSON = HERE / "data" / "lozere.geojson"
LOCAL_DEM = HERE / "data" / "mnt_25m_lozere.tif"
OUT = HERE / "data" / "dem_150m_l93.tif"
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
MARGIN_DEG = 0.02


def tile_tag(lat0: int, lon0: int) -> str:
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}_00_{ew}{abs(lon0):03d}_00"


def tile_urls(lat0: int, lon0: int):
    tag = tile_tag(lat0, lon0)
    # La grille 30 m s'appelle COG_10 sur le bucket ; on essaie aussi COG_30 par prudence.
    for res in ("10", "30"):
        name = f"Copernicus_DSM_COG_{res}_{tag}_DEM"
        yield f"{BASE}/{name}/{name}.tif"


def download_tile(lat0: int, lon0: int, dest: Path) -> bool:
    for url in tile_urls(lat0, lon0):
        r = requests.get(url, stream=True, timeout=300)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            print(f"  OK  {url} ({dest.stat().st_size / 1e6:.1f} Mo)")
            return True
        print(f"  --  {url} -> HTTP {r.status_code}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="MNT GeoTIFF à utiliser au lieu de Copernicus")
    args = ap.parse_args()
    src_path = Path(args.src) if args.src else (LOCAL_DEM if LOCAL_DEM.exists() else None)

    import rasterio
    from pyproj import Transformer
    from rasterio.merge import merge
    from rasterio.warp import Resampling, reproject
    from shapely.ops import transform as shp_transform

    to_l93 = Transformer.from_crs(4326, 2154, always_xy=True)
    to_wgs = Transformer.from_crs(2154, 4326, always_xy=True)
    geom = bs.load_geometry(GEOJSON)
    geom_l93 = shp_transform(to_l93.transform, geom)
    transform, X, Y = bs.target_grid(geom_l93)
    h, w = X.shape

    # Emprise de la grille en lon/lat -> liste des tuiles de 1° x 1° nécessaires
    lons, lats = to_wgs.transform([X.min(), X.max(), X.min(), X.max()], [Y.min(), Y.min(), Y.max(), Y.max()])
    lon_min, lon_max = min(lons) - MARGIN_DEG, max(lons) + MARGIN_DEG
    lat_min, lat_max = min(lats) - MARGIN_DEG, max(lats) + MARGIN_DEG
    tiles = [(la, lo) for la in range(math.floor(lat_min), math.floor(lat_max) + 1)
             for lo in range(math.floor(lon_min), math.floor(lon_max) + 1)]
    print(f"Grille {w}x{h} mailles ; tuiles Copernicus nécessaires : {[tile_tag(*t) for t in tiles]}")

    if src_path is not None:
        print(f"MNT source : {src_path}")
        with rasterio.open(src_path) as src:
            arr = src.read(1).astype("float32")
            src_crs, mtr = src.crs, src.transform
            if src.nodata is not None and not np.isnan(src.nodata):
                arr[arr == src.nodata] = np.nan
            print(f"  {arr.shape[1]}x{arr.shape[0]} px, CRS {src_crs}, "
                  f"maille {abs(mtr.a):.6g} x {abs(mtr.e):.6g} (unités du CRS)")
        arr[arr == 0] = np.nan          # 0 = donnée manquante (aucun point de la Lozère n'est à 0 m)
        mosaic = arr[None]
    else:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for la, lo in tiles:
                dest = Path(tmp) / f"{tile_tag(la, lo)}.tif"
                if download_tile(la, lo, dest):
                    paths.append(dest)
                else:
                    print(f"  !! tuile {tile_tag(la, lo)} introuvable (ignorée)")
            if not paths:
                sys.exit("Aucune tuile Copernicus téléchargée : vérifier l'accès réseau et les noms de tuiles.")
            srcs = [rasterio.open(p) for p in paths]
            mosaic, mtr = merge(srcs)
            src_crs = srcs[0].crs
            for s_ in srcs:
                s_.close()

    dem = np.full((h, w), np.nan, dtype="float32")
    reproject(mosaic[0].astype("float32"), dem, src_transform=mtr, src_crs=src_crs,
              dst_transform=transform, dst_crs="EPSG:2154",
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)

    holes = ~np.isfinite(dem)
    if holes.any() and (~holes).any():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
        dem = dem[tuple(idx)]                    # comble chaque trou par la maille valide la plus proche
        print(f"  {int(holes.sum())} maille(s) sans donnée comblée(s) par le plus proche voisin")
    ok = np.isfinite(dem)
    print(f"MNT 150 m : min {np.nanmin(dem):.0f} m, max {np.nanmax(dem):.0f} m, "
          f"mailles valides {100 * ok.mean():.1f} %")
    # Contrôle de plausibilité : la Lozère va d'environ 200 m (gorges) à 1699 m (Mont Lozère)
    if not (100 < np.nanmin(dem) < 600 and 1400 < np.nanmax(dem) < 1900):
        print("ATTENTION : altitudes inattendues pour la Lozère, à vérifier.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs="EPSG:2154", transform=transform, nodata=np.nan,
                       compress="deflate", predictor=3) as dst:
        dst.write(dem, 1)
    print(f"Écrit : {OUT} ({OUT.stat().st_size / 1e6:.2f} Mo)")


if __name__ == "__main__":
    main()
