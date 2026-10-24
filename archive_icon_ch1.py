#!/usr/bin/env python3
"""
Archivage ICON-CH1-EPS (run déterministe) recadré sur la Lozère.

Pourquoi : MeteoSwiss ne garde que les dernières 24 h en open data, et la
grille est icosaédrique native (pas de lat/lon dans les GRIB de prévision).
Ce script :
  1. télécharge les fichiers statiques de la grille (lat/lon des mailles),
  2. vérifie que la Lozère est bien couverte (--check-domain),
  3. télécharge T_2M pour les échéances 0..33 h d'un run,
  4. ne garde que les mailles de la Lozère (+ marge) dans un .npz compressé,
  5. supprime le GRIB brut.

Dépendances : pip install requests numpy eccodes
(le paquet pip "eccodes" nécessite la bibliothèque ecCodes installée sur le système)

Usage :
  python archive_icon_ch1.py --check-domain
  python archive_icon_ch1.py                      # dernier run complet disponible
  python archive_icon_ch1.py --ref 2026-10-24T12  # run précis (UTC)
  python archive_icon_ch1.py --vars T_2M U_10M V_10M CLCT

Cron conseillé : toutes les 3 h (les données disparaissent après 24 h).
"""
import argparse
import datetime as dt
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
import eccodes

STAC = "https://data.geo.admin.ch/api/stac/v1"
COLLECTION = "ch.meteoschweiz.ogd-forecasting-icon-ch1"
HORIZONTAL_ASSET_ID = "horizontal_constants_icon-ch1-eps.grib2"

# Lozère (approx. 2,99-4,02°E ; 44,11-44,98°N) + marge d'environ 0,15°
BBOX = dict(lon_min=2.85, lon_max=4.20, lat_min=43.95, lat_max=45.10)
MAX_HORIZON_H = 33

OUT_DIR = Path("archive_icon_ch1")
GRID_CACHE = OUT_DIR / "grille_lozere.npz"


# ----------------------------------------------------------------------------
# Accès API
# ----------------------------------------------------------------------------
def horizon_variants(h: int):
    """Durée ISO 8601 ; deux écritures possibles, on essaie les deux."""
    d, hh = divmod(h, 24)
    variants = [f"P{d}DT{hh:02d}H00M00S"]
    if d:
        variants.append(f"P0DT{h:02d}H00M00S")
    return variants


def find_item(ref: dt.datetime, variable: str, h: int, perturbed: bool = False):
    for hz in horizon_variants(h):
        body = {
            "collections": [COLLECTION],
            "forecast:reference_datetime": ref.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast:variable": variable,
            "forecast:perturbed": perturbed,
            "forecast:horizon": hz,
        }
        r = requests.post(f"{STAC}/search", json=body, timeout=60)
        r.raise_for_status()
        feats = r.json().get("features", [])
        if feats:
            return feats[0]
    return None


def asset_url(feature) -> str:
    return next(iter(feature["assets"].values()))["href"]


def download(url: str, dest: Path):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def find_latest_complete_run(variable: str = "T_2M") -> dt.datetime:
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    ref = now - dt.timedelta(hours=now.hour % 3)
    for _ in range(24 // 3 + 1):
        if find_item(ref, variable, MAX_HORIZON_H):
            return ref
        ref -= dt.timedelta(hours=3)
    raise RuntimeError("Aucun run complet trouvé sur les dernières 24 h.")


# ----------------------------------------------------------------------------
# Grille
# ----------------------------------------------------------------------------
def read_full_grid(grib_path: Path):
    """Lit lat/lon des mailles dans le fichier statique. Robuste à l'absence des définitions ICON d'ecCodes."""
    named, others = {}, []
    with open(grib_path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                try:
                    name = eccodes.codes_get(gid, "shortName")
                except Exception:
                    name = "unknown"
                vals = eccodes.codes_get_values(gid)
            finally:
                eccodes.codes_release(gid)
            if name in ("CLAT", "CLON"):
                named[name] = vals
            else:
                others.append((name, vals))
    if "CLAT" in named and "CLON" in named:
        lat, lon = named["CLAT"], named["CLON"]
    else:
        # shortName non reconnus : on identifie les deux champs par leurs valeurs
        # (sur ce domaine la latitude est toujours supérieure à la longitude).
        cands = [(n, v) for n, v in list(named.items()) + others if v.size > 1000]
        print(f"shortName CLAT/CLON non reconnus (messages : {[n for n, _ in cands]}) ; "
              "identification par les valeurs.")
        if len(cands) < 2:
            raise RuntimeError(f"Fichier de grille inattendu : {len(cands)} champ(s) exploitable(s).")
        cands = cands[:2]
        cands.sort(key=lambda nv: float(np.nanmean(nv[1])))
        lon, lat = cands[0][1], cands[1][1]
    # Les constantes ICON peuvent être en radians
    if np.nanmax(np.abs(lat)) <= np.pi / 2 + 1e-6:
        lat, lon = np.degrees(lat), np.degrees(lon)
    if not (30 < float(np.nanmean(lat)) < 60 and -10 < float(np.nanmean(lon)) < 30):
        raise RuntimeError(f"Coordonnées incohérentes (lat moyenne {np.nanmean(lat):.1f}, "
                           f"lon moyenne {np.nanmean(lon):.1f}).")
    return lat, lon


def get_grid(force: bool = False):
    if GRID_CACHE.exists() and not force:
        d = np.load(GRID_CACHE)
        return d["idx"], d["lat"], d["lon"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{STAC}/collections/{COLLECTION}/assets", timeout=60)
    r.raise_for_status()
    assets = r.json()["assets"]
    href = next(a["href"] for a in assets if a["id"] == HORIZONTAL_ASSET_ID)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "horizontal.grib2"
        download(href, p)
        lat, lon = read_full_grid(p)
    print(f"Grille complète : {lat.size} mailles ; "
          f"lat {lat.min():.2f}..{lat.max():.2f} ; lon {lon.min():.2f}..{lon.max():.2f}")
    mask = (
        (lon >= BBOX["lon_min"]) & (lon <= BBOX["lon_max"])
        & (lat >= BBOX["lat_min"]) & (lat <= BBOX["lat_max"])
    )
    idx = np.flatnonzero(mask)
    np.savez_compressed(GRID_CACHE, idx=idx, lat=lat[idx], lon=lon[idx],
                        dom=np.array([lat.min(), lat.max(), lon.min(), lon.max()]))
    return idx, lat[idx], lon[idx]


def check_domain():
    idx, lat, lon = get_grid(force=True)
    dom = np.load(GRID_CACHE)["dom"]
    print(f"Mailles dans l'emprise Lozère (+marge) : {idx.size}")
    if idx.size == 0:
        print("=> La Lozère n'est PAS dans le domaine ICON-CH1. Voir ICON-CH2 ou AROME.")
        return
    print(f"Emprise des mailles retenues : lat {lat.min():.2f}..{lat.max():.2f}, "
          f"lon {lon.min():.2f}..{lon.max():.2f}")
    margin_s = BBOX["lat_min"] - dom[0]
    print(f"Marge approx. au bord sud du domaine : {margin_s:.2f}° "
          f"(~{margin_s * 111:.0f} km). Le bord du domaine peut être peu fiable.")
    if idx.size < 5000:
        print("Attention : peu de mailles -> domaine probablement tronqué sur la Lozère.")


# ----------------------------------------------------------------------------
# Archivage d'un run
# ----------------------------------------------------------------------------
def read_first_values(grib_path: Path) -> np.ndarray:
    with open(grib_path, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        try:
            return eccodes.codes_get_values(gid)
        finally:
            eccodes.codes_release(gid)


def archive_run(ref: dt.datetime, variables):
    idx, lat, lon = get_grid()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for var in variables:
        out = OUT_DIR / f"{ref:%Y%m%dT%H}Z_{var}.npz"
        if out.exists():
            print(f"{out.name} déjà présent, ignoré.")
            continue
        fields, leads = [], []
        with tempfile.TemporaryDirectory() as tmp:
            for h in range(MAX_HORIZON_H + 1):
                feat = find_item(ref, var, h)
                if feat is None:
                    print(f"  {var} +{h:02d}h indisponible, arrêt de cette variable.")
                    break
                p = Path(tmp) / f"{var}_{h:02d}.grib2"
                download(asset_url(feat), p)
                fields.append(read_first_values(p)[idx].astype("float32"))
                leads.append(h)
                p.unlink()
        if not fields:
            continue
        np.savez_compressed(
            out, values=np.stack(fields), lead_hours=np.array(leads),
            lat=lat, lon=lon, ref_utc=ref.strftime("%Y-%m-%dT%H:%M:%SZ"), variable=var,
        )
        print(f"{out.name} écrit ({len(leads)} échéances, {idx.size} mailles).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-domain", action="store_true")
    ap.add_argument("--ref", help="Run UTC, ex. 2026-10-24T12")
    ap.add_argument("--vars", nargs="+", default=["T_2M"])
    args = ap.parse_args()

    if args.check_domain:
        check_domain()
        return
    if args.ref:
        ref = dt.datetime.fromisoformat(args.ref).replace(tzinfo=dt.timezone.utc)
    else:
        ref = find_latest_complete_run(args.vars[0])
    print(f"Run : {ref:%Y-%m-%d %H} UTC")
    archive_run(ref, args.vars)


if __name__ == "__main__":
    sys.exit(main())
