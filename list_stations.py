#!/usr/bin/env python3
"""
Liste les stations Infoclimat (réseau StatIC + stations officielles) situées dans la Lozère.

Source : https://www.infoclimat.fr/opendata/stations_xhr.php?format=geojson  (métadonnées en temps réel)
Sortie : data/stations_infoclimat.json  [{id, name, lat, lon, elevation, license}, ...]

Les noms de champs du GeoJSON ne sont pas documentés : on essaie plusieurs clés usuelles et, si l'identifiant
reste introuvable, le script affiche les champs rencontrés pour qu'on adapte.

Lancement : workflow GitHub « Lister les stations Infoclimat », ou  python list_stations.py
Dépendances : requests shapely pyproj
"""
import json
import sys
from pathlib import Path

import requests

import build_site as bs

HERE = Path(__file__).resolve().parent
URL = "https://www.infoclimat.fr/opendata/stations_xhr.php?format=geojson"
OUT = HERE / "data" / "stations_infoclimat.json"
BUFFER_DEG = 0.01        # ~1 km de marge autour du contour (stations en bordure)


def first(props, keys, default=None):
    for k in keys:
        if k in props and props[k] not in (None, ""):
            return props[k]
    return default


def main():
    from shapely.geometry import Point

    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    gj = r.json()
    feats = gj["features"] if isinstance(gj, dict) and "features" in gj else gj
    print(f"{len(feats)} stations dans la liste Infoclimat")

    geom = bs.load_geometry(HERE / "data" / "lozere.geojson").buffer(BUFFER_DEG)
    kept, seen_keys = [], set()
    for f in feats:
        props = f.get("properties", {}) if isinstance(f, dict) else {}
        seen_keys |= set(props)
        try:
            lon, lat = f["geometry"]["coordinates"][:2]
        except (KeyError, TypeError, IndexError):
            lat = first(props, ("latitude", "lat"))
            lon = first(props, ("longitude", "lon", "lng"))
        if lat is None or lon is None:
            continue
        if not geom.contains(Point(float(lon), float(lat))):
            continue
        sid = first(props, ("id", "id_station", "station_id", "code", "sid"))
        if sid is None:
            sys.exit(f"Identifiant de station introuvable ; champs disponibles : {sorted(seen_keys)}")
        lic = first(props, ("license", "licence"), {})
        if isinstance(lic, dict):
            lic = lic.get("license") or lic.get("code")
        kept.append({
            "id": str(sid),
            "name": str(first(props, ("name", "nom", "label", "title"), sid)),
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            "elevation": first(props, ("elevation", "altitude", "alt", "elev")),
            "license": lic,
        })

    kept.sort(key=lambda s: s["id"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(kept)} stations en Lozère -> {OUT}")
    for s in kept:
        print(f"  {s['id']:>7s}  {s['name'][:32]:32s}  {s['lat']:.3f} {s['lon']:.3f}  {s['elevation']} m  {s['license']}")
    if not kept:
        print(f"Aucune station retenue. Champs rencontrés : {sorted(seen_keys)}")


if __name__ == "__main__":
    main()
