#!/usr/bin/env python3
"""
Liste des communes de la Lozère de 300 habitants et plus, avec leur position, pour étiqueter la carte.

Source : API « Découpage administratif » de data.gouv.fr (geo.api.gouv.fr) : nom, population légale (INSEE) et centre
de chaque commune du département 48.  https://geo.api.gouv.fr/communes?codeDepartement=48&fields=nom,code,population,centre
Sortie : data/places.json  [{"name", "lat", "lon", "pop"}, ...] triée par population décroissante.

Lancement : étape du workflow de publication (téléchargée si le fichier n'existe pas), ou
    python prepare_places.py [--min-pop 300] [--force]
Dépendances : requests
"""
import argparse
import json
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "places.json"
URL = "https://geo.api.gouv.fr/communes"


def fetch_places(min_pop=300, dept="48"):
    r = requests.get(URL, params={"codeDepartement": dept, "fields": "nom,code,population,centre",
                                  "format": "json", "geometry": "centre"}, timeout=60)
    r.raise_for_status()
    places = []
    for c in r.json():
        pop = c.get("population")
        centre = c.get("centre") or {}
        coords = centre.get("coordinates")
        if pop is None or not coords or pop < min_pop:
            continue
        places.append({"name": c["nom"], "lat": round(float(coords[1]), 5), "lon": round(float(coords[0]), 5),
                       "pop": int(pop)})
    places.sort(key=lambda p: -p["pop"])
    return places


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pop", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="refaire même si data/places.json existe")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT.name} existe déjà : rien à faire.")
        return
    places = fetch_places(args.min_pop)
    if len(places) < 10:
        raise SystemExit(f"Seulement {len(places)} communes reçues : réponse inattendue de l'API, fichier non écrit.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(places, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(places)} communes de {args.min_pop} habitants et plus -> {OUT}")
    for p in places[:8]:
        print(f"  {p['name']:30s} {p['pop']:6d} hab.  {p['lat']:.3f} {p['lon']:.3f}")


if __name__ == "__main__":
    main()
