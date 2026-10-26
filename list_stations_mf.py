#!/usr/bin/env python3
"""
Liste les stations Météo-France du département de la Lozère qui ont des mesures de température récentes
(jeux de données horaires « stations principales » et « stations complémentaires », open data).

Sortie : data/stations_meteofrance.json  [{id, name, lat, lon, elevation, license, dataset}, ...]
Affiche aussi la date du dernier relevé de chaque station (pour connaître le retard des données).

Lancement : workflow GitHub « Lister les stations », ou  python list_stations_mf.py
Dépendances : requests
"""
import datetime as dt
import json
from pathlib import Path

import obs

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "stations_meteofrance.json"
DEPT = "48"
LOOKBACK_DAYS = 45
MIN_RECORDS = 100        # relevés horaires de température minimum sur la période


def main():
    today = dt.datetime.now(dt.timezone.utc).date()
    stations, series = obs.fetch_meteofrance(today - dt.timedelta(days=LOOKBACK_DAYS), today, DEPT)
    print(f"{len(stations)} stations Météo-France avec des relevés sur {LOOKBACK_DAYS} jours")
    kept = []
    for sid, meta in sorted(stations.items()):
        rows = [p for p in series.get(sid, []) if p.get("temp") is not None or p.get("tmin") is not None]
        if len(rows) < MIN_RECORDS or meta["lat"] is None or meta["lon"] is None:
            print(f"  ignorée : {sid} {meta['name']} ({len(rows)} relevés de température)")
            continue
        kept.append({"id": sid, "name": meta["name"], "lat": round(meta["lat"], 5), "lon": round(meta["lon"], 5),
                     "elevation": meta["elevation"], "license": obs.MF_LICENSE, "dataset": meta["dataset"]})
        print(f"  {sid}  {meta['name'][:30]:30s} {meta['lat']:.3f} {meta['lon']:.3f}  {meta['elevation']} m  "
              f"{len(rows)} relevés, dernier : {rows[-1]['t']:%d/%m %H} UTC  [{meta['dataset']}]")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(kept)} stations retenues -> {OUT}")


if __name__ == "__main__":
    main()
