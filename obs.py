"""
Observations des stations.

  - Infoclimat : API opendata (clé dans la variable d'environnement INFOCLIMAT_TOKEN, jamais dans le code).
    https://www.infoclimat.fr/opendata/?version=2&method=get&format=json&stations[]=ID&start=AAAA-MM-JJ&end=AAAA-MM-JJ&token=...
    Réponse : {"status": "OK", "stations": [...], "hourly": {"ID": [ {"dh_utc": "...", "temperature": ..., ...} ]}}
    Les données sont au pas d'environ 10 min (stations amateurs StatIC) ou horaire (SYNOP).
  - Datacake : à brancher dans fetch_datacake() (voir plus bas).

Licences : certaines stations sont en CC BY-NC (usage non commercial) ; citer « infoclimat.fr » comme source.
"""
import datetime as dt
import time

import requests

INFOCLIMAT_URL = "https://www.infoclimat.fr/opendata/"
UTC = dt.timezone.utc


def parse_rows(rows):
    """Lignes de l'API -> liste de dicts {t (UTC), temp (°C), wind_ms, cloud_oct, rad_wm2}."""
    out = []
    for r in rows or []:
        try:
            t = dt.datetime.strptime(r["dh_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        except (KeyError, ValueError, TypeError):
            continue
        wind = r.get("vent_moyen")                      # km/h dans l'API
        out.append({
            "t": t,
            "temp": r.get("temperature"),
            "wind_ms": None if wind is None else float(wind) / 3.6,
            "cloud_oct": r.get("nebulosite"),
            "rad_wm2": r.get("radiations", r.get("uv")),
        })
    return out


def dedupe(series):
    """Supprime les relevés en double (même instant) et trie par date."""
    seen = {}
    for p in series:
        seen[p["t"]] = p
    return [seen[t] for t in sorted(seen)]


def fetch_infoclimat(ids, start, end, token, chunk=10, pause=1.0):
    """
    ids : identifiants de stations ; start, end : 'AAAA-MM-JJ' (jours UTC).
    LIMITE DE L'API : 7 jours consécutifs au maximum par requête (sauf pour ses propres stations).
    Renvoie {id: [observations]} ; lève une erreur explicite si l'API répond autrement que « OK ».
    """
    out = {}
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        params = [("version", "2"), ("method", "get"), ("format", "json"),
                  ("start", start), ("end", end), ("token", token)] + [("stations[]", s) for s in part]
        r = requests.get(INFOCLIMAT_URL, params=params, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Infoclimat : HTTP {r.status_code} pour {len(part)} stations du {start} au {end}. "
                               f"Réponse : {r.text[:300]!r}")
        js = r.json()
        if js.get("status") != "OK":
            raise RuntimeError(f"Infoclimat : statut {js.get('status')!r}, erreurs {js.get('errors')}")
        for sid, rows in (js.get("hourly") or {}).items():
            out.setdefault(sid, []).extend(parse_rows(rows))
        time.sleep(pause)
    return {sid: dedupe(v) for sid, v in out.items()}


def fetch_datacake(stations_cfg, start, end):
    """
    À BRANCHER : récupération des 4 stations Datacake (sous abri, normes OMM).
    Doit renvoyer {id_station: [ {t (UTC, aware), temp (°C), wind_ms, ...} ]}, comme fetch_infoclimat.
    Tant que ce n'est pas écrit, les stations Datacake sont ignorées (message dans le journal).
    """
    if stations_cfg:
        print("Datacake : connecteur non encore écrit, stations ignorées.")
    return {}


def window_extreme(series, t0, t1, kind, min_cov=0.8):
    """
    Minimum ('min') ou maximum ('max') de la température dans [t0, t1] (datetimes UTC).
    Renvoie None si la couverture est inférieure à min_cov (part des relevés attendus), sinon
    {"value", "coverage", "step_min"} (step_min = pas de mesure : un pas horaire sous-estime les extrêmes).
    """
    pts = sorted((p["t"], p["temp"]) for p in series if p.get("temp") is not None and t0 <= p["t"] <= t1)
    if len(pts) < 3:
        return None
    steps = sorted((b[0] - a[0]).total_seconds() / 60 for a, b in zip(pts, pts[1:]))
    step = min(max(steps[len(steps) // 2], 5.0), 60.0)
    expected = (t1 - t0).total_seconds() / 60 / step + 1
    cov = len(pts) / expected
    if cov < min_cov:
        return None
    vals = [v for _, v in pts]
    return {"value": min(vals) if kind == "min" else max(vals), "coverage": round(cov, 2), "step_min": step}


def mean_wind(series, t0, t1):
    """Vent moyen (m/s) sur [t0, t1], ou None s'il manque les données."""
    v = [p["wind_ms"] for p in series if p.get("wind_ms") is not None and t0 <= p["t"] <= t1]
    return sum(v) / len(v) if len(v) >= 6 else None
