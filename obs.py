"""
Observations des stations.

  - Infoclimat : API opendata (clé dans la variable d'environnement INFOCLIMAT_TOKEN, jamais dans le code).
    https://www.infoclimat.fr/opendata/?version=2&method=get&format=json&stations[]=ID&start=AAAA-MM-JJ&end=AAAA-MM-JJ&token=...
    Réponse : {"status": "OK", "stations": [...], "hourly": {"ID": [ {"dh_utc": "...", "temperature": ..., ...} ]}}
    Les données sont au pas d'environ 10 min (stations amateurs StatIC) ou horaire (SYNOP).
  - Météo-France : données climatologiques de base HORAIRES (open data, Licence Ouverte), par département,
    via l'API de data.gouv.fr ; heures en UTC ; mise à jour quotidienne (donc avec un jour de retard environ).
  - Datacake : à brancher dans fetch_datacake() (voir plus bas).

Licences : certaines stations sont en CC BY-NC (usage non commercial) ; citer « infoclimat.fr » comme source.
"""
import csv
import datetime as dt
import gzip
import io
import re
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


# ----------------------------------------------------------------------------- Météo-France (open data)
DATAGOUV_API = "https://www.data.gouv.fr/api/1/datasets/"
MF_DATASETS = {
    "principales": "donnees-climatologiques-de-base-horaires",
    "complementaires": "donnees-climatologiques-de-base-horaires-stations-complementaires",
}
MF_LICENSE = "Licence Ouverte (Météo-France)"


def select_resources(resources, dept, y0, y1):
    """
    Fichiers départementaux (csv.gz) du jeu de données dont la période recoupe [y0, y1].
    Les noms ne sont pas figés : on repère le numéro de département et, s'ils figurent, les années du titre.
    """
    out = []
    for r in resources:
        title = r.get("title") or ""
        url = r.get("url") or ""
        fmt = (r.get("format") or "") + url
        if "csv" not in fmt.lower() or r.get("type", "main") != "main":
            continue
        if not re.search(rf"(?:^|[_\-/]){re.escape(dept)}(?:[_\-.]|$)", title) and \
           not re.search(rf"(?:departement_|[/_]H_|[/_]HOR_){re.escape(dept)}[_.]", url):
            continue
        years = [int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", title)]
        if years and (max(years) < y0 or min(years) > y1):
            continue
        out.append(r)
    return out


def parse_mf_csv(lines, d0=None, d1=None):
    """
    Lignes d'un CSV horaire Météo-France (séparateur « ; ») -> (stations, séries).
    Colonnes utilisées : NUM_POSTE, NOM_USUEL, LAT, LON, ALTI, AAAAMMJJHH (UTC), T, TN, TX, FF.
    Renvoie ({id: {name, lat, lon, elevation}}, {id: [observations]}).
    """
    reader = csv.DictReader(lines, delimiter=";")
    cols = reader.fieldnames or []
    missing = [c for c in ("NUM_POSTE", "AAAAMMJJHH") if c not in cols]
    if missing or not any(c in cols for c in ("T", "TN", "TX")):
        raise RuntimeError(f"CSV Météo-France inattendu : colonnes {cols[:40]}")

    def num(row, k):
        v = row.get(k)
        if v is None or v.strip() == "":
            return None
        try:
            return float(v.replace(",", "."))
        except ValueError:
            return None

    stations, series = {}, {}
    for row in reader:
        try:
            t = dt.datetime.strptime(row["AAAAMMJJHH"].strip(), "%Y%m%d%H").replace(tzinfo=UTC)
        except (ValueError, KeyError, AttributeError):
            continue
        if (d0 and t.date() < d0) or (d1 and t.date() > d1):
            continue
        sid = row["NUM_POSTE"].strip()
        temp, tmin, tmax = num(row, "T"), num(row, "TN"), num(row, "TX")
        if temp is None and tmin is None and tmax is None:
            continue
        if sid not in stations:
            stations[sid] = {"name": (row.get("NOM_USUEL") or sid).strip(), "lat": num(row, "LAT"),
                             "lon": num(row, "LON"), "elevation": num(row, "ALTI")}
        series.setdefault(sid, []).append({"t": t, "temp": temp, "tmin": tmin, "tmax": tmax,
                                           "wind_ms": num(row, "FF")})
    return stations, series


def fetch_meteofrance(d0, d1, dept="48", datasets=("principales", "complementaires")):
    """
    Observations horaires des stations Météo-France du département entre les dates d0 et d1 (objets date).
    Renvoie (stations, séries) ; les identifiants sont ceux de NUM_POSTE (ex. « 48095001 »).
    Hypothèse à confirmer au premier lancement : TN/TX = extrêmes de l'heure ; T = température à l'heure pleine.
    """
    stations, series = {}, {}
    for name in datasets:
        slug = MF_DATASETS[name]
        r = requests.get(DATAGOUV_API + slug + "/", timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"data.gouv.fr : HTTP {r.status_code} pour {slug} ({r.text[:200]!r})")
        files = select_resources(r.json().get("resources", []), dept, d0.year, d1.year)
        print(f"  Météo-France ({name}) : {len(files)} fichier(s) pour le département {dept}")
        for f in files:
            resp = requests.get(f["url"], timeout=300)
            resp.raise_for_status()
            text = gzip.decompress(resp.content).decode("utf-8", errors="replace")
            st, se = parse_mf_csv(io.StringIO(text), d0, d1)
            for sid, meta in st.items():
                meta["dataset"] = name
                stations.setdefault(sid, meta)
            for sid, rows in se.items():
                series[sid] = dedupe(series.get(sid, []) + rows)
    return stations, series


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
    key = "tmin" if kind == "min" else "tmax"
    # extrême propre à chaque relevé quand il existe (Météo-France : TN/TX horaires), sinon la température
    pts = sorted((p["t"], p[key] if p.get(key) is not None else p["temp"]) for p in series
                 if (p.get(key) is not None or p.get("temp") is not None) and t0 <= p["t"] <= t1)
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
