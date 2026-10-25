#!/usr/bin/env python3
"""
Compare les prévisions archivées (history/forecast_*.json) aux observations des stations.

Pour chaque jour dont les fenêtres sont terminées et pas encore vérifié :
  - récupère les observations (Infoclimat via INFOCLIMAT_TOKEN ; Datacake à brancher dans obs.py),
  - calcule Tn et Tx observés sur les MÊMES fenêtres que la prévision (couverture >= 80 % exigée),
  - ajoute une ligne par station dans history/verification.csv :
        prévision corrigée (tn_fc, tx_fc), ICON seul (tn_icon, tx_icon), indice d'inversion, motif, vent nocturne observé.
Le journal affiche biais et erreur absolue moyenne, avec et sans correction.

Lancement : workflow GitHub « Vérifier les prévisions » (chaque soir), ou  python verify.py
"""
import csv
import datetime as dt
import json
import os
from pathlib import Path

import obs

HERE = Path(__file__).resolve().parent
HIST = HERE / "history"
CSV_PATH = HIST / "verification.csv"
UTC = dt.timezone.utc
FIELDS = ["day", "id", "name", "source", "z_station", "lat", "lon",
          "tn_obs", "tx_obs", "tn_fc", "tx_fc", "tn_icon", "tx_icon", "alpha", "motif", "dz",
          "wind_night_obs", "cov_tn", "cov_tx", "step_min"]


def parse_utc(s):
    return dt.datetime.fromisoformat(s)


def rows_for_day(rec, series):
    """Lignes de vérification d'un jour, à partir de la prévision archivée et des séries observées {id: [obs]}."""
    from prepare_motif import solar_elevation
    tn0, tn1 = parse_utc(rec["tn_window_utc"][0]), parse_utc(rec["tn_window_utc"][1])
    tx0 = parse_utc(rec["tx_window_utc"][0])
    tx1 = parse_utc(rec["tx_truncated_at_utc"] or rec["tx_window_utc"][1])
    rows = []
    for st in rec["stations"]:
        s = series.get(st["id"])
        if not s:
            continue
        tn = obs.window_extreme(s, tn0, tn1, "min")
        tx = obs.window_extreme(s, tx0, tx1, "max")
        if tn is None and tx is None:
            continue
        night = [p["wind_ms"] for p in s
                 if p.get("wind_ms") is not None and tn0 <= p["t"] <= tn0 + dt.timedelta(hours=15)
                 and solar_elevation(float(st["lat"]), float(st["lon"]), p["t"]) < -0.5]
        wind = round(sum(night) / len(night), 2) if len(night) >= 6 else None
        rows.append({
            "day": rec["day"], "id": st["id"], "name": st.get("name"), "source": st["source"],
            "z_station": st.get("z_station"), "lat": st.get("lat"), "lon": st.get("lon"),
            "tn_obs": None if tn is None else round(tn["value"], 1),
            "tx_obs": None if tx is None else round(tx["value"], 1),
            "tn_fc": st.get("tn"), "tx_fc": st.get("tx"),
            "tn_icon": st.get("tn_icon"), "tx_icon": st.get("tx_icon"),
            "alpha": st.get("alpha"), "motif": st.get("motif"), "dz": st.get("dz"),
            "wind_night_obs": wind,
            "cov_tn": None if tn is None else tn["coverage"], "cov_tx": None if tx is None else tx["coverage"],
            "step_min": (tn or tx)["step_min"],
        })
    return rows


def summarize(rows):
    """Biais et erreur absolue moyenne (prévision - observation), avec et sans correction."""
    for var in ("tn", "tx"):
        pairs = [(r[f"{var}_fc"], r[f"{var}_icon"], r[f"{var}_obs"]) for r in rows
                 if r[f"{var}_obs"] is not None and r[f"{var}_fc"] is not None and r[f"{var}_icon"] is not None]
        if not pairs:
            print(f"  {var.upper()} : aucune station exploitable")
            continue
        n = len(pairs)
        b_fc = sum(f - o for f, _, o in pairs) / n
        b_ic = sum(i - o for _, i, o in pairs) / n
        m_fc = sum(abs(f - o) for f, _, o in pairs) / n
        m_ic = sum(abs(i - o) for _, i, o in pairs) / n
        print(f"  {var.upper()} ({n} stations) : biais {b_fc:+.2f} °C (ICON seul {b_ic:+.2f}) ; "
              f"erreur absolue moyenne {m_fc:.2f} °C (ICON seul {m_ic:.2f})")


def done_days():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return {r["day"] for r in csv.DictReader(f)}


def append_rows(rows):
    HIST.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in FIELDS})


def main():
    token = os.environ.get("INFOCLIMAT_TOKEN")
    now = dt.datetime.now(UTC)
    done = done_days()
    files = sorted(HIST.glob("forecast_*.json"))
    if not files:
        print("Aucune prévision archivée dans history/ : rien à vérifier.")
        return
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec["day"] in done:
            continue
        tn1 = parse_utc(rec["tn_window_utc"][1])
        tx1 = parse_utc(rec["tx_truncated_at_utc"] or rec["tx_window_utc"][1])
        ready = max(tn1, tx1) + dt.timedelta(hours=1)
        if now < ready:
            print(f"{rec['day']} : fenêtres pas terminées (prêt après {ready:%d/%m %H:%M} UTC).")
            continue
        ic_ids = [s["id"] for s in rec["stations"] if s["source"] == "infoclimat"]
        dc = [s for s in rec["stations"] if s["source"] == "datacake"]
        start = (parse_utc(rec["tn_window_utc"][0]).date() - dt.timedelta(days=1)).isoformat()
        end = (tx1.date() + dt.timedelta(days=1)).isoformat()
        series = {}
        if ic_ids:
            if not token:
                print(f"{rec['day']} : INFOCLIMAT_TOKEN absent, stations Infoclimat ignorées.")
            else:
                series.update(obs.fetch_infoclimat(ic_ids, start, end, token))
        series.update(obs.fetch_datacake(dc, start, end))
        rows = rows_for_day(rec, series)
        print(f"{rec['day']} : {len(rows)} stations vérifiées sur {len(rec['stations'])}")
        if rows:
            summarize(rows)
            append_rows(rows)


if __name__ == "__main__":
    main()
