#!/usr/bin/env python3
"""
Ajuste chaque jour la correction radiative sur les stations, sans attendre des semaines.

Chaque soir, avant de construire la carte, on regarde la NUIT qui vient de se terminer (de 18 h locale à 9 h locale) :
  - on compare le minimum observé à chaque station (Météo-France, Datacake ; Infoclimat si la clé est utilisable)
    à la prévision archivée la veille (history/forecast_AAAA-MM-JJ.json) ;
  - si la nuit était radiative (indice moyen >= 0,5) et que le motif froid était assez contrasté, on estime par
    moindres carrés le gain à appliquer au motif froid et le biais général d'ICON :
        Tn_obs - (prévision sans correction radiative) = a * (biais + gain * motif_froid + GAIN_POS * motif_doux)
  - le gain et le biais de la carte sont mis à jour par moyenne mobile exponentielle (poids 60 % puis 50 % pour les
    deux premières nuits, ensuite 35 %),
    bornés (gain entre 1 et 5, biais entre -4 et +2 °C), dans history/correction_state.json.

Lancement : étape du workflow de publication (avant build_site.py), ou  python update_state.py
"""
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

import model
import obs

HERE = Path(__file__).resolve().parent
HIST = HERE / "history"
STATE_PATH = HIST / model.STATE_NAME
UTC = dt.timezone.utc
ETA = 0.35               # poids de la nuit dans la moyenne mobile
MIN_STATIONS = 8         # stations minimales pour estimer une nuit
MIN_MEAN_ALPHA = 0.5     # nuit assez radiative
MIN_SPREAD = 0.25        # contraste minimal du motif froid pondéré (écart-type de a * motif froid)
NIGHT_HOURS = 15         # de 18 h locale à 9 h locale
RETRY_DAYS = 3           # une nuit dont les données arrivent en retard est retentée pendant 3 jours


def parse_utc(s):
    return dt.datetime.fromisoformat(s)


def estimate_correction(stations, tn_obs):
    """
    stations : stations de la prévision archivée (tn, motif, alpha, motif_term) ; tn_obs : {id: minimum observé}.
    Renvoie {"gain", "bias", "n", "rmse", "alpha_mean"} ou {"skip": raison}.
    """
    rows = []
    for st in stations:
        o = tn_obs.get(st["id"])
        if o is None or any(st.get(k) is None for k in ("tn", "motif_term", "alpha", "motif")):
            continue
        a, m = float(st["alpha"]), float(st["motif"])
        base = float(st["tn"]) - float(st["motif_term"])                  # prévision sans correction radiative
        r = o - base - a * model.GAIN_POS * max(m, 0.0)                    # ce que gain et biais doivent expliquer
        rows.append((a, a * min(m, 0.0), r))
    if len(rows) < MIN_STATIONS:
        return {"skip": f"{len(rows)} stations seulement", "retry": True}
    A = np.array(rows, dtype=float)
    if A[:, 0].mean() < MIN_MEAN_ALPHA:
        return {"skip": f"nuit peu radiative (indice moyen {A[:, 0].mean():.2f})"}
    if A[:, 1].std() < MIN_SPREAD:
        return {"skip": "motif froid trop peu contrasté sur les stations de cette nuit"}
    X, y = A[:, :2], A[:, 2]
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < 2:
        return {"skip": "estimation impossible (colonnes dépendantes)"}
    resid = y - X @ coef
    return {"bias": float(coef[0]), "gain": float(coef[1]), "n": len(rows),
            "rmse": float(np.sqrt(np.mean(resid ** 2))), "alpha_mean": float(A[:, 0].mean())}


def update_state(state, est, day):
    g = float(np.clip(est["gain"], -1.0, 8.0))
    b = float(np.clip(est["bias"], -6.0, 4.0))
    old_g, old_b = state["gain_neg"], state["bias"]
    eta = {0: 0.6, 1: 0.5}.get(state["n_updates"], ETA)       # les premières nuits pèsent davantage : convergence rapide
    state["gain_neg"] = float(np.clip((1 - eta) * old_g + eta * g, *model.GAIN_BOUNDS))
    state["bias"] = float(np.clip((1 - eta) * old_b + eta * b, *model.BIAS_BOUNDS))
    state["n_updates"] += 1
    state["log"] = (state["log"] + [{"day": day, "n": est["n"], "alpha_mean": round(est["alpha_mean"], 2),
                                     "gain_nuit": round(g, 2), "biais_nuit": round(b, 2), "rmse": round(est["rmse"], 2),
                                     "gain": round(state["gain_neg"], 2), "bias": round(state["bias"], 2)}])[-30:]
    return old_g, old_b


def save_state(state):
    HIST.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def observed_night_minima(rec, d_start, d_end, t0, t1):
    series = {}
    st = rec["stations"]
    ic = [s["id"] for s in st if s["source"] == "infoclimat"]
    if ic and os.environ.get("INFOCLIMAT_TOKEN"):
        try:
            series.update(obs.fetch_infoclimat(ic, d_start.isoformat(), d_end.isoformat(), os.environ["INFOCLIMAT_TOKEN"]))
        except obs.InfoclimatIPError:
            print("  Infoclimat : clé liée à une IP, stations ignorées.")
    if any(s["source"] == "meteofrance" for s in st):
        try:
            series.update(obs.fetch_meteofrance(d_start, d_end)[1])
        except Exception as e:
            print(f"  Météo-France indisponible ({type(e).__name__}: {e})")
    dc = [s for s in st if s["source"] == "datacake"]
    if dc:
        try:
            series.update(obs.fetch_datacake(dc, d_start.isoformat(), d_end.isoformat()))
        except Exception as e:
            print(f"  Datacake indisponible ({type(e).__name__}: {e})")
    out = {}
    for s in st:
        w = obs.window_extreme(series.get(s["id"], []), t0, t1, "min", min_cov=0.8)
        if w is not None:
            out[s["id"]] = w["value"]
    return out


def main():
    now = dt.datetime.now(UTC)
    state = model.load_state(STATE_PATH)
    files = sorted(HIST.glob("forecast_*.json"))
    print(f"Maintenant : {now:%d/%m %H:%M} UTC. État actuel : gain x{state['gain_neg']:.2f}, biais {state['bias']:+.2f} °C, "
          f"{state['n_updates']} ajustement(s).")
    if not files:
        print("Aucune prévision archivée dans history/ : correction inchangée.")
        return
    print(f"{len(files)} prévision(s) archivée(s) : {', '.join(f.stem.replace('forecast_', '') for f in files)}")
    changed = False
    treated = False
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        day = rec["day"]
        if day in state["days_done"]:
            continue
        t0 = parse_utc(rec["tn_window_utc"][0])
        t1 = t0 + dt.timedelta(hours=NIGHT_HOURS)
        ready = t1 + dt.timedelta(hours=1)
        if now < ready:
            print(f"  {day} : nuit pas encore terminée (exploitable après {ready:%d/%m %H:%M} UTC, "
                  f"fin de nuit à {t1:%H:%M} UTC + délai des données).")
            continue
        if now > t1 + dt.timedelta(days=RETRY_DAYS):
            print(f"  {day} : trop ancienne, ignorée.")
            state["days_done"] = (state["days_done"] + [day])[-60:]
            changed = True
            continue
        if not any(s.get("motif_term") is not None for s in rec["stations"]):
            print(f"  {day} : archive d'un ancien format (sans motif_term), ignorée.")
            state["days_done"] = (state["days_done"] + [day])[-60:]
            changed = True
            continue
        d0 = t0.date() - dt.timedelta(days=1)
        d1 = t1.date() + dt.timedelta(days=1)
        tn_obs = observed_night_minima(rec, d0, d1, t0, t1)
        est = estimate_correction(rec["stations"], tn_obs)
        treated = True
        print(f"Nuit du {day} : {len(tn_obs)} stations avec minimum observé sur {len(rec['stations'])}")
        dc = [s for s in rec["stations"] if s["id"].startswith("DC-") and s["id"] in tn_obs]
        for s in dc:
            print(f"  {s['id']:13s} observé {tn_obs[s['id']]:+5.1f} °C ; prévu {s['tn']:+5.1f} (ICON seul {s['tn_icon']:+5.1f})")
        if "skip" in est:
            print(f"  correction inchangée : {est['skip']}")
            if not est.get("retry") or now > t1 + dt.timedelta(days=1):
                state["days_done"] = (state["days_done"] + [day])[-60:]
                changed = True
            continue
        old_g, old_b = update_state(state, est, day)
        state["days_done"] = (state["days_done"] + [day])[-60:]
        changed = True
        print(f"  gain estimé pour la nuit : x{est['gain']:.2f} ; biais {est['bias']:+.2f} °C ({est['n']} stations, "
              f"écart résiduel {est['rmse']:.2f} °C, indice moyen {est['alpha_mean']:.2f})")
        print(f"  correction de la carte : gain x{old_g:.2f} -> x{state['gain_neg']:.2f} ; biais {old_b:+.2f} -> {state['bias']:+.2f} °C")
    if not treated:
        print("Aucune nuit exploitable pour le moment : correction inchangée.")
    if changed:
        save_state(state)


if __name__ == "__main__":
    main()
