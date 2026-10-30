"""
Modèle de correction 150 m (version 2 : correction FORTE en conditions radiatives).

  Tn(x) = Tn_ICON(x) + (1 - a(x)) * G * dz(x)  +  a(x) * [ b + gain_neg * min(motif(x), 0) + GAIN_POS * max(motif(x), 0) ]
  Tx(x) = Tx_ICON(x) + G_jour * dz(x)

  a(x)      indice de conditions radiatives (ciel clair, vent faible), de 0 à 1, calculé avec ICON ;
  motif(x)  structure fine de la thermographie ECOSTRESS (froid dans les cuvettes, doux sur les reliefs) ;
  gain_neg  AMPLIFICATION du froid des cuvettes : ICON surestime fortement les minima dans les cuvettes à froid,
            parfois de 5 à 10 °C, alors que le motif brut ne dépasse guère -3 °C. Valeur de départ 1,8 (un motif de
            -2,5 °C devient environ -4,5 °C), puis ajustée chaque jour sur les stations ORDINAIRES (Météo-France...) ;
            les cuvettes à froid instrumentées (Datacake) ne servent pas à cet ajustement : chacune apprend sa propre
            amplitude (ancrage), transmise aux pixels voisins de relief semblable ;
            le refroidissement est plafonné progressivement vers 9 °C (linéaire jusqu'à 5,5 °C) ;
  b         biais général d'ICON en conditions radiatives, ajusté de la même façon (0 au départ) ;
  dz(x)     écart entre le MNT à 150 m et le relief vu par ICON ; G = -6,5 °C/km (correction d'altitude classique,
            appliquée surtout quand les conditions ne sont pas radiatives).

Les stations Datacake (cuvettes à froid) sont toutes dans la queue froide du motif (de -2,1 à -2,9 °C en avril,
jusqu'à -5,8 °C en juin) : le motif désigne bien ces lieux ; c'est son amplitude qu'il fallait multiplier.

RÈGLE A PRIORI de a (ajustable avec calibrate.py) :
  a = f_vent * f_nuages sur les heures de nuit
  f_vent   = 1 si vent à 10 m <= 3 m/s ; 0 si >= 6 m/s (dans une cuvette l'air reste calme bien au-delà du vent d'ICON)
  f_nuages = 1 si nébulosité <= 30 % ; 0 si >= 70 %
"""
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

TZ = ZoneInfo("Europe/Paris")
G_NUIT = -6.5e-3      # °C par mètre
G_JOUR = -6.5e-3
MAX_DZ = 600.0        # écart d'altitude maximal pris en compte (m)

WIND_FULL, WIND_ZERO = 3.0, 6.0         # m/s
CLOUD_FULL, CLOUD_ZERO = 0.30, 0.70     # fraction de ciel couvert

GAIN_NEG_PRIOR = 1.8                    # amplification du motif froid hors stations, valeur de départ (ajustée chaque jour)
COLD_KNEE, COLD_MAX = 5.5, 9.0           # refroidissement linéaire jusqu'à 5,5 °C, puis plafonné progressivement vers 9 °C
GAIN_POS = 0.7                          # motif doux (reliefs) : peu amplifié
BIAS_PRIOR = 0.0
GAIN_BOUNDS = (1.0, 5.0)
BIAS_BOUNDS = (-4.0, 2.0)
STATE_NAME = "correction_state.json"

# Ancrage sur les stations : chaque station apprend sa propre amplitude A (°C par unité d'indice radiatif) et la
# transmet aux pixels voisins de relief semblable (même motif, même altitude).
STATION_RADIUS = 1500.0                 # m, portée du noyau gaussien
STATION_MOTIF_SCALE = 1.0               # °C : écart de motif au-delà duquel l'influence décroît
STATION_DZ_SCALE = 60.0                 # m : écart d'altitude au-delà duquel l'influence décroît
STATION_CAP = 9.0                       # °C : valeur maximale du terme d'ancrage
STATION_A_BOUNDS = (-10.0, 3.0)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def night_leads(lead_hours, run_utc, lat, lon):
    """Échéances (booléens) de la nuit qui suit le soir J : de 18 h locale J à 9 h locale J+1, soleil sous l'horizon."""
    from prepare_motif import solar_elevation
    day = run_utc.astimezone(TZ).date()
    start = dt.datetime.combine(day, dt.time(18), TZ)
    end = start + dt.timedelta(hours=15)
    out = []
    for h in lead_hours:
        v = run_utc + dt.timedelta(hours=int(h))
        out.append(bool(start <= v <= end and solar_elevation(lat, lon, v) < -0.5))
    return np.array(out)


def alpha_mesh(wind, cloud, night, n_pts):
    """
    wind  : (n_échéances, n_mailles) en m/s, ou None
    cloud : (n_échéances, n_mailles) en fraction ou en %, ou None
    night : (n_échéances,) booléens
    Renvoie (alpha, vent_moyen, nébulosité_moyenne) par maille. Un facteur indisponible vaut 0,5 (neutre) ;
    si les deux manquent, alpha = 0 (on retombe sur la correction d'altitude classique).
    """
    if wind is None and cloud is None:
        return np.zeros(n_pts), None, None
    if night.sum() < 4:
        raise ValueError(f"Seulement {int(night.sum())} échéances de nuit : run trop court ?")

    w = wind[night].mean(axis=0) if wind is not None else None
    c = cloud[night].mean(axis=0) if cloud is not None else None
    if c is not None and np.nanmax(c) > 1.5:
        c = c / 100.0
    f_w = 1.0 - smoothstep((w - WIND_FULL) / (WIND_ZERO - WIND_FULL)) if w is not None else np.full(n_pts, 0.5)
    f_c = 1.0 - smoothstep((c - CLOUD_FULL) / (CLOUD_ZERO - CLOUD_FULL)) if c is not None else np.full(n_pts, 0.5)
    return f_w * f_c, w, c


def hsurf_ok(z_icon, dem):
    """Vérifie que le champ 'h' du fichier de grille est bien l'altitude du relief ICON (comparée au MNT)."""
    m = np.isfinite(z_icon) & np.isfinite(dem)
    if m.sum() < 1000:
        return False, float("nan"), float("nan")
    r = float(np.corrcoef(z_icon[m], dem[m])[0, 1])
    mad = float(np.mean(np.abs(z_icon[m] - dem[m])))
    return (r > 0.9 and mad < 200.0), r, mad


def default_state():
    return {"gain_neg": GAIN_NEG_PRIOR, "bias": BIAS_PRIOR, "n_updates": 0, "days_done": [], "log": [], "stations": {}}


def load_state(path):
    """État de la correction (gain et biais ajustés sur les stations) ; valeurs de départ si le fichier n'existe pas."""
    st = default_state()
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        for k in st:
            if k in saved:
                st[k] = saved[k]
    except (OSError, ValueError):
        pass
    st["gain_neg"] = float(np.clip(st["gain_neg"], *GAIN_BOUNDS))
    st["bias"] = float(np.clip(st["bias"], *BIAS_BOUNDS))
    return st


def soft_cap(x, knee=COLD_KNEE, cap=COLD_MAX):
    """Refroidissement (positif) : inchangé jusqu'à 'knee', puis compressé vers 'cap' (pas d'emballement aux extrêmes)."""
    x = np.asarray(x, dtype=float)
    span = cap - knee
    return np.where(x <= knee, x, knee + span * np.tanh((x - knee) / span))


def regional_alpha(alpha_pts, q=0.75):
    """
    Indice radiatif RÉGIONAL : le quart le plus calme des mailles d'ICON. Dans une cuvette l'air reste calme même
    quand ICON annonce du vent sur le plateau voisin ; un indice purement local annulait la correction (nuit du 28/10 :
    indice nul à Saint-Paul alors que la station a relevé -6,3 °C).
    """
    return float(np.quantile(np.asarray(alpha_pts, dtype=float), q))


def station_field(X, Y, motif, dem, stations, state, step=150.0):
    """
    Terme d'ancrage S(x) en °C par unité d'indice radiatif : moyenne, pondérée par la distance et la ressemblance de
    relief, des amplitudes A apprises aux stations (state["stations"][id]["A"]).
    stations : liste de dicts {id, x, y, motif, dem} en Lambert-93, motif et altitude pris au pixel de la station.
    """
    num = np.zeros(X.shape)
    den = np.zeros(X.shape)
    x0, y0 = X[0, 0], Y[0, 0]
    n = int(3 * STATION_RADIUS / step) + 1
    for st in stations:
        rec = state.get("stations", {}).get(st["id"])
        if not rec:
            continue
        c = int(round((st["x"] - x0) / step))
        r = int(round((y0 - st["y"]) / step))
        r0, r1, c0, c1 = max(r - n, 0), min(r + n + 1, X.shape[0]), max(c - n, 0), min(c + n + 1, X.shape[1])
        if r0 >= r1 or c0 >= c1:
            continue
        sl = (slice(r0, r1), slice(c0, c1))
        d2 = (X[sl] - st["x"]) ** 2 + (Y[sl] - st["y"]) ** 2
        w = (np.exp(-d2 / STATION_RADIUS ** 2) * np.exp(-np.abs(motif[sl] - st["motif"]) / STATION_MOTIF_SCALE)
             * np.exp(-np.abs(dem[sl] - st["dem"]) / STATION_DZ_SCALE))
        num[sl] += w * rec["A"]
        den[sl] += w
    return np.clip(num / np.maximum(1.0, den), -STATION_CAP, STATION_CAP)


def radiative_term(alpha, motif, state):
    """
    Correction de Tn due aux conditions radiatives :
        a * ( biais - plafond(gain_neg * |motif froid|) + GAIN_POS * motif doux )
    Le plafond évite que les pixels au motif le plus extrême produisent des minima irréalistes.
    """
    a = np.clip(alpha, 0.0, 1.0)
    m = np.asarray(motif, dtype=float)
    cold = -soft_cap(state["gain_neg"] * np.maximum(-m, 0.0))
    return a * (state["bias"] + cold + GAIN_POS * np.maximum(m, 0.0))


def apply_correction(tn, tx, dem, z_icon, alpha, motif, state=None, station=None):
    state = state or default_state()
    dz = np.clip(dem - z_icon, -MAX_DZ, MAX_DZ)
    a = np.clip(alpha, 0.0, 1.0)
    tn2 = tn + (1.0 - a) * G_NUIT * dz + radiative_term(alpha, motif, state)
    if station is not None:
        tn2 = tn2 + a * station                       # ancrage sur les stations (appris chaque nuit)
    tx2 = tx + G_JOUR * dz
    return tn2.astype("float32"), tx2.astype("float32")
