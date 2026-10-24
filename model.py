"""
Modèle de correction 150 m (version 1, règle a priori).

  Tn(x) = Tn_ICON(x) + (1 - a(x)) * G * dz(x) + a(x) * motif(x)
  Tx(x) = Tx_ICON(x) + G_jour * dz(x)

  dz(x)    = altitude du MNT à 150 m - altitude du relief tel que le voit ICON
  G        = gradient vertical standard (-6,5 °C/km) : la correction d'altitude "normale"
  motif(x) = structure fine de la thermographie ECOSTRESS du 18/04/2026 (voir prepare_motif.py)
  a(x)     = indice d'inversion entre 0 (aucune) et 1 (nuit radiative idéale), calculé à partir d'ICON.

Pourquoi ce mélange : sur la nuit de la thermographie, la structure fine ne suit PAS le gradient
d'altitude normal (pente mesurée : +2 °C/km environ, R² 0,02). Quand la nuit est radiative (a proche de 1),
on applique donc le motif observé ; quand le vent ou les nuages brassent l'air (a proche de 0),
on revient à une correction d'altitude classique.

RÈGLE A PRIORI pour a (non calibrée sur des observations : à affiner) :
  a = f_vent * f_nuages, calculés sur les heures de nuit qui précèdent le minimum
  f_vent   = 1 si vent à 10 m <= 2 m/s ; 0 si >= 5 m/s ; progressif entre les deux
  f_nuages = 1 si nébulosité <= 20 % ; 0 si >= 80 % ; progressif entre les deux
"""
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

TZ = ZoneInfo("Europe/Paris")
G_NUIT = -6.5e-3      # °C par mètre
G_JOUR = -6.5e-3
MAX_DZ = 600.0        # écart d'altitude maximal pris en compte (m)

WIND_FULL, WIND_ZERO = 2.0, 5.0         # m/s
CLOUD_FULL, CLOUD_ZERO = 0.20, 0.80     # fraction de ciel couvert


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


def apply_correction(tn, tx, dem, z_icon, alpha, motif):
    dz = np.clip(dem - z_icon, -MAX_DZ, MAX_DZ)
    a = np.clip(alpha, 0.0, 1.0)
    tn2 = tn + (1.0 - a) * G_NUIT * dz + a * motif
    tx2 = tx + G_JOUR * dz
    return tn2.astype("float32"), tx2.astype("float32")
