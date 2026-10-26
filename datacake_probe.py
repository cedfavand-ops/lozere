#!/usr/bin/env python3
"""
Sonde de l'API Datacake (GraphQL) : à lancer UNE fois pour fixer la requête d'historique.

Les jetons ne sont jamais dans le code : secret GitHub DATACAKE_TOKENS, un objet JSON
    {"<identifiant de l'appareil>": "<jeton>", ...}
Ce script n'affiche JAMAIS un jeton ; il affiche le schéma de l'API et un extrait de la réponse
d'une requête d'historique des 24 dernières heures, pour la première station.

Lancement : workflow GitHub « Sonder l'API Datacake ».
"""
import datetime as dt
import json
import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
URL = "https://api.datacake.co/graphql/"


def gql(token, query):
    r = requests.post(URL, json={"query": query}, headers={"Authorization": f"Token {token}",
                                                          "Content-Type": "application/json"}, timeout=120)
    try:
        js = r.json()
    except ValueError:
        js = {"errors": [f"réponse non JSON (HTTP {r.status_code}) : {r.text[:200]!r}"]}
    return r.status_code, js


def show(title, js, limit=1500):
    txt = json.dumps(js, ensure_ascii=False)
    print(f"--- {title}\n{txt[:limit]}{' ...' if len(txt) > limit else ''}")


def token_for(tokens, st):
    """Jeton d'une station : clé = identifiant de l'appareil (conseillé), ou identifiant de station (DC-...)."""
    for key in (st["device"], st["id"], st.get("name")):
        v = tokens.get(key)
        if isinstance(v, dict):
            v = v.get("token")
        if v:
            return v
    return None


def main():
    raw = os.environ.get("DATACAKE_TOKENS")
    if not raw:
        sys.exit("Secret DATACAKE_TOKENS absent (ou vide).")
    try:
        tokens = json.loads(raw)
        assert isinstance(tokens, dict)
    except (ValueError, AssertionError):
        sys.exit("Le secret DATACAKE_TOKENS n'est pas un objet JSON valide : "
                 '{"identifiant appareil": "jeton", ...} (guillemets droits, virgules entre les paires).')
    stations = json.loads((HERE / "data" / "stations_datacake.json").read_text(encoding="utf-8"))
    # Les clés du secret ne sont pas secrètes (seules les valeurs le sont) : on les affiche pour le diagnostic
    print(f"Clés trouvées dans DATACAKE_TOKENS ({len(tokens)}) : {sorted(tokens)}")
    print("Clés attendues (identifiant de l'appareil) :")
    for s_ in stations:
        print(f"  {s_['id']:14s} -> {s_['device']}   jeton trouvé : {'oui' if token_for(tokens, s_) else 'NON'}")
    ok = [s_ for s_ in stations if token_for(tokens, s_)]
    if not ok:
        sys.exit("Aucune station n'a de jeton : les clés du secret doivent être les identifiants d'appareil ci-dessus.")
    st = ok[0]
    token = token_for(tokens, st)
    print(f"Station sondée : {st['id']} ({st['name']})")
    dev, field = st["device"], st["field"]
    print(f"Station sondée : {st['id']} ({st['name']}) ; jetons disponibles pour {len(tokens)} appareil(s)")

    code, js = gql(token, "query { user { email } }")
    print(f"--- authentification : HTTP {code} ; utilisateur reconnu : {bool((js.get('data') or {}).get('user'))}")

    code, js = gql(token, f'query {{ device(deviceId: "{dev}") {{ id verboseName online '
                          f'currentMeasurements(allActiveFields: true) {{ field {{ fieldName }} value }} }} }}')
    show(f"appareil et mesures actuelles (HTTP {code})", js)

    code, js = gql(token, '{ __type(name: "DeviceType") { fields { name args { name type { name kind ofType { name kind } } } } } }')
    fields = (((js.get("data") or {}).get("__type") or {}).get("fields")) or []
    hist = [f for f in fields if "hist" in f["name"].lower() or "measure" in f["name"].lower()]
    show("champs d'historique du type DeviceType (introspection)", hist if hist else js, 3000)

    now = dt.datetime.now(dt.timezone.utc)
    t0 = (now - dt.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    t1 = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    variants = {
        "timerange 1d, raw": f'history(fields: ["{field}"], timerange: "1d", resolution: "raw")',
        "timerange 1d, 1h": f'history(fields: ["{field}"], timerange: "1d", resolution: "1h")',
        "début/fin, raw": f'history(fields: ["{field}"], timerangeStart: "{t0}", timerangeEnd: "{t1}", resolution: "raw")',
        "début/fin, 1h": f'history(fields: ["{field}"], timerangeStart: "{t0}", timerangeEnd: "{t1}", resolution: "1h")',
    }
    for name, h in variants.items():
        code, js = gql(token, f'query {{ device(deviceId: "{dev}") {{ {h} }} }}')
        ok = bool((js.get("data") or {}).get("device")) and not js.get("errors")
        show(f"historique, variante « {name} » : {'OK' if ok else 'ÉCHEC'} (HTTP {code})", js, 1200)
        if ok:
            break


if __name__ == "__main__":
    main()
