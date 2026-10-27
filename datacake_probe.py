#!/usr/bin/env python3
"""
Sonde de l'API Datacake (GraphQL) : à lancer pour fixer la façon de LIRE l'historique.

Secrets GitHub (jamais affichés) :
  DATACAKE_TOKENS : objet JSON {"<identifiant appareil>": "<jeton de l'espace de travail>", ...}
  (repli : DATACAKE_TOKEN, un jeton unique)

Le script :
  1. nettoie le jeton (espaces, guillemets, préfixe « Token ») et n'affiche que sa FORME (longueur, hexadécimal) ;
  2. teste trois façons d'envoyer le jeton (Token, Bearer, brut) et affiche le code HTTP de chacune ;
  3. avec la première qui est acceptée, essaie : liste des appareils de l'espace de travail, mesures actuelles,
     schéma des requêtes d'historique, puis quatre formes de requête d'historique sur 24 h.
"""
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
URL = "https://api.datacake.co/graphql/"


def clean(tok):
    t = str(tok).strip().strip("\"'").strip()
    if t.lower().startswith("token "):
        t = t[6:].strip()
    return t


def shape(tok):
    return (f"longueur {len(tok)}, hexadécimal : {'oui' if re.fullmatch(r'[0-9a-fA-F]+', tok) else 'non'}, "
            f"espace ou retour à la ligne : {'oui' if re.search(r'\s', tok) else 'non'}")


def post(token, query, scheme="Token"):
    auth = {"Token": f"Token {token}", "Bearer": f"Bearer {token}", "Brut": token}[scheme]
    r = requests.post(URL, json={"query": query}, headers={"Authorization": auth, "Content-Type": "application/json"},
                      timeout=120)
    try:
        js = r.json()
    except ValueError:
        js = {"errors": [f"réponse non JSON (HTTP {r.status_code}) : {r.text[:200]!r}"]}
    return r.status_code, js


def show(title, js, limit=1500):
    txt = json.dumps(js, ensure_ascii=False)
    print(f"--- {title}\n{txt[:limit]}{' ...' if len(txt) > limit else ''}")


def token_for(st, single, tokens):
    if single:
        return clean(single)
    for key in (st["device"], st["id"], st.get("name")):
        v = tokens.get(key)
        if isinstance(v, dict):
            v = v.get("token")
        if v:
            return clean(v)
    return None


def main():
    single = (os.environ.get("DATACAKE_TOKEN") or "").strip() or None
    tokens = {}
    if os.environ.get("DATACAKE_TOKENS"):
        try:
            tokens = json.loads(os.environ["DATACAKE_TOKENS"])
            assert isinstance(tokens, dict)
        except (ValueError, AssertionError):
            sys.exit("Le secret DATACAKE_TOKENS n'est pas un objet JSON valide.")
    if not single and not tokens:
        sys.exit("Aucun secret DATACAKE_TOKENS / DATACAKE_TOKEN.")
    stations = json.loads((HERE / "data" / "stations_datacake.json").read_text(encoding="utf-8"))
    print("Forme des jetons (la valeur n'est jamais affichée) :")
    usable = []
    for s in stations:
        t = token_for(s, single, tokens)
        print(f"  {s['id']:14s} : {'absent' if not t else shape(t)}")
        if t:
            usable.append((s, t))
    if not usable:
        sys.exit("Aucune station n'a de jeton.")
    st, token = usable[0]
    dev, field, ws = st["device"], st["field"], st.get("workspace") or ""
    print(f"Station sondée : {st['id']} ({st['name']}) ; espace de travail connu : {'oui' if ws else 'non'}")

    scheme = None
    for sch in ("Token", "Bearer", "Brut"):
        code, js = post(token, "query { __typename }", sch)
        code2, js2 = post(token, "query { user { id } }", sch)
        print(f"--- authentification « {sch} » : requête simple HTTP {code} ; "
              f"utilisateur reconnu : {bool((js2.get('data') or {}).get('user'))} (HTTP {code2})")
        if code == 200 and scheme is None:
            scheme = sch
    if scheme is None:
        print("Aucune façon d'envoyer le jeton n'est acceptée (HTTP 401 partout) : le jeton n'est pas reconnu par\n"
              "cette API. Vérifiez qu'il correspond bien à l'espace de travail de la station, et copiez-moi le code\n"
              "(avec les secrets remplacés par XXX) qui lit vos stations dans votre autre projet.")
        sys.exit(1)
    print(f"Façon retenue : {scheme}")

    if ws:
        code, js = post(token, f'query {{ allDevices(inWorkspace: "{ws}") {{ id serialNumber verboseName }} }}', scheme)
        show(f"appareils de l'espace de travail (HTTP {code})", js)
    code, js = post(token, f'query {{ device(deviceId: "{dev}") {{ id verboseName online '
                           f'currentMeasurements(allActiveFields: true) {{ field {{ fieldName }} value }} }} }}', scheme)
    show(f"appareil et mesures actuelles (HTTP {code})", js)
    code, js = post(token, '{ __type(name: "DeviceType") { fields { name args { name type { name kind ofType { name kind } } } } } }', scheme)
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
        code, js = post(token, f'query {{ device(deviceId: "{dev}") {{ {h} }} }}', scheme)
        good = bool((js.get("data") or {}).get("device")) and not js.get("errors")
        show(f"historique, variante « {name} » : {'OK' if good else 'ÉCHEC'} (HTTP {code})", js, 1200)
        if good:
            break


if __name__ == "__main__":
    main()
