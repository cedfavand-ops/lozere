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
    """Retire espaces, guillemets, chevrons < > (restes d'un modèle de saisie) et un éventuel préfixe « Token »."""
    t = str(tok).strip().strip("\"'").strip()
    if t.lower().startswith("token "):
        t = t[6:]
    return re.sub(r"[\s<>\"']", "", t)


def had_junk(raw, cleaned):
    return str(raw).strip() != cleaned


def shape(tok):
    cls = []
    if re.search(r"[a-z]", tok): cls.append("minuscules")
    if re.search(r"[A-Z]", tok): cls.append("majuscules")
    if re.search(r"\d", tok): cls.append("chiffres")
    sym = sorted(set(re.findall(r"[^A-Za-z0-9]", tok)))
    return (f"longueur {len(tok)}, hexadécimal : {'oui' if re.fullmatch(r'[0-9a-fA-F]+', tok) else 'non'}, "
            f"contient {'+'.join(cls) or 'rien'}, symboles : {''.join(sym) or 'aucun'}, "
            f"points : {tok.count('.')}, commence par « eyJ » (jeton JWT) : {'oui' if tok.startswith('eyJ') else 'non'}")


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


def discover(st, token):
    """Cherche une API REST : affiche seulement les codes HTTP (404 = chemin inexistant, 401/403 = chemin existant)."""
    base = "https://api.datacake.co"
    dev = st["device"]
    paths = ["/", "/rest/", "/rest/v1/", "/v1/", "/api/", "/api/v1/", "/openapi.json", "/rest/openapi.json",
             "/api/schema/", f"/rest/v1/devices/{dev}/", f"/v1/devices/{dev}/", f"/api/v1/devices/{dev}/",
             f"/rest/devices/{dev}/", f"/devices/{dev}/"]
    print("--- exploration de api.datacake.co (aucun jeton affiché)")
    for path in paths:
        row = []
        for sch in ("sans jeton", "Token", "Bearer"):
            hdr = {} if sch == "sans jeton" else {"Authorization": f"{sch} {token}"}
            try:
                r = requests.get(base + path, headers=hdr, timeout=30)
                row.append(f"{sch}: {r.status_code}")
                if r.status_code == 200 and sch == "sans jeton":
                    print(f"    corps de {path} : {r.text[:200]!r}")
            except requests.RequestException as e:
                row.append(f"{sch}: erreur {type(e).__name__}")
        print(f"  GET {path.replace(dev, '<appareil>'):42s} " + " | ".join(row))


def rest_probe(st, token):
    """API REST v1 (https://api.datacake.co/v1/) : statuts et début des réponses, sans afficher le jeton."""
    base = "https://api.datacake.co/v1"
    dev, field = st["device"], st["field"]
    now = dt.datetime.now(dt.timezone.utc)
    t0 = (now - dt.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cands = [f"/devices/{dev}/", f"/devices/{dev}/measurements/", f"/devices/{dev}/history/",
             f"/devices/{dev}/fields/", f"/devices/{dev}/measurements/?field={field}&start={t0}"]
    print("--- API REST v1")
    for path in cands:
        try:
            r = requests.get(base + path, headers={"Authorization": f"Token {token}"}, timeout=60)
        except requests.RequestException as e:
            print(f"  GET {path.replace(dev, '<appareil>')[:60]:60s} erreur {type(e).__name__}")
            continue
        print(f"  GET {path.replace(dev, '<appareil>')[:60]:60s} HTTP {r.status_code}")
        if r.status_code == 200:
            print(f"      {r.text[:700]!r}")


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
        raw = None if single else next((tokens.get(k) for k in (s['device'], s['id'], s.get('name')) if tokens.get(k)), None)
        if t and raw is not None and had_junk(raw if isinstance(raw, str) else raw.get('token', ''), t):
            print("                   -> caractères superflus retirés avant l'envoi (chevrons, espaces, guillemets ou préfixe)")
        if t:
            usable.append((s, t))
    if not usable:
        sys.exit("Aucune station n'a de jeton.")
    st, token = usable[0]
    dev, field, ws = st["device"], st["field"], st.get("workspace") or ""
    print(f"Station sondée : {st['id']} ({st['name']}) ; espace de travail connu : {'oui' if ws else 'non'}")

    rest_probe(st, token)

    scheme = None
    for sch in ("Token", "Bearer", "Brut"):
        code, js = post(token, "query { __typename }", sch)
        code2, js2 = post(token, "query { user { id } }", sch)
        user_ok = bool((js2.get("data") or {}).get("user"))
        print(f"--- authentification « {sch} » : requête simple HTTP {code} ; utilisateur reconnu : {user_ok} (HTTP {code2})")
        if code == 200 and not user_ok:
            print("    (HTTP 200 sans utilisateur reconnu = accès ANONYME, pas une authentification réussie)")
        if code == 200 and user_ok and scheme is None:
            scheme = sch
    if scheme is None:
        print("Aucune façon d'envoyer le jeton n'authentifie un utilisateur sur l'API GraphQL.\n"
              "Ce jeton est peut-être celui d'une autre API (REST) : exploration ci-dessous.")
        discover(st, token)
        print("Copiez-moi aussi le code (secrets remplacés par XXX) qui lit vos stations dans votre autre projet.")
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
        "timerangestart/end, raw": f'history(fields: ["{field}"], timerangestart: "{t0}", timerangeend: "{t1}", resolution: "raw")',
        "timerangestart/end, 1h": f'history(fields: ["{field}"], timerangestart: "{t0}", timerangeend: "{t1}", resolution: "1h")',
        "timerangestart/end sans résolution": f'history(fields: ["{field}"], timerangestart: "{t0}", timerangeend: "{t1}")',
    }
    for name, h in variants.items():
        code, js = post(token, f'query {{ device(deviceId: "{dev}") {{ {h} }} }}', scheme)
        good = bool((js.get("data") or {}).get("device")) and not js.get("errors")
        show(f"historique, variante « {name} » : {'OK' if good else 'ÉCHEC'} (HTTP {code})", js, 1200)
        if good:
            break


if __name__ == "__main__":
    main()
