"""Smoke del router Basalt: /v1/models, poi una chat per modello (tempo, token, tok/s, inizio della risposta).
API key da BASALT_API_KEY, altrimenti dal keyring (servizio basalt, utente api-key); mai stampata.
URL da BASALT_URL (default http://127.0.0.1:8080/v1).

Uso:  python smoke_api.py [MODEL ...]"""
import json
import os
import sys
import time
import urllib.request

KEY = os.environ.get("BASALT_API_KEY")
if not KEY:
    import keyring
    KEY = keyring.get_password("basalt", "api-key")
BASE = os.environ.get("BASALT_URL", "http://127.0.0.1:8080/v1")


def call(path, body=None, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


models = [m["id"] for m in call("/models")["data"]]
print("models:", models)
want = sys.argv[1:] or models
prompt = "Spiegami in 5 righe perché lo Stretto di Hormuz conta per il petrolio."
for mid in want:
    for run in (1, 2):   # the first call of a model includes its swap
        t0 = time.time()
        r = call("/chat/completions", {"model": mid, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300,
                                       "temperature": 0})
        dt = time.time() - t0
        u = r.get("usage", {})
        txt = (r["choices"][0]["message"].get("content") or "").strip().replace("\n", " ")
        print(f"[{mid}] run {run}: {dt:.1f} s, completion {u.get('completion_tokens')} tok "
              f"({u.get('completion_tokens', 0) / dt:.1f} tok/s wall) | {txt[:140]!r}")
