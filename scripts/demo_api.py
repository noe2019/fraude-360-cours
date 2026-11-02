"""Demonstration du service de scoring temps reel.

Demarre l'API dans un processus separe, lui envoie des transactions reelles
tirees de la couche GOLD (des fraudes et des transactions legitimes), affiche
les decisions et mesure la latence.

    python scripts/demo_api.py            # 20 transactions
    python scripts/demo_api.py --n 100    # 100 transactions
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from fraude360.config import charger_config  # noqa: E402
from fraude360.modeling import donnees  # noqa: E402
from fraude360.utils.journal import lire_json, titre  # noqa: E402


def attendre_service(url: str, essais: int = 60) -> bool:
    for _ in range(essais):
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def principal() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--part-fraude", type=float, default=0.4)
    args = p.parse_args()

    cfg = charger_config()
    url = f"http://{cfg['service']['hote']}:{cfg['service']['port']}"
    resume = lire_json(cfg.chemin("artefacts") / "modele" / "resume_courant.json")
    variables = resume["variables_numeriques"] + resume["variables_categorielles"]

    titre("DEMONSTRATION DU SERVICE DE SCORING TEMPS REEL")
    print("Demarrage du service ...")
    proc = subprocess.Popen([sys.executable, "-m", "fraude360.serving.api"],
                            cwd=str(RACINE), env={**__import__("os").environ,
                                                  "PYTHONPATH": str(RACINE / "src")},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if not attendre_service(url):
            print("Le service n'a pas demarre. Verifier les dependances (fastapi, uvicorn).")
            return 1
        print(f"Service operationnel : {url}/docs\n")
        print(json.dumps(requests.get(f"{url}/modele").json(), indent=2, ensure_ascii=False))

        pdf = donnees.charger_gold(cfg)
        recent = pdf[pdf.jour_index >= pdf.jour_index.max() - 20]
        n_fraude = int(args.n * args.part_fraude)
        ech = pd.concat([
            recent[recent.est_fraude == 1].sample(min(n_fraude,
                                                      int((recent.est_fraude == 1).sum())),
                                                  random_state=7),
            recent[recent.est_fraude == 0].sample(args.n - n_fraude, random_state=7),
        ]).sample(frac=1, random_state=7)

        print(f"\n{'transaction':<16}{'montant':>10}{'reel':>10}{'score':>10}"
              f"{'decision':>12}  regles")
        print("-" * 88)
        latences, justes = [], 0
        for _, r in ech.iterrows():
            charge = {v: (None if pd.isna(r[v]) else
                          (bool(r[v]) if isinstance(r[v], (bool, np.bool_)) else
                           (float(r[v]) if isinstance(r[v], (int, float, np.number))
                            else str(r[v]))))
                      for v in variables}
            rep = requests.post(f"{url}/score", json={
                "transaction_id": r["transaction_id"], "variables": charge}, timeout=10)
            rep.raise_for_status()
            d = rep.json()
            latences.append(d["latence_ms"])
            reel = "FRAUDE" if r["est_fraude"] == 1 else "legitime"
            bon = ((r["est_fraude"] == 1 and d["decision"] != "APPROUVER")
                   or (r["est_fraude"] == 0 and d["decision"] == "APPROUVER"))
            justes += int(bon)
            print(f"{r['transaction_id']:<16}{r['montant']:>10.2f}{reel:>10}"
                  f"{d['score_modele']:>10.4f}{d['decision']:>12}  "
                  f"{','.join(d['regles_declenchees'])}")

        print("-" * 88)
        print(f"Decisions coherentes avec la verite terrain : {justes}/{len(ech)}")
        print(f"Latence : mediane {np.median(latences):.1f} ms | "
              f"95e centile {np.percentile(latences, 95):.1f} ms | "
              f"cible {cfg['service']['latence_cible_ms']} ms")
        print(f"\nJournal des decisions : "
              f"{cfg.chemin('artefacts') / 'service' / 'decisions.jsonl'}")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(principal())
