"""
test_api.py — Teste completo de todos os endpoints
====================================================
Uso:
    python test_api.py
    python test_api.py --url http://meu-servidor:8000
"""

import sys
import json
import time
import argparse
import requests

# ============================================================
# Config
# ============================================================

parser = argparse.ArgumentParser(description="Testa a ML API")
parser.add_argument("--url", default="http://localhost:8000", help="URL base da API")
ARGS = parser.parse_args()

API_URL = ARGS.url.rstrip("/")
HEADERS = {"Content-Type": "application/json"}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94mℹ\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"
LINE = "─" * 60

passed = 0
failed = 0


def section(title: str):
    print(f"\n{BOLD}{LINE}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{LINE}{RESET}")


def ok(msg: str, data=None):
    global passed
    passed += 1
    print(f"  {PASS}  {msg}")
    if data:
        print(f"     {json.dumps(data, ensure_ascii=False, indent=6)[:300]}")


def fail(msg: str, err=None):
    global failed
    failed += 1
    print(f"  {FAIL}  {msg}")
    if err:
        print(f"     → {err}")


def get(path: str) -> dict:
    r = requests.get(f"{API_URL}{path}", headers=HEADERS, timeout=8)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict) -> dict:
    r = requests.post(f"{API_URL}{path}", headers=HEADERS, json=body, timeout=8)
    r.raise_for_status()
    return r.json()


# ============================================================
# Testes
# ============================================================

section("1 — Health & Info")

try:
    d = get("/health")
    assert d["status"] in ("operational", "degraded")
    ok(f"/health  →  status={d['status']}, models_loaded={d['models_loaded']}")
except Exception as e:
    fail("/health", e)

try:
    d = get("/")
    assert "name" in d
    ok(f"/   →  {d['name']}")
except Exception as e:
    fail("/", e)

try:
    d = get("/ml/status")
    assert "pipelines" in d
    modes = {k: v["mode"] for k, v in d["pipelines"].items()}
    ok(f"/ml/status  →  modos: {modes}")
except Exception as e:
    fail("/ml/status", e)


section("2 — Métricas")

for pipeline, keys in [
    ("classification", ["accuracy", "precision"]),
    ("regression",     ["rmse", "r2"]),
    ("clustering",     ["silhouette_score"]),
    ("recommendation", ["hit_rate"]),
]:
    try:
        d = get(f"/ml/{pipeline}/metrics")
        for k in keys:
            assert d.get(k) is not None, f"campo {k!r} ausente"
        vals = {k: round(d[k], 3) for k in keys}
        ok(f"/ml/{pipeline}/metrics  →  {vals}")
    except Exception as e:
        fail(f"/ml/{pipeline}/metrics", e)


section("3 — Análise detalhada")

try:
    d = get("/ml/classification/confusion-matrix")
    assert "matrix" in d
    ok(f"/ml/classification/confusion-matrix  →  acurácia={d.get('accuracy')}")
except Exception as e:
    fail("/ml/classification/confusion-matrix", e)

try:
    d = get("/ml/clustering/characteristics")
    n = len(d.get("clusters", []))
    ok(f"/ml/clustering/characteristics  →  {n} clusters")
except Exception as e:
    fail("/ml/clustering/characteristics", e)

try:
    d = get("/ml/recommendation/top-skills")
    n = len(d.get("skills", []))
    ok(f"/ml/recommendation/top-skills  →  {n} habilidades")
except Exception as e:
    fail("/ml/recommendation/top-skills", e)


section("4 — Predições")

try:
    t0 = time.time()
    d = post("/ml/classification/predict", {"text": "Python, Machine Learning, SQL"})
    ms = (time.time() - t0) * 1000
    assert "prediction" in d and "confidence" in d
    ok(f"/ml/classification/predict  →  prediction={d['prediction']}, conf={d['confidence']:.2f}  ({ms:.0f}ms)")
except Exception as e:
    fail("/ml/classification/predict", e)

try:
    t0 = time.time()
    d = post("/ml/regression/predict", {"text": "Vaga sênior de data science com Python"})
    ms = (time.time() - t0) * 1000
    assert "prediction" in d
    ok(f"/ml/regression/predict  →  {d['prediction']:.1f} skills  ({ms:.0f}ms)")
except Exception as e:
    fail("/ml/regression/predict", e)

try:
    t0 = time.time()
    d = post("/ml/recommendation/predict", {"job_id": "JOB001", "top_k": 3})
    ms = (time.time() - t0) * 1000
    n = len(d.get("recommendations", []))
    ok(f"/ml/recommendation/predict  →  {n} recomendações  ({ms:.0f}ms)")
except Exception as e:
    fail("/ml/recommendation/predict", e)


section("5 — Resultado")

total = passed + failed
print(f"\n  Total: {total}  |  {PASS} {passed} passou  |  {FAIL if failed else INFO} {failed} falhou\n")

if failed > 0:
    print("  Verifique se o servidor está rodando:")
    print(f"  uvicorn ml_server:app --reload --port 8000\n")
    sys.exit(1)
else:
    print("  Todos os endpoints OK! Dashboard pronto para usar.\n")
