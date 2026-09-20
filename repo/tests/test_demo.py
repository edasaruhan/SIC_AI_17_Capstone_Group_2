"""`scripts/demo.py` testleri.

Demo'nun tek isi commit'li JSON'lari OKUMAK. En tehlikeli ariza, bir sayinin
betigin icine kopyalanmasi olurdu: artefakt guncellenir, demo eski sayiyi
basmaya devam eder ve sunumda yanlis rakam soylenir. Buradaki testler ciktinin
JSON'lardan ayrisamayacagini kilitliyor.

Ikinci sart: demo TEMIZ BIR KLONDA, veri dosyasi olmadan ve bagimlilik
kurulmadan kosmali - jurinin onunde `pip install` beklenmez.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "scripts" / "demo.py"
RESULTS = REPO / "reports" / "results"


def _load():
    spec = importlib.util.spec_from_file_location("demo_betigi", DEMO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = _load()


def _json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# ----------------------------------------------- sayilar JSON'dan geliyor
def test_the_headline_contrast_comes_straight_from_experiment_stats():
    """Birincil sonuc: Toys x SASRec, C1 - C4."""
    beklenen = (_json("experiment_stats.json")["results"]["Toys_and_Games/SASRec"]
                ["contrasts"]["C1-C4"]["metrics"]["recall@10"])

    c = demo.summary()["rq2_removing_gifts"]["Toys_and_Games/SASRec"]

    assert c["diff"] == beklenen["diff"]
    assert c["ci95"] == beklenen["ci"]
    assert c["interpretation"] == beklenen["interpretation"]


def test_every_gate_verdict_is_read_from_its_own_file():
    s = demo.summary()

    for kat, g in s["detector"]["gate1"].items():
        assert g["verdict"] == _json(f"gate1_{kat}.json")["verdict"]
    for hucre, v in s["experiment"]["gate2"].items():
        assert v == _json("gate2.json")["results"][hucre]["verdict"]


def test_the_human_validation_verdict_is_not_softened():
    """Hafta 4 kapisi INCOMPLETE ve demo bunu YAZMAK zorunda: tek
    etiketleyiciyle uyum olculmedi."""
    s = demo.summary()

    assert s["detector"]["human_validation"]["verdict"] == "INCOMPLETE"
    assert s["detector"]["human_validation"]["reliability_measured"] is False
    assert "INCOMPLETE" in demo.render(s)


def test_the_printed_text_carries_the_same_digits_as_the_json():
    """Bicimlendirme sirasinda sayi degismemeli."""
    beklenen = (_json("experiment_stats.json")["results"]["Toys_and_Games/SASRec"]
                ["contrasts"]["C1-C4"]["metrics"]["recall@10"]["diff"])

    metin = demo.render(demo.summary())

    assert f"{beklenen:+.6f}" in metin


def test_the_run_count_is_counted_not_typed():
    """72 sayisi betige YAZILMIYOR; kosu listelerinden sayiliyor."""
    stats = _json("experiment_stats.json")["results"]
    beklenen = sum(len(stats[h]["runs"]) for h in demo.CELLS)

    assert demo.summary()["experiment"]["n_runs"] == beklenen
    assert beklenen == 72


# ------------------------------------------------------ temiz klon sarti
def test_the_demo_imports_nothing_outside_the_standard_library():
    """Bagimlilik kurulmamis bir makinede de kosmali."""
    agac = ast.parse(DEMO.read_text(encoding="utf-8"))
    moduller = set()
    for d in ast.walk(agac):
        if isinstance(d, ast.Import):
            moduller |= {a.name.split(".")[0] for a in d.names}
        elif isinstance(d, ast.ImportFrom) and d.module:
            moduller.add(d.module.split(".")[0])

    assert moduller <= set(sys.stdlib_module_names), moduller


def test_the_demo_runs_as_a_plain_script():
    """`python scripts/demo.py` - paket kurulumu, config, veri yok."""
    p = subprocess.run([sys.executable, str(DEMO)], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", cwd=REPO)

    assert p.returncode == 0, p.stderr
    assert "RQ2" in p.stdout and "Kapi 2" in p.stdout


def test_json_mode_emits_the_same_summary():
    p = subprocess.run([sys.executable, str(DEMO), "--json"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", cwd=REPO)

    assert json.loads(p.stdout) == json.loads(json.dumps(demo.summary()))


# ------------------------------------------------------------ sessiz ariza
def test_a_missing_result_file_stops_the_demo_instead_of_printing_blanks(
    tmp_path: Path,
):
    """Eksik dosyayi atlayip bos bir tablo basmak, "olcum yok"u "sonuc yok"
    gibi gosterirdi."""
    with pytest.raises(FileNotFoundError, match="gate2.json"):
        demo.summary(tmp_path)
