"""Tek komutluk demo: projenin butun basliklarini commit'li sonuclardan basar.

    python scripts/demo.py
    python scripts/demo.py --json          # makineye okunur ozet

VERI DOSYASI GEREKTIRMEZ. Temiz bir klonda, sanal ortam bile kurmadan (yalniz
standart kutuphane) saniyeler icinde kosar: butun sayilar depoya commit edilmis
`reports/results/*.json` dosyalarindan okunur.

Neden var: 4,6 milyon satir etiketlendi, 72 deney kosuldu ve sonuclar 130'dan
fazla JSON'a dagildi. "Bu proje ne buldu" sorusunun cevabi, o dosyalari elle
acmayi gerektirmemeli. Sunumda canli gosterilebilecek tek komut budur.

Hicbir sayi bu dosyada YAZILI DEGIL - hepsi okunur. Bir artefakt degisirse
ciktisi da degisir; `tests/test_demo.py` ikisinin ayrismadigini kilitliyor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "reports" / "results"

CATEGORIES = ["Toys_and_Games", "Grocery_and_Gourmet_Food", "Video_Games", "All_Beauty"]
CELLS = [
    "Toys_and_Games/SASRec",
    "Toys_and_Games/BPR",
    "Grocery_and_Gourmet_Food/SASRec",
    "Grocery_and_Gourmet_Food/BPR",
]


def _read(results: Path, name: str) -> dict:
    yol = results / name
    if not yol.exists():
        raise FileNotFoundError(
            f"{yol} yok. Bu betik commit'li sonuclari okur; depoyu eksiksiz "
            "klonladiginizdan emin olun."
        )
    return json.loads(yol.read_text(encoding="utf-8"))


def summary(results: Path = RESULTS) -> dict:
    """Butun basliklar tek sozlukte. Saf okuma - hicbir sey hesaplanmaz."""
    gate2 = _read(results, "gate2.json")
    stats = _read(results, "experiment_stats.json")
    market = _read(results, "marketing_metrics.json")
    prev = _read(results, "prevalence.json")
    val = _read(results, "validation_500.json")

    gate1 = {}
    for kat in CATEGORIES:
        g = _read(results, f"gate1_{kat}.json")
        gate1[kat] = {
            "verdict": g["verdict"],
            "n_passed": g["n_passed"],
            "n_decided": g["n_decided"],
            "seasonality_ratio": g["criteria"]["1_seasonality"]["ratio"],
            "seasonality_ci95": g["criteria"]["1_seasonality"]["ci95"],
        }

    kalibre = prev["calibration"]["by_category"]
    rq1 = {
        kat: {
            "narrow_calibrated_pct": kalibre[kat]["narrow"]["calibrated_pct"],
            "narrow_ci95_pct": kalibre[kat]["narrow"]["ci95_pct"],
            "narrow_raw_pct": kalibre[kat]["narrow"]["raw_pct"],
        }
        for kat in CATEGORIES
        if kat in kalibre
    }

    def karsitlik(hucre: str, anahtar: str) -> dict | None:
        c = stats["results"].get(hucre, {}).get("contrasts", {}).get(anahtar)
        if c is None:
            return None
        metrik = c["primary_metric"]
        m = c["metrics"][metrik]
        return {
            "metric": metrik,
            "diff": m["diff"],
            "ci95": m["ci"],
            "relative_diff": m["relative_diff"],
            # `interpretation` yalnizca ONCEDEN KAYITLI yorum kuralinin
            # uygulandigi karsitliklarda var (plasebo eksenleri); digerlerinde
            # yon bilgisi tasiniyor ama karar cumlesi yok.
            "direction": m.get("direction"),
            "interpretation": m.get("interpretation"),
            "n_users": c["n_users"],
        }

    rq2 = {h: karsitlik(h, "C1-C4") for h in CELLS}
    rq3 = {h: karsitlik(h, "C3-C1") for h in CELLS}

    rq4 = {}
    for hucre in CELLS:
        r = market["results"].get(hucre, {})
        m2 = r.get("M2_half_life") or {}
        m1 = (r.get("M1_waste_share") or {}).get("C0-C1") or {}
        rq4[hucre] = {
            "half_life_interactions": (m2.get("fit") or {}).get("half_life"),
            "half_life_ci95": m2.get("half_life_interactions_ci"),
            "half_life_weeks": m2.get("half_life_weeks"),
            "waste_share_diff": m1.get("diff"),
            "waste_share_ci95": m1.get("ci"),
        }

    return {
        "provenance": {
            ad: _read(results, ad)["meta"].get("code_version")
            for ad in ("gate2.json", "experiment_stats.json",
                       "marketing_metrics.json", "prevalence.json")
        },
        "detector": {
            "gate1": gate1,
            "human_validation": {
                "verdict": val["verdict"],
                "n_annotators": prev["human_validation"]["n_annotators"],
                "reliability_measured": prev["human_validation"]["reliability_measured"],
                "macro_f1": val["measurements"]["llm_vs_human_sample_ci"]["macro_f1"],
            },
        },
        "experiment": {
            "gate2": {h: gate2["results"][h]["verdict"] for h in CELLS},
            "n_runs": sum(len(stats["results"][h]["runs"]) for h in CELLS),
        },
        "rq1_prevalence": rq1,
        "rq2_removing_gifts": rq2,
        "rq3_flagging_instead": rq3,
        "rq4_marketing": rq4,
        "dose_response": stats["dose_response"],
    }


# ------------------------------------------------------------------- bicim
def _pct(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def _ci(pair: list | None, digits: int = 4) -> str:
    if not pair:
        return "-"
    return f"[{pair[0]:.{digits}f}, {pair[1]:.{digits}f}]"


def render(s: dict) -> str:
    out: list[str] = []
    ek = out.append

    ek("=" * 78)
    ek("  HEDIYE KONTAMINASYONU - SIC AI Capstone, Grup 2")
    ek("  Bu cikti reports/results/ altindaki commit'li JSON'lardan okundu.")
    ek("=" * 78)

    ek("")
    ek("KAPILAR")
    for kat, g in s["detector"]["gate1"].items():
        ek(f"  Kapi 1  {kat:<26} {g['verdict']:<6} "
           f"({g['n_passed']}/{g['n_decided']})  "
           f"Aralik-Ocak/yaz = {g['seasonality_ratio']:.2f}x "
           f"{_ci(g['seasonality_ci95'], 2)}")
    hv = s["detector"]["human_validation"]
    ek(f"  Hafta 4 insan dogrulamasi     {hv['verdict']:<6} "
       f"makro-F1 {hv['macro_f1']['value']:.4f} {_ci(hv['macro_f1']['ci95'], 4)} "
       f"- {hv['n_annotators']} etiketleyici, guvenilirlik olculmedi")
    for hucre, v in s["experiment"]["gate2"].items():
        ek(f"  Kapi 2  {hucre:<26} {v}")
    ek(f"  Toplam kosu: {s['experiment']['n_runs']}")

    ek("")
    ek("RQ1 - Hediye alimlari ne kadar yaygin? (insan kalibrasyonlu, dar tanim)")
    for kat, r in s["rq1_prevalence"].items():
        ek(f"  {kat:<26} {r['narrow_calibrated_pct']:>5.1f}% "
           f"[{r['narrow_ci95_pct'][0]:.1f}, {r['narrow_ci95_pct'][1]:.1f}]"
           f"   (ham {r['narrow_raw_pct']:.1f}%)")

    ek("")
    ek("RQ2 - Hediye satirlarini silmek, ayni sayida RASTGELE satiri silmekten iyi mi?")
    ek("      (C1 - C4; pozitif = hediyeyi silmek daha iyi)")
    for hucre, c in s["rq2_removing_gifts"].items():
        if c is None:
            ek(f"  {hucre:<34} -")
            continue
        ek(f"  {hucre:<34} {c['metric']:<10} {c['diff']:+.6f} "
           f"{_ci(c['ci95'])}  ({_pct(c['relative_diff'], 1)})  "
           f"{c['interpretation'] or c['direction'] or ''}")

    ek("")
    ek("RQ3 - Silmek mi, isaretlemek mi? (C3 - C1; pozitif = isaretlemek daha iyi)")
    for hucre, c in s["rq3_flagging_instead"].items():
        if c is None:
            ek(f"  {hucre:<34} -")
            continue
        ek(f"  {hucre:<34} {c['metric']:<10} {c['diff']:+.6f} "
           f"{_ci(c['ci95'])}  {c['interpretation'] or c['direction'] or ''}")

    ek("")
    ek("RQ4 - Liste ne kadar sure kirli kaliyor? (M2 yari omur, M1 israf payi)")
    for hucre, r in s["rq4_marketing"].items():
        hl = r["half_life_interactions"]
        if hl is None:
            ek(f"  {hucre:<34} yari omur olculemedi")
            continue
        hafta = r["half_life_weeks"]
        ek(f"  {hucre:<34} yari omur {hl:.2f} kendi alimi "
           f"{_ci(r['half_life_ci95'], 2)} (~{hafta:.1f} hafta) | "
           f"israf payi {r['waste_share_diff']:+.4f} {_ci(r['waste_share_ci95'])}")

    ek("")
    ek("DOZ-YANIT (Toys etkisi Grocery'ninkinden buyuk mu?)")
    for model, d in s["dose_response"].items():
        for anahtar, v in d.items():
            m = v["metrics"].get("recall@10") or next(iter(v["metrics"].values()))
            ek(f"  {model:<8} {anahtar:<8} {v['x']} - {v['y']}: "
               f"{m['diff']:+.6f} {_ci(m['ci'])}")

    ek("")
    ek("PROVENANS (bu sayilari ureten kod surumu)")
    for ad, surum in s["provenance"].items():
        ek(f"  {ad:<26} {surum}")
    ek("")
    ek("Ayrinti, guven araliklari ve SINIRLILIKLAR: docs/SONUCLAR.md")
    ek("Figurler: reports/figures/ (F19 yayginlik, F21 karsitliklar, F22 yari omur)")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="Sonuc JSON'larinin klasoru (varsayilan reports/results)")
    parser.add_argument("--json", action="store_true",
                        help="Metin yerine makineye okunur ozet")
    args = parser.parse_args(argv)

    s = summary(args.results)
    if args.json:
        print(json.dumps(s, ensure_ascii=False, indent=2))
    else:
        print(render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
