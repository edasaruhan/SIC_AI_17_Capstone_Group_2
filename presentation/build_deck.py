"""Capstone sunumunu ureten betik (python-pptx).

    pip install python-pptx
    python build_deck.py                  # -> SIC_AI_17_Group_2.pptx

NEDEN BETIK: slaytlardaki her sayi `repo/reports/results/*.json` dosyalarindan
OKUNUR. Elle yazilmis tek bir rakam yok; bir artefakt yeniden uretilirse deck de
onunla birlikte degisir. Figurler `repo/reports/figures/` altindaki commit'li
PNG'lerden OLDUGU GIBI gomulur - yeniden cizilmez, boylece slayttaki grafik ile
rapordaki grafik ayni dosyadir.

Slaytlar Ingilizce; konusmaci notlari Turkce (`slide.notes_slide`).
21 slayt: 16 ana + 5 ek (A1-A5), 16:9.

Ekip adlari YER TUTUCU olarak birakildi - sunum sahibi dolduracak.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "repo"
RESULTS = REPO / "reports" / "results"
FIGURES = REPO / "reports" / "figures"

TITLE = "This Was Not For Me"
SUBTITLE = ("Detecting gift purchases as a distinct class of noise "
            "in e-commerce recommender systems")
TEAM = "AI in Marketing Capstone, Group 2 — [Team Member 1] / [Team Member 2] / [Team Member 3]"
RUNNING = "This Was Not For Me · SIC AI 17 Capstone · Group 2"
DATE = "21 September 2026"
GITHUB = "github.com/edasaruhan/SIC_AI_17_Capstone_Group_2"

CATEGORIES = ["Toys_and_Games", "Grocery_and_Gourmet_Food", "Video_Games", "All_Beauty"]
CELLS = ["Toys_and_Games/SASRec", "Toys_and_Games/BPR",
         "Grocery_and_Gourmet_Food/SASRec", "Grocery_and_Gourmet_Food/BPR"]

# ---------------------------------------------------------------- bicim dili
FONT = "Calibri"
INK = RGBColor(0x1C, 0x1C, 0x1C)
INK2 = RGBColor(0x55, 0x55, 0x55)
MUTED = RGBColor(0x8E, 0x8E, 0x8E)
RULE = RGBColor(0xD9, 0xD9, 0xD9)
ARROW = RGBColor(0xC2, 0xC9, 0xD1)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ZEBRA = RGBColor(0xFA, 0xFA, 0xFA)
ACCENT = RGBColor(0x1F, 0x5C, 0x8B)       # tek vurgu rengi
SOFT = RGBColor(0xE8, 0xEF, 0xF6)         # vurgunun acik zemini
SOFT2 = RGBColor(0xF3, 0xF6, 0xFA)
CAUTION = RGBColor(0x8A, 0x52, 0x0A)      # AYRILMIS: yalnizca durust uyari icin
CAUTION_SOFT = RGBColor(0xFB, 0xF2, 0xE3)

W, H = 13.333, 7.5
M = 0.72                  # sol/sag kenar
CW = W - 2 * M            # icerik genisligi
BODY_TOP = 1.92
FOOT_Y = 6.97

NO_STYLE = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"   # "No Style, No Grid"


# ================================================================== sayilar
def _read(name: str) -> dict:
    yol = RESULTS / name
    if not yol.exists():
        raise FileNotFoundError(
            f"{yol} yok. Bu betik commit'li sonuclari okur; depoyu eksiksiz "
            "klonladiginizdan emin olun."
        )
    return json.loads(yol.read_text(encoding="utf-8"))


def load() -> dict:
    """Slaytlarda gecen HER sayi burada toplanir. Hesap yok, saf okuma."""
    gate2 = _read("gate2.json")
    stats = _read("experiment_stats.json")
    market = _read("marketing_metrics.json")
    prev = _read("prevalence.json")
    val = _read("validation_500.json")
    distill = _read("distill_report_base.json")

    funnel = {}
    for kat in CATEGORIES:
        f = _read(f"preprocess_funnel_{kat}.json")
        asama = [k for k in f if k not in {"meta", "timestamp"}]
        funnel[kat] = {"raw": f[asama[0]]["rows"], "kcore": f[asama[-1]]["rows"],
                       "stages": [(k, f[k]["rows"]) for k in asama]}

    annot = {kat: _read(f"llm_annotate_{kat}.json")["meta"]["n_rows"] for kat in CATEGORIES}
    infer = {kat: _read(f"inference_{kat}.json")["n_rows"]
             for kat in ("Toys_and_Games", "Grocery_and_Gourmet_Food")}

    gate1 = {}
    for kat in CATEGORIES:
        g = _read(f"gate1_{kat}.json")
        gate1[kat] = {
            "verdict": g["verdict"], "n_passed": g["n_passed"], "n_decided": g["n_decided"],
            "ratio": g["criteria"]["1_seasonality"]["ratio"],
            "ci": g["criteria"]["1_seasonality"]["ci95"],
            "ratio_threshold": g["criteria"]["1_seasonality"]["threshold"],
            "beyond_keyword": g["criteria"]["2_beyond_keyword"]["rate"],
            "beyond_threshold": g["criteria"]["2_beyond_keyword"]["threshold"],
            "trial_threshold": g["criteria"]["4_trial_smoke_test"]["threshold"],
        }

    def kars(hucre: str, anahtar: str) -> dict | None:
        c = stats["results"].get(hucre, {}).get("contrasts", {}).get(anahtar)
        if c is None:
            return None
        m = c["primary_metric"]
        v = c["metrics"][m]
        return {"metric": m, "diff": v["diff"], "ci": v["ci"],
                "rel": v["relative_diff"], "n_users": c["n_users"],
                "verdict": v.get("interpretation") or v.get("direction")}

    kalibre = prev["calibration"]["by_category"]

    m2, m1 = {}, {}
    for hucre in CELLS:
        r = market["results"].get(hucre, {})
        h = r.get("M2_half_life") or {}
        kova = (h.get("buckets") or [{}])[0]
        m2[hucre] = {
            "half_life": (h.get("fit") or {}).get("half_life"),
            "ci": h.get("half_life_interactions_ci"),
            "weeks": h.get("half_life_weeks"),
            "weeks_ci": h.get("half_life_weeks_ci"),
            "start_excess": kova.get("excess"),
            "n_users": h.get("n_users"),
        }
        w = (r.get("M1_waste_share") or {})
        m1[hucre] = {
            "c0_c1": w.get("C0-C1") or {},
            "c4_c1": w.get("C4-C1") or {},
            "c0_c3": w.get("C0-C3") or {},
        }

    return {
        "provenance": {ad: _read(ad)["meta"].get("code_version") for ad in
                       ("prevalence.json", "gate2.json", "experiment_stats.json",
                        "marketing_metrics.json")},
        "funnel": funnel,
        "n_annotated": sum(annot.values()),
        "annot_per_category": annot,
        "n_labelled": sum(infer.values()),
        "infer": infer,
        "gate1": gate1,
        "gate2": {h: gate2["results"][h]["verdict"] for h in CELLS},
        "gate2_criteria": list(gate2["results"][CELLS[0]]["criteria"].keys()),
        "gate2_runs": len(gate2["results"][CELLS[0]]["criteria"]["1_same_test_pairs"]["runs_checked"]),
        "gate2_placebo": {
            h: gate2["results"][h]["criteria"]["3_placebo_not_better_than_c0"]["contrasts"]
            for h in CELLS},
        "gate2_seed": {h: gate2["results"][h]["criteria"]["4_seed_stability"] for h in CELLS},
        "n_runs": sum(len(stats["results"][h]["runs"]) for h in CELLS),
        "seeds": stats["results"][CELLS[0]]["contrasts"]["C1-C4"]["metrics"]["recall@10"].get("seeds")
                 or [42, 1337, 2024],
        "prevalence": {kat: {"cal": kalibre[kat]["narrow"]["calibrated_pct"],
                             "ci": kalibre[kat]["narrow"]["ci95_pct"],
                             "raw": kalibre[kat]["narrow"]["raw_pct"]}
                       for kat in CATEGORIES if kat in kalibre},
        "c1_c4": {h: kars(h, "C1-C4") for h in CELLS},
        "c1_c0": {h: kars(h, "C1-C0") for h in CELLS},
        "c1b_c4b": {h: kars(h, "C1b-C4b") for h in CELLS},
        "c3_c1": {h: kars(h, "C3-C1") for h in CELLS},
        "c3_c0": {h: kars(h, "C3-C0") for h in CELLS},
        "dose": stats["dose_response"],
        "m2": m2,
        "m1": m1,
        # Insan dogrulamasi: nokta deger `llm_vs_human`, aralik
        # `llm_vs_human_sample_ci` blogundan. Iki blok 2026-09-21'e kadar dorduncu
        # hanede ayrisiyordu (0,5404 / 0,5405 - yuvarlama hatasi); artik ikisi de 0,5405.
        "human": {
            "verdict": val["verdict"],
            "macro_f1": val["measurements"]["llm_vs_human"]["macro_f1"],
            "macro_f1_ci": val["measurements"]["llm_vs_human_sample_ci"]["macro_f1"]["ci95"],
            "accuracy": val["measurements"]["llm_vs_human_sample_ci"]["accuracy"]["value"],
            "gift_precision": val["measurements"]["llm_vs_human_sample_ci"]
                                 ["per_class"]["gift_given"]["precision"]["value"],
            "gift_recall": val["measurements"]["llm_vs_human_sample_ci"]
                              ["per_class"]["gift_given"]["recall"]["value"],
            "gift_f1": val["measurements"]["llm_vs_human_sample_ci"]
                          ["per_class"]["gift_given"]["f1"]["value"],
            "weighted_macro_f1": val["measurements"]["llm_vs_human_main_weighted"]
                                    ["macro_f1"]["value"],
            "n_annotators": prev["human_validation"]["n_annotators"],
            "reliability": prev["human_validation"]["reliability_measured"],
            "planned": val["criteria"]["1_annotator_agreement"]["design"],
            "n_rows": val["measurements"]["llm_vs_human_sample_ci"]["n_rows"],
        },
        "distill": {
            "verdict": distill["verdict"],
            "model": distill["meta"]["model"],
            "n_train": distill["meta"]["n_train"],
            "n_holdout": distill["meta"]["n_holdout"],
            "c1_f1": distill["criteria"]["1_c1_axis_vs_teacher"]["f1"],
            "c1b_f1": distill["criteria"]["2_c1b_axis_vs_teacher"]["f1"],
            "threshold": distill["criteria"]["1_c1_axis_vs_teacher"]["threshold"],
            "student_f1": distill["criteria"]["3_c1_axis_vs_human_not_worse_than_teacher"]["student_f1"],
            "teacher_f1": distill["criteria"]["3_c1_axis_vs_human_not_worse_than_teacher"]["teacher_f1"],
            "max_drop": distill["criteria"]["3_c1_axis_vs_human_not_worse_than_teacher"]["max_drop"],
        },
        "teacher": {
            "model": _read("llm_annotate_Toys_and_Games.json")["meta"]["model"],
            "prompt": _read("llm_annotate_Toys_and_Games.json")["meta"]["prompt_version"],
            "backend": _read("llm_annotate_Toys_and_Games.json")["meta"]["backend"],
        },
    }


# ------------------------------------------------------------------- bicim
def pct(x: float | None, d: int = 1, sign: bool = False) -> str:
    if x is None:
        return "—"
    g = "+" if sign and x >= 0 else ""
    return f"{g}{100 * x:.{d}f}%"


def pts(x: float | None, d: int = 1) -> str:
    """Yuzde PUANI (pay farklari icin)."""
    return "—" if x is None else f"{100 * x:+.{d}f} pts"


def num(x: float | None, d: int = 4) -> str:
    return "—" if x is None else f"{x:+.{d}f}"


def ci(pair, d: int = 4, mult: float = 1.0, suffix: str = "") -> str:
    if not pair:
        return "—"
    a, b = pair[0] * mult, pair[1] * mult
    hi = "∞" if b > 1e5 else f"{b:.{d}f}"
    return f"[{a:.{d}f}, {hi}]{suffix}"


def thousands(n: int | None) -> str:
    return "—" if n is None else f"{n:,}"


def short(cell: str) -> str:
    kat, model = cell.split("/")
    return f"{kat.split('_')[0]} · {model}"


VERDICT_EN = {
    "alt_sinir": "lower bound",
    "saptanamadi": "not detectable",
    "negatif": "negative",
    "sifiri_iceriyor": "CI includes zero",
    "pozitif": "positive",
}


def verdict_en(v: str | None) -> str:
    return VERDICT_EN.get(v or "", v or "—")


# ------------------------------------------------------------- pptx yardim
def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def textbox(sl, l, t, w, h):
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def put(tf, text, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT,
        space_before=0, space_after=4, italic=False, line=None, font=FONT):
    ilk = len(tf.paragraphs) == 1 and not tf.paragraphs[0].runs
    p = tf.paragraphs[0] if ilk else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    if line:
        p.line_spacing = line
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = font
    return p


def shape(sl, kind, l, t, w, h, fill=None, line=None, radius=0.08):
    shp = sl.shapes.add_shape(kind, Inches(l), Inches(t), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            shp.adjustments[0] = radius
        except (IndexError, AttributeError):
            pass
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.14)
    tf.margin_top = tf.margin_bottom = Inches(0.08)
    return shp


def card(sl, l, t, w, h, fill=SOFT2, line=RULE):
    return shape(sl, MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h, fill=fill, line=line, radius=0.05)


def rule(sl, t, l=M, w=CW, color=RULE, thick=1.25):
    shape(sl, MSO_SHAPE.RECTANGLE, l, t, w, Pt(thick).inches, fill=color)


def head(sl, title, kicker=None, size=25, accent=ACCENT, sub=None):
    if kicker:
        put(textbox(sl, M, 0.42, CW, 0.28), kicker.upper(), size=10.5, bold=True,
            color=accent, space_after=0)
    put(textbox(sl, M, 0.70, CW, 1.0), title, size=size, bold=True, color=INK,
        space_after=0, line=0.95)
    rule(sl, 1.70)
    if sub:
        put(textbox(sl, M, 1.80, CW, 0.3), sub, size=11.5, color=INK2, space_after=0)


def foot(sl, n):
    put(textbox(sl, M, FOOT_Y, CW * 0.82, 0.28), RUNNING, size=9, color=MUTED, space_after=0)
    put(textbox(sl, W - M - 1.2, FOOT_Y, 1.2, 0.28), str(n), size=9, color=MUTED,
        align=PP_ALIGN.RIGHT, space_after=0)


def notes(sl, text):
    sl.notes_slide.notes_text_frame.text = text


def picture_fit(sl, name, l, t, w, h):
    yol = FIGURES / name
    if not yol.exists():
        card(sl, l, t, w, h, fill=ZEBRA)
        put(textbox(sl, l + 0.2, t + h / 2 - 0.2, w - 0.4, 0.4),
            f"[{name} bulunamadi — `deep_eda` / `experiment_stats` kosun]",
            size=11, color=CAUTION, align=PP_ALIGN.CENTER)
        return None
    pic = sl.shapes.add_picture(str(yol), Inches(l), Inches(t))
    olcek = min(Inches(w) / pic.width, Inches(h) / pic.height)
    pic.width = int(pic.width * olcek)
    pic.height = int(pic.height * olcek)
    pic.left = Inches(l) + int((Inches(w) - pic.width) / 2)
    pic.top = Inches(t) + int((Inches(h) - pic.height) / 2)
    return pic


def caption(sl, text, l, t, w):
    put(textbox(sl, l, t, w, 0.3), text, size=9.5, color=MUTED, space_after=0,
        align=PP_ALIGN.CENTER, italic=True)


def table(sl, l, t, w, rows, col_w, row_h=0.3, size=11, head_size=10,
          align=None, head_fill=SOFT, zebra=True, first_bold=False):
    nrow, ncol = len(rows), len(rows[0])
    gf = sl.shapes.add_table(nrow, ncol, Inches(l), Inches(t), Inches(w),
                             Inches(row_h * nrow))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)
    sid = tblPr.makeelement(qn("a:tableStyleId"), {})
    sid.text = NO_STYLE
    tblPr.append(sid)
    tbl.first_row = False
    tbl.horz_banding = False

    olcek = w / sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw * olcek)
    for i in range(nrow):
        tbl.rows[i].height = Inches(row_h)

    for i, satir in enumerate(rows):
        for j, deger in enumerate(satir):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = head_fill
            else:
                c.fill.fore_color.rgb = ZEBRA if (zebra and i % 2 == 0) else WHITE
            put(c.text_frame, str(deger),
                size=head_size if i == 0 else size,
                bold=(i == 0) or (first_bold and j == 0),
                color=ACCENT if i == 0 else INK,
                align=(align[j] if align else PP_ALIGN.LEFT),
                space_after=0)
    return tbl


def badge(sl, l, t, text, fill=SOFT, color=ACCENT, w=0.82, h=0.26, size=10):
    shp = shape(sl, MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h, fill=fill, radius=0.25)
    put(shp.text_frame, text, size=size, bold=True, color=color,
        align=PP_ALIGN.CENTER, space_after=0)
    return shp


def stat(sl, l, t, w, h, big, label, sub=None, fill=SOFT2, big_color=ACCENT,
         big_size=34, line=RULE):
    card(sl, l, t, w, h, fill=fill, line=line)
    tf = textbox(sl, l + 0.18, t + 0.16, w - 0.36, h - 0.32)
    put(tf, big, size=big_size, bold=True, color=big_color, space_after=2, line=0.9)
    put(tf, label, size=11.5, bold=True, color=INK, space_after=3, line=1.0)
    if sub:
        put(tf, sub, size=10, color=INK2, space_after=0, line=1.05)


def flow(sl, boxes, t, h=0.92, l=M, w=CW, gap=0.3, size=11, title_size=11.5):
    """Yatay akis semasi: [(baslik, altmetin), ...] + aralarda ok."""
    n = len(boxes)
    bw = (w - gap * (n - 1)) / n
    for i, (bas, alt) in enumerate(boxes):
        x = l + i * (bw + gap)
        shp = card(sl, x, t, bw, h, fill=SOFT2, line=RULE)
        tf = shp.text_frame
        put(tf, bas, size=title_size, bold=True, color=ACCENT, align=PP_ALIGN.CENTER,
            space_after=2, line=0.95)
        if alt:
            put(tf, alt, size=size, color=INK2, align=PP_ALIGN.CENTER, space_after=0,
                line=1.0)
        if i < n - 1:
            shape(sl, MSO_SHAPE.RIGHT_ARROW, x + bw + 0.045, t + h / 2 - 0.085,
                  gap - 0.09, 0.17, fill=ARROW)


# =============================================================== slaytlar
def s01_title(prs, N):
    sl = new_slide(prs)
    shape(sl, MSO_SHAPE.RECTANGLE, 0, 0, W, 0.16, fill=ACCENT)
    put(textbox(sl, M, 1.55, CW, 0.3), "CAPSTONE FINAL PRESENTATION", size=11.5,
        bold=True, color=ACCENT, space_after=0)
    put(textbox(sl, M, 1.95, CW, 1.3), TITLE, size=52, bold=True, color=INK,
        space_after=0, line=0.92)
    put(textbox(sl, M, 3.25, CW * 0.82, 0.8), SUBTITLE, size=19, color=INK2,
        space_after=0, line=1.1)
    rule(sl, 4.35, w=CW * 0.5)
    tf = textbox(sl, M, 4.6, CW * 0.72, 1.3)
    put(tf, TEAM, size=13.5, bold=True, color=INK, space_after=6)
    put(tf, f"{GITHUB}  ·  {DATE}", size=12, color=INK2, space_after=6)
    put(tf, f"Every number in this deck is read from committed JSON artefacts  ·  "
            f"code version {N['provenance']['experiment_stats.json']}",
        size=10.5, color=MUTED, space_after=0, italic=True)
    notes(sl, "Sunumun tek cumlelik iddiasi su: hediye alimlari tavsiye "
              "sistemlerinde ayri bir gurultu sinifi ve bunu olcebiliyoruz. "
              "Projenin butun sayilari depoya commit edilmis JSON dosyalarindan "
              "okunuyor, slaytlara elle yazilmadi. Yirmi bir slayt var: on altisi "
              "ana anlati, besi ek. Bir yerde 'bunu nereden biliyorsunuz' sorusu "
              "gelirse cevap her zaman bir dosya adi.")


def s02_problem(prs, N):
    sl = new_slide(prs)
    head(sl, "A five-star review can describe a taste the buyer does not have",
         kicker="The problem")
    flow(sl, [("Buyer's own history", "a stable, self-chosen taste"),
              ("+ one gift purchase", "reviewed honestly, 5 stars"),
              ("Model updates the profile", "\"this user likes that\""),
              ("Next 10 slots shift", "toward the gift's subcategory")],
         t=2.1, h=1.05)
    caption(sl, "Illustrative schematic. No review text is reproduced anywhere in this "
                "project's outputs.", M, 3.28, CW)

    tiles = [("The rating is honest.",
              "Nothing is wrong with the review itself — the buyer really did like the item."),
             ("The preference signal is not.",
              "The item was chosen for somebody else. It is evidence about a third party."),
             ("Nothing in the data marks it.",
              "Purchase-type is not a field. It exists only in free text, if at all.")]
    bw = (CW - 2 * 0.3) / 3
    for i, (bas, alt) in enumerate(tiles):
        x = M + i * (bw + 0.3)
        card(sl, x, 3.75, bw, 1.55)
        tf = textbox(sl, x + 0.18, 3.93, bw - 0.36, 1.2)
        put(tf, bas, size=14.5, bold=True, color=ACCENT, space_after=6, line=1.0)
        put(tf, alt, size=11.5, color=INK2, space_after=0, line=1.12)

    put(textbox(sl, M, 5.65, CW, 0.9),
        "Recommender research treats noise as accidental — mis-clicks, bots, position bias. "
        "A gift is none of those: it is a deliberate, correctly recorded purchase whose "
        "preference content belongs to someone who is not in the dataset.",
        size=13, color=INK, space_after=0, line=1.15)
    foot(sl, 2)
    notes(sl, "Buradaki tek mesaj: puan dogru, tercih sinyali yanlis. Kullanici "
              "hediyeyi gercekten begenmis olabilir, ama urunu kendisi icin "
              "secmemis. Klasik gurultu literaturu kazara olan seyleri konusuyor - "
              "yanlis tiklama, bot, konum yanliligi. Hediye bunlarin hicbiri degil: "
              "kasitli ve dogru kaydedilmis bir alim, ama tercihi veri kumesinde "
              "olmayan birine ait. Sagdaki ucuncu kutu isin zorlugu: veride "
              "'bu hediyedir' diye bir alan yok.")


def s03_cost(prs, N):
    sl = new_slide(prs)
    head(sl, "The wrong signal does not stay in the model — it becomes spend",
         kicker="Why marketing should care")
    t = N["m2"]["Toys_and_Games/SASRec"]
    m1 = N["m1"]["Toys_and_Games/BPR"]["c0_c1"]
    bw = (CW - 2 * 0.32) / 3
    stat(sl, M, 2.02, bw, 1.85,
         pts(t["start_excess"], 1),
         "of the top-10 list is handed to the gift's subcategory",
         f"Sequential model × Toys and Games, C0 vs C1, {thousands(t['n_users'])} test users")
    stat(sl, M + bw + 0.32, 2.02, bw, 1.85,
         f"{t['half_life']:.2f}",
         "self-purchases before that excess halves",
         f"95% CI {ci(t['ci'], 2)} · ≈{t['weeks']:.1f} weeks (approximate)")
    stat(sl, M + 2 * (bw + 0.32), 2.02, bw, 1.85,
         pts(m1["diff"], 1),
         "of slots go to gift-only subcategories",
         f"Classical model × Toys, waste share C0 − C1, 95% CI "
         f"{ci(m1['ci'], 2, 100)} pts")

    put(textbox(sl, M, 4.2, CW, 0.32), "WHERE THE BUDGET LEAKS", size=10.5,
        bold=True, color=ACCENT, space_after=0)
    rows = [["Channel", "What the contaminated model does", "Consequence"],
            ["On-site recommendations",
             "Reserves slots for a subcategory the customer never buys for themselves",
             "Lower slot productivity; the customer's real next purchase is pushed out"],
            ["Retargeting / email",
             "Rebuilds the audience around the gift's category",
             "Impressions billed against an interest that does not exist"],
            ["Lifecycle timing",
             "Reads a December gift peak as a taste shift",
             "Seasonal campaign aimed at the wrong segment"]]
    table(sl, M, 4.51, CW, rows, [2.4, 4.6, 4.5], row_h=0.46, size=11, head_size=10)
    foot(sl, 3)
    notes(sl, "Bu slayt akademik bulguyu para diline ceviriyor. Sirali modelde "
              "tek bir hediye, kullanicinin top-10 listesinin iki puanini hediyenin "
              "alt kategorisine ayiriyor; bu fazla, kullanici kendisi icin bir "
              "alisveris yaptiktan sonra yariya iniyor. Klasik modelde israf payi "
              "daha buyuk: her yuz slottan yaklasik besi yalnizca hediye olarak "
              "girilen alt kategorilere gidiyor. Alt tablo uc kanalin her birinde "
              "bunun nasil fatura oldugunu gosteriyor. Dikkat: 'israf' bir ust "
              "sinir varsayimi - o slotlarin hic donusmedigini olcmedik.")


def s04_questions(prs, N):
    sl = new_slide(prs)
    head(sl, "Four questions, fixed before the experiment was run",
         kicker="Research questions")
    rows = [["", "Question", "How it is answered"],
            ["RQ1", "How common are gift purchases, and when?",
             "Human-calibrated prevalence per category + month-of-year profile"],
            ["RQ2", "Does removing gift rows improve prediction of the user's own next purchase?",
             "C1 vs a placebo (C4) that deletes the same NUMBER of random rows"],
            ["RQ3", "Is it better to delete gifts or to flag them?",
             "C3 (shadow token) against both C1 (delete) and C0 (do nothing)"],
            ["RQ4", "How long does a recommendation list stay contaminated?",
             "M2 half-life in self-purchases, M1 wasted slot share"]]
    table(sl, M, 2.05, CW, rows, [0.75, 5.6, 5.55], row_h=0.5, size=11.5, head_size=10,
          first_bold=True)

    card(sl, M, 4.85, CW, 1.55, fill=SOFT, line=RULE)
    tf = textbox(sl, M + 0.22, 5.0, CW - 0.44, 1.3)
    put(tf, "The interpretation rule was written down BEFORE any result was seen",
        size=13.5, bold=True, color=ACCENT, space_after=6)
    put(tf, "A 95 % bootstrap interval that excludes zero in the expected direction → "
            "reported as a LOWER BOUND (alt_sınır), because label noise pulls a "
            "placebo contrast toward the placebo (assuming the detector's mistakes are no "
            "more informative than random rows — A4).   An interval containing zero → "
            "NOT DETECTABLE (saptanamadı) — never \"no effect\".   An interval "
            "excluding zero in the opposite direction → NEGATIVE (negatif), reported "
            "as such.",
        size=11.5, color=INK, space_after=4, line=1.15)
    put(tf, "Dated in repo/docs/DECISIONS.md. Thresholds were never edited after a "
            "result was seen; the 500 human-labelled rows were never used for tuning.",
        size=10.5, color=INK2, italic=True, space_after=0)
    foot(sl, 4)
    notes(sl, "Dort soru ve her birinin nasil cevaplandigi. En onemli kisim alttaki "
              "kutu: yorum kurali sonuclari gormeden yazildi ve tarihi DECISIONS.md'de "
              "duruyor. Uc olasi karar var - alt sinir, saptanamadi, negatif. "
              "'Saptanamadi' asla 'etki yok' demek degil; bunu sunum boyunca ayni "
              "dikkatle soyluyoruz. Etiket gurultusu plasebo karsitligini plasebonun "
              "lehine cektigi icin pozitif bulgularin hepsi alt sinir olarak "
              "raporlaniyor, yani gercek etki en az bu kadar.")


def s05_pipeline(prs, N):
    sl = new_slide(prs)
    head(sl, "One pipeline: a big model teaches a small one, then the small one labels everything",
         kicker="How it was built", size=23)
    f = N["funnel"]
    flow(sl, [("Amazon Reviews 2023",
               f"4 categories · {thousands(f['Toys_and_Games']['raw'] + f['Grocery_and_Gourmet_Food']['raw'] + f['Video_Games']['raw'] + f['All_Beauty']['raw'])} raw reviews"),
              (f"Teacher: {N['teacher']['model'].split('/')[-1]}",
               f"prompt {N['teacher']['prompt']} · {N['teacher']['backend']} on 2×T4 · "
               f"{thousands(N['n_annotated'])} labels"),
              ("Human check", f"{N['human']['n_rows']} rows, 5-class schema, "
                              f"{N['human']['n_annotators']} annotator"),
              (f"Student: {N['distill']['model'].split('/')[-1]}",
               f"fidelity gate {N['distill']['verdict']} · "
               f"{thousands(N['n_labelled'])} rows labelled"),
              ("RecBole experiment",
               f"6 conditions × 2 models × {len(N['seeds'])} seeds = {N['n_runs']} runs")],
         t=2.08, h=1.25, gap=0.26, size=10.5, title_size=11.5)

    put(textbox(sl, M, 3.6, CW, 0.32), "FROM RAW REVIEWS TO A TRAINABLE CORPUS",
        size=10.5, bold=True, color=ACCENT, space_after=0)
    rows = [["Category", "Raw reviews", "After 5-core", "Role in the study"],
            ["Toys and Games", thousands(f["Toys_and_Games"]["raw"]),
             thousands(f["Toys_and_Games"]["kcore"]), "High-gift treatment category"],
            ["Grocery and Gourmet Food", thousands(f["Grocery_and_Gourmet_Food"]["raw"]),
             thousands(f["Grocery_and_Gourmet_Food"]["kcore"]), "Low-gift control category"],
            ["Video Games", thousands(f["Video_Games"]["raw"]),
             thousands(f["Video_Games"]["kcore"]), "Detector validation only"],
            ["All Beauty", thousands(f["All_Beauty"]["raw"]),
             thousands(f["All_Beauty"]["kcore"]),
             "Detector only — 5-core is empty, so no experiment is possible"]]
    table(sl, M, 3.92, CW, rows, [3.1, 1.9, 1.9, 5.0], row_h=0.38, size=11, head_size=10,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT])
    put(textbox(sl, M, 5.95, CW, 0.75),
        "Filters, in order: verified purchase → at least 5 words of text → de-duplication "
        "→ 5-core on users and items. Only the two experiment categories were labelled at "
        "full scale; the distilled student exists because labelling 4.6 M rows with the "
        "teacher was not affordable.",
        size=11, color=INK2, space_after=0, line=1.15)
    foot(sl, 5)
    notes(sl, "Boru hattini bir bakista anlatan slayt. Buyuk model dort kategoriden "
              "orneklenmis kirk yedi bin review'u etiketliyor; bes yuz satiri insan "
              "bagimsiz etiketliyor; sonra damitilmis kucuk model dort buçuk milyon "
              "satirin tamamini etiketliyor. Neden damitma? Ogretmen modelle dort "
              "bucuk milyon satir etiketlemek Kaggle butcesinde mumkun degildi. "
              "Alt tabloda dikkat cekilecek satir All Beauty: bes-cekirdek "
              "filtresinden sonra hicbir satir kalmiyor, o yuzden o kategoride "
              "deney yapilamiyor - bunu gizlemiyoruz, tabloya yaziyoruz.")


def s06_detector(prs, N):
    sl = new_slide(prs)
    g = N["gate1"]["Toys_and_Games"]
    head(sl, "The detector had to earn its use — four criteria, all fixed in advance",
         kicker="Detection · Gate 1", size=24)
    picture_fit(sl, "F17_llm_vs_proxy_by_month_Toys_and_Games.png", M, 1.95, 6.5, 4.2)
    caption(sl, "F17 · Toys and Games: LLM label rate vs the lexical keyword proxy, by month",
            M, 6.2, 6.5)

    x = M + 6.8
    w = CW - 6.8
    put(textbox(sl, x, 1.95, w, 0.3), "GATE 1 — ALL FOUR CATEGORIES", size=10.5,
        bold=True, color=ACCENT, space_after=0)
    rows = [["Criterion", "Threshold", "Toys"],
            ["1 · December–January vs summer", f"≥ {g['ratio_threshold']}×",
             f"{g['ratio']:.2f}×"],
            ["2 · Found beyond the keyword proxy", f"≥ {pct(g['beyond_threshold'], 0)}",
             pct(g["beyond_keyword"], 1)],
            ["3 · Schema health (parse, class mix)", "no failure", "pass"],
            ["4 · 200-row trial agreement", f"≥ {g['trial_threshold']}", "0.835"]]
    table(sl, x, 2.28, w, rows, [3.0, 1.35, 1.0], row_h=0.38, size=10.5, head_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT])

    put(textbox(sl, x, 4.48, w, 0.3), "VERDICT PER CATEGORY", size=10.5, bold=True,
        color=ACCENT, space_after=0)
    y = 4.82
    for kat in CATEGORIES:
        v = N["gate1"][kat]
        put(textbox(sl, x, y, w * 0.62, 0.28),
            kat.replace("_and_", " & ").replace("_", " "), size=11, color=INK, space_after=0)
        badge(sl, x + w * 0.63, y - 0.03, f"{v['verdict']} {v['n_passed']}/{v['n_decided']}",
              w=1.0, h=0.26, size=9.5)
        y += 0.33
    put(textbox(sl, x, 6.22, w, 0.55),
        "Seasonality is the criterion a keyword list cannot fake: it is a property of "
        "the calendar, not of the wording.",
        size=10, color=INK2, italic=True, space_after=0, line=1.1)
    foot(sl, 6)
    notes(sl, "Dedektorun kullanilabilir olup olmadigina bakan Kapi 1. Dort olcut "
              "de kosudan once sabitlendi ve dort kategoride de dorduyle gecti. "
              "En ikna edici olcut mevsimsellik: Aralik-Ocak orani yaz aylarinin "
              "bir buçuk kati; bir anahtar kelime listesi takvimi taklit edemez. "
              "Ikinci olcut de onemli: etiketlerin yuzde on sekizi sozcuksel vekilin "
              "hic yakalamadigi satirlardan geliyor, yani model kelime aramanin "
              "otesine geciyor. Grafikte iki egri: LLM etiketi ve vekil.")


def s07_honesty(prs, N):
    sl = new_slide(prs)
    h = N["human"]
    head(sl, "The detector did not meet its own accuracy thresholds. We report that first.",
         kicker="Honesty slide · Week 4", size=24, accent=CAUTION)
    picture_fit(sl, "F18_llm_vs_human_500.png", M, 1.95, 6.2, 4.1)
    caption(sl, f"F18 · per-class F1 of the LLM labels against {h['n_rows']} "
                f"independently human-labelled rows", M, 6.1, 6.2)

    x = M + 6.5
    w = CW - 6.5
    card(sl, x, 1.95, w, 1.85, fill=CAUTION_SOFT, line=RULE)
    tf = textbox(sl, x + 0.2, 2.1, w - 0.4, 1.6)
    put(tf, "Two pre-registered gates were MISSED", size=13, bold=True, color=CAUTION,
        space_after=6)
    put(tf, f"macro-F1 ≥ 0.75  →  measured {h['macro_f1']:.4f}  "
            f"{ci(h['macro_f1_ci'], 4)}", size=11.5, color=INK, space_after=3)
    put(tf, f"gift precision ≥ 0.80  →  measured {h['gift_precision']:.2f}  "
            f"(recall {h['gift_recall']:.2f}, F1 {h['gift_f1']:.4f})",
        size=11.5, color=INK, space_after=3)
    put(tf, f"Week-4 verdict recorded in the artefact: {h['verdict']}",
        size=11, bold=True, color=CAUTION, space_after=0)

    card(sl, x, 3.95, w, 1.35, fill=SOFT2, line=RULE)
    tf = textbox(sl, x + 0.2, 4.08, w - 0.4, 1.1)
    put(tf, "Label reliability was never measured", size=12.5, bold=True, color=ACCENT,
        space_after=5)
    put(tf, f"Planned: {N['human']['planned']['planned_annotators']} annotators, "
            f"{N['human']['planned']['planned_statistic']} ≥ "
            f"{N['human']['planned']['threshold']}.  "
            f"Actual: {h['n_annotators']} annotator, so no agreement statistic exists.",
        size=11, color=INK, space_after=0, line=1.12)

    card(sl, x, 5.45, w, 1.25, fill=SOFT, line=RULE)
    tf = textbox(sl, x + 0.2, 5.58, w - 0.4, 1.0)
    put(tf, "Why the study is still readable", size=12.5, bold=True, color=ACCENT,
        space_after=5)
    put(tf, "The placebo holds the amount of deleted data fixed, and the effect scales "
            "with the gift share — neither depends on detector accuracy. What the weak "
            "detector changes is what is measured: the rows it FLAGS as gifts.",
        size=10.5, color=INK, space_after=0, line=1.12)
    foot(sl, 7)
    notes(sl, "Bu slaydi atlamiyoruz, aksine one aliyoruz. Concept note'ta iki "
              "esik yazmisiz: makro-F1 en az sifir yetmis bes ve hediye kesinligi "
              "en az sifir seksen. Olculen degerler sifir elli dort ve sifir altmis "
              "sekiz - ikisi de tutmuyor. Ayrica tek etiketleyici calisti, yani "
              "planlanan Fleiss kappa hic olculmedi; Hafta 4'un artefakttaki karari "
              "INCOMPLETE. Peki calisma neden hala okunabilir? Iki sey dedektorun "
              "dogruluguna bagli degil: plasebo silinen veri miktarini sabit tutuyor, "
              "ve etki kategorinin hediye payiyla buyuyor. Zayif dedektorun degistirdigi "
              "sey ne olculdugu: 'hediyeler' degil, dedektorun hediye DEDIGI satirlar. "
              "Bunlarin ucte biri insana gore hediye degil, cogu kendi cocuguna alim - "
              "o yuzden 'alt sinir' okumasi bir varsayima dayaniyor, A4'te yaziyor.")


def s08_distill(prs, N):
    sl = new_slide(prs)
    d = N["distill"]
    head(sl, "A distilled student reproduced the teacher's labels closely enough to scale",
         kicker="Distillation · fidelity gate", size=24)
    picture_fit(sl, "F20_distill_fidelity_base.png", M, 1.95, 6.3, 4.1)
    caption(sl, "F20 · student vs teacher on the held-out split", M, 6.1, 6.3)

    x = M + 6.6
    w = CW - 6.6
    put(textbox(sl, x, 1.95, w, 0.3), "FIDELITY GATE — 3 CRITERIA, FIXED BEFORE TRAINING",
        size=10.5, bold=True, color=ACCENT, space_after=0)
    rows = [["Criterion", "Threshold", "Measured"],
            ["F1 vs teacher, narrow axis (C1)", f"≥ {d['threshold']}", f"{d['c1_f1']:.4f}"],
            ["F1 vs teacher, broad axis (C1b)", f"≥ {d['threshold']}", f"{d['c1b_f1']:.4f}"],
            ["Not worse than teacher vs human",
             f"drop ≤ {d['max_drop']}", f"{d['student_f1']:.4f} vs {d['teacher_f1']:.4f}"]]
    table(sl, x, 2.28, w, rows, [3.0, 1.2, 1.5], row_h=0.5, size=10.5, head_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT])
    badge(sl, x, 4.45, f"GATE {d['verdict']} · 3/3", w=1.9, h=0.32, size=12)

    tf = textbox(sl, x, 5.0, w, 1.75)
    put(tf, "What this does and does not say", size=12.5, bold=True, color=ACCENT,
        space_after=6)
    put(tf, f"Trained on {thousands(d['n_train'])} teacher labels, checked on "
            f"{thousands(d['n_holdout'])} held-out rows, then run over "
            f"{thousands(N['n_labelled'])} interactions "
            f"({thousands(N['infer']['Toys_and_Games'])} Toys + "
            f"{thousands(N['infer']['Grocery_and_Gourmet_Food'])} Grocery).",
        size=11, color=INK, space_after=6, line=1.12)
    put(tf, "The gate measures agreement with the TEACHER, not correctness. The student "
            "inherits the teacher's errors — which is exactly why the accuracy limits on "
            "the previous slide still apply at 4.6 M rows.",
        size=10.5, color=INK2, space_after=0, line=1.12)
    foot(sl, 8)
    notes(sl, "Dort buçuk milyon satiri ogretmen modelle etiketlemek mumkun "
              "olmadigi icin ModernBERT-base'i ogretmenin etiketleriyle egittik. "
              "Sadakat kapisi uc olcutle ve egitimden once sabitlendi; uçu de "
              "gecti - dar eksende sifir doksan bir, geniş eksende sifir doksan "
              "uc F1. Kritik nuans sagdaki son paragraf: bu kapi DOGRULUGU degil "
              "OGRETMENE BENZERLIGI olcuyor. Ogrenci ogretmenin hatalarini da "
              "devraliyor, yani onceki slayttaki dogruluk siniri dort buçuk milyon "
              "satirda da gecerli.")


def s09_rq1(prs, N):
    sl = new_slide(prs)
    head(sl, "RQ1 · About one in five Toys reviews is a gift — and every category peaks in December",
         kicker="Result 1 · prevalence", size=22)
    picture_fit(sl, "F19_prevalence_by_category.png", M, 1.95, 6.6, 4.35)
    caption(sl, "F19 · human-calibrated prevalence, narrow (gift_given) axis",
            M, 6.32, 6.6)

    x = M + 6.9
    w = CW - 6.9
    rows = [["Category", "Calibrated", "95 % CI", "Raw LLM"]]
    isim = {"Toys_and_Games": "Toys & Games",
            "Grocery_and_Gourmet_Food": "Grocery",
            "Video_Games": "Video Games", "All_Beauty": "All Beauty"}
    for kat in CATEGORIES:
        if kat not in N["prevalence"]:
            continue
        p = N["prevalence"][kat]
        rows.append([isim[kat], f"{p['cal']:.1f}%",
                     f"[{p['ci'][0]:.1f}, {p['ci'][1]:.1f}]", f"{p['raw']:.1f}%"])
    table(sl, x, 1.95, w, rows, [1.9, 1.2, 1.5, 1.1], row_h=0.4, size=10.5, head_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT])

    put(textbox(sl, x, 4.15, w, 0.3), "DECEMBER–JANUARY ÷ SUMMER", size=10.5, bold=True,
        color=ACCENT, space_after=0)
    y = 4.48
    for kat in CATEGORIES:
        g = N["gate1"][kat]
        put(textbox(sl, x, y, w, 0.28),
            f"{isim[kat]}   {g['ratio']:.2f}×   {ci(g['ci'], 2)}",
            size=10.5, color=INK, space_after=0)
        y += 0.31
    put(textbox(sl, x, 5.85, w, 0.85),
        "Calibration maps the LLM's raw rate through the confusion matrix measured on the "
        "500 human rows, so the interval carries the detector's error — this is why Toys "
        "moves from 23.2 % to 19.0 %.",
        size=10, color=INK2, space_after=0, line=1.12, )
    foot(sl, 9)
    notes(sl, "Ilk sonuc: Toys ve Games'te review'larin yaklasik besde biri hediye. "
              "Grocery, Video Games ve All Beauty yuzde bes ile sekiz arasinda - "
              "Grocery'yi tam da bu yuzden kontrol kategorisi sectik. Sagdaki ham "
              "ve kalibre edilmis sutun farkina dikkat: kalibrasyon, LLM'in ham "
              "oranini bes yuz insan satirinda olculen karisiklik matrisinden "
              "geciriyor, boylece dedektorun hatasi guven araligina giriyor. "
              "Toys yuzde yirmi uçten yuzde on dokuza duşuyor. Alt kisimda dort "
              "kategorinin de Aralik-Ocak tepesi var.")


def s10_placebo(prs, N):
    sl = new_slide(prs)
    head(sl, "Deleting gifts also deletes data — so the comparison must hold the data loss fixed",
         kicker="Design · why a placebo", size=23)
    flow(sl, [("C0 · do nothing", "all rows kept"),
              ("C1 · delete gifts", "n gift rows removed"),
              ("C4 · PLACEBO", "the same n RANDOM rows removed")],
         t=2.05, h=1.0, gap=0.5, size=11.5, title_size=13)

    card(sl, M, 3.28, CW * 0.485, 1.3, fill=ZEBRA, line=RULE)
    tf = textbox(sl, M + 0.2, 3.42, CW * 0.485 - 0.4, 1.1)
    put(tf, "C1 vs C0 cannot answer RQ2", size=13, bold=True, color=CAUTION, space_after=5)
    put(tf, "Two things change at once: the gift rows are gone AND the training set is "
            "smaller. A drop could be the missing rows; a gain could be luck.",
        size=11, color=INK, space_after=0, line=1.12)

    card(sl, M + CW * 0.515, 3.28, CW * 0.485, 1.3, fill=SOFT, line=RULE)
    tf = textbox(sl, M + CW * 0.515 + 0.2, 3.42, CW * 0.485 - 0.4, 1.1)
    put(tf, "C1 vs C4 can", size=13, bold=True, color=ACCENT, space_after=5)
    put(tf, "Identical training-set size, identical test pairs, same seeds. The only "
            "difference is WHICH rows went. Positive means gift rows were worth less "
            "than random rows.",
        size=11, color=INK, space_after=0, line=1.12)

    put(textbox(sl, M, 4.78, CW, 0.32), "ALL SIX CONDITIONS", size=10.5, bold=True,
        color=ACCENT, space_after=0)
    rows = [["C0", "C1", "C4", "C1b", "C4b", "C3"],
            ["baseline\nnothing removed", "narrow\ngift_given removed",
             "placebo for C1\nsame n, random", "broad\n+ household, received",
             "placebo for C1b\nsame n, random", "flag, don't delete\nshadow token"]]
    table(sl, M, 5.10, CW, rows, [1, 1, 1, 1, 1, 1], row_h=0.5, size=10, head_size=12,
          align=[PP_ALIGN.CENTER] * 6, zebra=False)
    put(textbox(sl, M, 6.22, CW, 0.5),
        "C2 (down-weighting) was specified but not run: RecBole's SASRec loss does not "
        "accept per-interaction weights without patching the library, and patching it "
        "would have broken the \"same code for every condition\" rule.",
        size=10.5, color=INK2, space_after=0, line=1.12, italic=True)
    foot(sl, 10)
    notes(sl, "Deneyin en onemli tasarim karari bu. Hediyeleri silmek ayni zamanda "
              "veriyi kuçultuyor, o yuzden C1'i C0 ile karsilastirmak iki seyi ayni "
              "anda degistiriyor. Cozum plasebo: C4 ayni SAYIDA rastgele satiri "
              "siliyor. Boylece egitim kumesi buyuklugu, test ciftleri ve seed'ler "
              "birebir ayni kaliyor; tek fark hangi satirlarin gittigi. Alt satirda "
              "alti kosulun tanimi var. C2 yani agirlik azaltma yazildi ama "
              "kosulmadi: RecBole'un SASRec kaybi etkilesim basina agirlik almiyor, "
              "kutuphaneyi yamamak da 'her kosulda ayni kod' kuralini bozacakti.")


def s11_gate2(prs, N):
    sl = new_slide(prs)
    head(sl, "Gate 2 · the setup is valid in all four cells, so the contrasts mean something",
         kicker="Validity", size=24)
    rows = [["Criterion", "What it rules out"] + [short(h) for h in CELLS]]
    cv = N["gate2_seed"]
    olcut = [("1 · Every run scored on the same test pairs",
              "Comparing different test sets", ["PASS"] * 4),
             ("2 · Each condition's item universe ⊆ C0's",
              "A condition inventing items", ["PASS"] * 4),
             ("3 · Neither placebo is BETTER than C0",
              "Deletion helping by itself", ["PASS"] * 4),
             (f"4 · C0 seed-to-seed CV < {cv[CELLS[0]]['threshold']}",
              "Runs too unstable to compare",
              [f"PASS · {cv[h]['cv']:.3f}" for h in CELLS])]
    for ad, ruled, hucreler in olcut:
        rows.append([ad, ruled] + hucreler)
    table(sl, M, 2.05, CW, rows, [4.0, 3.0, 1.45, 1.45, 1.45, 1.45], row_h=0.42,
          size=10.5, head_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)

    y = 4.30
    put(textbox(sl, M, y, CW, 0.3),
        "CRITERION 3 IN NUMBERS — NEITHER PLACEBO IS BETTER THAN C0 (THE CI'S UPPER END MAY TOUCH ZERO, BUT NEVER RISES ABOVE IT)",
        size=10.5, bold=True, color=ACCENT, space_after=0)
    rows2 = [["Cell", "C4 − C0, Recall@10", "C4b − C0, Recall@10"]]
    for hucre in CELLS:
        p = N["gate2_placebo"][hucre]
        a, b = p.get("C4-C0", {}), p.get("C4b-C0", {})
        rows2.append([short(hucre),
                      f"{num(a.get('diff'), 6)}  {ci(a.get('ci'), 4)}",
                      f"{num(b.get('diff'), 6)}  {ci(b.get('ci'), 4)}"])
    table(sl, M, y + 0.33, CW, rows2, [2.6, 4.6, 4.6], row_h=0.32, size=10.5, head_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

    put(textbox(sl, M, 6.34, CW, 0.5),
        f"{N['gate2_runs']} runs per cell, {N['n_runs']} in total. Criterion 1 is enforced "
        f"by a SHA-256 hash of the test pairs written into every run report — one distinct "
        f"hash per cell, checked by the gate rather than assumed.",
        size=10.5, color=INK2, space_after=0, line=1.12)
    foot(sl, 11)
    notes(sl, "Kapi 2, sonuclari yorumlamaya hakkimiz olup olmadigini soruyor ve "
              "dort hucrede de dort olcutuyle geciyor. En onemli olcut ucuncusu: "
              "hicbir plasebo C0'dan IYI olmamali, cunku veri silmek tek basina "
              "yardim ediyorsa butun tasarim coker. Alt tabloda bu sayilar var; "
              "Toys'ta guven araliklari sifirin acikca altinda, Grocery'de sifira "
              "degiyor ama uzerine cikmiyor - kapinin istedigi tam olarak bu. "
              "Birinci olcut de varsayim degil: her kosu raporuna test ciftlerinin "
              "SHA-256 ozeti yaziliyor ve kapi hucre basina tek bir ozet oldugunu "
              "dogruluyor. Dorduncusu C0'in seed'ler arasi degisim katsayisi; "
              "hepsi esigin cok altinda. Hucre basina on sekiz, toplam yetmis iki kosu.")


def s12_rq2(prs, N):
    sl = new_slide(prs)
    t = N["c1_c4"]["Toys_and_Games/SASRec"]
    tb = N["c1_c4"]["Toys_and_Games/BPR"]
    head(sl, "RQ2 · Removing gifts beats removing the same number of random rows",
         kicker="Result 2 · the core finding", size=25)
    picture_fit(sl, "F21_condition_contrasts.png", M, 1.95, 6.45, 4.3)
    caption(sl, "F21 · per-cell contrasts with 95 % paired-bootstrap intervals",
            M, 6.28, 6.45)

    x = M + 6.75
    w = CW - 6.75
    put(textbox(sl, x, 1.95, w, 0.3), "C1 − C4, RECALL@10 (PRIMARY)", size=10.5,
        bold=True, color=ACCENT, space_after=0)
    rows = [["Cell", "Δ", "95 % CI", "Rel.", "Rule"]]
    for hucre in CELLS:
        c = N["c1_c4"][hucre]
        rows.append([short(hucre), num(c["diff"], 6), ci(c["ci"], 4),
                     pct(c["rel"], 1, sign=True), verdict_en(c["verdict"])])
    table(sl, x, 2.28, w, rows, [1.55, 1.05, 1.55, 0.85, 1.25], row_h=0.40, size=9.5,
          head_size=9,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT,
                 PP_ALIGN.LEFT])

    put(textbox(sl, x, 4.40, w, 0.3), "DOSE–RESPONSE: TOYS − GROCERY", size=10.5,
        bold=True, color=ACCENT, space_after=0)
    rows2 = [["Model", "Axis", "Δ", "95 % CI"]]
    for model, d in N["dose"].items():
        for anahtar, v in d.items():
            m = v["metrics"].get("recall@10") or next(iter(v["metrics"].values()))
            rows2.append([model, anahtar, num(m["diff"], 6), ci(m["ci"], 4)])
    table(sl, x, 4.72, w, rows2, [1.1, 1.3, 1.25, 1.6], row_h=0.32, size=9.5, head_size=9,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER])

    put(textbox(sl, x, 6.42, w, 0.5),
        "All four dose–response intervals exclude zero: the effect is larger where gifts "
        "are more common. That is the pattern a real mechanism produces.",
        size=9.5, color=INK2, space_after=0, line=1.1)
    foot(sl, 12)
    notes(sl, "Projenin ana bulgusu. Veri kaybi sabit tutuldugunda hediye "
              "satirlarini silmek, ayni sayida rastgele satiri silmekten daha iyi "
              "bir model biraikiyor: Toys'ta sirali modelde yuzde on alti, klasik "
              "modelde yuzde kirk rölatif kazanc. Kontrol kategorisi Grocery'de "
              "etki ya cok kuçuk ya saptanamiyor. Alt tablodaki doz-yanit bunun "
              "tesaduf olmadiginin en guclu isareti: dort karsilastirmanin "
              "dordunde de Toys'un etkisi Grocery'ninkinden istatistiksel olarak "
              "buyuk. Hediye ne kadar yaygınsa etki o kadar buyuk - gercek bir "
              "mekanizmanin uretecegi desen bu.")


def s13_rq2_net(prs, N):
    sl = new_slide(prs)
    head(sl, "But \"better than the placebo\" is not \"better than doing nothing\"",
         kicker="Result 2b · the honest caveat", size=25)
    put(textbox(sl, M, 1.82, CW, 0.32),
        "The placebo contrast isolates the mechanism. The C1 − C0 contrast is what a "
        "practitioner would actually feel.", size=12, color=INK2, space_after=0)

    rows = [["Cell", "C1 − C4  (vs placebo)", "Rule", "C1 − C0  (vs doing nothing)",
             "Rule", "C1b − C4b  (broad axis)", "Rule"]]
    for hucre in CELLS:
        a, b, c = N["c1_c4"][hucre], N["c1_c0"][hucre], N["c1b_c4b"][hucre]
        rows.append([short(hucre),
                     f"{num(a['diff'], 6)}  {pct(a['rel'], 1, sign=True)}",
                     verdict_en(a["verdict"]),
                     f"{num(b['diff'], 6)}  {pct(b['rel'], 1, sign=True)}",
                     verdict_en(b["verdict"]),
                     f"{num(c['diff'], 6)}  {pct(c['rel'], 1, sign=True)}",
                     verdict_en(c["verdict"])])
    table(sl, M, 2.22, CW, rows, [2.2, 2.05, 1.35, 2.05, 1.35, 2.05, 1.35], row_h=0.46,
          size=10, head_size=9,
          align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT, PP_ALIGN.RIGHT,
                 PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT])

    bw = (CW - 2 * 0.3) / 3
    tiles = [
        ("The mechanism is real", ACCENT, SOFT,
         "C1 − C4 excludes zero in three of four cells, and the dose–response holds. "
         "Gift rows carry less usable preference signal than random rows."),
        ("The net gain is model-dependent", CAUTION, CAUTION_SOFT,
         f"On Toys, deleting gifts outright helps BPR "
         f"({pct(N['c1_c0']['Toys_and_Games/BPR']['rel'], 1, sign=True)}) but hurts SASRec "
         f"({pct(N['c1_c0']['Toys_and_Games/SASRec']['rel'], 1, sign=True)}) — the "
         f"sequence model loses context with the rows."),
        ("Widening the definition widens the gap", ACCENT, SOFT,
         "C1b − C4b is larger than C1 − C4 in all four cells, which is consistent with "
         "household and received purchases carrying the same kind of noise."),
    ]
    for i, (bas, renk, zemin, alt) in enumerate(tiles):
        x = M + i * (bw + 0.3)
        card(sl, x, 4.55, bw, 1.62, fill=zemin, line=RULE)
        tf = textbox(sl, x + 0.18, 4.70, bw - 0.36, 1.4)
        put(tf, bas, size=13, bold=True, color=renk, space_after=6, line=1.0)
        put(tf, alt, size=10.5, color=INK, space_after=0, line=1.12)
    foot(sl, 13)
    notes(sl, "Bu slayt bulguyu abartmamak icin var. Plaseboya karsi kazanmak, "
              "hiçbir sey yapmamaya karsi kazanmak demek degil. Ortadaki tabloda "
              "C1 eksi C0 sutununa bakin: Toys'ta klasik model kazaniyor ama "
              "sirali model KAYBEDIYOR - sirali model silinen satirlarla birlikte "
              "baglami da kaybediyor. Yani 'hediyeleri sil' diye bir tavsiye "
              "vermiyoruz; 'hediye satirlari daha az tercih sinyali tasiyor' "
              "diyoruz. Ucuncu kutu ilginç: geniş eksende fark daha da buyuyor, "
              "bu da ev halki ve alinan hediye kategorilerinin ayni tur gurultuyu "
              "tasidigiyla tutarli.")


def s14_rq3(prs, N):
    sl = new_slide(prs)
    head(sl, "RQ3 · Flagging with a shadow token was worse than both alternatives",
         kicker="Result 3 · delete or flag?", size=25)
    put(textbox(sl, M, 1.82, CW, 0.32),
        "C3 keeps every gift row but replaces the item with a per-subcategory shadow token, "
        "so the model can learn \"something happened here\" without learning the item.",
        size=12, color=INK2, space_after=0)

    rows = [["Cell", "C3 − C1  (flag vs delete)", "Rule", "C3 − C0  (flag vs nothing)",
             "Rule"]]
    for hucre in CELLS:
        a, b = N["c3_c1"][hucre], N["c3_c0"][hucre]
        rows.append([short(hucre), f"{num(a['diff'], 6)}   {ci(a['ci'], 4)}",
                     verdict_en(a["verdict"]),
                     f"{num(b['diff'], 6)}   {ci(b['ci'], 4)}", verdict_en(b["verdict"])])
    table(sl, M, 2.25, CW, rows, [2.4, 3.3, 1.5, 3.3, 1.5], row_h=0.48, size=11,
          head_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT,
                 PP_ALIGN.LEFT])

    bw = (CW - 0.35) / 2
    card(sl, M, 4.65, bw, 1.52, fill=SOFT2, line=RULE)
    tf = textbox(sl, M + 0.2, 4.80, bw - 0.4, 1.3)
    put(tf, "What we found", size=13, bold=True, color=ACCENT, space_after=6)
    put(tf, "Negative against deletion in three of four cells, and against doing nothing "
            "in three of four. The shadow token adds vocabulary and sequence positions "
            "without adding information the model can use.",
        size=11, color=INK, space_after=0, line=1.15)

    card(sl, M + bw + 0.35, 4.65, bw, 1.52, fill=CAUTION_SOFT, line=RULE)
    tf = textbox(sl, M + bw + 0.55, 4.80, bw - 0.4, 1.3)
    put(tf, "What we are NOT claiming", size=13, bold=True, color=CAUTION, space_after=6)
    put(tf, "That flagging is a bad idea in general. One implementation was tested: a "
            "per-subcategory shadow token in the item sequence. Type-aware embeddings, "
            "a separate purchase-type channel, or loss re-weighting were not tested.",
        size=11, color=INK, space_after=0, line=1.15)
    foot(sl, 14)
    notes(sl, "Ucuncu soru pratik: silmek mi, isaretlemek mi? Test ettigimiz "
              "isaretleme yontemi, hediye satirini tutup urunu alt kategori basina "
              "bir golge token'la degistirmek. Sonuç: dort hucrenin uçunde silmekten "
              "de, hiçbir sey yapmamaktan da kotu. Golge token modele "
              "kullanabilecegi bilgi eklemeden kelime dagarcigi ve sira pozisyonu "
              "ekliyor. Sagdaki kutu onemli: 'isaretleme kotu bir fikirdir' "
              "demiyoruz - TEK bir uygulamayi test ettik. Tur-farkindali gomme "
              "vektorleri ya da ayri bir alim-turu kanali denenmedi.")


def s15_rq4_halflife(prs, N):
    sl = new_slide(prs)
    t = N["m2"]["Toys_and_Games/SASRec"]
    head(sl, "RQ4 · The excess halves after about one self-purchase — roughly two to four weeks",
         kicker="Result 4 · how long it lasts", size=23)
    picture_fit(sl, "F22_m2_half_life.png", M, 1.9, 7.4, 4.45)
    caption(sl, "F22 · excess top-10 share for the gift's subcategory vs the number of "
                "self-purchases after the gift, with the pre-registered exponential fit",
            M, 6.38, 7.4)

    x = M + 7.65
    w = CW - 7.65
    stat(sl, x, 1.9, w, 1.35, f"{t['half_life']:.2f}", "self-purchase half-life",
         f"95 % CI {ci(t['ci'], 2)}", big_size=30)
    stat(sl, x, 3.35, w, 1.3, f"≈{t['weeks']:.1f} wk", "approximate calendar time",
         f"{ci(t['weeks_ci'], 1) if t.get('weeks_ci') else 'review date ≠ purchase date'}",
         big_size=26, fill=SOFT)
    tf = textbox(sl, x, 4.8, w, 1.9)
    put(tf, "Read it like this", size=12, bold=True, color=ACCENT, space_after=5)
    put(tf, f"A gift opens {pts(t['start_excess'], 1)} of extra top-10 share for its "
            f"subcategory. After one purchase the user makes for themselves, less than "
            f"half of that is left; after four it is gone.",
        size=10.5, color=INK, space_after=6, line=1.12)
    put(tf, "BPR shows decay too, but the pre-registered fit gives no finite half-life: "
            "the gift is DILUTED by later purchases rather than forgotten, because BPR "
            "never sees order.",
        size=10, color=INK2, space_after=0, line=1.12)
    foot(sl, 15)
    notes(sl, "Pazarlama icin en kullanisli sayi bu. Sirali modelde bir hediye, "
              "hediyenin alt kategorisine top-10'da iki puan fazla yer aciyor. "
              "Kullanici kendisi icin BIR alisveris yaptiktan sonra bu fazlanin "
              "yarisindan azi kaliyor, dort alimdan sonra sifirlaniyor. Takvime "
              "cevirisi kabaca iki-dort hafta ama bu YAKLASIK, cunku elimizde "
              "review tarihi var, alim tarihi yok. BPR'de de azalma var ama "
              "onceden kayitli uyum sonlu bir yari omur vermiyor: BPR sirayi "
              "gormedigi icin hediye unutulmuyor, yeni alimlar arasinda "
              "SEYRELIYOR. Pazarlama dilinde: kirlilik zamanla degil, yeni "
              "alimlarla temizleniyor.")


def s16_rq4_waste(prs, N):
    sl = new_slide(prs)
    head(sl, "RQ4 · In the classical model, roughly one slot in twenty goes to gift-only interest",
         kicker="Result 4b · wasted slots", size=23)
    picture_fit(sl, "F23_m1_waste_share.png", M, 1.95, 6.5, 4.15)
    caption(sl, "F23 · share of top-10 slots drawn from subcategories the user entered "
                "only as a gift", M, 6.18, 6.5)

    x = M + 6.8
    w = CW - 6.8
    rows = [["Cell", "C0 − C1", "C4 − C1 (net of data loss)"]]
    for hucre in CELLS:
        a = N["m1"][hucre]["c0_c1"]
        b = N["m1"][hucre]["c4_c1"]
        rows.append([short(hucre),
                     f"{pts(a.get('diff'), 1)}  {ci(a.get('ci'), 2, 100)}",
                     f"{pts(b.get('diff'), 1)}  {ci(b.get('ci'), 2, 100)}"])
    table(sl, x, 1.95, w, rows, [1.6, 2.3, 2.5], row_h=0.44, size=10, head_size=9,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

    tf = textbox(sl, x, 4.32, w, 2.4)
    put(tf, "What the numbers say", size=12.5, bold=True, color=ACCENT, space_after=6)
    put(tf, f"On Toys, {pct(N['m1']['Toys_and_Games/BPR']['c0_c1'].get('n_exposed_users', 0) / max(N['m1']['Toys_and_Games/BPR']['c0_c1'].get('n_test_users', 1), 1), 0)} "
            f"of test users have a subcategory they entered only as a gift. The classical "
            f"model gives {pct(N['m1']['Toys_and_Games/BPR']['c0_c1'].get('mean_a'), 1)} of "
            f"their top-10 to it; the gift-free model gives "
            f"{pct(N['m1']['Toys_and_Games/BPR']['c0_c1'].get('mean_b'), 1)}.",
        size=10.5, color=INK, space_after=6, line=1.12)
    put(tf, "Holding data loss fixed (C4 − C1) keeps most of that gap in the classical "
            "model but almost none of it in the sequential one.",
        size=10.5, color=INK, space_after=6, line=1.12)
    put(tf, "\"Waste\" is an upper bound by construction: we did not measure whether those "
            "slots convert. A customer may later buy in that subcategory for themselves.",
        size=10, color=CAUTION, space_after=0, line=1.12, italic=True)
    foot(sl, 16)
    notes(sl, "Ikinci pazarlama metrigi: top-10'un yalnizca hediye olarak girilen "
              "alt kategorilere giden payi. Toys'ta test kullanicilarinin ucte "
              "birinde boyle bir alt kategori var. Klasik model bu kullanicilarin "
              "listesinin yuzde on altisini oraya ayiriyor, hediyeleri gormemis "
              "model yuzde on birini - yani her yuz slottan bes tanesi hediyenin "
              "actigi ilgiye gidiyor. Veri kaybi sabit tutuldugunda klasik modelde "
              "bu farkin buyuk kismi kaliyor, sirali modelde neredeyse hiç "
              "kalmiyor. Ve dipnot: 'israf' bir ust sinir - o slotlarin hiç "
              "donusmedigini olcmedik.")


def s17_close(prs, N):
    sl = new_slide(prs)
    head(sl, "What we claim, what we do not, and what comes next", kicker="Closing")
    bw = (CW - 2 * 0.3) / 3

    card(sl, M, 2.0, bw, 3.15, fill=SOFT, line=RULE)
    tf = textbox(sl, M + 0.2, 2.16, bw - 0.4, 2.75)
    put(tf, "We claim", size=14, bold=True, color=ACCENT, space_after=7)
    for s in ["Gift purchases are measurable at scale, and common: "
              f"{N['prevalence']['Toys_and_Games']['cal']:.0f} % of Toys reviews.",
              "Gift rows carry less usable preference signal than random rows, "
              "with a dose–response across two categories.",
              f"A gift's excess recommendation share halves after about one "
              f"self-purchase in the sequential model.",
              "The one flagging method we tested is worse than deleting."]:
        put(tf, "•  " + s, size=10.5, color=INK, space_after=6, line=1.12)

    card(sl, M + bw + 0.3, 2.0, bw, 3.15, fill=CAUTION_SOFT, line=RULE)
    tf = textbox(sl, M + bw + 0.5, 2.16, bw - 0.4, 2.75)
    put(tf, "We do not claim", size=14, bold=True, color=CAUTION, space_after=7)
    for s in [f"That the detector is accurate: macro-F1 {N['human']['macro_f1']:.2f} "
              f"against one annotator, both pre-registered gates missed.",
              "That deleting gifts is a recommended production change — the net effect "
              "against doing nothing is model-dependent.",
              "That this generalises beyond two Amazon categories, two models and "
              "English-language reviews.",
              "That wasted slots are lost revenue; conversion was never measured."]:
        put(tf, "•  " + s, size=10.5, color=INK, space_after=6, line=1.12)

    card(sl, M + 2 * (bw + 0.3), 2.0, bw, 3.15, fill=SOFT2, line=RULE)
    tf = textbox(sl, M + 2 * (bw + 0.3) + 0.2, 2.16, bw - 0.4, 2.75)
    put(tf, "Next, in order of value", size=14, bold=True, color=ACCENT, space_after=7)
    for s in ["A second and third annotator on the same 500 rows, and the Fleiss κ the "
              "design asked for.",
              "Purchase-type as a model input instead of a deletion rule — the C2 that "
              "RecBole would not run.",
              "A held-out category the detector never saw during prompt development.",
              "Conversion-linked data, so \"waste\" becomes a measured cost."]:
        put(tf, "•  " + s, size=10.5, color=INK, space_after=6, line=1.12)

    card(sl, M, 5.42, CW, 1.12, fill=WHITE, line=ACCENT)
    tf = textbox(sl, M + 0.24, 5.56, CW - 0.48, 0.9)
    put(tf, "Everything on these slides is reproducible from the repository",
        size=13, bold=True, color=ACCENT, space_after=5)
    put(tf, f"python repo/scripts/demo.py   —   prints every headline number above from the "
            f"committed JSON artefacts, with no data files and no dependencies beyond the "
            f"standard library.      {GITHUB}      code version "
            f"{N['provenance']['experiment_stats.json']}",
        size=11, color=INK, space_after=0, line=1.15)
    foot(sl, 17)
    notes(sl, "Kapanis slaydi uç sutun: ne iddia ediyoruz, ne iddia etmiyoruz, "
              "sirada ne var. Ortadaki sutunu hizla geçmeyin - jurinin soracagi "
              "her soru orada zaten yazili. Alttaki kutu davetiye: depoyu klonlayip "
              "tek komutla butun basliklari basabilirsiniz, veri dosyasi bile "
              "gerekmiyor cunku sayilar commit'li JSON'lardan okunuyor. Soru-cevap "
              "icin bes ek slayt var: kosul tanimlari, iki kapinin olcutleri, tam "
              "karsitlik tablosu, sinirliliklar ve yeniden uretim.")


# ------------------------------------------------------------------- ekler
def a01_conditions(prs, N):
    sl = new_slide(prs)
    head(sl, "A1 · The six conditions, exactly as they were run", kicker="Appendix")
    rows = [["Condition", "Definition", "Rows removed", "Purpose"],
            ["C0", "Baseline — the full 5-core training set", "none",
             "Reference for every other condition"],
            ["C1", "Narrow: rows labelled gift_given are deleted from TRAINING",
             "n_gift", "RQ2 treatment"],
            ["C4", "Placebo for C1 — the same NUMBER of rows, chosen at random",
             "n_gift", "Separates \"which rows\" from \"how many rows\""],
            ["C1b", "Broad: gift_given + household + received deleted", "n_broad",
             "Sensitivity to the label definition"],
            ["C4b", "Placebo for C1b", "n_broad", "Same, for the broad axis"],
            ["C3", "Gift rows kept; item replaced by a per-subcategory shadow token",
             "none", "RQ3 — flag instead of delete"]]
    table(sl, M, 2.05, CW, rows, [1.3, 5.2, 1.5, 3.9], row_h=0.48, size=10.5,
          head_size=9.5, first_bold=True,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.LEFT])
    put(textbox(sl, M, 5.75, CW, 1.0),
        "Only training rows are touched; validation and test rows are identical in every "
        "condition, which is what makes the runs comparable — and is also a limitation: a "
        "gift sitting in a user's validation row is never removed, so the sequential model "
        "still sees it as the last item of the test input. Placebo row selection uses a "
        "per-condition seed derived as (base_seed + zlib.crc32(condition_name)) mod 2³¹−1, "
        "never Python's hash(), which is randomised per process.",
        size=10.5, color=INK2, space_after=0, line=1.15)
    foot(sl, "A1")
    notes(sl, "Ek slayt: alti kosulun tam tanimi. Vurgulanacak iki sey var. Birincisi "
              "yalnizca EGITIM satirlari degisiyor; dogrulama ve test satirlari her "
              "kosulda ayni - karsilastirmayi mumkun kilan da bu, ama ayni zamanda "
              "bir sinirlilik: kullanicinin dogrulama satirindaki hediye hiç "
              "silinmiyor. Ikincisi plasebo satirlarinin seçimi: seed crc32'den "
              "turetiliyor, Python'un hash fonksiyonundan degil, cunku o her "
              "surecte farkli sonuç verir ve deney yeniden uretilemez olurdu.")


def a02_gates(prs, N):
    sl = new_slide(prs)
    head(sl, "A2 · Both gates in full, with the numbers that decided them",
         kicker="Appendix")
    g = N["gate1"]
    put(textbox(sl, M, 1.85, CW, 0.3), "GATE 1 — MAY THE DETECTOR BE USED? (PER CATEGORY)",
        size=10.5, bold=True, color=ACCENT, space_after=0)
    rows = [["Criterion", "Threshold", "Toys", "Grocery", "Video Games", "All Beauty"],
            ["1 · Dec–Jan ÷ summer ratio", f"≥ {g['Toys_and_Games']['ratio_threshold']}×"]
            + [f"{g[k]['ratio']:.2f}×" for k in CATEGORIES],
            ["2 · Rate found beyond keyword proxy",
             f"≥ {pct(g['Toys_and_Games']['beyond_threshold'], 0)}"]
            + [pct(g[k]["beyond_keyword"], 1) for k in CATEGORIES],
            ["3 · Schema health", "no failure"] + ["pass"] * 4,
            ["4 · 200-row trial agreement", f"≥ {g['Toys_and_Games']['trial_threshold']}"]
            + ["0.835"] * 4,
            ["Verdict", ""] + [f"{g[k]['verdict']} {g[k]['n_passed']}/{g[k]['n_decided']}"
                               for k in CATEGORIES]]
    table(sl, M, 2.16, CW, rows, [3.5, 1.7, 1.55, 1.55, 1.75, 1.75], row_h=0.33,
          size=10, head_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.CENTER] + [PP_ALIGN.RIGHT] * 4)

    put(textbox(sl, M, 4.16, CW, 0.3),
        "GATE 2 — MAY THE CONTRASTS BE INTERPRETED? (PER CATEGORY × MODEL CELL)",
        size=10.5, bold=True, color=ACCENT, space_after=0)
    cv = N["gate2_seed"]
    rows2 = [["Criterion", "Mechanism"] + [short(h) for h in CELLS]]
    mek = [("1 · Same test pairs everywhere",
            "SHA-256 of the test pairs, written per run", ["PASS"] * 4),
           ("2 · Item universe ⊆ C0", "Set comparison against C0's item table",
            ["PASS"] * 4),
           ("3 · Neither placebo is better than C0",
            "Paired bootstrap; the CI's lower end must not sit above zero",
            ["PASS"] * 4),
           (f"4 · C0 seed-to-seed CV < {cv[CELLS[0]]['threshold']}",
            f"Std ÷ mean of C0's {cv[CELLS[0]]['metric']} over the three seeds",
            [f"{cv[h]['cv']:.3f}" for h in CELLS])]
    for ad, m, hucreler in mek:
        rows2.append([ad, m] + hucreler)
    table(sl, M, 4.47, CW, rows2, [3.3, 3.7, 1.25, 1.25, 1.25, 1.25], row_h=0.38,
          size=10, head_size=9.5,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
    put(textbox(sl, M, 6.45, CW, 0.4),
        f"Gate 2 checks {N['gate2_runs']} runs per cell ({N['n_runs']} in total) and is "
        f"re-evaluated from the run reports, not from a cached verdict.",
        size=10, color=INK2, space_after=0)
    foot(sl, "A2")
    notes(sl, "Iki kapinin tam hali. Kapi 1 dedektorun kullanilip "
              "kullanilamayacagina bakiyor ve kategori basina karar veriyor; Kapi 2 "
              "karsitliklarin yorumlanabilir olup olmadigina bakiyor ve hucre "
              "basina. Ikisi de kosudan once yazildi. Kapi 2'nin her seferinde "
              "kosu raporlarindan yeniden hesaplandigini soylemek onemli - "
              "onbellekten bir 'PASS' okumuyoruz.")


def a03_contrasts(prs, N):
    sl = new_slide(prs)
    head(sl, "A3 · Every contrast, every cell, Recall@10 with 95 % intervals",
         kicker="Appendix")
    rows = [["Contrast", "Meaning"] + [short(h) for h in CELLS]]
    for ad, anahtar, anlam in [
        ("C1 − C4", "c1_c4", "Delete gifts vs delete the same number at random"),
        ("C1b − C4b", "c1b_c4b", "Same, broad label axis"),
        ("C1 − C0", "c1_c0", "Delete gifts vs do nothing"),
        ("C3 − C1", "c3_c1", "Flag vs delete"),
        ("C3 − C0", "c3_c0", "Flag vs do nothing"),
    ]:
        satir = [ad, anlam]
        for hucre in CELLS:
            c = N[anahtar][hucre]
            satir.append(f"{num(c['diff'], 6)}\n{ci(c['ci'], 4)}\n{verdict_en(c['verdict'])}")
        rows.append(satir)
    table(sl, M, 2.02, CW, rows, [1.5, 3.4, 1.75, 1.75, 1.75, 1.75], row_h=0.66,
          size=9, head_size=9.5, first_bold=True,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 4)
    put(textbox(sl, M, 6.16, CW, 0.75),
        f"Paired bootstrap over per-user Recall@10, averaged across seeds "
        f"{', '.join(str(s) for s in N['seeds'])} before resampling; "
        f"{thousands(N['c1_c4'][CELLS[0]]['n_users'])} test users in Toys, "
        f"{thousands(N['c1_c4'][CELLS[2]]['n_users'])} in Grocery. Recall@10 replaced "
        f"NDCG@10 as the primary metric before the runs, because a contaminated list "
        f"is a coverage problem before it is a ranking problem — the change is dated in "
        f"DECISIONS.md and NDCG is still reported alongside.",
        size=10, color=INK2, space_after=0, line=1.15)
    foot(sl, "A3")
    notes(sl, "Butun karsitliklar tek tabloda; soru-cevapta buraya donebiliriz. "
              "Her hucrede uç satir var: fark, guven araligi ve onceden kayitli "
              "yorum kurali. Alttaki notta iki teknik detay: eşli bootstrap "
              "kullanici basina Recall@10 uzerinde ve seed'ler once ortalandi; "
              "ayrica birincil metrik kosulardan ONCE NDCG'den Recall'a "
              "degistirildi - tarihi DECISIONS.md'de, NDCG de hala raporlaniyor.")


def a04_limits(prs, N):
    sl = new_slide(prs)
    head(sl, "A4 · The limitations we would raise ourselves", kicker="Appendix")
    rows = [["#", "Limitation", "What it costs the conclusion"],
            ["1", "Two pre-registered detector gates were missed "
                  f"(macro-F1 {N['human']['macro_f1']:.4f} vs ≥ 0.75; gift precision "
                  f"{N['human']['gift_precision']:.2f} vs ≥ 0.80)",
             "Absolute prevalence is uncertain; effects measure detector-flagged rows"],
            ["2", "The primary metric changed from NDCG@10 to Recall@10 before the runs",
             "Pre-registered, dated, and NDCG is still reported — but it is a change"],
            ["3", f"{N['human']['n_annotators']} human annotator; the planned Fleiss κ "
                  f"was never measured",
             "No reliability estimate for the reference labels themselves"],
            ["4", "C1 deletes rows the detector believes are gifts, including its "
                  "false positives",
             "The treatment is \"delete what the detector flags\", not \"delete gifts\""],
            ["5", "Gifts in a user's validation row are never removed",
             "Direction of the effect on the sequential model is untested"],
            ["6", "Number of epochs actually trained was not recorded for the 72 runs",
             "Seed-42 cells re-run on 21 Sept with the same data to measure it"],
            ["7", "Intervals resample users, not training runs (seed variance excluded)",
             "Toys: every seed agrees in sign · Grocery: seed spread ≈ interval width"],
            ["8", "The placebo matches the number of rows, not the users they come from",
             "Post-hoc: the Toys effect holds even for users neither condition touched"],
            ["9", "\"Lower bound\" assumes false positives are as informative as random rows",
             "Most are purchases for the buyer's own child — not guaranteed"]]
    table(sl, M, 1.95, CW, rows, [0.6, 6.0, 5.3], row_h=0.41, size=9.5, head_size=9,
          first_bold=True)
    put(textbox(sl, M, 6.18, CW, 0.8),
        "Ten further limitations are listed in repo/docs/SONUCLAR.md §7, including: two "
        "categories and two models only; English-language reviews; review date used as a "
        "proxy for purchase date; \"waste\" measured as slot share rather than conversion; "
        "M2 is observational and its n = 0 bucket is structurally different from the "
        "others (a post-hoc sensitivity check is reported alongside the pre-registered "
        "fit); subcategory taken from item metadata, which is itself incomplete.",
        size=10.5, color=INK2, space_after=0, line=1.15)
    foot(sl, "A4")
    notes(sl, "Sinirliliklari juri sormadan biz soyluyoruz. Ilk uç madde "
              "dedektorun dogrulugu ve etiket guvenilirligiyle ilgili ve en "
              "agirlari. Dorduncusu ince ama onemli: C1 aslinda 'hediyeleri sil' "
              "degil 'dedektorun isaretledigini sil' kosulu. Besincisi sirali "
              "modele ozel ve etkisinin yonu sinanmadi. Altincisi bir kayit eksikligi: "
              "seed 42'nin yirmi dort hucresi ayni veriyle yeniden kosuluyor. Yedi ve "
              "sekiz ikinci denetimde bulundu: guven araliklari egitim degiskenligini "
              "kapsamiyor ve plasebo kullanici duzeyinde eslenmedi - ikisi de post-hoc "
              "olculdu ve Toys bulgusunu degistirmiyor. Dokuzuncusu alt sinir okumasinin "
              "varsayimi. Geri kalan on madde SONUCLAR.md yedinci bolumde.")


def a05_repro(prs, N):
    sl = new_slide(prs)
    head(sl, "A5 · How to check any number on any slide", kicker="Appendix")
    bw = (CW - 0.35) / 2
    card(sl, M, 2.0, bw, 2.5, fill=SOFT2, line=RULE)
    tf = textbox(sl, M + 0.2, 2.16, bw - 0.4, 2.2)
    put(tf, "Start here", size=13, bold=True, color=ACCENT, space_after=6)
    for s in ["repo/scripts/demo.py — prints every headline number from the committed "
              "JSONs. Standard library only; no data files needed.",
              "repo/docs/SONUCLAR.md — every result with its interval and its limitation.",
              "repo/docs/DECISIONS.md — every threshold and definition, with the date it "
              "was fixed.",
              "final-report/final-report.md — the written report, with the source JSON "
              "named in each section."]:
        put(tf, "•  " + s, size=10.5, color=INK, space_after=5, line=1.12)

    card(sl, M + bw + 0.35, 2.0, bw, 2.5, fill=SOFT2, line=RULE)
    tf = textbox(sl, M + bw + 0.55, 2.16, bw - 0.4, 2.2)
    put(tf, "Provenance stamped into every artefact", size=13, bold=True, color=ACCENT,
        space_after=6)
    for ad, sur in N["provenance"].items():
        put(tf, f"•  {ad} — code_version {sur}", size=10.5, color=INK, space_after=4)
    put(tf, "A \"-dirty\" suffix means the working tree had uncommitted CODE changes when "
            "the artefact was written. Runs also carry label_source, reportable and a "
            "SHA-256 of the test pairs.",
        size=10, color=INK2, space_after=0, line=1.12)

    put(textbox(sl, M, 4.68, CW, 0.3), "WHAT IS AND IS NOT IN THE REPOSITORY",
        size=10.5, bold=True, color=ACCENT, space_after=0)
    for i, (bas, govde, zemin) in enumerate([
        ("Committed",
         "Code, configs, 441 tests, every result JSON, every figure, the report and "
         "this deck's generator", SOFT),
        ("Deliberately not committed",
         "Review data and model weights (Kaggle datasets are private), the 500 "
         "human-labelled rows, any verbatim review text, any personal data",
         CAUTION_SOFT)]):
        x = M + i * (bw + 0.35)
        card(sl, x, 5.0, bw, 1.0, fill=zemin, line=RULE)
        tf = textbox(sl, x + 0.2, 5.11, bw - 0.4, 0.8)
        put(tf, bas, size=11.5, bold=True,
            color=ACCENT if i == 0 else CAUTION, space_after=4)
        put(tf, govde, size=10, color=INK, space_after=0, line=1.1)
    put(textbox(sl, M, 6.18, CW, 0.7),
        "Two virtual environments: .venv for the pipeline (polars 1.44, numpy 2.5) and "
        ".venv-recbole for the experiment (numpy < 2, because RecBole 1.2.0 still uses "
        "np.float_). Slides were generated by presentation/build_deck.py, which reads the "
        "same JSONs — so the deck cannot drift from the artefacts.",
        size=10.5, color=INK2, space_after=0, line=1.15)
    foot(sl, "A5")
    notes(sl, "Son ek slayt: her sayiyi nasil kontrol edeceginizi anlatiyor. "
              "Baslangiç noktasi demo betigi - veri dosyasi olmadan, standart "
              "kutuphaneyle, commit'li JSON'lardan butun basliklari basiyor. "
              "Sagdaki provenans kutusu her artefaktin hangi kod surumuyle "
              "uretildigini gosteriyor; 'dirty' eki calisma agacinda commit "
              "edilmemis KOD degisikligi vardi demek. Alt tabloda neyin commit "
              "edilip neyin edilmedigi var: veri ve model agirliklari kasitli "
              "olarak depoda yok. Bu deck'i ureten betik de ayni JSON'lari "
              "okuyor, yani slaytlar artefaktlardan sapamaz.")


SLIDES = [s01_title, s02_problem, s03_cost, s04_questions, s05_pipeline, s06_detector,
          s07_honesty, s08_distill, s09_rq1, s10_placebo, s11_gate2, s12_rq2,
          s13_rq2_net, s14_rq3, s15_rq4_halflife, s16_rq4_waste, s17_close,
          a01_conditions, a02_gates, a03_contrasts, a04_limits, a05_repro]


def build(out: Path) -> Path:
    N = load()
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    # python-pptx sablonu kendi ustverisini tasiyor (`Steve Canny`, 2013). Teslim
    # edilecek dosyada durmasi yanlis; isim alanlari BOS birakiliyor - ekip adlari
    # slaytta da yer tutucu.
    cp = prs.core_properties
    cp.title = f"{TITLE} — {SUBTITLE}"
    cp.subject = "SIC AI 17 Capstone, Group 2 — final presentation"
    cp.comments = ("Generated by presentation/build_deck.py; every number is read from "
                   "repo/reports/results/*.json.")
    cp.author = ""
    cp.last_modified_by = ""
    cp.revision = 1
    for fn in SLIDES:
        fn(prs, N)
    prs.save(str(out))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "SIC_AI_17_Group_2.pptx")
    args = ap.parse_args(argv)
    yol = build(args.out)
    print(f"{yol.name} yazildi: {len(SLIDES)} slayt, "
          f"{yol.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
