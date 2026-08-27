"""LLM annotation kosucusu testleri.

Burada model DOGRULANMIYOR - kosucu dogrulaniyor. Sinanan sey: satir kaybolmuyor
mu, kesinti sonrasi devam ediyor mu, bozuk cikti sessizce yutuluyor mu,
prefix caching'in dayanagi olan ortak onek gercekten var mi.

Bunlarin hepsi 47.200 satirlik bir kosuda fark edilmesi PAHALI olan arizalar:
her biri ancak sonuclara bakildiginda ortaya cikardi.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from gift_contamination.analysis.keyword_scan import scan_category
from gift_contamination.config import Config
from gift_contamination.data.sampling import annotation_sample_path, build_sample
from gift_contamination.detection.llm_annotate import (
    FAILED_ROW,
    KEY_COLUMNS,
    LABEL_COLUMNS,
    MAX_REVIEW_CHARS,
    StubBackend,
    annotate,
    annotation_path,
    annotation_stats_path,
    build_prompts,
    parse_one,
    shard_dir,
    shared_prefix_len,
)
from gift_contamination.detection.prompting import load_prompt
from gift_contamination.detection.schema import PurchaseType


@pytest.fixture
def sampled(cfg: Config) -> Config:
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    return cfg


# ------------------------------------------------------------------ uctan uca
def test_every_input_row_gets_a_label(sampled: Config):
    """Girdi satir sayisi = cikti satir sayisi.

    Eksik satir yaygınlık hesabini sessizce kaydirir: 11.800'den 11.600'e dusen
    bir cikti yalnizca 'biraz az' degil, hangi satirlarin dustugune bagli olarak
    YANLI bir orandir.
    """
    n_in = pl.read_parquet(annotate(sampled, "pilot", backend_name="stub")).height
    n_sample = pl.read_parquet(annotation_sample_path(sampled, "pilot")).height

    assert n_in == n_sample


def test_output_carries_no_review_text(sampled: Config):
    """Cikti birebir review metni TASIMAZ (CLAUDE.md bolum 8.1).

    Metin ornekleme parquet'inde duruyor; ikilemek hem gereksiz hem de birebir
    metin tasiyan ikinci bir dosya yaratir.
    """
    df = pl.read_parquet(annotate(sampled, "pilot", backend_name="stub"))

    assert "text" not in df.columns
    assert set(KEY_COLUMNS) <= set(df.columns)
    assert set(LABEL_COLUMNS) <= set(df.columns)


def test_run_records_prompt_version_and_model(sampled: Config):
    """Hangi etiketin hangi prompt'la uretildigi kayittan okunabilmeli."""
    df = pl.read_parquet(annotate(sampled, "pilot", backend_name="stub"))

    assert df["prompt_version"].n_unique() == 1
    assert df["prompt_version"][0] == load_prompt(sampled).version
    assert df["model"][0] == "stub-model"


# ------------------------------------------------------------- kesinti/devam
def test_interrupted_run_resumes_instead_of_restarting(sampled: Config):
    """Yarim kalan kosu bastan baslamamali.

    Kaggle oturumu 12 saatte kesilir. 70 dakikalik bir kosuyu bir kopmada
    bastan almak kabul edilemez; parcalar diskte duruyorsa atlanmali.
    """
    annotate(sampled, "pilot", backend_name="stub")
    shards = shard_dir(sampled, "pilot")
    parts = sorted(shards.glob("part_*.parquet"))
    assert len(parts) > 1, "shard_rows=3 ile birden fazla parca olusmaliydi"

    # Son parcayi sil: kosu tam orada kesilmis gibi.
    dropped = pl.read_parquet(parts[-1])
    parts[-1].unlink()
    annotation_path(sampled, "pilot").unlink()

    calls: list[int] = []

    class CountingStub(StubBackend):
        def generate(self, prompts, reviews):
            calls.append(len(prompts))
            return super().generate(prompts, reviews)

    from gift_contamination.detection import llm_annotate

    llm_annotate.BACKENDS["counting"] = CountingStub
    try:
        out = pl.read_parquet(annotate(sampled, "pilot", backend_name="counting"))
    finally:
        del llm_annotate.BACKENDS["counting"]

    # Yalnizca silinen parcanin satirlari yeniden uretilmis olmali.
    assert sum(calls) == dropped.height
    assert out.height == pl.read_parquet(
        annotation_sample_path(sampled, "pilot")
    ).height


def test_force_discards_the_shards(sampled: Config):
    annotate(sampled, "pilot", backend_name="stub")
    n_before = len(list(shard_dir(sampled, "pilot").glob("part_*.parquet")))

    annotate(sampled, "pilot", backend_name="stub", force=True)

    assert len(list(shard_dir(sampled, "pilot").glob("part_*.parquet"))) == n_before


# ------------------------------------------------------------------ parse yolu
def test_unparseable_output_is_written_not_dropped():
    """Bozuk cikti satiri dusurmez; `unclear` + parse_ok=False olarak yazilir."""
    record, ok, downgraded = parse_one("bu JSON degil", "herhangi bir review")

    assert ok is False
    assert downgraded is False
    assert record == FAILED_ROW
    assert record["purchase_type"] == PurchaseType.UNCLEAR.value


def test_span_not_in_text_is_downgraded():
    """Uydurma kanit `unclear`'a duser (CLAUDE.md bolum 9)."""
    raw = json.dumps({
        "purchase_type": "gift_given", "confidence": "high", "recipient": "child",
        "occasion": "birthday", "evidence_span": "metinde olmayan bir cumle",
    })

    record, ok, downgraded = parse_one(raw, "I bought this for myself and love it.")

    assert ok is True
    assert downgraded is True
    assert record["purchase_type"] == PurchaseType.UNCLEAR.value


def test_verbatim_span_survives():
    review = "I got this for my daughter's birthday and she loved it."
    raw = json.dumps({
        "purchase_type": "gift_given", "confidence": "high", "recipient": "child",
        "occasion": "birthday", "evidence_span": "for my daughter's birthday",
    })

    record, ok, downgraded = parse_one(raw, review)

    assert (ok, downgraded) == (True, False)
    assert record["purchase_type"] == PurchaseType.GIFT_GIVEN.value


def test_parse_failures_are_counted_in_the_report(sampled: Config):
    """Basarisizlik orani raporlanmali; sessiz atlamak yok."""

    class BrokenStub(StubBackend):
        def generate(self, prompts, reviews):
            from gift_contamination.detection.llm_annotate import Generation

            return [Generation("{bozuk", 3) for _ in prompts]

    from gift_contamination.detection import llm_annotate

    llm_annotate.BACKENDS["broken"] = BrokenStub
    try:
        annotate(sampled, "pilot", backend_name="broken")
    finally:
        del llm_annotate.BACKENDS["broken"]

    report = json.loads(
        annotation_stats_path(sampled, "pilot").read_text(encoding="utf-8")
    )
    assert report["quality"]["parse_fail_rate"] == 1.0
    assert report["quality"]["n_retried"] == report["meta"]["n_rows"]


# -------------------------------------------------------------- prefix caching
def test_prompts_share_a_long_prefix(sampled: Config):
    """Prefix caching'in dayanagi olculur, VARSAYILMAZ.

    Ortak onek sifira yakinsa caching hicbir ise yaramaz ve 47.200 satirda
    ~116M gereksiz prefill token uretilir. Bu testin kirilmasi, prompt'a satir
    basina degisen bir sey (tarih, satir no, rastgele ornek) sizdigini gosterir.
    """
    df = pl.read_parquet(
        annotation_sample_path(sampled, "pilot")
    )
    prompt = load_prompt(sampled)

    prompts, _, _ = build_prompts(df, prompt, StubBackend(sampled, prompt))
    shared = shared_prefix_len(prompts)

    assert shared > 2000, f"ortak onek yalnizca {shared} karakter"
    assert shared >= len(prompt.system) * 0.9


def test_shared_prefix_is_zero_when_prompts_diverge():
    assert shared_prefix_len(["abc", "xyz"]) == 0
    assert shared_prefix_len(["same-start-A", "same-start-B"]) == len("same-start-")


# ------------------------------------------------------------------- kirpma
def test_long_reviews_are_truncated_and_counted(sampled: Config):
    """max_model_len'i asan review kosu ortasinda patlamamali, kirpilmali."""
    prompt = load_prompt(sampled)
    df = pl.DataFrame({
        "text": ["x" * (MAX_REVIEW_CHARS + 500)],
        "title": ["t"], "category": ["All_Beauty"], "product_title": ["p"],
    })

    _, reviews, n_truncated = build_prompts(df, prompt, StubBackend(sampled, prompt))

    assert n_truncated == 1
    assert len(reviews[0]) == MAX_REVIEW_CHARS


def test_missing_product_title_gets_a_placeholder(sampled: Config):
    """Bos urun adi prompt'a 'Product: ' diye girmemeli.

    Metadata join'i 47.200 satirda 2 bos baslik birakiyor; bos string LLM'in
    bir sinyal sanabilecegi bir bosluk uretir.
    """
    prompt = load_prompt(sampled)
    df = pl.DataFrame({
        "text": ["kisa bir review"], "title": ["t"],
        "category": ["All_Beauty"], "product_title": [None],
    })

    prompts, _, _ = build_prompts(df, prompt, StubBackend(sampled, prompt))

    assert "(unknown)" in prompts[0]


# ------------------------------------------------------------------- idempotans
def test_second_run_skips_without_force(sampled: Config):
    first = annotate(sampled, "pilot", backend_name="stub")
    stamp = first.stat().st_mtime_ns

    annotate(sampled, "pilot", backend_name="stub")

    assert first.stat().st_mtime_ns == stamp


def test_report_separates_frames(sampled: Config):
    """Yaygınlık yalnizca `main`den okunur; boost cerceveleri ayri sayilmali."""
    annotate(sampled, "pilot", backend_name="stub")

    report = json.loads(
        annotation_stats_path(sampled, "pilot").read_text(encoding="utf-8")
    )

    assert "main" in report["by_frame"]
    assert set(report["by_frame"]) <= {"main", "boost", "boost_received"}


def test_stub_output_cannot_be_mistaken_for_a_real_run(sampled: Config):
    """Kuru kosu ciktisi diskte kalirsa gercek kosu SESSIZCE atlanmamali.

    Bu, sonuclara bakilana kadar fark edilmeyecek bir ariza olurdu: elde taklit
    etiketlerle dolu bir parquet, raporda gercekmis gibi.
    """
    annotate(sampled, "pilot", backend_name="stub")

    assert pl.read_parquet(annotation_path(sampled, "pilot"))["backend"][0] == "stub"
    with pytest.raises(RuntimeError, match="stub"):
        annotate(sampled, "pilot", backend_name="vllm")
