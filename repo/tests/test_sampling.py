"""Ornekleme testleri.

Buradaki asil risk dogruluk degil YANLILIK. Ornek yanliysa kod hatasiz calisir,
sayilar makul gorunur ve yaygınlık tahmini sessizce yanlis cikar. En kritik test
`test_main_frame_is_unbiased` ile `test_frames_are_disjoint`: ikisi birden
`boost` satirlarinin yaygınlık hesabina sizmasini engelliyor. Pilot kategoride
sizarsa oran %2.13 yerine %15.01 gorunuyor - yedi kat.
"""

from __future__ import annotations

import polars as pl
import pytest

from gift_contamination.analysis.keyword_scan import PROXY_COL, scan_category
from gift_contamination.config import Config
from gift_contamination.data.sampling import (
    FRAME_BOOST,
    FRAME_MAIN,
    FRAME_RECEIVED,
    LABELS,
    SUPPORTED_STRATA,
    TRIAL_STRATA,
    _check_strata,
    _stratum_seed,
    _trial_strata,
    _draw_boost,
    _draw_main,
    _length_bucket,
    allocate_by_share,
    allocation_path,
    annotation_sample_path,
    build_sample,
    build_trial,
    build_trial_source,
    build_validation,
    proportional_allocation,
    trial_ids_path,
    validation_ids_path,
)
from gift_contamination.detection.llm_annotate import annotation_path
from gift_contamination.utils.io import read_json, write_json

SEED = 42


# ------------------------------------------------------------------ tahsis
def test_allocation_sums_to_request():
    counts = {"a": 500, "b": 300, "c": 200}

    alloc = proportional_allocation(counts, 100)

    assert sum(alloc.values()) == 100


def test_allocation_is_proportional():
    counts = {"a": 500, "b": 300, "c": 200}

    alloc = proportional_allocation(counts, 100)

    assert (alloc["a"], alloc["b"], alloc["c"]) == (50, 30, 20)


def test_allocation_never_exceeds_cell_population():
    """Kucuk hucreden var olandan fazlasi istenemez."""
    counts = {"tiny": 2, "big": 998}

    alloc = proportional_allocation(counts, 500)

    assert alloc["tiny"] <= 2
    assert sum(alloc.values()) == 500


def test_allocation_caps_at_population():
    counts = {"a": 3, "b": 3}

    alloc = proportional_allocation(counts, 100)

    assert sum(alloc.values()) == 6


def test_allocation_gives_small_cells_a_chance():
    """En buyuk kalan yontemi kucuk hucreleri sistematik olarak sifirlamamali.

    Duz floor() ile 1/1000'lik bir hucre her zaman 0 alir ve o katman ornekte
    hic temsil edilmez.
    """
    counts = {f"c{i}": 1 for i in range(10)}

    alloc = proportional_allocation(counts, 5)

    assert sum(alloc.values()) == 5
    assert sum(1 for v in alloc.values() if v == 1) == 5


def test_allocation_handles_degenerate_input():
    assert proportional_allocation({}, 10) == {}
    assert proportional_allocation({"a": 0}, 10) == {"a": 0}
    assert proportional_allocation({"a": 5}, 0) == {"a": 0}


def test_allocation_is_deterministic():
    counts = {"a": 7, "b": 7, "c": 7}

    assert proportional_allocation(counts, 10) == proportional_allocation(counts, 10)


# --------------------------------------------------- sabit paya gore tahsis
def test_share_allocation_hits_the_exact_total():
    """round() ile pay pay yuvarlamak toplami kaydirir.

    Python bankaci yuvarlamasi kullaniyor: round(12.5) == 12. 200 istenen bir
    sette kategori basina 49 satir cikar ve dosya adi 200 derken icinde 196
    satir olur.
    """
    alloc = allocate_by_share({"proxy": 0.5, "speculative": 0.25, "unflagged": 0.25}, 50)

    assert sum(alloc.values()) == 50
    assert alloc["proxy"] == 25


def test_share_allocation_respects_caps():
    alloc = allocate_by_share({"a": 0.5, "b": 0.5}, 100, caps={"a": 10, "b": 500})

    assert alloc["a"] == 10
    assert sum(alloc.values()) == 100


def test_share_allocation_stops_when_every_cap_is_reached():
    alloc = allocate_by_share({"a": 0.5, "b": 0.5}, 100, caps={"a": 3, "b": 4})

    assert alloc == {"a": 3, "b": 4}


def test_equal_shares_split_evenly():
    roles = {r: 0.25 for r in ("pilot", "high", "mid", "low")}

    alloc = allocate_by_share(roles, 200)

    assert sum(alloc.values()) == 200
    assert set(alloc.values()) == {50}


# ------------------------------------------------------------- uzunluk kovasi
@pytest.mark.parametrize(
    ("n_words", "expected"), [(1, "<15"), (14, "<15"), (15, "15-40"), (39, "15-40"), (40, ">=40"), (900, ">=40")]
)
def test_length_bucket_boundaries(n_words: int, expected: str):
    got = (
        pl.DataFrame({"n_words": [n_words]})
        .select(_length_bucket([15, 40]))
        .item()
    )

    assert got == expected


# --------------------------------------------------------------- cekim mantigi
def _synthetic(n: int = 4000, rng_seed: int = 0) -> pl.DataFrame:
    """Bilinen yaygınlıga sahip sentetik havuz.

    Vekil bayragi katmanla ILISKILI uretiliyor (rating 5'te uc kati sik):
    iliskisiz olsaydi her ornekleme semasi yansiz cikar ve test hicbir sey
    kanitlamazdi.

    Degerler GERCEK rastgele - modulo aritmetigiyle uretilmiyor. `i % 100` ile
    `i % 5` deterministik olarak iliskili oldugundan, modulo tabanli bir havuzda
    katman ICI varyans kayboluyor ve olcum yansizligi degil kurgunun artefaktini
    yansitiyor.
    """
    import random

    rng = random.Random(rng_seed)
    rows = []
    for i in range(n):
        rating = rng.randint(1, 5)
        is_proxy = rng.random() < (0.20 if rating == 5 else 0.06)
        # Hediye ALMIS satirlar seyrek ama sifir degil - korpusta %0.08-0.28.
        is_received = (not is_proxy) and rng.random() < 0.04
        rows.append(
            {
                "row_id": i,
                "rating": float(rating),
                "month": rng.randint(1, 12),
                "n_words": rng.randint(5, 80),
                "kw_gift_evidence": is_proxy or is_received,
                "kw_gift_speculative": rng.random() < 0.03,
                "kw_gift_received": is_received,
                # Vekilin TANIMI: kanit var VE alinmis degil.
                PROXY_COL: is_proxy,
            }
        )
    return (
        pl.DataFrame(rows)
        .with_columns(_length_bucket([15, 40]))
        .with_columns(
            pl.concat_str(
                [
                    pl.col("month").cast(pl.String),
                    pl.col("rating").cast(pl.Int64).cast(pl.String),
                    pl.col("len_bucket"),
                ],
                separator="|",
            ).alias("stratum_id")
        )
        .drop("len_bucket")
    )


def test_main_frame_is_unbiased():
    """Orantili tahsis kendinden agirlikli: duz ortalama gercek orani vermeli.

    Yansizlik BEKLENEN DEGERIN ozelligi, tek bir cekimin degil - tek cekim
    orneklem hatasi kadar sapar. O yuzden bircok popülasyon uzerinden ortalama
    sapmaya bakiyoruz. Tek cekimde ~0.009'luk SRS standart hatasi normalken,
    ortalama sapma sifira yakin olmak zorunda.
    """
    deviations = []
    for pop_seed in range(12):
        thin = _synthetic(rng_seed=pop_seed)
        main, _ = _draw_main(thin, 1000, SEED)
        assert main.height == 1000
        deviations.append(main[PROXY_COL].mean() - thin[PROXY_COL].mean())

    mean_deviation = sum(deviations) / len(deviations)

    assert mean_deviation == pytest.approx(0.0, abs=0.003), (
        f"ortalama sapma {mean_deviation:+.4f} - tahmin edici yanli"
    )
    assert max(abs(d) for d in deviations) < 0.03, "tek cekim sapmasi cok buyuk"


def test_main_frame_preserves_stratum_proportions():
    thin = _synthetic()
    main, report = _draw_main(thin, 1000, SEED)

    for cell in report["cells"]:
        share_pop = cell["population"] / report["population"]
        share_alloc = cell["allocated"] / report["n_drawn"]
        assert share_alloc == pytest.approx(share_pop, abs=0.01)


def test_boost_frame_is_all_proxy():
    thin = _synthetic()
    main, _ = _draw_main(thin, 500, SEED)

    boost = _draw_boost(thin, main, 100, SEED)

    assert boost.height == 100
    assert boost[PROXY_COL].all()


def test_frames_are_disjoint():
    """Ayni satir iki cercevede birden gecerse iki kez etiketlenir ve sayilir."""
    thin = _synthetic()
    main, _ = _draw_main(thin, 500, SEED)
    boost = _draw_boost(thin, main, 100, SEED)

    assert set(main["row_id"]) & set(boost["row_id"]) == set()


def test_pooling_the_frames_inflates_the_rate():
    """Cercevelerin karistirilmasinin ZARARLI oldugunu sayiyla gosteren test.

    Bu bir regresyon kalkani: biri `sample_frame` ayrimini kaldirmaya kalkarsa,
    kaybedilen seyin ne oldugu burada yaziyor.
    """
    thin = _synthetic()
    truth = thin[PROXY_COL].mean()
    main, _ = _draw_main(thin, 500, SEED)
    boost = _draw_boost(thin, main, 100, SEED)

    pooled = pl.concat([main, boost])[PROXY_COL].mean()

    assert main[PROXY_COL].mean() == pytest.approx(truth, abs=0.03)
    assert pooled > truth * 1.5, "havuzlama orani sismeliydi; test kurgusu bozuk"


def test_same_seed_draws_same_rows():
    thin = _synthetic()

    first, _ = _draw_main(thin, 300, SEED)
    second, _ = _draw_main(thin, 300, SEED)

    assert first["row_id"].to_list() == second["row_id"].to_list()


def test_different_seed_draws_different_rows():
    thin = _synthetic()

    first, _ = _draw_main(thin, 300, SEED)
    other, _ = _draw_main(thin, 300, SEED + 1)

    assert first["row_id"].to_list() != other["row_id"].to_list()


# ---------------------------------------------------- katman kapsamı (Bulgu 2)
def test_trial_strata_partition_the_frame():
    """Dort katman AYRIK ve TUKETICI olmali: her satir tam olarak birinde.

    2026-08-27 denetiminin buldugu hata: uc katmanli eski halde
    `kw_gift_received` satirlari HICBIRINE dusmuyordu. 200 satirlik deneme
    gecisinde "hediye ALMIS" vakasindan sifir ornek cikti - oysa
    "receiving a gift is not giving one" prompt'un uc kritik ayrimindan biri.
    """
    thin = _synthetic()

    pools = _trial_strata(thin)

    ids = [set(p["row_id"].to_list()) for p in pools.values()]
    union = set().union(*ids)
    assert len(union) == sum(len(s) for s in ids), "katmanlar cakisiyor"
    assert union == set(thin["row_id"].to_list()), "kapsanmayan satir var"


def test_received_rows_are_reachable_by_the_trial_set():
    """Regresyon kalkani: 'received' katmani bos donerse deneme seti o
    vakayi hic gormez ve prompt kurali sinanmadan dogrulanmis sayilir."""
    thin = _synthetic()

    received = _trial_strata(thin)["received"]

    assert received.height > 0
    assert received["kw_gift_received"].all()
    assert not received[PROXY_COL].any(), "vekil received satirlarini dislamali"


def test_trial_shares_sum_to_one():
    assert sum(TRIAL_STRATA.values()) == pytest.approx(1.0)


# --------------------------------------------------- katman seed'i (Bulgu 3)
def test_different_strata_draw_different_positions():
    """Ortak seed ayni boyutlu havuzlarda BIREBIR ayni konumlari sectiriyordu.

    2026-08-27 denetimi: iki farkli katman da [275, 607, 687, 702, 851]
    konumlarini secmisti. Nokta tahmini yansiz kaliyordu ama katmanlar arasi
    bagimsizlik yoktu - ve bootstrap guven araliklari bunu varsayiyor.
    """
    a = _stratum_seed(SEED, "5|2|<15")
    b = _stratum_seed(SEED, "9|2|>=40")

    assert a != b

    pool = pl.DataFrame({"row_id": list(range(400))})
    pos_a = sorted(pool.sample(5, seed=a, shuffle=True)["row_id"].to_list())
    pos_b = sorted(pool.sample(5, seed=b, shuffle=True)["row_id"].to_list())
    assert pos_a != pos_b, "turetilmis seed'ler ayni konumlari veriyor"


def test_stratum_seed_is_stable_across_processes():
    """`hash()` PYTHONHASHSEED ile degisir; crc32 degismez.

    hash() kullanilsaydi ornek her sureçte farkli cikardi ve `seed: 42` ile
    yeniden uretilebilirlik iddiasi yanlis olurdu.
    """
    assert _stratum_seed(42, "1|5|<15") == _stratum_seed(42, "1|5|<15")
    assert _stratum_seed(42, "1|5|<15") == 439307199  # crc32, surece bagli degil
    assert 0 <= _stratum_seed(42, "x") < 2**31 - 1


# ------------------------------------------------- config <-> kod (Bulgu 6a)
def test_config_strata_must_match_the_code(cfg: Config):
    """`sampling.strata` degistirilip cikti degismemesi sessizce gecmemeli."""
    _check_strata(cfg)  # mevcut config gecmeli

    cfg._data["sampling"]["strata"] = ["month", "rating"]
    with pytest.raises(ValueError, match="sampling.strata"):
        _check_strata(cfg)
    cfg._data["sampling"]["strata"] = list(SUPPORTED_STRATA)


# ------------------------------------------------------------------ uctan uca
def test_build_sample_end_to_end(cfg: Config):
    scan_category(cfg, "pilot")

    dest = build_sample(cfg, "pilot")
    sample = pl.read_parquet(dest)

    assert dest == annotation_sample_path(cfg, "pilot")
    assert sample.height > 0
    assert set(sample["sample_frame"]).issubset(
        {FRAME_MAIN, FRAME_BOOST, FRAME_RECEIVED}
    )
    # Metin tasinmali: LLM'e gidecek olan bu.
    assert sample["text"].null_count() == 0
    # row_id benzersiz -> cerceveler ayrik
    assert sample["row_id"].n_unique() == sample.height


def test_sample_carries_the_product_name(cfg: Config):
    """Urun adi ornege girmeli - LLM prompt'unun yeni girdisi bu.

    Fixture'daki `i5` urununun basligi kasitli olarak BOS: eksik urun adi yolu
    da gecilsin ve sayac sinansin. Kacirma "kapsam tamdir" diye varsayilmaz,
    olculup tahsis raporuna yazilir.
    """
    import json

    scan_category(cfg, "pilot")
    sample = pl.read_parquet(build_sample(cfg, "pilot"))
    report = json.loads(allocation_path(cfg, "pilot").read_text(encoding="utf-8"))

    assert {"parent_asin", "product_title", "product_category"} <= set(sample.columns)
    assert sample["product_title"].null_count() == 0, "null degil bos string olmali"
    # i5'in basligi bos: sayac onu gormeli
    n_blank = int((sample["product_title"].str.strip_chars() == "").sum())
    assert report["n_missing_product_title"] == n_blank
    # ...ama hepsi bos olmamali, yoksa join hic calismamis demektir
    assert n_blank < sample.height


def test_build_sample_is_idempotent(cfg: Config):
    """Cikti varsa yeniden hesaplanmamali (CLAUDE.md bolum 7, birinci yari)."""
    scan_category(cfg, "pilot")
    first = pl.read_parquet(build_sample(cfg, "pilot"))

    second = pl.read_parquet(build_sample(cfg, "pilot"))

    assert first.equals(second)


def test_build_sample_recompute_is_deterministic(cfg: Config):
    """`--force` ile yeniden hesaplama AYNI sonucu vermeli.

    Idempotency iddiasinin ikinci yarisi ve asil onemli olan bu. Onceki test
    `should_skip` yoluna girip dosyayi kendisiyle karsilastiriyor; yeniden
    hesaplamanin deterministik oldugunu HIC sinamiyordu (2026-08-27 denetimi).
    """
    scan_category(cfg, "pilot")
    first = pl.read_parquet(build_sample(cfg, "pilot", force=True))

    second = pl.read_parquet(build_sample(cfg, "pilot", force=True))

    assert first.equals(second)


def test_trial_ids_are_recorded_for_exclusion(cfg: Config):
    """Prompt bu satirlarda ayarlanacak; Hafta 4 dogrulamasi onlari dislamali."""
    import json

    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")

    build_trial(cfg, ["pilot"], 4)
    meta = json.loads(trial_ids_path(cfg).read_text(encoding="utf-8"))

    assert meta["exclude_from_validation"] is True
    assert sum(len(v) for v in meta["excluded"].values()) == meta["n"]
    for ids in meta["excluded"].values():
        assert ids == sorted(ids)


def test_exclusion_key_is_scoped_by_category(cfg: Config):
    """`row_id` her kategoride 0'dan basliyor: TEK BASINA benzersiz DEGIL.

    2026-08-27 denetimi: dort kategorinin ornekleri arasinda 78 ortak row_id
    olculdu. Duz bir row_id listesiyle dislama yapmak baska kategorilerde masum
    satirlari da atar - ve Hafta 4'un dogrulama seti tam olarak bu listeyi
    kullanacak.
    """
    import json

    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    build_trial(cfg, ["pilot"], 4)

    meta = json.loads(trial_ids_path(cfg).read_text(encoding="utf-8"))

    assert "row_ids" not in meta, "duz row_id listesi geri gelmis - anahtar belirsiz"
    assert set(meta["excluded"]) == {cfg.category_slug("pilot")}
    assert "(category, row_id)" in meta["note"]


# ------------------------------------- deneme setinin LLM'e verilecek hali
def test_trial_source_recovers_the_product_title(cfg: Config):
    """Deneme CSV'si urun adi TASIMIYOR; LLM uretimde onu goruyor.

    Getirilmezse Kapi 1'in uyum sayisi, prompt'un gercekte kostugu girdiden
    farkli bir girdiyi olcerdi.
    """
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    build_trial(cfg, ["pilot"], 4)

    out = pl.read_parquet(build_trial_source(cfg, 4))

    assert out.height == 4
    assert "product_title" in out.columns
    assert "trial_id" in out.columns
    # Fixture metadata'sinda `i5`in basligi kasitli olarak bos; digerleri dolu.
    assert (out["product_title"].str.strip_chars() != "").any()


def test_trial_source_text_matches_the_labeled_csv(cfg: Config):
    """`row_id` korpus capinda kararli olmali.

    Insan etiketi bir review'a bagli; join baska bir satiri getirirse etiket
    baska bir metne baglanir ve uyum sayisi sessizce anlamsizlasir.
    """
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    csv = pl.read_csv(build_trial(cfg, ["pilot"], 4))

    out = pl.read_parquet(build_trial_source(cfg, 4))

    joined = csv.select("trial_id", pl.col("text").alias("csv_text")).join(
        out.select("trial_id", "text"), on="trial_id", how="inner"
    )
    assert joined.height == 4
    assert (joined["csv_text"] == joined["text"]).all()


def test_trial_source_refuses_a_row_id_that_drifted(cfg: Config):
    """Kaymis bir `row_id` SESSIZCE gecmemeli."""
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    csv_path = build_trial(cfg, ["pilot"], 4)
    # Tek satirin metnini boz: korpusla artik tutmuyor.
    bozuk = pl.read_csv(csv_path).with_columns(
        pl.when(pl.col("trial_id") == 1)
        .then(pl.lit("baska bir review metni"))
        .otherwise(pl.col("text"))
        .alias("text")
    )
    bozuk.write_csv(csv_path)

    with pytest.raises(RuntimeError, match="tutmuyor"):
        build_trial_source(cfg, 4, force=True)


# ------------------------------------------------ Hafta 4: dogrulama seti


def _fake_annotations(
    cfg: Config, role: str = "pilot", *, backend: str = "vllm",
    labels: list[str] | None = None,
) -> pl.DataFrame:
    """Ornekten sahte bir LLM cikti parquet'i uretir.

    Gercek annotation ciktisi review METNINI tasimiyor (boyut icin dusuruldu);
    fixture de o sozlesmeyi taklit etmeli, yoksa `build_validation`in ornege
    geri join atmasi test edilmemis kalir.
    """
    sample = pl.read_parquet(annotation_sample_path(cfg, role))
    n = sample.height
    labels = labels or [LABELS[i % len(LABELS)] for i in range(n)]
    out = sample.select(
        "row_id", "category", "sample_frame", "month", "rating", "n_words",
    ).with_columns(
        pl.Series("purchase_type", labels[:n]),
        pl.lit("high").alias("confidence"),
        pl.lit("child").alias("recipient"),
        pl.lit("none").alias("occasion"),
        pl.lit("kanit").alias("evidence_span"),
        pl.lit(backend).alias("backend"),
    )
    dest = annotation_path(cfg, role)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dest)
    return out


def _prepare(cfg: Config, **kw) -> pl.DataFrame:
    scan_category(cfg, "pilot")
    build_sample(cfg, "pilot")
    return _fake_annotations(cfg, **kw)


def test_validation_strata_come_from_the_llm_label(cfg: Config):
    """Katmanlar sozcuksel vekilden degil, modelin KENDI etiketinden cikmali.

    Olculecek sey detektorun dogrulugu; katman da onun kararlarini kapsamali.
    Vekile gore katmanlarsak modelin vekilden ayrildigi yerleri hic gormeyiz -
    ki tam olarak orasi olculmek istenen yer.
    """
    ann = _prepare(cfg)
    hedef = {k: v for k, v in cfg.get("validation.strata").items() if v}

    out = pl.read_csv(build_validation(cfg, ["pilot"], 5))

    cekilen = dict(out.group_by("val_stratum").len().iter_rows())
    assert cekilen == hedef
    # Katman etiketi gercekten modelin etiketi mi
    eslesme = out.join(
        ann.select("row_id", "purchase_type"), on="row_id", how="inner"
    )
    assert (eslesme["val_stratum"] == eslesme["purchase_type"]).all()


def test_validation_excludes_the_prompt_trial_rows(cfg: Config):
    """Prompt o satirlar okunarak yazildi; ayni satirlarla dogrulamak,
    prompt'u kendi test setine fit etmek olur."""
    _prepare(cfg)
    trial = pl.read_csv(build_trial(cfg, ["pilot"], 4))
    yasak = set(trial["row_id"].to_list())

    out = pl.read_csv(build_validation(cfg, ["pilot"], 5))

    assert not (set(out["row_id"].to_list()) & yasak)
    rapor = read_json(validation_ids_path(cfg))
    assert rapor["n_dropped_as_trial_rows"]["pilot"] == len(yasak)


def test_validation_reports_the_drop_it_computed_not_the_one_it_claims(cfg: Config):
    """Hesaplanmamis bir iddia, kod degisince sessizce yalan soyler.

    2026-08-27 denetiminin 7. bulgusu aynen buydu (`frames_disjoint`).
    Deneme setinin satirlari ornegin ICINDE olmayabilir - o zaman DUSEN sayi
    sifirdir, mevcut ID sayisi degil.
    """
    _prepare(cfg)
    # Deneme kaydini elle kur: hicbiri ornekte olmayan row_id'ler.
    write_json(
        {"excluded": {"Test_Cat": [90001, 90002, 90003]}},
        trial_ids_path(cfg),
        None,
    )

    build_validation(cfg, ["pilot"], 5)

    rapor = read_json(validation_ids_path(cfg))
    assert rapor["trial_ids_available"] == 3
    # Bakildi ve SIFIR dustu - anahtarin yoklugu bunu soyleyemezdi.
    assert rapor["n_dropped_as_trial_rows"] == {"pilot": 0}


def test_validation_exclusion_is_scoped_by_category(cfg: Config):
    """`row_id` her kategoride 0'dan basliyor - dislama CIFT anahtarli olmali.

    Duz bir row_id listesi baska kategorilerde masum satirlari da atardi
    (olculdu: dort kategori arasinda 78 ortak deger).
    """
    ann = _prepare(cfg)
    baska = sorted(ann["row_id"].to_list())[:2]
    write_json({"excluded": {"Baska_Kategori": baska}}, trial_ids_path(cfg), None)

    out = pl.read_csv(build_validation(cfg, ["pilot"], 5))

    # Baska kategoride dislanan id'ler BURADA masum
    assert read_json(validation_ids_path(cfg))["n_dropped_as_trial_rows"] == {"pilot": 0}
    assert out.height == 5


def test_stub_annotations_cannot_seed_the_validation_set(cfg: Config):
    """Kuru kosu etiketleri RASTGELE.

    Onlardan cekilmis bir set uc kisinin saatlerini cope atar ve bunu ancak
    etiketleme bittikten sonra fark ederiz.
    """
    _prepare(cfg, backend="stub")

    with pytest.raises(RuntimeError, match="Kuru kosu"):
        build_validation(cfg, ["pilot"], 5)


def test_validation_strata_must_cover_every_label(cfg: Config):
    """Semaya sinif eklenip config'e eklenmezse o sinif hic olculmez - sessizce."""
    _prepare(cfg)
    cfg._data["validation"]["strata"].pop("received")

    with pytest.raises(ValueError, match="ayrisiyor"):
        build_validation(cfg, ["pilot"], 5)


def test_validation_carries_the_review_text(cfg: Config):
    """Annotation ciktisi metni tasimiyor; etiketleyen kisi metni okuyacak.

    Ornege geri join atilmazsa sayfa bos metinle uretilir ve bu ancak
    etiketleme baslayinca fark edilir.
    """
    _prepare(cfg)

    out = pl.read_csv(build_validation(cfg, ["pilot"], 5))

    assert (out["text"].str.strip_chars().str.len_chars() > 0).all()
    assert PROXY_COL in out.columns  # vekil karsilastirmasi icin lazim
