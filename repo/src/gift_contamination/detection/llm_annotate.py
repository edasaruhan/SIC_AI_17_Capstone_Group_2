"""vLLM ile toplu LLM annotation (Hafta 3, asama [B]).

Girdi  : data/interim/<slug>_annotation_sample.parquet   (11.800 satir/kategori)
Cikti  : data/annotations/<slug>_llm.parquet             (satir basina etiket)
Rapor  : reports/results/llm_annotate_<slug>.json        (throughput + kalite)

ISLETIM: Kaggle T4. Yerel GTX 1650 Ti 4 GB, Qwen3-4B fp16 ~8 GB - sigmiyor.
Yerelde `--backend stub` ile butun boru hatti (yollar, sema, sayaclar, parquet)
GPU'suz kosturulabilir; yalnizca uretim adimi taklit edilir.

DORT TASARIM KARARI; ilk ucu birer arizadan, dorduncusu donanimdan geliyor:

1. PREFIX CACHING PAZARLIK KONUSU DEGIL (CLAUDE.md bolum 6). SYSTEM blogu ~2.450
   token, review medyani ~30 token: her satirin prefill'inin %99'u ayni. Bunun
   ISE YARAMASI icin paylasilan onek her satirda BAYT BAYT AYNI olmali - bu
   yuzden chat template'i burada, tokenizer ile, bir kez uygulanip
   `llm.generate(list[str])` cagriliyor. `llm.chat()` surumler arasi degisken
   davraniyor ve oneki dogrulamak imkansizlasiyor. Gerceklesen onek uzunlugu
   olculup rapora yaziliyor - varsayilmiyor.

2. KOSU KESINTIYE DAYANIKLI. Kaggle oturumu 12 saatte kesilir, GPU OOM olabilir,
   internet duser. Cikti `shard_rows` satirlik parcalar halinde yazilir; yeniden
   baslatildiginda bitmis `row_id`'ler atlanir. 70 dakikalik bir kosuyu bir
   kopmada bastan almak kabul edilemez.

3. BASARISIZ SATIR SESSIZCE DUSMEZ. Parse hatasi `retry_on_parse_fail` kadar
   tekrar denenir; hala basarisizsa satir `unclear` + `parse_ok=False` olarak
   YAZILIR ve orani raporlanir. Cikti satir sayisi girdiye esit olmak zorunda -
   aksi halde eksik satirlar yaygınlık hesabini sessizce kaydirir.

4. IKI GPU VERI-PARALEL KULLANILIR, TENSOR-PARALEL DEGIL. Kaggle iki T4 veriyor
   ve Qwen3-4B fp16 (~8 GB) TEK karta sigiyor. Modeli bolmenin (TP) tek yaptigi
   sey PCIe uzerinden all-reduce maliyeti eklemek - T4'lerde NVLink yok. Onun
   yerine iki BAGIMSIZ surec kosuyor, her biri satirlarin yarisini aliyor:
   GPU'lar arasi hic iletisim yok, her surecin kendi TAM prefix cache'i var,
   hizlanma ~2x. Kaggle kotasi OTURUM saati olarak sayildigi icin bu kotayi da
   yariya indiriyor. Bolme (`_worker_slice`) deterministik ve yeniden
   baslatmada AYNI - degisirse bir isci digerinin yarim biraktigi satirlari
   asla gormez ve kosu hic bitmez. TP yalnizca model tek karta sigmazsa
   gerekir; `tensor_parallel_size` o gun icin duruyor ve kod
   `gpus x tp <= mevcut kart` kontrolunu yapiyor.

Kullanim:
    # GPU'suz kuru kosu (gercek veri, taklit model)
    python -m gift_contamination.detection.llm_annotate --category high \\
        --backend stub --limit 200

    # Kaggle: iki T4'u de kullanir (detection.gpus: auto)
    python -m gift_contamination.detection.llm_annotate --category high

    # Kapi 1'in dorduncu olcutu: elle etiketlenmis 200 satirlik deneme seti
    python -m gift_contamination.detection.llm_annotate --trial 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from ..config import Config, add_standard_args, resolve_roles
from ..data.sampling import annotation_sample_path, trial_source_path
from ..utils.io import code_version, read_json, should_skip, write_json  # noqa: F401 - distill/inference buradan okuyor
from ..utils.logging import get_logger, log_output
from .prompting import Prompt, load_prompt
from .schema import (
    Confidence,
    GiftAnnotation,
    Occasion,
    PurchaseType,
    Recipient,
    downgrade_if_span_missing,
    load_json_schema,
)

log = get_logger("detection.llm_annotate")

# Urun adi metadata join'inden bos gelebilir (47.200 satirda 2 olculdu). Bos
# string birakmak prompt'ta "Product: " gibi sakat bir satir uretir; LLM bunu
# bir sinyal sanabilir. Acik bir yer tutucu daha durust.
UNKNOWN_PRODUCT = "(unknown)"

# Review metni kirpma siniri. max_model_len 4096 = 2.450 (system) + review + 220
# (cikti). 6.000 karakter ~1.500 token, yani tavanin altinda guvenli pay birakir.
# Bilgi kaybi ihmal edilebilir: kanit govdenin medyan %1-3'unde duruyor ve 512
# token'da bile kayip %0,013-0,044 (data-research bolum 4.8, T5). Kirpilan satir
# sayisi yine de sayilip raporlanir - "ihmal edilebilir" bir varsayim degil olcum.
MAX_REVIEW_CHARS = 6000

# Cikti parquet'i review METNINI TASIMAZ. Metin zaten ornekleme parquet'inde;
# (row_id, category) ile join'lenir. Ikilemek hem yeri buyutur hem de birebir
# review metni tasiyan ikinci bir dosya yaratir (CLAUDE.md bolum 8.1).
KEY_COLUMNS = ["row_id", "category", "sample_frame", "stratum_id", "month", "rating",
               "n_words", "parent_asin"]
LABEL_COLUMNS = ["purchase_type", "confidence", "recipient", "occasion", "evidence_span"]

# Parse tamamen basarisiz olan satirin yerine yazilan kayit. `unclear` seciliyor
# cunku "karar veremedik" durumunu ifade eden sinif bu; `parse_ok=False` ile
# ayirt edilir ve yaygınlık hesabindan cikarilabilir.
FAILED_ROW = {
    "purchase_type": PurchaseType.UNCLEAR.value,
    "confidence": Confidence.LOW.value,
    "recipient": Recipient.UNKNOWN.value,
    "occasion": Occasion.UNKNOWN.value,
    "evidence_span": "",
}


# --------------------------------------------------------------------- yollar
def annotation_path(cfg: Config, role: str) -> Path:
    return cfg.path("annotations", f"{cfg.category_slug(role)}_llm.parquet")


def annotation_stats_path(cfg: Config, role: str) -> Path:
    return cfg.path("results", f"llm_annotate_{cfg.category_slug(role)}.json")


def shard_dir(cfg: Config, role: str) -> Path:
    """Yarim kalan kosunun parcalari. Kosu bitince silinmez - kanit olarak kalir."""
    return cfg.path("annotations", "_shards", cfg.category_slug(role))




# ---------------------------------------------------------------- backend'ler
@dataclass(frozen=True)
class Generation:
    text: str
    n_output_tokens: int


class StubBackend:
    """GPU'suz kuru kosu. Modeli taklit eder, boru hattini ETMEZ.

    Kural tabanli ve deterministik: metinde hediye sozcugu varsa `gift_given`,
    yoksa `self`. Amaci dogru etiket uretmek DEGIL - yol cozumleme, sema
    dogrulama, span kontrolu, parca yazimi ve sayaclarin gercek veri uzerinde
    calistigini GPU olmadan gostermek. Uretilen span metinden birebir alinir,
    boylece `downgrade_if_span_missing` yolu da gercekten sinanir.
    """

    name = "stub"
    # Tokenizer yok: token sayisi UYDURULMAZ, None kalir. Rapora "0 token"
    # yazmak yanlis bir kesinlik verirdi.
    shared_prefix_tokens = None

    def __init__(self, *_args, **_kwargs) -> None:
        log.warning("STUB backend: etiketler taklit, sonuc ANALIZ EDILEMEZ")

    def generate(self, prompts: list[str], reviews: list[str]) -> list[Generation]:
        out = []
        for review in reviews:
            words = review.split()
            lower = review.lower()
            hit = next((w for w in ("gift", "present", "birthday") if w in lower), None)
            if hit:
                start = lower.index(hit)
                span = review[start : start + 40].strip()
                payload = {**FAILED_ROW, "purchase_type": PurchaseType.GIFT_GIVEN.value,
                           "confidence": Confidence.MEDIUM.value, "evidence_span": span}
            else:
                payload = {**FAILED_ROW, "purchase_type": PurchaseType.SELF.value,
                           "occasion": Occasion.NONE.value,
                           "evidence_span": " ".join(words[:8])}
            body = json.dumps(payload)
            out.append(Generation(body, len(body) // 4))
        return out


class VllmBackend:
    """Gercek uretim. vLLM `import` edilmesi PAHALI, o yuzden burada yapiliyor.

    Chat template'i vLLM'e degil tokenizer'a uygulatiyoruz: `llm.generate()` duz
    string alir ve boylece paylasilan onegin her satirda ayni oldugunu
    DOGRULAYABILIRIZ. Prefix caching'in ise yaramasi buna bagli.
    """

    name = "vllm"

    def __init__(self, cfg: Config, prompt: Prompt) -> None:
        from transformers import AutoTokenizer  # noqa: PLC0415 - agir import
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        # vLLM structured output API'si 0.10'da yeniden adlandirildi
        # (`guided_decoding` -> `structured_outputs`). Kaggle'da surumu biz
        # secmiyoruz ve 10 dakikalik kurulumdan sonra ImportError almak pahali,
        # o yuzden ikisi de deneniyor. Hangisinin kullanildigi loglanir.
        try:
            from vllm.sampling_params import StructuredOutputsParams  # noqa: PLC0415

            structured = {"structured_outputs": StructuredOutputsParams}
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams  # noqa: PLC0415

            structured = {"guided_decoding": GuidedDecodingParams}
        [(sp_kwarg, sp_class)] = structured.items()
        log.info("vLLM structured output API: %s", sp_kwarg)

        model = cfg.get("detection.primary_model")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(
            model=model,
            dtype=cfg.get("detection.dtype"),
            max_model_len=cfg.get("detection.max_model_len"),
            # Normalde 1: model tek T4'e siginca tensor parallel yalnizca
            # PCIe all-reduce maliyeti ekler. Coklu GPU kazanci VERI paralel
            # sureclerden geliyor (bkz. `_resolve_gpus`).
            tensor_parallel_size=cfg.get("detection.tensor_parallel_size"),
            # ZORUNLU. Kapaliysa 47.200 satirda ~116M gereksiz prefill token.
            enable_prefix_caching=cfg.get("detection.enable_prefix_caching"),
            gpu_memory_utilization=cfg.get("detection.gpu_memory_utilization"),
        )
        self.params = SamplingParams(
            temperature=cfg.get("detection.temperature"),
            max_tokens=cfg.get("detection.max_new_tokens"),
            **{sp_kwarg: sp_class(json=load_json_schema(cfg))},
        )
        # SYSTEM blogunun tokenizer'a gore gercek uzunlugu. Dokumandaki ~2.450
        # bir tahmindi; rapora bu sayi girer.
        self.shared_prefix_tokens = len(self.tokenizer.encode(prompt.system))

    def generate(self, prompts: list[str], reviews: list[str]) -> list[Generation]:
        outs = self.llm.generate(prompts, self.params)
        return [
            Generation(o.outputs[0].text, len(o.outputs[0].token_ids))
            for o in outs
        ]


BACKENDS = {"stub": StubBackend, "vllm": VllmBackend}


# ------------------------------------------------------------------ hazirlik
def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_REVIEW_CHARS:
        return text, False
    return text[:MAX_REVIEW_CHARS], True


def build_prompts(
    df: pl.DataFrame, prompt: Prompt, backend
) -> tuple[list[str], list[str], int]:
    """(chat template uygulanmis prompt'lar, kirpilmis review metinleri, kirpilan)."""
    rendered: list[str] = []
    reviews: list[str] = []
    n_truncated = 0

    apply_template = getattr(backend, "tokenizer", None)
    for row in df.iter_rows(named=True):
        text, cut = _truncate(row["text"] or "")
        n_truncated += cut
        reviews.append(text)
        messages = prompt.messages(
            product_title=row["product_title"] or UNKNOWN_PRODUCT,
            category=row["category"],
            title=row["title"] or "",
            text=text,
        )
        if apply_template is None:
            # stub: gercek chat template yok ama SYSTEM yine de ONE konuyor.
            # Aksi halde kuru kosuda ortak onek olcumu anlamsiz cikar (yalnizca
            # USER blogu kalir) ve prefix caching kontrolu sinanamaz.
            rendered.append("\n".join(m["content"] for m in messages))
        else:
            rendered.append(
                apply_template.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
    return rendered, reviews, n_truncated


def shared_prefix_len(prompts: list[str]) -> int:
    """Butun prompt'larin ortak onekinin KARAKTER uzunlugu.

    Prefix caching'in dayanagi bu; olculmezse "%99 ortak" bir iddia olarak kalir.
    Sifir cikarsa caching hicbir ise yaramiyor demektir ve kosu durdurulmalidir.
    """
    if len(prompts) < 2:
        return len(prompts[0]) if prompts else 0
    first, last = min(prompts), max(prompts)   # sozluksel uc noktalar yeter
    i = 0
    while i < min(len(first), len(last)) and first[i] == last[i]:
        i += 1
    return i


# --------------------------------------------------------------------- parse
def parse_one(raw: str, review: str) -> tuple[dict, bool, bool]:
    """(kayit, parse_ok, downgraded).

    Iki katman: guided decoding semayi zorlar, Pydantic donen nesneyi dogrular.
    Ikisi de gecerse `evidence_span` metinde birebir aranir.
    """
    try:
        obj = GiftAnnotation.model_validate_json(raw)
    except (ValidationError, ValueError):
        return dict(FAILED_ROW), False, False

    obj, downgraded = downgrade_if_span_missing(obj, review)
    record = {k: (v.value if hasattr(v, "value") else v)
              for k, v in obj.model_dump().items()}
    return record, True, downgraded


# ----------------------------------------------------------------- ana akislar


def _label_chunk(
    chunk: pl.DataFrame, prompt: Prompt, backend, retries: int, stats: dict,
    *, tag: str = "",
) -> tuple[list[dict], list[bool], list[bool], int]:
    """Bir parcayi etiketler: prompt uret -> uret -> parse -> basarisizlari tekrar dene.

    Hem gercek kosu hem deneme kosusu (Kapi 1 olcut 4) BURADAN geciyor. Iki ayri
    kopya olsaydi uyum sayisi zamanla uretimde kosan boru hattini olcmeyi
    birakabilirdi ve bunu fark etmenin bir yolu olmazdi.

    `stats` YERINDE guncelleniyor; donen deger (kayitlar, parse_ok, downgrade,
    ortak onek uzunlugu).
    """
    prompts, reviews, n_cut = build_prompts(chunk, prompt, backend)
    stats["n_truncated"] += n_cut

    gens = backend.generate(prompts, reviews)
    records, ok_flags, down_flags = [], [], []
    for gen, review in zip(gens, reviews):
        record, ok, down = parse_one(gen.text, review)
        # Tek bir tekrar denemesi tum parcayi yeniden uretmeyi gerektirir;
        # bunun yerine basarisiz satirlar toplanip ayrica deneniyor (asagida).
        records.append(record)
        ok_flags.append(ok)
        down_flags.append(down)
        stats["n_output_tokens"] += gen.n_output_tokens

    if retries and not all(ok_flags):
        idx = [j for j, ok in enumerate(ok_flags) if not ok]
        stats["n_retried"] += len(idx)
        log.warning("%s%s satir yeniden deneniyor", tag, len(idx))
        regen = backend.generate([prompts[j] for j in idx], [reviews[j] for j in idx])
        for j, gen in zip(idx, regen):
            record, ok, down = parse_one(gen.text, reviews[j])
            records[j], ok_flags[j], down_flags[j] = record, ok, down
            stats["n_output_tokens"] += gen.n_output_tokens

    stats["n_parse_fail"] += ok_flags.count(False)
    stats["n_downgraded"] += down_flags.count(True)
    return records, ok_flags, down_flags, shared_prefix_len(prompts)


def _attach_labels(
    chunk: pl.DataFrame, records: list[dict], ok_flags: list[bool],
    down_flags: list[bool], *, cfg: Config, prompt: Prompt, backend,
    keep: list[str],
) -> pl.DataFrame:
    """Etiketleri girdi satirlarina yapistirir. Kolon sozlesmesi TEK YERDE."""
    return chunk.select([c for c in keep if c in chunk.columns]).with_columns(
        **{
            col: pl.Series(col, [r[col] for r in records], dtype=pl.String)
            for col in LABEL_COLUMNS
        },
        parse_ok=pl.Series("parse_ok", ok_flags, dtype=pl.Boolean),
        span_downgraded=pl.Series("span_downgraded", down_flags, dtype=pl.Boolean),
    ).with_columns(
        pl.lit(prompt.version).alias("prompt_version"),
        pl.lit(cfg.get("detection.primary_model")).alias("model"),
        # Taklit etiket gercek etiketten AYIRT EDILEBILIR olmali; kuru kosu
        # ciktisi yanlislikla analize girmesin.
        pl.lit(backend.name).alias("backend"),
    )


def _done_row_ids(shards: Path, worker: int) -> set[int]:
    """Bu iscinin daha once bitirdigi satirlar.

    YALNIZCA kendi parcalarina bakiyor. Dilimler ayrik oldugu icin bu dogru;
    hepsine bakmak ayrica log'u yanlis yapardi ("5.900 satir zaten etiketli"
    derken aslinda digerinin satirlarini sayardi).

    Isci sayisi kosular arasi degisirse (once 1 GPU, sonra 2) bolme kayar ve
    bir kisim satir yeniden uretilir. Cikti yine dogru - `merge` tekrarlari
    `unique` ile ayikliyor - sadece bir miktar is bosa gider.
    """
    if not shards.exists():
        return set()
    done: set[int] = set()
    for part in sorted(shards.glob(f"part_w{worker}_*.parquet")):
        done |= set(pl.read_parquet(part, columns=["row_id"])["row_id"].to_list())
    if done:
        log.info("isci %s: %s satir zaten etiketli, atlaniyor", worker, f"{len(done):,}")
    return done


def _stale_backend(dest: Path, backend_name: str) -> str | None:
    """Diskteki cikti baska bir backend'le uretildiyse onun adi.

    Kritik: `stub` ile kuru kosu yapip ciktiyi silmeyi unutmak, gercek kosunun
    idempotans yuzunden SESSIZCE atlanmasina yol acar - ve elimizde taklit
    etiketlerle dolu bir parquet kalir. Bu, sonuclara bakilana kadar fark
    edilmeyecek bir ariza; o yuzden atlama karari backend'e de bakiyor.
    """
    if not dest.exists():
        return None
    try:
        existing = pl.read_parquet(dest, columns=["backend"])["backend"][0]
    except Exception:  # noqa: BLE001 - eski/bozuk dosya: karar veremiyoruz
        return None
    return None if existing == backend_name else existing


def _worker_slice(df: pl.DataFrame, worker: int, workers: int) -> pl.DataFrame:
    """Isciler arasinda deterministik, ayrik, dengeli bolme.

    `row_id`e gore siralayip her n'inciyi almak, `row_id % n`den daha iyi:
    ikincisi ham dosyadaki id dagilimina bagli, bu degil. Bolme YENIDEN
    BASLATMADA DA AYNI kalmak zorunda - degisirse bir isci digerinin yarim
    biraktigi satirlari asla gormez ve kosu hic bitmez.
    """
    if workers == 1:
        return df
    return df.sort("row_id").gather_every(workers, offset=worker)


def _resolve_gpus(cfg: Config, backend_name: str) -> int:
    """Kac veri-paralel surec kosacak.

    Kaggle iki T4 veriyor. Model tek karta sigdigi icin (Qwen3-4B fp16 ~8 GB,
    T4 16 GB) TENSOR parallel DEGIL VERI parallel dogru secim: T4'lerde NVLink
    yok, all-reduce PCIe uzerinden gidiyor ve 4B'lik bir modelde iletisim
    maliyeti kazanci yiyor. Iki bagimsiz surec ise birbiriyle hic konusmuyor,
    her birinin kendi tam prefix cache'i oluyor ve hizlanma ~2x.

    Kaggle kotasi OTURUM saati olarak sayiliyor, GPU basina degil - yani iki
    GPU kullanmak kotayi da yariya indiriyor.
    """
    requested = cfg.get("detection.gpus")
    tp = cfg.get("detection.tensor_parallel_size")
    if backend_name != "vllm":
        return 1   # kuru kosuda GPU yok, surec cogaltmak anlamsiz

    try:
        import torch  # noqa: PLC0415 - yalnizca LLM ortaminda var

        available = torch.cuda.device_count()
    except ImportError:
        if requested == "auto":
            log.warning("torch yok, gpus=auto -> 1")
            return 1
        available = None

    n = available if requested == "auto" else int(requested)
    n = max(1, n // tp)   # her surec `tp` kart tuketiyor

    if available is not None and n * tp > available:
        raise RuntimeError(
            f"gpus({n}) x tensor_parallel_size({tp}) = {n * tp} kart isteniyor "
            f"ama {available} kart var."
        )
    if n > 1:
        log.info("veri-paralel: %s surec x %s kart", n, tp)
    return n


def _refuse_partial_output(dest: Path, src: Path, limit: int | None, force: bool) -> None:
    """`--limit` ile uretilmis kismi cikti, tam kosuyu SESSIZCE atlatmamali.

    Duman testi tam bu tuzagi acti: `--limit 200` ciktisi diskte kalinca tam
    kosu idempotans yuzunden atlanir ve elde 200 satirlik bir dosya kalir.
    Backend kontrolu bunu yakalamaz - ikisi de `vllm`. Kismi ciktinin satir
    sayisi ornekleme boyutundan kucuk oldugu icin burada yakalaniyor.
    """
    if force or not dest.exists() or not src.exists():
        return
    have = pl.read_parquet(dest, columns=["row_id"]).height
    want = limit if limit is not None else pl.read_parquet(
        src, columns=["row_id"]
    ).height
    if have < want:
        raise RuntimeError(
            f"{dest.name} yalnizca {have:,} satir iceriyor, {want:,} isteniyor "
            "(muhtemelen bir --limit kosusundan kalma). Sessizce atlamiyorum - "
            "ya dosyayi silin ya --force verin."
        )


def annotate(
    cfg: Config,
    role: str,
    *,
    backend_name: str = "vllm",
    limit: int | None = None,
    force: bool = False,
    gpus: int | None = None,
) -> Path:
    """Tek giris noktasi. `gpus > 1` ise kendini alt sureclerde cogaltir."""
    dest = annotation_path(cfg, role)
    stale = _stale_backend(dest, backend_name)
    if stale is not None and not force:
        raise RuntimeError(
            f"{dest.name} '{stale}' backend'iyle uretilmis, simdi '{backend_name}' "
            f"isteniyor. Sessizce atlamiyorum - ya dosyayi silin ya --force verin."
        )
    src = annotation_sample_path(cfg, role)
    _refuse_partial_output(dest, src, limit, force)
    if should_skip(dest, force, log):
        return dest

    if not src.exists():
        raise FileNotFoundError(
            f"Ornekleme yok: {src}\n"
            "Once cekin: python -m gift_contamination.data.sampling "
            f"--category {role}"
        )

    shards = shard_dir(cfg, role)
    if force and shards.exists():
        for leftover in shards.iterdir():
            leftover.unlink()
    shards.mkdir(parents=True, exist_ok=True)

    n_gpus = gpus if gpus is not None else _resolve_gpus(cfg, backend_name)
    if backend_name != "vllm" and n_gpus > 1:
        # Kuru kosuda GPU yok; surec cogaltmak yalnizca gurultu uretir.
        log.warning("backend=%s icin gpus=%s yok sayildi", backend_name, n_gpus)
        n_gpus = 1
    started = time.perf_counter()
    if n_gpus > 1:
        _spawn_workers(cfg, role, backend_name, limit, n_gpus)
    else:
        run_worker(cfg, role, backend_name=backend_name, limit=limit,
                   worker=0, workers=1)
    return merge(cfg, role, limit=limit, elapsed=time.perf_counter() - started)


def _spawn_workers(
    cfg: Config, role: str, backend_name: str, limit: int | None, n_gpus: int
) -> None:
    """Her GPU icin bir alt surec; hepsi bitene kadar beklenir.

    vLLM'in kendi coklu-GPU backend'i (ray/multiproc) yerine duz surec
    cogaltmayi tercih ediyoruz: her cocuk sirradan bir TEK GPU vLLM ornegi,
    yani Kaggle'da kirilacak bir sey yok.
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    # Cocuk taze bir yorumlayici; notebook'un sys.path'ini gormez.
    src_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    # `tensor_parallel_size > 1` ise her isciye BIRDEN COK kart dusuyor;
    # tek kart vermek TP'yi sessizce bozardi.
    tp = cfg.get("detection.tensor_parallel_size")
    procs = []
    for w in range(n_gpus):
        devices = ",".join(str(w * tp + k) for k in range(tp))
        child_env = dict(env, CUDA_VISIBLE_DEVICES=devices)
        cmd = [
            sys.executable, "-m", "gift_contamination.detection.llm_annotate",
            "--config", str(cfg.source), "--category", role,
            "--backend", backend_name,
            "--worker", str(w), "--workers", str(n_gpus), "--no-merge",
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        log.info("GPU %s -> alt surec baslatiliyor", devices)
        procs.append((w, subprocess.Popen(cmd, env=child_env)))

    failed = [w for w, p in procs if p.wait() != 0]
    if failed:
        raise RuntimeError(
            f"{len(failed)} isci basarisiz oldu (GPU {failed}). Parcalar "
            f"{shard_dir(cfg, role)} altinda duruyor - ayni komutu tekrar "
            "calistirmak kaldigi yerden devam eder."
        )


def run_worker(
    cfg: Config,
    role: str,
    *,
    backend_name: str,
    limit: int | None,
    worker: int,
    workers: int,
) -> None:
    """Tek bir iscinin isi: kendi dilimini etiketleyip parcalarini yazmak."""
    df = pl.read_parquet(annotation_sample_path(cfg, role))
    if limit is not None:
        df = df.head(limit)
    df = _worker_slice(df, worker, workers)

    shards = shard_dir(cfg, role)
    shards.mkdir(parents=True, exist_ok=True)
    done = _done_row_ids(shards, worker)
    todo = df.filter(~pl.col("row_id").is_in(list(done))) if done else df

    prompt = load_prompt(cfg)
    backend = BACKENDS[backend_name](cfg, prompt)
    log.info(
        "isci %s/%s | %s | model=%s | prompt=%s | %s satir etiketlenecek",
        worker, workers, backend.name, cfg.get("detection.primary_model"),
        prompt.version, f"{todo.height:,}",
    )

    shard_rows = cfg.get("detection.shard_rows")
    retries = cfg.get("detection.retry_on_parse_fail")
    started = time.perf_counter()
    stats = {"n_parse_fail": 0, "n_downgraded": 0, "n_truncated": 0,
             "n_output_tokens": 0, "n_retried": 0}
    prefix_chars = 0

    # Parca adi isciyi tasiyor: iki surec ayni dosyaya yazamaz.
    n_existing = len(list(shards.glob(f"part_w{worker}_*.parquet")))
    for i, chunk in enumerate(todo.iter_slices(shard_rows)):
        records, ok_flags, down_flags, prefix = _label_chunk(
            chunk, prompt, backend, retries, stats, tag=f"parca {i}: ",
        )
        prefix_chars = max(prefix_chars, prefix)
        out = _attach_labels(
            chunk, records, ok_flags, down_flags,
            cfg=cfg, prompt=prompt, backend=backend, keep=KEY_COLUMNS,
        )
        part = shards / f"part_w{worker}_{n_existing + i:05d}.parquet"
        out.write_parquet(part)

        elapsed = time.perf_counter() - started
        n_seen = min((i + 1) * shard_rows, todo.height)
        log.info(
            "isci %s | parca %s | %s satir | %.1f satir/sn",
            worker, part.name, f"{n_seen:,}", n_seen / elapsed,
        )

    # Isci kendi sayaclarini yazar; birlestirme onlari toplar. Wall-clock
    # ISCININ degil EBEVEYNIN saatinden alinir - paralel kosuda toplam sure
    # iscilerin toplami degil en yavas iscinin suresidir.
    stats |= {"shared_prefix_chars": prefix_chars, "backend": backend.name,
              "prompt_version": prompt.version,
              "system_prompt_tokens": getattr(backend, "shared_prefix_tokens", None),
              # YALNIZCA uretim suresi - motor kurulumu haric. `started` backend
              # insa edildikten SONRA baslatiliyor. Ikisini ayirmak sart: 200
              # satirlik duman testinde kurulum (indirme + torch.compile + CUDA
              # graph) 252 sn'nin ~185'ini yiyordu ve toplam hiz 0,79 satir/sn
              # gorunuyordu; gercek uretim hizi 3,0 satir/sn idi. Ayirmazsak
              # kucuk kosudan buyuk kosuya uzatma yapilamaz.
              "generation_s": time.perf_counter() - started,
              "n_rows_generated": todo.height}
    write_json(stats, shards / f"stats_w{worker}.json")


# --------------------------------------------------- deneme kosusu (Kapi 1 / 4)
def trial_annotation_path(cfg: Config, n: int = 200) -> Path:
    return cfg.path("annotations", f"prompt_trial_{n}_llm.parquet")


def trial_stats_path(cfg: Config, n: int = 200) -> Path:
    return cfg.path("results", f"llm_annotate_prompt_trial_{n}.json")


def annotate_trial(
    cfg: Config, n: int = 200, *, backend_name: str = "vllm", force: bool = False,
) -> Path:
    """Elle etiketlenmis deneme setini LLM'e etiketletir (Kapi 1, olcut 4).

    TEK SUREC, PARCALAMA YOK. 200 satir tek T4'te ~1 dakika uretim; kesintiye
    dayaniklilik makinesi burada hicbir sey kazandirmaz, yalnizca yuzey ekler.
    Uretim kosusunun aksine bu dosya kucuk ve yeniden uretmek ucuz.

    Etiketleme `_label_chunk` uzerinden gidiyor - yani gercek kosuyla AYNI
    prompt, ayni chat template, ayni parse ve ayni tekrar deneme mantigi. Uyum
    sayisinin olctugu sey uretimde kosan boru hatti olmak zorunda.
    """
    dest = trial_annotation_path(cfg, n)
    stale = _stale_backend(dest, backend_name)
    if stale is not None and not force:
        raise RuntimeError(
            f"{dest.name} '{stale}' backend'iyle uretilmis, simdi '{backend_name}' "
            f"isteniyor. Sessizce atlamiyorum - ya dosyayi silin ya --force verin."
        )
    if should_skip(dest, force, log):
        return dest

    src = trial_source_path(cfg, n)
    if not src.exists():
        raise FileNotFoundError(
            f"Deneme kaynagi yok: {src}\n"
            "Once uretin: python -m gift_contamination.data.sampling "
            f"--trial-source {n}"
        )

    df = pl.read_parquet(src)
    prompt = load_prompt(cfg)
    started_all = time.perf_counter()
    backend = BACKENDS[backend_name](cfg, prompt)
    startup_s = time.perf_counter() - started_all

    log.info(
        "deneme kosusu | %s | model=%s | prompt=%s | %s satir",
        backend.name, cfg.get("detection.primary_model"), prompt.version,
        f"{df.height:,}",
    )

    stats = {"n_parse_fail": 0, "n_downgraded": 0, "n_truncated": 0,
             "n_output_tokens": 0, "n_retried": 0}
    retries = cfg.get("detection.retry_on_parse_fail")
    started = time.perf_counter()
    parts, prefix_chars = [], 0
    for chunk in df.iter_slices(cfg.get("detection.shard_rows")):
        records, ok_flags, down_flags, prefix = _label_chunk(
            chunk, prompt, backend, retries, stats, tag="deneme: ",
        )
        prefix_chars = max(prefix_chars, prefix)
        parts.append(_attach_labels(
            chunk, records, ok_flags, down_flags,
            cfg=cfg, prompt=prompt, backend=backend,
            keep=["trial_id", *KEY_COLUMNS],
        ))
    generation_s = time.perf_counter() - started

    final = pl.concat(parts, how="vertical").sort("trial_id")
    if final.height != df.height:
        raise RuntimeError(
            f"satir sayisi tutmuyor: girdi {df.height}, cikti {final.height}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(dest)
    log_output(log, dest, n_rows=final.height)

    write_json(
        {
            "meta": {
                "n": n,
                "model": cfg.get("detection.primary_model"),
                "prompt_version": prompt.version,
                "backend": backend.name,
                "code_version": code_version(),
                "n_rows": final.height,
            },
            "throughput": {
                "generation_s": round(generation_s, 1),
                "startup_s": round(startup_s, 1),
                "rows_per_s_generating": round(final.height / generation_s, 2)
                if generation_s else None,
                "output_tokens": stats["n_output_tokens"],
                "shared_prefix_chars": prefix_chars,
                "system_prompt_tokens": getattr(backend, "shared_prefix_tokens", None),
            },
            "quality": {
                "parse_fail_rate": round(stats["n_parse_fail"] / final.height, 6),
                "n_parse_fail": stats["n_parse_fail"],
                "n_retried": stats["n_retried"],
                "span_downgrade_rate": round(stats["n_downgraded"] / final.height, 6),
                "n_span_downgraded": stats["n_downgraded"],
                "n_truncated_reviews": stats["n_truncated"],
            },
            "by_purchase_type": dict(sorted(
                final.group_by("purchase_type").len().iter_rows()
            )),
        },
        trial_stats_path(cfg, n),
        log,
    )
    return dest


def merge(cfg: Config, role: str, *, limit: int | None, elapsed: float) -> Path:
    """Butun iscilerin parcalarini tek parquet'e toplar ve raporu yazar."""
    dest = annotation_path(cfg, role)
    shards = shard_dir(cfg, role)
    parts = sorted(shards.glob("part_w*.parquet"))
    if not parts:
        raise RuntimeError(f"Birlestirilecek parca yok: {shards}")

    final = pl.concat([pl.read_parquet(p) for p in parts], how="vertical").unique(
        subset=["row_id", "category"], keep="last", maintain_order=True
    ).sort("row_id")

    expected = pl.read_parquet(annotation_sample_path(cfg, role), columns=["row_id"]).height
    if limit is not None:
        expected = min(expected, limit)
    # Girdi = cikti. Esit degilse satir kaybolmus demektir ve bu sessizce
    # gecilemez: eksik satirlar yaygınlık hesabini kaydirir. Veri-paralel
    # kosuda bu kontrol ayrica bir isciyi sessizce kaybetmedigimizi de gosterir.
    if final.height != expected:
        raise RuntimeError(
            f"satir sayisi tutmuyor: beklenen {expected}, cikti {final.height}. "
            f"Parcalar {shards} altinda duruyor, veri kaybolmadi - ayni komutu "
            "tekrar calistirmak eksigi tamamlar."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(dest)
    log_output(log, dest, n_rows=final.height)

    stats = _sum_worker_stats(shards)
    _write_stats(cfg, role, final, stats, elapsed, limit=limit, n_workers=len(
        list(shards.glob("stats_w*.json"))
    ))
    return dest


def _sum_worker_stats(shards: Path) -> dict:
    files = sorted(shards.glob("stats_w*.json"))
    total = {"n_parse_fail": 0, "n_downgraded": 0, "n_truncated": 0,
             "n_output_tokens": 0, "n_retried": 0, "n_rows_generated": 0}
    # Bu ikisi TOPLANMAZ: onek her iscide ayni, sure ise paralel gectigi icin
    # toplami degil EN YAVAS isciyi gosterir.
    peak = {"shared_prefix_chars": 0, "generation_s": 0.0}
    meta = {}
    for f in files:
        s = read_json(f)
        for k in total:
            total[k] += s.get(k, 0)
        for k in peak:
            peak[k] = max(peak[k], s.get(k, 0))
        meta = {"backend": s["backend"], "prompt_version": s["prompt_version"],
                "system_prompt_tokens": s["system_prompt_tokens"]}
    return total | peak | meta


def _write_stats(cfg, role, final, stats, elapsed, *, limit, n_workers) -> None:
    """Kosu raporu. Kapi 1 bu dosyayi okur; tahmin degil olcum yazilir."""
    n = final.height
    # Uretim suresi en yavas isciden; sifirsa (devam ettirilen kosuda hicbir
    # isci uretim yapmadiysa) duvar saatine dus.
    gen_s = stats.get("generation_s") or elapsed
    # `group_by` anahtari TUPLE dondurur ("main",); dict anahtari olarak
    # kullanilirsa json.dump patlar. Denetimde ayni desen `build_trial`da da
    # cikmisti - o yuzden burada da acikca [0] aliniyor.
    by_frame = {
        frame[0]: dict(sorted(
            g.group_by("purchase_type").len().sort("len", descending=True).iter_rows()
        ))
        for frame, g in sorted(
            final.group_by("sample_frame"), key=lambda kv: kv[0][0]
        )
    } if "sample_frame" in final.columns else {}

    report = {
        "meta": {
            "category": cfg.category_slug(role),
            "model": cfg.get("detection.primary_model"),
            "prompt_version": stats["prompt_version"],
            "backend": stats["backend"],
            "code_version": code_version(),
            "n_rows": n,
            "n_workers": n_workers,
            # Kismi kosu KENDINI TANITMALI. Duman testinin 200 satirlik raporu
            # ile tam kosunun 11.800'luk raporu aksi halde yapisal olarak ayni
            # gorunur ve yanlislikla analize girebilir.
            "limit": limit,
            "is_partial": bool(limit is not None),
        },
        # KURULUM ile URETIM AYRI raporlanir. Motor kurulumu (model indirme,
        # torch.compile, CUDA graph capture) satir sayisindan BAGIMSIZ sabit bir
        # maliyet: 200 satirlik duman testinde 252 sn'nin ~185'ini yiyip toplam
        # hizi 0,79 satir/sn gosteriyordu, oysa uretim 3,0 satir/sn gidiyordu.
        # Kucuk kosudan buyuk kosuya uzatma YALNIZCA `rows_per_s_generating`
        # ile yapilir; `rows_per_s` tek seferlik bu kosunun gercek maliyeti.
        "throughput": {
            "elapsed_s": round(elapsed, 1),
            "generation_s": round(gen_s, 1),
            "startup_s": round(max(elapsed - gen_s, 0.0), 1),
            "rows_per_s": round(n / elapsed, 2) if elapsed else None,
            "rows_per_s_generating": round(n / gen_s, 2) if gen_s else None,
            "rows_per_s_per_gpu": round(n / gen_s / n_workers, 2)
            if gen_s and n_workers else None,
            "output_tokens": stats["n_output_tokens"],
            "output_tokens_per_s": round(stats["n_output_tokens"] / gen_s, 1)
            if gen_s else None,
            # Prefix caching'in dayanagi. Sifira yakinsa caching ise yaramiyor.
            "shared_prefix_chars": stats["shared_prefix_chars"],
            "system_prompt_tokens": stats["system_prompt_tokens"],
        },
        "quality": {
            "parse_fail_rate": round(stats["n_parse_fail"] / n, 6) if n else None,
            "n_parse_fail": stats["n_parse_fail"],
            "n_retried": stats["n_retried"],
            "span_downgrade_rate": round(stats["n_downgraded"] / n, 6) if n else None,
            "n_span_downgraded": stats["n_downgraded"],
            "n_truncated_reviews": stats["n_truncated"],
        },
        "by_purchase_type": dict(sorted(
            final.group_by("purchase_type").len().sort("len", descending=True).iter_rows()
        )),
        # Yaygınlık YALNIZCA `main`'den okunur; boost cerceveleri kendi sectigimiz
        # seyi olcer (sampling.py, ayni gerekce).
        "by_frame": by_frame,
    }
    write_json(report, annotation_stats_path(cfg, role), log)

    tp = report["throughput"]
    if tp["rows_per_s_generating"]:
        full = pl.read_parquet(
            annotation_sample_path(cfg, role), columns=["row_id"]
        ).height
        log.info(
            "uretim %.2f satir/sn (%s GPU) | kurulum %.0f sn tek seferlik | "
            "tam kategori (%s satir) tahmini ~%.0f dk",
            tp["rows_per_s_generating"], n_workers, tp["startup_s"], f"{full:,}",
            (full / tp["rows_per_s_generating"] + tp["startup_s"]) / 60,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    parser.add_argument(
        "--backend", choices=sorted(BACKENDS), default="vllm",
        help="'stub' GPU'suz kuru kosu icin; etiketler taklit edilir",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Yalnizca ilk N satir (duman testi)",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Kac veri-paralel surec. Varsayilan config'ten (detection.gpus, "
             "'auto' = torch.cuda.device_count()). Kaggle'da 2.",
    )
    parser.add_argument(
        "--trial", type=int, default=None, metavar="N",
        help="Kategori yerine N satirlik deneme setini etiketle (Kapi 1, olcut 4)",
    )
    # Asagidaki ucu ALT SURECLER icin; elle verilmesi gerekmez.
    parser.add_argument("--worker", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--no-merge", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    if args.trial:
        annotate_trial(cfg, args.trial, backend_name=args.backend, force=args.force)
        return 0

    for role in resolve_roles(cfg, args.category):
        if args.no_merge:
            # Alt surec: yalnizca kendi dilimini etiketler, birlestirmez.
            run_worker(cfg, role, backend_name=args.backend, limit=args.limit,
                       worker=args.worker, workers=args.workers)
            continue
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        annotate(cfg, role, backend_name=args.backend, limit=args.limit,
                 force=args.force,
                 gpus=int(args.gpus) if args.gpus is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
