"""vLLM ile toplu LLM annotation (Hafta 3, asama [B]).

Girdi  : data/interim/<slug>_annotation_sample.parquet   (11.800 satir/kategori)
Cikti  : data/annotations/<slug>_llm.parquet             (satir basina etiket)
Rapor  : reports/results/llm_annotate_<slug>.json        (throughput + kalite)

ISLETIM: Kaggle T4. Yerel GTX 1650 Ti 4 GB, Qwen3-4B fp16 ~8 GB - sigmiyor.
Yerelde `--backend stub` ile butun boru hatti (yollar, sema, sayaclar, parquet)
GPU'suz kosturulabilir; yalnizca uretim adimi taklit edilir.

UC TASARIM KARARI, ucu de bir arizadan geliyor:

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

Kullanim:
    # GPU'suz kuru kosu (gercek veri, taklit model)
    python -m gift_contamination.detection.llm_annotate --category high \\
        --backend stub --limit 200

    # Kaggle T4
    python -m gift_contamination.detection.llm_annotate --category high
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from ..config import Config, add_standard_args, resolve_roles
from ..data.sampling import annotation_sample_path
from ..utils.io import should_skip, write_json
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
def _done_row_ids(shards: Path) -> set[int]:
    if not shards.exists():
        return set()
    done: set[int] = set()
    for part in sorted(shards.glob("part_*.parquet")):
        done |= set(pl.read_parquet(part, columns=["row_id"])["row_id"].to_list())
    if done:
        log.info("yarim kosu bulundu: %s satir zaten etiketli, atlaniyor", f"{len(done):,}")
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


def annotate(
    cfg: Config,
    role: str,
    *,
    backend_name: str = "vllm",
    limit: int | None = None,
    force: bool = False,
) -> Path:
    dest = annotation_path(cfg, role)
    stale = _stale_backend(dest, backend_name)
    if stale is not None and not force:
        raise RuntimeError(
            f"{dest.name} '{stale}' backend'iyle uretilmis, simdi '{backend_name}' "
            f"isteniyor. Sessizce atlamiyorum - ya dosyayi silin ya --force verin."
        )
    if should_skip(dest, force, log):
        return dest

    src = annotation_sample_path(cfg, role)
    if not src.exists():
        raise FileNotFoundError(
            f"Ornekleme yok: {src}\n"
            "Once cekin: python -m gift_contamination.data.sampling "
            f"--category {role}"
        )

    df = pl.read_parquet(src)
    if limit is not None:
        df = df.head(limit)

    shards = shard_dir(cfg, role)
    if force and shards.exists():
        for part in shards.glob("part_*.parquet"):
            part.unlink()
    shards.mkdir(parents=True, exist_ok=True)

    done = _done_row_ids(shards)
    todo = df.filter(~pl.col("row_id").is_in(list(done))) if done else df

    prompt = load_prompt(cfg)
    backend = BACKENDS[backend_name](cfg, prompt)
    log.info(
        "%s | model=%s | prompt=%s | %s satir etiketlenecek",
        backend.name, cfg.get("detection.primary_model"), prompt.version,
        f"{todo.height:,}",
    )

    shard_rows = cfg.get("detection.shard_rows")
    retries = cfg.get("detection.retry_on_parse_fail")
    started = time.perf_counter()
    stats = {"n_parse_fail": 0, "n_downgraded": 0, "n_truncated": 0,
             "n_output_tokens": 0, "n_retried": 0}
    prefix_chars = 0

    n_existing = len(list(shards.glob("part_*.parquet")))
    for i, chunk in enumerate(todo.iter_slices(shard_rows)):
        prompts, reviews, n_cut = build_prompts(chunk, prompt, backend)
        stats["n_truncated"] += n_cut
        prefix_chars = max(prefix_chars, shared_prefix_len(prompts))

        gens = backend.generate(prompts, reviews)
        records, ok_flags, down_flags = [], [], []
        for gen, review in zip(gens, reviews):
            record, ok, down = parse_one(gen.text, review)
            # Tek bir tekrar denemesi tum parcayi yeniden uretmeyi gerektirir;
            # bunun yerine basarisiz satirlar toplanip ayrica denenir (asagida).
            records.append(record)
            ok_flags.append(ok)
            down_flags.append(down)
            stats["n_output_tokens"] += gen.n_output_tokens

        if retries and not all(ok_flags):
            idx = [j for j, ok in enumerate(ok_flags) if not ok]
            stats["n_retried"] += len(idx)
            log.warning("parca %s: %s satir yeniden deneniyor", i, len(idx))
            regen = backend.generate([prompts[j] for j in idx], [reviews[j] for j in idx])
            for j, gen in zip(idx, regen):
                record, ok, down = parse_one(gen.text, reviews[j])
                records[j], ok_flags[j], down_flags[j] = record, ok, down
                stats["n_output_tokens"] += gen.n_output_tokens

        stats["n_parse_fail"] += ok_flags.count(False)
        stats["n_downgraded"] += down_flags.count(True)

        out = chunk.select(
            [c for c in KEY_COLUMNS if c in chunk.columns]
        ).with_columns(
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
        part = shards / f"part_{n_existing + i:05d}.parquet"
        out.write_parquet(part)

        elapsed = time.perf_counter() - started
        n_seen = (i + 1) * shard_rows
        log.info(
            "parca %s yazildi | %s satir | %.1f satir/sn",
            part.name, f"{min(n_seen, todo.height):,}", min(n_seen, todo.height) / elapsed,
        )

    final = pl.concat(
        [pl.read_parquet(p) for p in sorted(shards.glob("part_*.parquet"))],
        how="vertical",
    ).unique(subset=["row_id", "category"], keep="last", maintain_order=True)

    # Girdi = cikti. Esit degilse satir kaybolmus demektir ve bu sessizce
    # gecilemez: eksik satirlar yaygınlık hesabini kaydirir.
    if final.height != df.height:
        raise RuntimeError(
            f"satir sayisi tutmuyor: girdi {df.height}, cikti {final.height}. "
            f"Parcalar {shards} altinda duruyor, veri kaybolmadi."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(dest)
    log_output(log, dest, n_rows=final.height)

    _write_stats(cfg, role, final, prompt, backend, stats, prefix_chars,
                 time.perf_counter() - started)
    return dest


def _write_stats(cfg, role, final, prompt, backend, stats, prefix_chars, elapsed) -> None:
    """Kosu raporu. Kapi 1 bu dosyayi okur; tahmin degil olcum yazilir."""
    n = final.height
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
            "prompt_version": prompt.version,
            "backend": backend.name,
            "n_rows": n,
        },
        "throughput": {
            "elapsed_s": round(elapsed, 1),
            "rows_per_s": round(n / elapsed, 2) if elapsed else None,
            "output_tokens": stats["n_output_tokens"],
            "output_tokens_per_s": round(stats["n_output_tokens"] / elapsed, 1)
            if elapsed else None,
            # Prefix caching'in dayanagi. Sifira yakinsa caching ise yaramiyor.
            "shared_prefix_chars": prefix_chars,
            "system_prompt_tokens": getattr(backend, "shared_prefix_tokens", None),
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
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    for role in resolve_roles(cfg, args.category):
        log.info("=== kategori: %s (%s) ===", role, cfg.category_slug(role))
        annotate(cfg, role, backend_name=args.backend, limit=args.limit,
                 force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
