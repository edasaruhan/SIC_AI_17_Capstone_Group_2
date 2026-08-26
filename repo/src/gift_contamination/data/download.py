"""Amazon Reviews 2023 ham review dosyalarını HuggingFace'ten indirir.

`datasets.load_dataset(..., trust_remote_code=True)` KULLANILMIYOR. datasets 4.0
`trust_remote_code` parametresini, 4.5 ise loading-script desteğini tamamen kaldırdı;
McAuley-Lab reposu script tabanlı olduğu için README ve `docs/PROJECT_SPEC.md §16`
içindeki örnek çağrı bugün `RuntimeError` veriyor. Onun yerine ham `.jsonl` dosyası
doğrudan `hf_hub_download` ile çekiliyor: daha hızlı, kesintiden devam ediyor ve
idempotency dosyanın varlığından geliyor. Gerekçe: `docs/DECISIONS.md`.

Kullanım:
    python -m gift_contamination.data.download --config configs/base.yaml --category pilot
    python -m gift_contamination.data.download --category all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..config import Config, add_standard_args, resolve_roles
from ..utils.io import relative_to_repo, write_json
from ..utils.logging import get_logger, human_bytes

log = get_logger("data.download")


def remote_review_file(cfg: Config, role: str) -> str:
    """HF reposundaki dosya yolu, örn. 'raw/review_categories/All_Beauty.jsonl'."""
    template = cfg.get("dataset.review_file_template")
    return template.format(slug=cfg.category_slug(role))


def raw_review_path(cfg: Config, role: str) -> Path:
    """İndirilen dosyanın kanonik yerel yolu. Tüm alt modüller bunu kullanır."""
    return cfg.path("raw") / remote_review_file(cfg, role)


def remote_meta_file(cfg: Config, role: str) -> str:
    """Ürün metadata dosyası, örn. 'raw/meta_categories/meta_All_Beauty.jsonl'."""
    template = cfg.get("dataset.meta_file_template")
    return template.format(slug=cfg.category_slug(role))


def raw_meta_path(cfg: Config, role: str) -> Path:
    return cfg.path("raw") / remote_meta_file(cfg, role)


def probe_schema(path: Path, required: list[str], n_lines: int) -> dict:
    """İlk N satırı okuyup beklenen alanların varlığını doğrular.

    HF veri kartı ile McAuley Lab sitesi arasında alan adı tutarsızlığı var
    (`timestamp` / `sort_timestamp`, `helpful_vote` / `helpful_votes`). Yanlış alanı
    sessizce okumaktansa burada gürültülü patlamak isteniyor.
    """
    seen: set[str] = set()
    types: dict[str, str] = {}
    n = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if n >= n_lines:
                break
            rec = json.loads(line)
            seen.update(rec)
            for k, v in rec.items():
                types.setdefault(k, type(v).__name__)
            n += 1

    missing = [f for f in required if f not in seen]
    if missing:
        raise ValueError(
            f"{path.name}: beklenen alanlar eksik: {missing}. "
            f"Dosyada bulunanlar: {sorted(seen)}. "
            "configs/base.yaml içindeki dataset.required_fields ile veri kartını karşılaştırın."
        )
    return {
        "file": relative_to_repo(path),
        "lines_probed": n,
        "fields": sorted(seen),
        "field_types": dict(sorted(types.items())),
        "required_present": required,
    }


def download_category(cfg: Config, role: str, *, force: bool = False) -> Path:
    from huggingface_hub import hf_hub_download

    slug = cfg.category_slug(role)
    dest = raw_review_path(cfg, role)
    remote = remote_review_file(cfg, role)

    if dest.exists() and not force:
        log.info(
            "atlanıyor (çıktı mevcut, --force ile ezebilirsiniz): %s | %s",
            dest,
            human_bytes(dest.stat().st_size),
        )
    else:
        local_dir = cfg.path("raw")
        local_dir.mkdir(parents=True, exist_ok=True)
        log.info("indiriliyor: %s [%s] -> %s", slug, remote, local_dir)
        got = hf_hub_download(
            repo_id=cfg.get("dataset.hf_repo"),
            repo_type="dataset",
            filename=remote,
            local_dir=str(local_dir),
            force_download=force,
            token=os.environ.get("HF_TOKEN") or None,
        )
        dest = Path(got)
        log.info("indirildi: %s | %s", dest, human_bytes(dest.stat().st_size))

    probe = probe_schema(
        dest,
        list(cfg.get("dataset.required_fields")),
        int(cfg.get("dataset.schema_probe_lines")),
    )
    log.info(
        "şema doğrulandı: %s | %d satır incelendi | %d alan",
        slug,
        probe["lines_probed"],
        len(probe["fields"]),
    )
    write_json(probe, cfg.path("results", f"schema_probe_{slug}.json"), log)
    return dest


def download_metadata(cfg: Config, role: str, *, force: bool = False) -> Path:
    """Ürün metadata dosyasını indirir; review indirmesiyle aynı sözleşme.

    Alan adları review dosyasınınkinden farklı ve varsayım yapmıyoruz -
    `probe_schema` metadata için de koşuyor ve eksik alanda gürültülü patlıyor.
    """
    from huggingface_hub import hf_hub_download

    slug = cfg.category_slug(role)
    dest = raw_meta_path(cfg, role)
    remote = remote_meta_file(cfg, role)

    if dest.exists() and not force:
        log.info(
            "atlanıyor (çıktı mevcut, --force ile ezebilirsiniz): %s | %s",
            dest,
            human_bytes(dest.stat().st_size),
        )
    else:
        local_dir = cfg.path("raw")
        local_dir.mkdir(parents=True, exist_ok=True)
        log.info("metadata indiriliyor: %s [%s]", slug, remote)
        got = hf_hub_download(
            repo_id=cfg.get("dataset.hf_repo"),
            repo_type="dataset",
            filename=remote,
            local_dir=str(local_dir),
            force_download=force,
            token=os.environ.get("HF_TOKEN") or None,
        )
        dest = Path(got)
        log.info("indirildi: %s | %s", dest, human_bytes(dest.stat().st_size))

    probe = probe_schema(
        dest,
        list(cfg.get("dataset.meta_required_fields")),
        int(cfg.get("dataset.schema_probe_lines")),
    )
    log.info(
        "metadata şeması doğrulandı: %s | %d satır | %d alan: %s",
        slug,
        probe["lines_probed"],
        len(probe["fields"]),
        ", ".join(probe["fields"]),
    )
    write_json(probe, cfg.path("results", f"schema_probe_meta_{slug}.json"), log)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_standard_args(parser)
    parser.add_argument(
        "--meta",
        action="store_true",
        help="Review yerine ürün metadata dosyasını indir",
    )
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    fetch = download_metadata if args.meta else download_category
    for role in resolve_roles(cfg, args.category):
        fetch(cfg, role, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
