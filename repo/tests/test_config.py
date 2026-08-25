"""Config yukleme ve yol cozumleme testleri.

Config projenin tek dogruluk kaynagi (CLAUDE.md bolum 2); bir yazim hatasinin sessizce
None donup deneyi bozmasi en pahali hata sinifi. Bu yuzden eksik anahtar hata atmali.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gift_contamination.config import (
    REPO_ROOT,
    Config,
    ConfigError,
    resolve_roles,
)


def test_real_config_loads():
    """Repodaki gercek configs/base.yaml yuklenebiliyor mu."""
    conf = Config.load()

    assert conf.get("dataset.hf_repo") == "McAuley-Lab/Amazon-Reviews-2023"
    assert conf.get("dataset.join_key") == "parent_asin", "asin ile join YASAK"
    assert set(conf.category_roles()) == {"pilot", "high", "mid", "low"}


def test_missing_key_raises_instead_of_returning_none():
    conf = Config({"a": {"b": 1}}, Path("test.yaml"))

    assert conf.get("a.b") == 1
    with pytest.raises(ConfigError, match="yazim-hatasi"):
        conf.get("a.yazim-hatasi")


def test_missing_key_returns_default_when_given():
    conf = Config({"a": {"b": 1}}, Path("test.yaml"))

    assert conf.get("a.yok", default="varsayilan") == "varsayilan"


def test_relative_paths_resolve_against_repo_root(cfg: Config):
    conf = Config({"paths": {"interim": "data/interim"}}, Path("test.yaml"))

    assert conf.path("interim") == REPO_ROOT / "data" / "interim"
    assert conf.path("interim", "x.parquet").name == "x.parquet"


def test_absolute_paths_are_left_alone(cfg: Config, tmp_path: Path):
    """Config mutlak yol veriyorsa (orn. veri baska diskte) dokunulmamali."""
    assert cfg.path("interim") == tmp_path / "interim"


def test_category_slug_strips_hf_prefix():
    conf = Config.load()

    assert conf.category_slug("pilot") == "All_Beauty"
    assert conf.category_slug("high") == "Toys_and_Games"


def test_unknown_category_role_raises():
    conf = Config.load()

    with pytest.raises(ConfigError, match="Bilinmeyen kategori"):
        conf.category_slug("boyle-bir-rol-yok")


def test_resolve_roles_expands_all():
    conf = Config.load()

    assert resolve_roles(conf, "all") == conf.category_roles()
    assert resolve_roles(conf, "pilot") == ["pilot"]
