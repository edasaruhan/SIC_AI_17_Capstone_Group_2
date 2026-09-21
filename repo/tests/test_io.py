"""`utils/io` testleri.

Buradaki alanlar birer IDDIA: commit edilen JSON "bu sayilar su kodla uretildi"
diyor. Yanlis bir iddiayi sessizce yazmak, hic yazmamaktan kotudur - ciktiya
bakan kisi onu dogru sanar. `code_version` tam bunu yapiyordu: calisma agacinda
commit'lenmemis degisiklikle uretilen cikti, temiz bir commit'ten gelmis gibi
damgalaniyordu (denetim 2026-09-20).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gift_contamination.utils import io


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True,
    )


@pytest.fixture
def depo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("ilk = 1\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "r.json").write_text("{}", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "ilk")
    monkeypatch.setattr(io, "REPO_ROOT", tmp_path)
    return tmp_path


def test_code_version_is_the_short_hash_when_the_tree_is_clean(depo: Path):
    surum = io.code_version()

    assert surum and len(surum) >= 7
    assert not surum.endswith("-dirty")


def test_code_version_says_dirty_when_the_code_changed(depo: Path):
    """Commit'lenmemis KOD degisikligiyle uretilen cikti temiz gorunmemeli."""
    (depo / "src" / "a.py").write_text("ilk = 2\n", encoding="utf-8")

    assert io.code_version().endswith("-dirty")


def test_writing_a_report_does_not_make_the_stamp_dirty(depo: Path):
    """`reports/` CIKTI. Bir raporu yazmak bir sonrakini 'dirty' damgalasaydi
    damga anlamini yitirirdi - soru "bu sayilari ureten KOD commit'li miydi"."""
    (depo / "reports" / "r.json").write_text('{"yeni": 1}', encoding="utf-8")

    assert not io.code_version().endswith("-dirty")


def test_a_new_uncommitted_code_file_makes_the_stamp_dirty(depo: Path):
    """Kod yolunda henuz commit'lenmemis YENI bir modul de koddur.

    Eskiden `--untracked-files=no` ile soruluyordu ve bu test tersini kilitliyordu:
    commit'lenmemis yeni bir modulun urettigi cikti, o modulu HIC icermeyen bir
    commit'le temiz damgalaniyordu (`robustness_posthoc.json` ilk kosusunda, denetim
    2026-09-21). `__pycache__` ve `egg-info` gitignore'da; sayilmazlar.
    """
    (depo / "src" / "yeni.py").write_text("x = 1\n", encoding="utf-8")

    assert io.code_version().endswith("-dirty")


def test_an_ignored_file_on_a_code_path_does_not_make_the_stamp_dirty(depo: Path):
    (depo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(depo, "add", ".gitignore")
    _git(depo, "commit", "-q", "-m", "ignore")
    (depo / "src" / "__pycache__").mkdir()
    (depo / "src" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00")

    assert not io.code_version().endswith("-dirty")


def test_code_version_is_none_outside_a_repository(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch):
    """Kayit alani, kapi degil: git yoksa kosu durmamali."""
    monkeypatch.setattr(io, "REPO_ROOT", tmp_path)

    assert io.code_version() is None


def test_relative_to_repo_never_leaks_the_os_user_name(depo: Path):
    """Commit edilen JSON'a mutlak yol girerse isletim sistemi kullanici adi
    da girer - proje bunu yasakliyor (CLAUDE.md 8)."""
    icerde = depo / "reports" / "results" / "x.json"

    assert io.relative_to_repo(icerde) == "reports/results/x.json"
    # Depo disindaki yol yalnizca dosya adina duser.
    assert io.relative_to_repo(Path("D:/baska/yer/x.json")) == "x.json"
