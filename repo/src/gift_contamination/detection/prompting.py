"""Prompt dosyasini okuma ve render etme.

`prompts/gift_detection_v1.md` bir KOD ARTEFAKTIDIR: prompt degisirse yeni dosya
acilir (`v2`), uzerine yazilmaz. Her annotation kaydinda hangi surumle uretildigi
`prompt_version` alaninda saklanir - prompt dosyasinin kendisi bunu sart kosuyor.
Bu modul o surumu dosyadan okur, boylece elle girilen bir sabitle kayabilecek bir
alan olmaz.

Dosya bicimi: `## SYSTEM` ve `## USER` basliklarinin altinda birer fenced blok.

Sessiz basarisizlik yok: eksik blok, eksik yer tutucu veya doldurulmamis yer
tutucu istisna atar. 40-60K'lik bir kosuda bos bir SYSTEM prompt'u ile devam
etmek, hatanin ancak sonuclara bakildiginda fark edilmesi demektir.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from ..config import REPO_ROOT, Config

# `## SYSTEM` ... ```<icerik>```   (dil etiketi opsiyonel)
_BLOCK = r"^##\s+{name}\s*$.*?^```[a-zA-Z]*\s*$\n(?P<body>.*?)^```\s*$"
_VERSION = re.compile(r"^\*\*Versiyon:\*\*\s*(?P<v>[^\s·]+)", re.MULTILINE)

# USER blogunda beklenen yer tutucular. Fazlasi veya eksigi hatadir.
REQUIRED_FIELDS = ("product_title", "category", "title", "text")
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class Prompt(NamedTuple):
    version: str
    system: str
    user_template: str
    source: Path

    def render(self, *, product_title: str, category: str, title: str, text: str) -> str:
        """USER blogunu doldurur. SYSTEM ayri gonderilir (chat rolleri).

        `str.format` KULLANILMIYOR, bilerek. Review metni kullanici icerigi ve
        icinde suslu parantez gecebiliyor - gercek bir ornek: "...the {LOOSE}
        piece..." `format` bunu bir yer tutucu sanip `KeyError` atiyordu ve hata
        11.800 satirlik kosunun ortasinda cikiyordu. Duz `replace` ile review
        metni asla sablon olarak yorumlanmaz.
        """
        filled = self.user_template
        for field, value in (
            ("product_title", product_title), ("category", category),
            ("title", title), ("text", text),
        ):
            filled = filled.replace("{" + field + "}", value)

        # Kontrol yalnizca BILDIGIMIZ yer tutucularda: doldurulmus metnin
        # icindeki `{LOOSE}` bizim sorunumuz degil, review'un icerigi.
        left = [f for f in REQUIRED_FIELDS if "{" + f + "}" in filled]
        if left:
            raise ValueError(
                f"{self.source.name}: doldurulmamis yer tutucu kaldi: {sorted(left)}"
            )
        return filled

    def messages(
        self, *, product_title: str, category: str, title: str, text: str
    ) -> list[dict[str, str]]:
        """vLLM / OpenAI uyumlu mesaj listesi."""
        return [
            {"role": "system", "content": self.system},
            {
                "role": "user",
                "content": self.render(
                    product_title=product_title, category=category, title=title, text=text
                ),
            },
        ]


def prompt_path(cfg: Config | None = None) -> Path:
    """Config'in gosterdigi prompt dosyasi.

    cfg verilmezse varsayilan config OKUNUR, dosya adi burada tekrarlanmaz.
    Tekrarlansaydi prompt versiyonu bumplendiginde config degisir ama bu
    fonksiyon eski dosyayi dondurmeye devam ederdi - ve fark ancak annotation
    ciktisindaki `prompt_version` alanina bakildiginda anlasilirdi.
    """
    raw = Path((cfg or Config.load()).get("detection.prompt_path"))
    return raw if raw.is_absolute() else REPO_ROOT / raw


def _block(markdown: str, name: str, source: Path) -> str:
    match = re.search(
        _BLOCK.format(name=name), markdown, re.MULTILINE | re.DOTALL
    )
    if not match:
        raise ValueError(
            f"{source.name}: '## {name}' basligi altinda fenced blok bulunamadi"
        )
    return match.group("body").strip()


def load_prompt(cfg: Config | None = None, path: Path | None = None) -> Prompt:
    src = path or prompt_path(cfg)
    if not src.exists():
        raise FileNotFoundError(f"Prompt dosyasi yok: {src}")
    markdown = src.read_text(encoding="utf-8")

    version_match = _VERSION.search(markdown)
    if not version_match:
        raise ValueError(
            f"{src.name}: '**Versiyon:**' satiri yok. Her annotation kaydi "
            "prompt surumunu tasimak zorunda."
        )

    user_template = _block(markdown, "USER", src)
    fields = set(_PLACEHOLDER.findall(user_template))
    if fields != set(REQUIRED_FIELDS):
        raise ValueError(
            f"{src.name}: USER blogundaki yer tutucular {sorted(fields)}, "
            f"beklenen {sorted(REQUIRED_FIELDS)}"
        )

    return Prompt(
        version=version_match.group("v"),
        system=_block(markdown, "SYSTEM", src),
        user_template=user_template,
        source=src,
    )


def render(
    product_title: str, category: str, title: str, text: str, cfg: Config | None = None
) -> str:
    """Tek seferlik kullanim icin kisayol; toplu kosuda `load_prompt` bir kez cagirilir."""
    return load_prompt(cfg).render(
        product_title=product_title, category=category, title=title, text=text
    )
