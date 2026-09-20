# Duman testi çıktıları — sonuç tablosuna GİREMEZ

Buradaki dosyalar 2026-08-29'da Video_Games üzerinde koşturulan **kurulum
provası**ndan kalma. Gerçek deneyle aynı klasörde durdukları sürece adlarından
ayırt edilemiyorlardı (denetim 2026-09-20), o yüzden ayrıldılar.

Neden raporlanamazlar:

| Alan | Değer | Anlamı |
|---|---|---|
| `label_source` | `proxy` | Etiketler sözcüksel vekilden, damıtılmış modelden değil |
| `reportable` | `false` | `conditions.REPORTABLE_LABEL_SOURCES` dışında |
| `meta.epochs` | 5 | Tam eğitim değil |
| `meta.device` | `cpu` | Matris 2× T4'te koştu |
| `code_version` | yok | Alan o tarihte bu üreticide yoktu |
| `split.test_pairs_sha256` | yok | Kapı 2'nin 1. ölçütü bu özeti gerektiriyor |

Video_Games deneye hiç girmedi: `experiment.categories` yalnızca `high` ve
`low` içeriyor, ve 5-core'da kontaminasyon oranı deneyi taşıyacak kadar yüksek
değil (`docs/SONUCLAR.md` §7, madde 6).

`condition_Video_Games_C3.json` yok — duman testi C3'ten önce yapıldı.
