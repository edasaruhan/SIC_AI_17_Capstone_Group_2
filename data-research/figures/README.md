# Figures — published copy

These 16 PNGs are **not maintained by hand**. They are written automatically by
the EDA commands, which copy every figure they produce into this folder so the
Data Research submission renders with its images without reaching outside its
own directory:

```yaml
# repo/configs/base.yaml
eda:
  publish_figures_to: ../data-research/figures
```

The canonical source is [`../../repo/reports/figures/`](../../repo/reports/figures/).
Both copies were byte-identical when last checked (20 September 2026).

To regenerate:

```bash
cd repo
python -m gift_contamination.analysis.eda       --config configs/base.yaml   # F1–F8
python -m gift_contamination.analysis.deep_eda  --config configs/base.yaml   # F9–F16
```

Edit a figure by changing the code that draws it, never the PNG — the next EDA
run overwrites this folder.
