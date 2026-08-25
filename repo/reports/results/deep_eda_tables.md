# Deep EDA tables

_Uretim: `python -m gift_contamination.analysis.deep_eda`_


## T5 detection surface

| category | gift-flagged reviews | evidence in title | median position in body | 90th pct position | evidence lost @512 tok | evidence lost @1024 tok | recipient recoverable | occasion recoverable |
|---|---|---|---|---|---|---|---|---|
| Toys and Games | 1,374,807 | 11.2% | 0.031 | 0.580 | 0.013% | 0.001% | 78.7% | 29.0% |
| Video Games | 145,020 | 9.8% | 0.011 | 0.464 | 0.044% | 0.008% | 87.7% | 25.6% |
| Grocery and Gourmet Food | 198,944 | 9.3% | 0.030 | 0.608 | 0.013% | 0.001% | 78.3% | 24.6% |
| All Beauty | 11,451 | 8.3% | 0.024 | 0.607 | 0.018% | 0.000% | 80.9% | 19.9% |


## T6 recipients

| category | recipient | n | share |
|---|---|---|---|
| Toys and Games | son | 273,736 | 29.1% |
| Toys and Games | daughter | 244,549 | 26.0% |
| Toys and Games | grandson | 138,015 | 14.7% |
| Toys and Games | granddaughter | 107,731 | 11.5% |
| Toys and Games | niece | 58,827 | 6.3% |
| Toys and Games | nephew | 57,386 | 6.1% |
| Toys and Games | kids | 35,761 | 3.8% |
| Toys and Games | kid | 23,272 | 2.5% |
| Video Games | son | 54,233 | 49.8% |
| Video Games | daughter | 13,074 | 12.0% |
| Video Games | grandson | 12,690 | 11.6% |
| Video Games | husband | 10,179 | 9.3% |
| Video Games | boyfriend | 5,946 | 5.5% |
| Video Games | nephew | 5,197 | 4.8% |
| Video Games | brother | 3,951 | 3.6% |
| Video Games | kids | 3,700 | 3.4% |
| Grocery and Gourmet Food | husband | 29,726 | 26.0% |
| Grocery and Gourmet Food | son | 22,098 | 19.3% |
| Grocery and Gourmet Food | daughter | 21,847 | 19.1% |
| Grocery and Gourmet Food | wife | 12,364 | 10.8% |
| Grocery and Gourmet Food | mom | 10,135 | 8.9% |
| Grocery and Gourmet Food | sister | 6,260 | 5.5% |
| Grocery and Gourmet Food | mother | 6,090 | 5.3% |
| Grocery and Gourmet Food | kids | 5,942 | 5.2% |
| All Beauty | daughter | 2,896 | 38.8% |
| All Beauty | husband | 1,095 | 14.7% |
| All Beauty | wife | 1,031 | 13.8% |
| All Beauty | son | 602 | 8.1% |
| All Beauty | mom | 574 | 7.7% |
| All Beauty | sister | 504 | 6.7% |
| All Beauty | granddaughter | 437 | 5.8% |
| All Beauty | niece | 332 | 4.4% |


## T6b occasions

| category | occasion | n | share |
|---|---|---|---|
| Toys and Games | birthday | 192,775 | 49.3% |
| Toys and Games | christmas | 182,652 | 46.7% |
| Toys and Games | xmas | 8,672 | 2.2% |
| Toys and Games | baby shower | 3,477 | 0.9% |
| Toys and Games | graduation | 1,673 | 0.4% |
| Toys and Games | valentine | 1,458 | 0.4% |
| Video Games | christmas | 23,002 | 62.8% |
| Video Games | birthday | 11,760 | 32.1% |
| Video Games | xmas | 1,454 | 4.0% |
| Video Games | anniversary | 193 | 0.5% |
| Video Games | father's day | 137 | 0.4% |
| Video Games | valentine's day | 106 | 0.3% |
| Grocery and Gourmet Food | birthday | 19,435 | 45.2% |
| Grocery and Gourmet Food | christmas | 19,078 | 44.4% |
| Grocery and Gourmet Food | wedding | 1,571 | 3.7% |
| Grocery and Gourmet Food | xmas | 990 | 2.3% |
| Grocery and Gourmet Food | valentine's day | 968 | 2.3% |
| Grocery and Gourmet Food | mother's day | 957 | 2.2% |
| All Beauty | christmas | 1,083 | 49.6% |
| All Beauty | birthday | 864 | 39.6% |
| All Beauty | wedding | 131 | 6.0% |
| All Beauty | xmas | 66 | 3.0% |
| All Beauty | mother's day | 21 | 1.0% |
| All Beauty | baby shower | 18 | 0.8% |


## T7 sequence feasibility

| category | users in k-core | with ≥1 gift | last item is a gift | entire sequence is gifts | **eval-eligible users** | mean contamination (gift users) |
|---|---|---|---|---|---|---|
| Toys and Games | 268,652 | 128,546 (47.8%) | 11.32% | 0.146% | **88.7%** | 24.2% |
| Video Games | 47,663 | 5,958 (12.5%) | 2.77% | 0.109% | **97.2%** | 22.5% |
| Grocery and Gourmet Food | 268,991 | 25,767 (9.6%) | 1.33% | 0.003% | **98.7%** | 14.7% |
| All Beauty | 0 | — | — | — | **n/a — no k-core** | — |


## T8 clean vs kcore

| category | clean corpus | k-core corpus | shift | k-core interactions |
|---|---|---|---|---|
| Toys and Games | 11.07% | 11.27% | +1.8% | 2,164,018 |
| Video Games | 4.40% | 2.68% | -39.2% | 368,476 |
| Grocery and Gourmet Food | 1.85% | 1.34% | -27.5% | 2,434,594 |
| All Beauty | 2.13% | — | k-core bos |  |


## T9 confounds

| category | gift rate, shortest 20% | gift rate, longest 20% | length ratio | unverified share (raw) | non-ASCII text | duplicated text, raw | duplicated text, after filters |
|---|---|---|---|---|---|---|---|
| Toys and Games | 8.23% | 12.56% | 1.53× | 8.6% | 11.75% | 15.90% | 3.47% |
| Video Games | 4.38% | 2.64% | 0.60× | 13.9% | 8.19% | 14.67% | 2.30% |
| Grocery and Gourmet Food | 0.96% | 2.27% | 2.36× | 8.0% | 10.92% | 14.89% | 1.96% |
| All Beauty | 1.39% | 2.37% | 1.70× | 9.5% | 14.83% | 10.09% | 0.49% |


## T10 cross category

| metric | value | of |
|---|---|---|
| customers observed in ≥2 of the four categories | 2,316,411 | 12,690,408 distinct customers (18.3%) |
| …of those, entered ≥1 category **only** via a gift | 207,155 | 8.9% of multi-category customers |
| customers in all four categories | 15,162 | upper bound on a full cross-category profile |


## T11 temporal resolution

| category | consecutive pairs on the same day | median gap (days) | **held-out item same day as previous** | median gap before held-out item (days) |
|---|---|---|---|---|
| Toys and Games | 43.0% | 15 | **31.2%** | 93 |
| Video Games | 42.3% | 17 | **31.5%** | 101 |
| Grocery and Gourmet Food | 34.8% | 34 | **24.5%** | 99 |


## T12 gift rating signature

| category | gift mean | rest mean | difference | gift 5★ | rest 5★ |
|---|---|---|---|---|---|
| Toys and Games | 4.517 | 4.104 | +0.412 | 77.8% | 63.7% |
| Video Games | 4.482 | 3.974 | +0.508 | 77.6% | 58.6% |
| Grocery and Gourmet Food | 4.490 | 4.049 | +0.442 | 79.0% | 64.8% |
| All Beauty | 4.542 | 3.870 | +0.673 | 78.6% | 57.2% |


## T13 duplication by length

| category | length (words) | reviews | duplicated text |
|---|---|---|---|
| Toys and Games | 1-2 | 1,146,836 | 92.5% |
| Toys and Games | 3-5 | 1,778,495 | 53.5% |
| Toys and Games | 6-10 | 2,368,574 | 9.7% |
| Toys and Games | 11-25 | 4,649,687 | 3.5% |
| Toys and Games | 26-50 | 3,527,746 | 2.9% |
| Toys and Games | 50+ | 2,789,068 | 2.8% |
| Video Games | 1-2 | 388,192 | 91.4% |
| Video Games | 3-5 | 409,901 | 44.6% |
| Video Games | 6-10 | 517,752 | 7.5% |
| Video Games | 11-25 | 1,076,198 | 3.7% |
| Video Games | 26-50 | 930,543 | 2.9% |
| Video Games | 50+ | 1,302,029 | 2.7% |
| Grocery and Gourmet Food | 1-2 | 1,222,977 | 91.9% |
| Grocery and Gourmet Food | 3-5 | 1,611,148 | 41.8% |
| Grocery and Gourmet Food | 6-10 | 2,193,877 | 5.4% |
| Grocery and Gourmet Food | 11-25 | 4,153,328 | 2.4% |
| Grocery and Gourmet Food | 26-50 | 2,981,719 | 2.3% |
| Grocery and Gourmet Food | 50+ | 2,155,471 | 2.4% |
| All Beauty | 1-2 | 50,058 | 81.9% |
| All Beauty | 3-5 | 68,096 | 25.0% |
| All Beauty | 6-10 | 101,006 | 2.8% |
| All Beauty | 11-25 | 201,482 | 2.0% |
| All Beauty | 26-50 | 153,037 | 2.1% |
| All Beauty | 50+ | 127,849 | 2.0% |


## T14 seasonality summary

| category | Jan | Feb | Jun–Sep trough (mean) | Nov | Dec | Dec ÷ summer |
|---|---|---|---|---|---|---|
| Toys and Games | 13.72% | 11.88% | 9.67% | 10.60% | 12.94% | 1.34× |
| Video Games | 6.97% | 4.97% | 3.51% | 3.42% | 5.90% | 1.68× |
| Grocery and Gourmet Food | 2.68% | 2.15% | 1.52% | 1.61% | 2.70% | 1.77× |
| All Beauty | 3.50% | 2.35% | 1.75% | 1.98% | 3.33% | 1.90× |
