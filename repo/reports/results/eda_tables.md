# EDA tables

_Uretim: `python -m gift_contamination.analysis.eda`_


## T1 preprocessing funnel

| category | stage | rows | users | items | retained |
|---|---|---|---|---|---|
| Toys and Games | 01_raw | 16,260,406 | 8,116,226 | 890,667 | 100.0% |
| Toys and Games | 02_verified_purchase | 14,859,195 | 7,678,450 | 855,903 | 91.4% |
| Toys and Games | 03_min_words_5 | 12,574,302 | 6,824,523 | 801,297 | 77.3% |
| Toys and Games | 04_dedup | 12,417,784 | 6,824,523 | 801,297 | 76.4% |
| Toys and Games | 05_k_core_5 | 2,164,018 | 268,652 | 103,618 | 13.3% |
| Video Games | 01_raw | 4,624,615 | 2,766,656 | 137,249 | 100.0% |
| Video Games | 02_verified_purchase | 3,982,807 | 2,507,761 | 129,039 | 86.1% |
| Video Games | 03_min_words_5 | 3,341,936 | 2,185,291 | 122,578 | 72.3% |
| Video Games | 04_dedup | 3,296,440 | 2,185,291 | 122,578 | 71.3% |
| Video Games | 05_k_core_5 | 368,476 | 47,663 | 14,606 | 8.0% |
| Grocery and Gourmet Food | 01_raw | 14,318,520 | 7,034,393 | 603,182 | 100.0% |
| Grocery and Gourmet Food | 02_verified_purchase | 13,176,519 | 6,645,166 | 572,844 | 92.0% |
| Grocery and Gourmet Food | 03_min_words_5 | 10,940,987 | 5,801,858 | 541,797 | 76.4% |
| Grocery and Gourmet Food | 04_dedup | 10,774,599 | 5,801,858 | 541,797 | 75.2% |
| Grocery and Gourmet Food | 05_k_core_5 | 2,434,594 | 268,991 | 94,863 | 17.0% |
| All Beauty | 01_raw | 701,528 | 631,986 | 112,565 | 100.0% |
| All Beauty | 02_verified_purchase | 634,969 | 584,592 | 106,811 | 90.5% |
| All Beauty | 03_min_words_5 | 543,045 | 503,388 | 99,296 | 77.4% |
| All Beauty | 04_dedup | 537,261 | 503,388 | 99,296 | 76.6% |
| All Beauty | 05_k_core_5 | 0 | 0 | 0 | 0.0% |


## T2 corpus profile

| category | interactions | users | items | density | int/user (mean) | int/user (median) | users ≥5 | int/item (mean) | items ≥5 | item Gini | median words | period |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Toys and Games | 12,417,784 | 6,824,523 | 801,297 | 2.27e-06 | 1.82 | 1 | 5.71% | 15.50 | 37.1% | 0.805 | 22 | 2000-08 → 2023-09 |
| Video Games | 3,296,440 | 2,185,291 | 122,578 | 1.23e-05 | 1.51 | 1 | 3.03% | 26.89 | 42.3% | 0.850 | 27 | 1999-03 → 2023-09 |
| Grocery and Gourmet Food | 10,774,599 | 5,801,858 | 541,797 | 3.43e-06 | 1.86 | 1 | 5.85% | 19.89 | 40.2% | 0.826 | 21 | 2001-08 → 2023-09 |
| All Beauty | 537,261 | 503,388 | 99,296 | 1.07e-05 | 1.07 | 1 | 0.08% | 5.41 | 21.6% | 0.684 | 22 | 2002-10 → 2023-09 |


## T3 keyword proxy rates

| category | reviews | gift evidence | speculative | received | **proxy rate** | naive single-regex | inflation |
|---|---|---|---|---|---|---|---|
| Toys and Games | 12,417,784 | 11.30% | 0.92% | 0.28% | **11.07%** | 11.91% | 1.07× |
| Video Games | 3,296,440 | 4.50% | 0.14% | 0.13% | **4.40%** | 4.61% | 1.05× |
| Grocery and Gourmet Food | 10,774,599 | 1.89% | 0.19% | 0.08% | **1.85%** | 2.08% | 1.12× |
| All Beauty | 537,261 | 2.18% | 0.15% | 0.08% | **2.13%** | 2.32% | 1.09× |
