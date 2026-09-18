# BangalASR-1000 benchmark results

This comparison covers the models listed in `models.json`.

| Rank | Model / decoder | Normalized WER | Normalized CER | Raw WER | Raw CER | RTF |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Speaklar Bangla ASR | **12.89%** | 3.62% | 13.19% | 3.74% | 0.0065 |
| 2 | Titu Conformer Large | 13.09% | **3.59%** | 13.35% | 3.71% | 0.0064 |
| 3 | AI4Bharat IndicConformer RNNT | 21.98% | 5.95% | 22.17% | 6.03% | 0.0121 |
| 4 | AI4Bharat IndicConformer CTC | 23.53% | 6.19% | 23.73% | 6.30% | 0.0055 |
| 5 | Titu FastConformer | 160.64% | 41.19% | 160.67% | 41.27% | 0.0036 |

All remaining models were evaluated on the same 1,000 recordings and 3,026 reference words. Lower WER/CER is better.
