# Bengali ASR benchmark

Reproducible Bengali automatic speech recognition benchmark for five publicly released NeMo checkpoints, evaluated on the BangalASR 1,000-recording test set. It computes corpus-level word error rate (WER), character error rate (CER), predictions, and real-time factor under a pinned protocol.

The BangalASR dataset is downloaded and validated: **1,000 recordings, 3,463.8 seconds (57.73 minutes)**. Every CSV reference matches one WAV. All source files are 16 kHz mono. No duplicate audio hashes or unused audio were found.

The complete audio/transcript mapping is [data/asr_bangal_1000.csv](data/asr_bangal_1000.csv). It contains 1,000 rows with `audio` and `text` columns. The corresponding WAV archive is [data/asr_bengali_1000_wav.zip](data/asr_bengali_1000_wav.zip).

**Status: the remaining models completed on all 1,000 recordings on a Tesla T4.** IndicConformer's CTC and RNNT decoders were evaluated separately. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for the ranked results and `results/` for all predictions, metrics, and run metadata.

## Benchmark results

Normalized corpus WER/CER on the 1,000-recording BangalASR test set; lower is better:

| Rank | Model / decoder | WER | CER |
|---:|---|---:|---:|
| 1 | Speaklar Bangla ASR | **12.89%** | 3.62% |
| 2 | Titu Conformer Large | 13.09% | **3.59%** |
| 3 | AI4Bharat IndicConformer RNNT | 21.98% | 5.95% |
| 4 | AI4Bharat IndicConformer CTC | 23.53% | 6.19% |
| 5 | Titu FastConformer | 160.64% | 41.19% |

Full methodology, raw scores, edit counts, runtime, and reproducibility metadata are in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

For an utterance-by-utterance comparison, use [results/all_transcripts.csv](results/all_transcripts.csv). Each of its 1,000 rows contains the audio path, the original reference transcript, and the prediction from Speaklar, Titu Large, Titu Fast, IndicConformer CTC, and IndicConformer RNNT.

## Models

`models.json` pins each Hugging Face repository to its inspected commit and selects an explicit checkpoint:

| CLI name | Repository | Environment |
|---|---|---|
| titu-large | hishab/titu_stt_bn_conformer_large | Standard |
| titu-fast | hishab/titu_stt_bn_fastconformer | Standard |
| speaklar | munzurul/speaklar_bangla_asr | Speaklar |
| indic | ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large | AI4Bharat |

Speaklar uses `speaklar_bangla_asr.nemo`, not an arbitrarily selected epoch. Indic uses CTC by default; RNNT is an explicitly labeled alternate run. Other models retain their checkpoint decoder defaults. This comparison measures published configurations, not identical decoding algorithms.

## Prepare or re-download

From this directory:

```bash
python -m pip install soundfile scipy
python download_data.py
python benchmark.py prepare
python -m unittest -v
```

Existing downloads are checked against SHA-256 hashes. Upstream changes fail verification. The manifest uses project-relative audio paths, so the whole directory can be moved to a GPU machine. The original CSV paths are repaired by exact basename matching, with duplicate names rejected.

## Standard models: Python 3.11 and GPU recommended

Create a clean environment, install a matching PyTorch/torchaudio pair for your CUDA driver, then:

```bash
python -m pip install -r requirements.txt
python benchmark.py run --model titu-large --device cuda
python benchmark.py run --model titu-fast --device cuda
```

Use `--device cpu` for CPU execution or reduce `--batch-size` if GPU memory is insufficient. Runs refuse to overwrite an existing output directory; choose a new `--output` directory for retries. A failed batch fails the run instead of silently dropping samples. Partial predictions remain available for debugging but receive no final metrics.

## Authentication

Private/gated checkpoints use `HF_TOKEN` from the process environment. The supplied credential was verified to access Speaklar and IndicConformer; it is not saved in this project. Set the variable through your runtime's secret manager or a hidden prompt:

```python
import getpass, os
os.environ['HF_TOKEN'] = getpass.getpass('Hugging Face token: ')
```

Run benchmark commands as children of that Python process, or export the variable securely in your shell. An account must have accepted the IndicConformer access conditions.

## Speaklar: separate environment

The publisher documents Python 3.10/3.11, Torch 2.6 and NeMo 2.4. Use a separate Python 3.11 environment:

```bash
python -m pip install -r requirements-speaklar.txt
python benchmark.py run --model speaklar --device cuda --limit 8 --output smoke
python benchmark.py run --model speaklar --device cuda
```

## IndicConformer: separate environment

The publisher requires the AI4Bharat NeMo fork. In a separate Python environment, follow its installation instructions:

```bash
git clone --branch nemo-v2 https://github.com/AI4Bharat/NeMo.git vendor/NeMo
cd vendor/NeMo
bash reinstall.sh
cd ../..
python -m pip install soundfile scipy huggingface_hub
python benchmark.py run --model indic --decoder ctc --device cuda --limit 8 --output smoke
python benchmark.py run --model indic --decoder ctc --device cuda
# Optional alternate decoder, reported separately:
python benchmark.py run --model indic --decoder rnnt --device cuda
```

This environment has not been installed or tested here. Record the fork commit and `pip freeze` alongside results, since the publisher's branch may change. The adapter checks for the fork's `language_id` API and fails clearly if loaded in an incompatible environment.

## Evaluation protocol and outputs

Each model evaluates the same 1,000 rows in CSV order with no fine-tuning. `--limit` creates a labeled subset for smoke tests only. Each run writes `predictions.jsonl`, `metrics.json`, and `run.json` under `results/<model>/`.

- Corpus WER = total word edit distance / total reference words; never the average of utterance WERs.
- CER counts Unicode code points excluding whitespace, not Bengali grapheme clusters.
- Raw scores collapse whitespace only. Normalized scores apply Unicode NFC, remove Unicode format characters, replace punctuation with spaces, and collapse whitespace on both references and predictions. Digits, spelling and Bengali vowel marks are preserved.
- RTF = timed inference-loop seconds / audio seconds. It includes audio loading, decoding and prediction writing; excludes model loading, downloads and one warmup. CUDA is synchronized. Compare speed only on the same hardware, environment and batch size.
- Metadata includes pinned model revision, manifest hash, package versions, device, batch size and completion status. Preserve `pip freeze` for exact dependency reproduction.

Dataset/model training overlap has not been audited. These scores should be described as performance on this dataset, not proven held-out generalization.

## Test data

The test data comes from [menon92/BangalASR, `data/asr-bengali-1000`](https://github.com/menon92/BangalASR/tree/main/data/asr-bengali-1000). The repository includes the original CSV and WAV archive under `data/`. To download and verify a fresh copy instead, run:

```bash
python download_data.py
python benchmark.py prepare
```

`download_data.py` verifies the upstream CSV and WAV archive with SHA-256 checksums before extraction. The benchmark does not require a Hugging Face or GitHub token for this public test data. A Hugging Face token is only needed for gated model repositories such as IndicConformer; provide it through `HF_TOKEN` and never commit it.
