# Semantic E2E-VGuard

This repository combines E2E-VGuard timbre protection with the T-SemAttack
semantic objective. The project-owned Python code is split into reusable logic
under `core/` and executable workflows under `scripts/`.

## Layout

```text
core/
  guard.py          optimization loop and timbre/quality objectives
  encoders.py       S3, HuBERT, Whisper, and MFCC semantic ensemble
  masking.py        psychoacoustic masking model
  modeling.py       VITS model adapter
  evaluation.py     ASR, audio quality, speaker, TTS, and robustness evaluation
  utils.py          runtime, device, CSV, and JSON helpers
scripts/
  protect.py        single-file and manifest protection
  evaluate.py       ASR, TTS, speaker, and robustness evaluation
  prepare.py        model, dataset, and listening-test preparation
  synthesize.py     paired-seed XTTS-v2 generation
tts_models/          vendored model components required by the core
```

The previous top-level entrypoints were merged into these four scripts. Run all
commands from this directory, for example:

```bash
cd /home/ljh/ciscn/code/src
python -m scripts.protect --help
```

## Recommended Pipeline

The current main result is `lq25_large_balanced`: 100 optimization steps,
HuBERT-large, Whisper-large-v3, a `4/255` bound, and a 25 dB SNR constraint.
It is the default batch preset and reproduces the configuration recorded in
`outputs/06_large_balanced_20260804/full50/commands/`.

Protect a manifest:

```bash
python -m scripts.protect batch \
  --manifest /home/ljh/ciscn/data/full50_hf_20260619_143424/manifest.csv \
  --quality_preset lq25_large_balanced \
  --device cuda \
  --output_dir /home/ljh/ciscn/outputs/full50_lq25_large_balanced
```

The historical `q18_perceptual` and `q24_perceptual` presets remain available
for comparison. Arguments after `--` override preset values.

Protect one audio file with explicit settings:

```bash
python -m scripts.protect single \
  --input_wav path/to/input.wav \
  --output_wav path/to/protected.wav \
  --epochs 100 \
  --device cuda
```

Evaluate direct ASR degradation and objective audio quality:

```bash
python -m scripts.evaluate asr \
  --manifest /home/ljh/ciscn/outputs/full50_lq25_large_balanced/protected_lq25_large_balanced.csv \
  --asr_models openai-whisper:medium,facebook/wav2vec2-base-960h \
  --device cuda \
  --output_dir /home/ljh/ciscn/outputs/full50_lq25_large_balanced/asr_eval
```

Generate paired-seed XTTS-v2 samples:

```bash
python -m scripts.synthesize \
  --references /home/ljh/ciscn/outputs/full50_lq25_large_balanced/protected_lq25_large_balanced.csv \
  --target_text "This is a controlled downstream text to speech evaluation." \
  --output_dir /home/ljh/ciscn/outputs/tts_lq25_large_balanced \
  --device cuda
```

Evaluate post-TTS ASR and speaker similarity against original clean speakers:

```bash
python -m scripts.evaluate tts \
  --manifest /home/ljh/ciscn/outputs/tts_lq25_large_balanced/tts_manifest.csv \
  --asr_models openai-whisper:medium,facebook/wav2vec2-base-960h \
  --speaker_metric ecapa \
  --device cuda \
  --output_dir /home/ljh/ciscn/outputs/tts_lq25_large_balanced/eval
```

## Supporting Commands

Download required weights:

```bash
python -m scripts.prepare models
```

Fetch a small known-transcript LibriTTS subset:

```bash
python -m scripts.prepare dataset \
  --split dev.clean \
  --max_items 5 \
  --output_dir /home/ljh/ciscn/data/libritts_devclean_small
```

Create robustness transforms and evaluate them:

```bash
python -m scripts.evaluate robustness \
  --manifest /home/ljh/ciscn/outputs/full50_lq25_large_balanced/protected_lq25_large_balanced.csv \
  --source_condition lq25_large_balanced \
  --output_dir /home/ljh/ciscn/outputs/robustness

python -m scripts.evaluate asr \
  --manifest /home/ljh/ciscn/outputs/robustness/robust_manifest.csv \
  --asr_models openai-whisper:medium,facebook/wav2vec2-base-960h \
  --device cuda \
  --output_dir /home/ljh/ciscn/outputs/robustness/asr_eval
```

Create a blinded listening set:

```bash
python -m scripts.prepare listening \
  --manifest outputs/conditions.csv \
  --conditions clean q18_perceptual q24_perceptual lq25_large_balanced \
  --sample_count 20 \
  --output_dir outputs/listening_test
```

See `outputs/README.md` for the result index. Existing CSV, JSON, manifest, and
audio evidence is intentionally independent of this source layout refactor.

## Environment

The remote environment uses `/home/ljh/ciscn/.venv/bin/python`. Prefer the
configured Hugging Face and PyPI mirrors:

```bash
cd /home/ljh/ciscn
export HF_ENDPOINT=https://hf-mirror.com
export UV_HTTP_TIMEOUT=120
uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r code/src/requirements.txt
```

XTTS-v2 is optional and still uses the environment adaptation in
`scripts/synthesize.py`.
