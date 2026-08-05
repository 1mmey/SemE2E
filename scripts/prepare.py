from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download

from core.utils import read_csv_rows, write_csv_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def _download_files(repo_id: str, files: list[str], download_path: Path) -> None:
    download_path.mkdir(parents=True, exist_ok=True)
    for filename in files:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(download_path),
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded {filename}")


def download_models(_args: argparse.Namespace) -> None:
    print("Downloading final semantic surrogate models...")
    for repo_id, target in (
        ("facebook/hubert-large-ll60k", ROOT / "checkpoints" / "hf" / "facebook" / "hubert-large-ll60k"),
        ("openai/whisper-large-v3", ROOT / "checkpoints" / "hf" / "openai" / "whisper-large-v3"),
    ):
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )

    print("Downloading GPT-SoVITS SoVITS checkpoint...")
    _download_files(
        "lj1995/GPT-SoVITS",
        ["gsv-v2final-pretrained/s2G2333k.pth"],
        ROOT / "checkpoints" / "GSV" / "base_models",
    )
    print("Downloading WavLM...")
    snapshot_download(
        repo_id="microsoft/wavlm-base-plus",
        local_dir=str(ROOT / "checkpoints" / "wavlm"),
        local_dir_use_symlinks=False,
    )
    print("Downloading CosyVoice encoders...")
    cosyvoice_dir = ROOT / "checkpoints" / "CosyVoice" / "base_models" / "CosyVoice-300M"
    _download_files(
        "FunAudioLLM/CosyVoice-300M",
        ["campplus.onnx", "speech_tokenizer_v1.onnx"],
        cosyvoice_dir,
    )
    tokenizer_source = cosyvoice_dir / "speech_tokenizer_v1.onnx"
    tokenizer_target = ROOT / "checkpoints" / "CosyVoice" / "speech_tokenizer_v1.onnx"
    tokenizer_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tokenizer_source, tokenizer_target)

    print("Downloading VITS checkpoint...")
    _download_files(
        "csukuangfj/vits-ljs",
        ["pretrained_ljs.pth"],
        ROOT / "checkpoints" / "VITS",
    )


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip()).strip("_") or "item"


def _iter_libritts_rows(args: argparse.Namespace):
    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")
    from datasets import Audio, load_dataset

    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    for row_index, row in enumerate(dataset):
        if row_index < args.offset:
            continue
        if row_index >= args.max_scan:
            break
        yield row_index, row


def fetch_dataset(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    selected = []
    for row_index, row in _iter_libritts_rows(args):
        text = row["text_normalized"].strip()
        if not args.min_text_chars <= len(text) <= args.max_text_chars:
            continue
        wav_path = args.output_dir / "wavs" / f"{_slug(row['id'])}.wav"
        audio_bytes = (row.get("audio") or {}).get("bytes")
        if not audio_bytes:
            raise ValueError(f"row {row.get('id', '<unknown>')} has no audio bytes")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(audio_bytes)
        info = sf.info(str(wav_path))
        duration = float(info.frames) / float(info.samplerate)
        if not args.min_duration <= duration <= args.max_duration:
            continue
        selected.append(
            {
                "row_idx": row_index,
                "id": row["id"],
                "speaker_id": row["speaker_id"],
                "chapter_id": row["chapter_id"],
                "split": args.split,
                "audio": str(wav_path.resolve()),
                "duration_s": f"{duration:.4f}",
                "text_normalized": text,
                "text_original": row["text_original"].strip(),
                "source_path": row["path"],
            }
        )
        print(f"selected {row['id']}: {duration:.2f}s | {text}", flush=True)
        if len(selected) >= args.max_items:
            break
    if not selected:
        raise SystemExit("no rows selected")
    manifest = args.manifest or (args.output_dir / "manifest.csv")
    write_csv_rows(
        selected,
        manifest,
        ["row_idx", "id", "speaker_id", "chapter_id", "split", "audio", "duration_s", "text_normalized", "text_original", "source_path"],
    )
    print(f"Wrote {manifest}")


def prepare_listening_test(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.manifest.resolve(), required={"id", "condition", "audio"})
    by_sample = {}
    for row in rows:
        by_sample.setdefault(row["id"], {})[row["condition"]] = row
    eligible = sorted(
        sample_id
        for sample_id, condition_rows in by_sample.items()
        if all(condition in condition_rows for condition in args.conditions)
    )
    if args.sample_count < 1 or len(eligible) < args.sample_count:
        raise ValueError(f"requested {args.sample_count} samples but only {len(eligible)} are eligible")

    random_generator = random.Random(args.seed)
    selected = random_generator.sample(eligible, args.sample_count)
    trials = [(sample_id, condition) for sample_id in selected for condition in args.conditions]
    random_generator.shuffle(trials)
    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    reference_paths = {}
    for index, sample_id in enumerate(selected, start=1):
        clean_row = by_sample[sample_id].get("clean")
        if clean_row is None:
            raise ValueError(f"sample {sample_id} has no clean reference")
        source = Path(clean_row.get("clean_audio") or clean_row["audio"]).resolve()
        destination = audio_dir / f"R{index:03d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        reference_paths[sample_id] = destination

    response_rows = []
    key_rows = []
    for index, (sample_id, condition) in enumerate(trials, start=1):
        row = by_sample[sample_id][condition]
        source = Path(row["audio"]).resolve()
        trial_id = f"T{index:03d}"
        destination = audio_dir / f"{trial_id}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        reference = reference_paths[sample_id].relative_to(output_dir).as_posix()
        candidate = destination.relative_to(output_dir).as_posix()
        response_rows.append(
            {
                "trial_id": trial_id,
                "reference_audio": reference,
                "candidate_audio": candidate,
                "quality_mos_1_5": "",
                "content_consistency_1_5": "",
                "timbre_similarity_1_5": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "trial_id": trial_id,
                "sample_id": sample_id,
                "condition": condition,
                "reference_audio": reference,
                "candidate_audio": candidate,
                "reference_text": row.get("reference_text", ""),
            }
        )
    write_csv_rows(
        response_rows,
        output_dir / "response_sheet.csv",
        [
            "trial_id",
            "reference_audio",
            "candidate_audio",
            "quality_mos_1_5",
            "content_consistency_1_5",
            "timbre_similarity_1_5",
            "notes",
        ],
    )
    write_csv_rows(
        key_rows,
        output_dir / "answer_key.csv",
        [
            "trial_id",
            "sample_id",
            "condition",
            "reference_audio",
            "candidate_audio",
            "reference_text",
        ],
    )
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "manifest": str(args.manifest.resolve()),
                "conditions": args.conditions,
                "sample_count": args.sample_count,
                "trial_count": len(trials),
                "seed": args.seed,
                "selected_sample_ids": selected,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Wrote {len(trials)} blinded trials to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model, dataset, and listening-test preparation.")
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="Download project model weights.")
    models.set_defaults(func=download_models)

    dataset = commands.add_parser("dataset", help="Fetch a small LibriTTS subset.")
    dataset.add_argument("--dataset", default="mythicinfinity/libritts")
    dataset.add_argument("--split", default="dev.clean")
    dataset.add_argument("--config", default="all")
    dataset.add_argument("--hf_endpoint", default=DEFAULT_HF_ENDPOINT)
    dataset.add_argument("--offset", type=int, default=0)
    dataset.add_argument("--max_scan", type=int, default=500)
    dataset.add_argument("--max_items", type=int, default=5)
    dataset.add_argument("--min_duration", type=float, default=2.0)
    dataset.add_argument("--max_duration", type=float, default=8.0)
    dataset.add_argument("--min_text_chars", type=int, default=25)
    dataset.add_argument("--max_text_chars", type=int, default=180)
    dataset.add_argument("--output_dir", type=Path, default=ROOT / "data" / "libritts_subset")
    dataset.add_argument("--manifest", type=Path, default=None)
    dataset.set_defaults(func=fetch_dataset)

    listening = commands.add_parser("listening", help="Create a blinded paired listening set.")
    listening.add_argument("--manifest", type=Path, required=True)
    listening.add_argument("--conditions", nargs="+", required=True)
    listening.add_argument("--sample_count", type=int, default=20)
    listening.add_argument("--seed", type=int, default=20260804)
    listening.add_argument("--output_dir", type=Path, required=True)
    listening.set_defaults(func=prepare_listening_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
