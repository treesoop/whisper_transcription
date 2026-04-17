#!/usr/bin/env python3
"""
pyannote.audio diarization을 직접 실행한다.

ASR/align 단계는 생략하고, regular diarization + exclusive diarization interval을 JSON으로 저장한다.
"""

import argparse
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np


warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")


def log(message: str) -> None:
    print(message, flush=True)


def load_audio_ffmpeg(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        output = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg audio decode 실패: {stderr}") from exc
    return np.frombuffer(output, np.int16).astype(np.float32) / 32768.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct pyannote diarization helper")
    parser.add_argument("audio", help="오디오 파일 경로")
    parser.add_argument("--output-json", required=True, help="출력 JSON 경로")
    parser.add_argument("--hf-token", required=True, help="HuggingFace token")
    parser.add_argument("--diarize-model", required=True, help="pyannote diarization model")
    parser.add_argument("--device", default="mps", help="torch device")
    parser.add_argument("--cache-dir", help="모델 캐시 디렉토리")
    parser.add_argument("--min-speakers", type=int, help="최소 화자 수")
    parser.add_argument("--max-speakers", type=int, help="최대 화자 수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        args.diarize_model,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    if args.device:
        pipeline.to(torch.device(args.device))

    last_step = {"name": None}

    def on_progress(step_name, step_artifact=None, file=None, total=None, completed=None) -> None:
        if step_name != last_step["name"]:
            last_step["name"] = step_name
            log(f"pyannote step: {step_name}")
        if total is not None and completed is not None and total > 0:
            percent = min((completed / total) * 100.0, 100.0)
            log(f"pyannote progress: {percent:.1f}%")

    waveform = load_audio_ffmpeg(args.audio)
    audio_input = {
        "waveform": torch.from_numpy(waveform[None, :]),
        "sample_rate": 16000,
    }

    output = pipeline(
        audio_input,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        hook=on_progress,
    )

    def annotation_to_intervals(annotation) -> list[dict]:
        return [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]

    regular_intervals = annotation_to_intervals(output.speaker_diarization)
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
    exclusive_intervals = (
        annotation_to_intervals(exclusive_annotation)
        if exclusive_annotation is not None
        else []
    )

    payload = {
        "intervals": regular_intervals,
        "exclusive_intervals": exclusive_intervals,
    }
    if exclusive_intervals:
        payload["preferred"] = "exclusive_intervals"
    else:
        payload["preferred"] = "intervals"

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    log(
        "pyannote diarization saved: "
        f"{output_path} (exclusive={len(exclusive_intervals) > 0})"
    )


if __name__ == "__main__":
    main()
