#!/usr/bin/env python3
"""
mlx_whisper 전사 + whispermlx 화자 분리 병합 파이프라인

전사는 mlx_whisper의 high-quality 결과를 사용하고,
화자 분리는 whispermlx의 pyannote diarization을 사용해 병합한다.

업그레이드 포인트:
- word timestamp 기반 화자 배정
- 재실행을 빠르게 하는 artifact/cache 저장
- HF 토큰 환경변수 지원
- 긴 오디오 자동 chunking
- txt/json/srt/vtt 출력 지원

Usage:
    python3 transcribe.py meeting.m4a
    python3 transcribe.py meeting.m4a --diarize
    python3 transcribe.py meeting.m4a --diarize --formats txt,json,srt
    python3 transcribe.py meeting.m4a --diarize -o ./output
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_TRANSCRIBE_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_OUTPUT_FORMATS = ("txt",)
SUPPORTED_OUTPUT_FORMATS = ("txt", "json", "srt", "vtt")
HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
UNKNOWN_SPEAKER = "UNKNOWN"
DEFAULT_CHUNK_MINUTES = 30.0
DEFAULT_AUTO_CHUNK_MINUTES = 60.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 1.5
DEFAULT_SEGMENT_MERGE_GAP = 0.8
DEFAULT_SEGMENT_BREAK_GAP = 1.2
DEFAULT_LOCAL_CONFIG_NAME = ".transcribe.local.json"


@dataclass
class ChunkSpec:
    index: int
    logical_start: float
    logical_end: float
    extract_start: float
    extract_end: float
    output_audio_path: Path
    is_chunked: bool


def bool_str(value: bool) -> str:
    return "True" if value else "False"


def log(message: str) -> None:
    print(message, flush=True)


def format_elapsed(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def find_executable(name: str, extra_paths: Optional[Sequence[str]] = None) -> str:
    """PATH와 알려진 설치 경로에서 실행 파일 찾기"""
    import shutil

    found = shutil.which(name)
    if found:
        return found

    for path in extra_paths or []:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded

    raise FileNotFoundError(f"{name} 실행 파일을 찾을 수 없습니다.")


def find_mlx_whisper() -> str:
    try:
        return find_executable(
            "mlx_whisper",
            extra_paths=[
                "~/Library/Python/3.9/bin/mlx_whisper",
                "~/Library/Python/3.10/bin/mlx_whisper",
                "~/Library/Python/3.11/bin/mlx_whisper",
                "~/Library/Python/3.12/bin/mlx_whisper",
                "~/.local/bin/mlx_whisper",
            ],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "mlx_whisper를 찾을 수 없습니다. `pip install mlx-whisper` 후 다시 시도하세요."
        ) from exc


def find_whispermlx() -> str:
    try:
        return find_executable(
            "whispermlx",
            extra_paths=[
                "~/.local/bin/whispermlx",
                "~/Library/Python/3.10/bin/whispermlx",
                "~/Library/Python/3.11/bin/whispermlx",
                "~/Library/Python/3.12/bin/whispermlx",
            ],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "whispermlx를 찾을 수 없습니다. `uv tool install whispermlx --python 3.12` 후 다시 시도하세요."
        ) from exc


def find_whispermlx_python() -> str:
    whispermlx_path = Path(find_whispermlx()).resolve()
    python_path = whispermlx_path.parent / "python"
    if python_path.is_file() and os.access(python_path, os.X_OK):
        return str(python_path)
    raise FileNotFoundError("whispermlx 전용 Python 실행 파일을 찾을 수 없습니다.")


def require_audio_file(audio_path: Path) -> None:
    if not audio_path.is_file():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")


def require_ffmpeg() -> str:
    try:
        return find_executable(
            "ffmpeg",
            extra_paths=[
                "/opt/homebrew/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
            ],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg를 찾을 수 없습니다. `brew install ffmpeg` 후 다시 시도하세요."
        ) from exc


def require_ffprobe() -> str:
    try:
        return find_executable(
            "ffprobe",
            extra_paths=[
                "/opt/homebrew/bin/ffprobe",
                "/usr/local/bin/ffprobe",
            ],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffprobe를 찾을 수 없습니다. ffmpeg 설치를 확인하세요."
        ) from exc


def run_preflight(diarize: bool) -> None:
    errors = []

    checks = [require_ffmpeg, require_ffprobe, find_mlx_whisper]
    if diarize:
        checks.append(find_whispermlx)

    for check in checks:
        try:
            check()
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if errors:
        unique_errors = list(dict.fromkeys(errors))
        raise RuntimeError("필수 의존성이 없습니다:\n- " + "\n- ".join(unique_errors))


def run_cli(cmd: List[str], step_name: str) -> None:
    """외부 CLI 실행, 진행 로그는 실시간으로 그대로 보여준다."""
    try:
        completed = subprocess.run(
            cmd,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{step_name} 실패 (exit code: {exc.returncode})") from exc


def read_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_formats(value: str) -> List[str]:
    raw = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw:
        return list(DEFAULT_OUTPUT_FORMATS)
    if "all" in raw:
        return list(SUPPORTED_OUTPUT_FORMATS)

    invalid = [fmt for fmt in raw if fmt not in SUPPORTED_OUTPUT_FORMATS]
    if invalid:
        raise ValueError(
            "지원하지 않는 출력 형식: "
            + ", ".join(invalid)
            + f" (지원: {', '.join(SUPPORTED_OUTPUT_FORMATS)})"
        )

    ordered = []
    for fmt in SUPPORTED_OUTPUT_FORMATS:
        if fmt in raw:
            ordered.append(fmt)
    return ordered


def resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    if cli_value:
        return cli_value

    for env_name in HF_TOKEN_ENV_NAMES:
        token = os.getenv(env_name)
        if token:
            return token
    return None


def load_local_config(config_path: Path) -> Dict:
    if not config_path.is_file():
        return {}
    try:
        with config_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"로컬 설정 JSON을 읽지 못했습니다: {config_path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"로컬 설정 형식이 올바르지 않습니다: {config_path}")
    return data


def resolve_bool_option(current: Optional[bool], config: Dict, key: str, default: bool = False) -> bool:
    if current is not None:
        return current
    value = config.get(key, default)
    return bool(value)


def mask_sensitive_command(command: str) -> str:
    masked = re.sub(r"(--hf-token\s+)(\S+)", r"\1***", command)
    masked = re.sub(r"(--hf_token\s+)(\S+)", r"\1***", masked)
    for env_name in HF_TOKEN_ENV_NAMES:
        masked = re.sub(rf"({env_name}=)(\S+)", r"\1***", masked)
    return masked


def find_output_json(output_dir: Path, preferred_stems: Optional[Sequence[str]] = None) -> Path:
    for stem in preferred_stems or []:
        candidate = output_dir / f"{stem}.json"
        if candidate.is_file():
            return candidate

    json_files = sorted(path for path in output_dir.iterdir() if path.suffix == ".json")
    if len(json_files) == 1:
        return json_files[0]

    raise FileNotFoundError(f"결과 JSON을 찾을 수 없습니다: {output_dir}")


def warn_about_other_runs() -> None:
    cmd = ["ps", "-Ao", "pid=,command="]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return

    current_pid = os.getpid()
    matches = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        command = parts[1]
        if pid == current_pid:
            continue
        if "transcribe.py" in command or "whispermlx" in command:
            matches.append((pid, mask_sensitive_command(command)))

    if not matches:
        return

    log("주의: 이미 실행 중인 전사/화자분리 프로세스가 있습니다. 병렬 실행은 매우 느려질 수 있습니다.")
    for pid, command in matches[:5]:
        log(f"- PID {pid}: {command}")


def get_audio_duration(audio_path: Path) -> float:
    ffprobe = require_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("오디오 길이를 읽지 못했습니다.") from exc

    output = completed.stdout.strip()
    if not output:
        raise RuntimeError("오디오 길이를 읽지 못했습니다.")
    return float(output)


def plan_chunks(audio_path: Path, artifact_dir: Path, args: argparse.Namespace) -> Tuple[float, List[ChunkSpec]]:
    duration = get_audio_duration(audio_path)

    chunk_seconds = max(0.0, args.chunk_minutes * 60.0)
    auto_chunk_seconds = max(0.0, args.auto_chunk_minutes * 60.0)
    overlap = max(0.0, args.chunk_overlap_seconds)

    should_chunk = (
        not args.no_auto_chunk
        and chunk_seconds > 0
        and auto_chunk_seconds > 0
        and duration > auto_chunk_seconds
    )

    if not should_chunk:
        return duration, [
            ChunkSpec(
                index=0,
                logical_start=0.0,
                logical_end=duration,
                extract_start=0.0,
                extract_end=duration,
                output_audio_path=audio_path,
                is_chunked=False,
            )
        ]

    chunks_dir = artifact_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    logical_start = 0.0
    index = 0
    while logical_start < duration - 0.001:
        logical_end = min(duration, logical_start + chunk_seconds)
        extract_start = max(0.0, logical_start - (overlap if logical_start > 0 else 0.0))
        extract_end = min(duration, logical_end + (overlap if logical_end < duration else 0.0))
        chunks.append(
            ChunkSpec(
                index=index,
                logical_start=logical_start,
                logical_end=logical_end,
                extract_start=extract_start,
                extract_end=extract_end,
                output_audio_path=chunks_dir / f"chunk_{index:03d}.wav",
                is_chunked=True,
            )
        )
        logical_start = logical_end
        index += 1

    return duration, chunks


def materialize_chunk(audio_path: Path, chunk: ChunkSpec, force: bool) -> Path:
    if not chunk.is_chunked:
        return audio_path

    if chunk.output_audio_path.is_file() and not force:
        return chunk.output_audio_path

    ffmpeg = require_ffmpeg()
    chunk.output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, chunk.extract_end - chunk.extract_start)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{chunk.extract_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(chunk.output_audio_path),
    ]
    log(
        f"[chunk {chunk.index + 1}] 오디오 분할 중 "
        f"({format_clock(chunk.logical_start)} - {format_clock(chunk.logical_end)})..."
    )
    run_cli(cmd, f"chunk {chunk.index + 1} 오디오 분할")
    return chunk.output_audio_path


def prepare_work_audio(audio_path: Path, artifact_dir: Path, force: bool) -> Path:
    work_audio_path = artifact_dir / "_work_audio" / "source_16k_mono.wav"
    if work_audio_path.is_file() and not force:
        log(f"work audio cache 사용: {work_audio_path}")
        return work_audio_path

    ffmpeg = require_ffmpeg()
    work_audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(work_audio_path),
    ]
    log("공유 작업용 오디오 준비 중 (16kHz mono WAV)...")
    run_cli(cmd, "공유 작업용 오디오 준비")
    return work_audio_path


def load_or_run_mlx(
    audio_path: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
    word_timestamps: bool,
) -> Dict:
    output_stem = "mlx_words" if word_timestamps else "mlx_segments"
    json_path = artifact_dir / "mlx" / f"{output_stem}.json"
    if json_path.is_file() and not args.force:
        log(f"mlx cache 사용: {json_path}")
        return read_json(json_path)

    output_dir = json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    mlx_whisper_cli = find_mlx_whisper()
    cmd = [
        mlx_whisper_cli,
        "--model",
        args.transcribe_model,
        "--condition-on-previous-text",
        "False",
        "--verbose",
        "False",
        "-f",
        "json",
        "-o",
        str(output_dir),
        "--output-name",
        output_stem,
        str(audio_path),
    ]

    if word_timestamps:
        cmd.extend(
            [
                "--word-timestamps",
                "True",
                "--hallucination-silence-threshold",
                "1",
            ]
        )

    if args.language:
        cmd.extend(["--language", args.language])
    if args.initial_prompt:
        cmd.extend(["--initial-prompt", args.initial_prompt])

    run_cli(cmd, "mlx_whisper 전사")
    output_json = find_output_json(output_dir, preferred_stems=("mlx", audio_path.stem))
    return read_json(output_json)


def load_or_run_diarization(
    audio_path: Path,
    artifact_dir: Path,
    hf_token: str,
    args: argparse.Namespace,
) -> List[Dict]:
    output_dir = artifact_dir / "whispermlx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "diarization_intervals.json"
    if output_json.is_file() and not args.force:
        log(f"diarization cache 사용: {output_json}")
        payload = read_json(output_json)
        preferred_key = payload.get("preferred")
        if preferred_key and payload.get(preferred_key):
            return payload.get(preferred_key, [])
        return payload.get("exclusive_intervals") or payload.get("intervals", [])

    helper_script = Path(__file__).with_name("diarize_segments.py")
    whispermlx_python = find_whispermlx_python()
    shared_model_cache = artifact_dir.parent / "_pyannote_cache"
    shared_model_cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        whispermlx_python,
        str(helper_script),
        str(audio_path),
        "--output-json",
        str(output_json),
        "--hf-token",
        hf_token,
        "--diarize-model",
        args.diarize_model,
        "--device",
        "mps",
        "--cache-dir",
        str(shared_model_cache),
    ]
    if args.min_speakers is not None:
        cmd.extend(["--min-speakers", str(args.min_speakers)])
    if args.max_speakers is not None:
        cmd.extend(["--max-speakers", str(args.max_speakers)])

    run_cli(cmd, "pyannote 화자 분리")
    payload = read_json(output_json)
    preferred_key = payload.get("preferred")
    if preferred_key and payload.get(preferred_key):
        return payload.get(preferred_key, [])
    return payload.get("exclusive_intervals") or payload.get("intervals", [])


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def word_text(word: Dict) -> str:
    return str(word.get("word") or word.get("text") or "")


def rebuild_text_from_words(words: Sequence[Dict]) -> str:
    text = "".join(word_text(word) for word in words).strip()
    if text:
        return text
    return " ".join(word_text(word).strip() for word in words if word_text(word).strip()).strip()


def normalize_word(word: Dict) -> Dict:
    start = coerce_float(word.get("start"))
    end = coerce_float(word.get("end"), start)
    normalized = {
        "start": start,
        "end": max(start, end),
        "text": word_text(word),
    }
    if "speaker" in word and word.get("speaker") is not None:
        normalized["speaker"] = str(word["speaker"])
    if "score" in word and word.get("score") is not None:
        normalized["score"] = word["score"]
    return normalized


def normalize_segment(segment: Dict) -> Dict:
    start = coerce_float(segment.get("start"))
    end = coerce_float(segment.get("end"), start)
    normalized = {
        "start": start,
        "end": max(start, end),
        "text": str(segment.get("text") or "").strip(),
    }
    if "speaker" in segment and segment.get("speaker") is not None:
        normalized["speaker"] = str(segment["speaker"])

    words = []
    for word in segment.get("words", []) or []:
        normalized_word = normalize_word(word)
        if normalized_word["text"]:
            words.append(normalized_word)

    if words:
        normalized["words"] = words
        normalized["start"] = words[0]["start"]
        normalized["end"] = words[-1]["end"]
        normalized["text"] = rebuild_text_from_words(words)

    return normalized


def normalize_result(payload: Dict) -> Dict:
    return {
        "language": payload.get("language"),
        "segments": [normalize_segment(segment) for segment in payload.get("segments", []) or []],
    }


def shift_segments(segments: Sequence[Dict], offset: float) -> List[Dict]:
    shifted = []
    for segment in segments:
        item = {
            "start": segment["start"] + offset,
            "end": segment["end"] + offset,
            "text": segment.get("text", ""),
        }
        if "speaker" in segment:
            item["speaker"] = segment["speaker"]

        if segment.get("words"):
            item["words"] = []
            for word in segment["words"]:
                shifted_word = dict(word)
                shifted_word["start"] = word["start"] + offset
                shifted_word["end"] = word["end"] + offset
                item["words"].append(shifted_word)
            item["text"] = rebuild_text_from_words(item["words"])

        shifted.append(item)
    return shifted


def midpoint(start: float, end: float) -> float:
    if end <= start:
        return start
    return start + ((end - start) / 2.0)


def in_window_by_midpoint(start: float, end: float, window_start: float, window_end: float) -> bool:
    point = midpoint(start, end)
    if point == window_end and window_end > window_start:
        return False
    return window_start <= point < window_end


def trim_segments_to_window(segments: Sequence[Dict], window_start: float, window_end: float) -> List[Dict]:
    trimmed = []
    for segment in segments:
        words = []
        for word in segment.get("words", []) or []:
            if in_window_by_midpoint(word["start"], word["end"], window_start, window_end):
                words.append(dict(word))

        if words:
            item = {
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": rebuild_text_from_words(words),
                "words": words,
            }
            if "speaker" in segment:
                item["speaker"] = segment["speaker"]
            trimmed.append(item)
            continue

        if not in_window_by_midpoint(segment["start"], segment["end"], window_start, window_end):
            continue

        item = dict(segment)
        item["start"] = max(segment["start"], window_start)
        item["end"] = min(segment["end"], window_end)
        trimmed.append(item)

    return trimmed


def flatten_words(segments: Sequence[Dict]) -> List[Dict]:
    words = []
    for segment in segments:
        for word in segment.get("words", []) or []:
            words.append(dict(word))
    return words


def merge_speaker_intervals(intervals: Sequence[Dict], max_gap: float = 0.15) -> List[Dict]:
    ordered = sorted(
        (
            {
                "start": coerce_float(interval["start"]),
                "end": max(coerce_float(interval["start"]), coerce_float(interval["end"], coerce_float(interval["start"]))),
                "speaker": str(interval["speaker"]),
            }
            for interval in intervals
            if interval.get("speaker")
        ),
        key=lambda item: (item["start"], item["end"]),
    )

    merged = []
    for interval in ordered:
        if interval["end"] <= interval["start"]:
            continue
        if (
            merged
            and merged[-1]["speaker"] == interval["speaker"]
            and interval["start"] <= merged[-1]["end"] + max_gap
        ):
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        else:
            merged.append(dict(interval))
    return merged


def trim_intervals_to_window(intervals: Sequence[Dict], window_start: float, window_end: float) -> List[Dict]:
    trimmed = []
    for interval in intervals:
        start = max(interval["start"], window_start)
        end = min(interval["end"], window_end)
        if end <= start:
            continue
        trimmed.append(
            {
                "start": start,
                "end": end,
                "speaker": interval["speaker"],
            }
        )
    return merge_speaker_intervals(trimmed)


def extract_speaker_intervals(diarized_payload: Dict) -> List[Dict]:
    normalized = normalize_result(diarized_payload)
    intervals = []

    for segment in normalized["segments"]:
        used_word_level = False
        for word in segment.get("words", []) or []:
            speaker = word.get("speaker") or segment.get("speaker")
            if not speaker:
                continue
            intervals.append(
                {
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": speaker,
                }
            )
            used_word_level = True

        if not used_word_level and segment.get("speaker"):
            intervals.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": segment["speaker"],
                }
            )

    return merge_speaker_intervals(intervals)


def extend_merged_intervals(
    merged: List[Dict],
    new_intervals: Sequence[Dict],
    max_gap: float = 0.15,
) -> List[Dict]:
    for interval in new_intervals:
        if not merged:
            merged.append(dict(interval))
            continue

        previous = merged[-1]
        if (
            previous["speaker"] == interval["speaker"]
            and interval["start"] <= previous["end"] + max_gap
        ):
            previous["end"] = max(previous["end"], interval["end"])
        else:
            merged.append(dict(interval))
    return merged


def choose_nearest_speaker(
    query_start: float,
    query_end: float,
    previous_interval: Optional[Dict],
    next_interval: Optional[Dict],
) -> Optional[str]:
    query_mid = midpoint(query_start, query_end)
    candidates = []

    if previous_interval is not None:
        candidates.append((abs(query_mid - previous_interval["end"]), previous_interval["speaker"]))
    if next_interval is not None:
        candidates.append((abs(next_interval["start"] - query_mid), next_interval["speaker"]))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def assign_speaker_for_span(
    start: float,
    end: float,
    intervals: Sequence[Dict],
    interval_idx: int,
    previous_interval: Optional[Dict],
) -> Tuple[str, int, Optional[Dict]]:
    span_end = max(start, end)

    while interval_idx < len(intervals) and intervals[interval_idx]["end"] <= start:
        previous_interval = intervals[interval_idx]
        interval_idx += 1

    best_speaker = None
    best_overlap = 0.0
    scan_idx = interval_idx
    while scan_idx < len(intervals) and intervals[scan_idx]["start"] < span_end:
        overlap = max(
            0.0,
            min(span_end, intervals[scan_idx]["end"]) - max(start, intervals[scan_idx]["start"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = intervals[scan_idx]["speaker"]
        scan_idx += 1

    if best_speaker:
        return best_speaker, interval_idx, previous_interval

    next_interval = intervals[interval_idx] if interval_idx < len(intervals) else None
    nearest = choose_nearest_speaker(start, span_end, previous_interval, next_interval)
    return nearest or UNKNOWN_SPEAKER, interval_idx, previous_interval


def dominant_speaker_from_words(words: Sequence[Dict]) -> str:
    weights: Dict[str, float] = {}
    for word in words:
        speaker = str(word.get("speaker") or UNKNOWN_SPEAKER)
        duration = max(0.05, word["end"] - word["start"])
        text_weight = max(1, len(word_text(word).strip()))
        weights[speaker] = weights.get(speaker, 0.0) + (duration * text_weight)

    if not weights:
        return UNKNOWN_SPEAKER

    known_weights = {speaker: weight for speaker, weight in weights.items() if speaker != UNKNOWN_SPEAKER}
    if known_weights:
        return max(known_weights.items(), key=lambda item: item[1])[0]
    return max(weights.items(), key=lambda item: item[1])[0]


def assign_speakers_to_segments(segments: Sequence[Dict], intervals: Sequence[Dict]) -> List[Dict]:
    if not intervals:
        output = []
        for segment in segments:
            item = dict(segment)
            if item.get("words"):
                item["words"] = [dict(word) for word in item["words"]]
                for word in item["words"]:
                    word["speaker"] = UNKNOWN_SPEAKER
            item["speaker"] = UNKNOWN_SPEAKER
            output.append(item)
        return output

    assigned = []
    interval_idx = 0
    previous_interval = None

    for segment in segments:
        item = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment.get("text", ""),
        }
        if segment.get("words"):
            words = []
            for word in segment["words"]:
                labeled_word = dict(word)
                speaker, interval_idx, previous_interval = assign_speaker_for_span(
                    labeled_word["start"],
                    labeled_word["end"],
                    intervals,
                    interval_idx,
                    previous_interval,
                )
                labeled_word["speaker"] = speaker
                words.append(labeled_word)

            item["words"] = words
            item["speaker"] = dominant_speaker_from_words(words)
            item["text"] = rebuild_text_from_words(words)
            item["start"] = words[0]["start"]
            item["end"] = words[-1]["end"]
        else:
            speaker, interval_idx, previous_interval = assign_speaker_for_span(
                item["start"],
                item["end"],
                intervals,
                interval_idx,
                previous_interval,
            )
            item["speaker"] = speaker

        assigned.append(item)

    return assigned


def make_segment_from_words(words: Sequence[Dict], speaker: str) -> Dict:
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "speaker": speaker or UNKNOWN_SPEAKER,
        "text": rebuild_text_from_words(words),
        "words": [dict(word) for word in words],
    }


def split_segments_by_speaker(segments: Sequence[Dict], break_gap: float) -> List[Dict]:
    output = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            item = dict(segment)
            item["speaker"] = str(item.get("speaker") or UNKNOWN_SPEAKER)
            output.append(item)
            continue

        current_words = [dict(words[0])]
        current_speaker = str(words[0].get("speaker") or segment.get("speaker") or UNKNOWN_SPEAKER)

        for word in words[1:]:
            next_word = dict(word)
            next_speaker = str(next_word.get("speaker") or current_speaker or UNKNOWN_SPEAKER)
            gap = max(0.0, next_word["start"] - current_words[-1]["end"])
            if next_speaker != current_speaker or gap > break_gap:
                output.append(make_segment_from_words(current_words, current_speaker))
                current_words = [next_word]
                current_speaker = next_speaker
            else:
                current_words.append(next_word)

        if current_words:
            output.append(make_segment_from_words(current_words, current_speaker))

    return output


def smooth_unknown_segments(segments: Sequence[Dict]) -> List[Dict]:
    smoothed = [dict(segment) for segment in segments]

    for segment in smoothed:
        if segment.get("words"):
            segment["words"] = [dict(word) for word in segment["words"]]

    for idx, segment in enumerate(smoothed):
        speaker = str(segment.get("speaker") or UNKNOWN_SPEAKER)
        if speaker != UNKNOWN_SPEAKER:
            continue

        duration = max(0.0, segment["end"] - segment["start"])
        short_text = len(segment.get("text", "").strip()) <= 12
        short_segment = duration <= 1.2 or short_text
        if not short_segment:
            continue

        prev_speaker = None
        next_speaker = None

        if idx > 0:
            prev_value = str(smoothed[idx - 1].get("speaker") or UNKNOWN_SPEAKER)
            if prev_value != UNKNOWN_SPEAKER:
                prev_speaker = prev_value
        if idx + 1 < len(smoothed):
            next_value = str(smoothed[idx + 1].get("speaker") or UNKNOWN_SPEAKER)
            if next_value != UNKNOWN_SPEAKER:
                next_speaker = next_value

        replacement = None
        if prev_speaker and prev_speaker == next_speaker:
            replacement = prev_speaker
        elif prev_speaker and not next_speaker:
            replacement = prev_speaker
        elif next_speaker and not prev_speaker:
            replacement = next_speaker

        if replacement:
            segment["speaker"] = replacement
            for word in segment.get("words", []) or []:
                word["speaker"] = replacement

    return smoothed


def merge_adjacent_segments(segments: Sequence[Dict], max_gap: float) -> List[Dict]:
    merged = []
    for segment in segments:
        item = dict(segment)
        if item.get("words"):
            item["words"] = [dict(word) for word in item["words"]]

        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]
        same_speaker = str(previous.get("speaker") or "") == str(item.get("speaker") or "")
        gap = max(0.0, item["start"] - previous["end"])
        can_merge = same_speaker and gap <= max_gap

        if not can_merge:
            merged.append(item)
            continue

        previous["end"] = max(previous["end"], item["end"])

        if previous.get("words") and item.get("words"):
            previous["words"].extend(item["words"])
            previous["text"] = rebuild_text_from_words(previous["words"])
        else:
            left = previous.get("text", "").rstrip()
            right = item.get("text", "").lstrip()
            if left and right:
                previous["text"] = f"{left} {right}"
            else:
                previous["text"] = left or right

    return merged


def build_plain_segments(segments: Sequence[Dict], max_gap: float) -> List[Dict]:
    plain_segments = []
    for segment in segments:
        item = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment.get("text", "").strip(),
        }
        if segment.get("words"):
            item["words"] = [dict(word) for word in segment["words"]]
            item["text"] = rebuild_text_from_words(item["words"])
            item["start"] = item["words"][0]["start"]
            item["end"] = item["words"][-1]["end"]
        plain_segments.append(item)
    return merge_adjacent_segments(plain_segments, max_gap=max_gap)


def format_clock(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_subtitle_timestamp(seconds: float, kind: str) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, ms = divmod(milliseconds, 1000)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    separator = "," if kind == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def write_text_output(segments: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_speakers = any(segment.get("speaker") for segment in segments)
    with output_path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            timestamp = format_clock(segment["start"])
            text = segment.get("text", "").strip()
            if not text:
                continue
            if has_speakers and segment.get("speaker"):
                handle.write(f"[{segment['speaker']}] {timestamp} {text}\n")
            else:
                handle.write(f"{timestamp} {text}\n")


def write_subtitle_output(segments: Sequence[Dict], output_path: Path, kind: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if kind == "vtt":
            handle.write("WEBVTT\n\n")

        counter = 1
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue

            if kind == "srt":
                handle.write(f"{counter}\n")
            start = format_subtitle_timestamp(segment["start"], kind)
            end = format_subtitle_timestamp(segment["end"], kind)
            speaker_prefix = f"[{segment['speaker']}] " if segment.get("speaker") else ""
            handle.write(f"{start} --> {end}\n")
            handle.write(f"{speaker_prefix}{text}\n\n")
            counter += 1


def build_public_payload(
    audio_path: Path,
    segments: Sequence[Dict],
    language: Optional[str],
    intervals: Sequence[Dict],
    duration: float,
    chunks: Sequence[ChunkSpec],
    args: argparse.Namespace,
) -> Dict:
    speaker_names = sorted(
        {
            str(segment.get("speaker"))
            for segment in segments
            if segment.get("speaker") and str(segment.get("speaker")) != UNKNOWN_SPEAKER
        }
    )
    word_count = sum(len(segment.get("words", []) or []) for segment in segments)

    return {
        "audio": str(audio_path),
        "language": language,
        "diarized": args.diarize,
        "models": {
            "transcription": args.transcribe_model,
            "diarize_asr": None,
            "diarization": args.diarize_model if args.diarize else None,
        },
        "chunking": {
            "enabled": len(chunks) > 1,
            "chunk_count": len(chunks),
            "chunk_minutes": args.chunk_minutes,
            "auto_chunk_minutes": args.auto_chunk_minutes,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
        },
        "stats": {
            "duration_seconds": round(duration, 3),
            "segment_count": len(segments),
            "word_count": word_count,
            "speaker_count": len(speaker_names),
            "speakers": speaker_names,
            "speaker_interval_count": len(intervals),
        },
        "segments": list(segments),
    }


def write_requested_outputs(
    output_dir: Path,
    stem: str,
    segments: Sequence[Dict],
    payload: Dict,
    formats: Sequence[str],
) -> List[Path]:
    written = []
    for fmt in formats:
        output_path = output_dir / f"{stem}.{fmt}"
        if fmt == "txt":
            write_text_output(segments, output_path)
        elif fmt == "json":
            write_json(output_path, payload)
        elif fmt in ("srt", "vtt"):
            write_subtitle_output(segments, output_path, fmt)
        else:
            continue
        written.append(output_path)
    return written


def compute_current_segments(
    transcription_segments: Sequence[Dict],
    speaker_intervals: Sequence[Dict],
    diarize: bool,
) -> List[Dict]:
    if diarize:
        labeled_segments = assign_speakers_to_segments(transcription_segments, speaker_intervals)
        split_segments = split_segments_by_speaker(
            labeled_segments,
            break_gap=DEFAULT_SEGMENT_BREAK_GAP,
        )
        smoothed_segments = smooth_unknown_segments(split_segments)
        return merge_adjacent_segments(
            smoothed_segments,
            max_gap=DEFAULT_SEGMENT_MERGE_GAP,
        )

    return build_plain_segments(
        transcription_segments,
        max_gap=min(0.35, DEFAULT_SEGMENT_MERGE_GAP),
    )


def write_progress_outputs(
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    audio_path: Path,
    detected_language: Optional[str],
    diarize: bool,
    args: argparse.Namespace,
    duration: float,
    chunks: Sequence[ChunkSpec],
    transcription_segments: Sequence[Dict],
    speaker_intervals: Sequence[Dict],
) -> Tuple[List[Dict], Dict, List[Path]]:
    current_segments = compute_current_segments(
        transcription_segments=transcription_segments,
        speaker_intervals=speaker_intervals,
        diarize=diarize,
    )
    payload = build_public_payload(
        audio_path=audio_path,
        segments=current_segments,
        language=detected_language,
        intervals=speaker_intervals,
        duration=duration,
        chunks=chunks,
        args=args,
    )
    written_outputs = write_requested_outputs(
        output_dir=output_dir,
        stem=stem,
        segments=current_segments,
        payload=payload,
        formats=formats,
    )
    return current_segments, payload, written_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mlx_whisper + whispermlx 화자 분리")
    parser.add_argument("audio", nargs="?", help="오디오 파일 경로")
    parser.add_argument(
        "--config",
        help=f"로컬 설정 JSON 경로 (기본: ./{DEFAULT_LOCAL_CONFIG_NAME})",
    )
    parser.add_argument("--diarize", action="store_true", default=None, help="화자 분리 활성화")
    parser.add_argument("--hf-token", help="HuggingFace 토큰 (미입력 시 환경변수 탐색)")
    parser.add_argument("-o", "--output-dir", help="출력 디렉토리")
    parser.add_argument(
        "--formats",
        help="출력 형식(csv): txt,json,srt,vtt 또는 all",
    )
    parser.add_argument(
        "--transcribe-model",
        default=DEFAULT_TRANSCRIBE_MODEL,
        help="mlx_whisper 전사 모델",
    )
    parser.add_argument(
        "--diarize-asr-model",
        default="small",
        help="호환성용 옵션. 현재는 사용되지 않음",
    )
    parser.add_argument(
        "--diarize-model",
        default=DEFAULT_DIARIZE_MODEL,
        help="pyannote diarization 모델",
    )
    parser.add_argument("--language", help="언어 고정 (기본: 자동 감지)")
    parser.add_argument("--initial-prompt", help="전사 첫 프롬프트")
    parser.add_argument("--min-speakers", type=int, help="최소 화자 수")
    parser.add_argument("--max-speakers", type=int, help="최대 화자 수")
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=DEFAULT_CHUNK_MINUTES,
        help="자동 chunking 시 chunk 길이(분)",
    )
    parser.add_argument(
        "--auto-chunk-minutes",
        type=float,
        default=DEFAULT_AUTO_CHUNK_MINUTES,
        help="이 길이(분)를 넘으면 자동 chunking",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=DEFAULT_CHUNK_OVERLAP_SECONDS,
        help="chunk 경계 overlap(초)",
    )
    parser.add_argument("--no-auto-chunk", action="store_true", help="자동 chunking 비활성화")
    parser.add_argument("--force", action="store_true", help="cache/artifact 무시하고 재실행")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else Path.cwd() / DEFAULT_LOCAL_CONFIG_NAME
    )
    local_config = load_local_config(config_path)

    audio_value = args.audio or local_config.get("audio")
    if not audio_value:
        raise RuntimeError(
            "오디오 파일 경로가 없습니다. "
            "CLI 인자 또는 로컬 설정 파일의 `audio`를 지정하세요."
        )

    args.diarize = resolve_bool_option(args.diarize, local_config, "diarize", default=False)
    args.hf_token = args.hf_token or local_config.get("hf_token")
    args.output_dir = args.output_dir or local_config.get("output_dir") or "."
    args.formats = args.formats or local_config.get("formats") or "txt"

    audio_path = Path(audio_value).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        formats = parse_formats(args.formats)
        require_audio_file(audio_path)
        run_preflight(args.diarize)
        warn_about_other_runs()

        hf_token = resolve_hf_token(args.hf_token)
        if args.diarize and not hf_token:
            raise RuntimeError(
                "--diarize 사용 시 HF 토큰이 필요합니다. "
                "--hf-token 또는 HF_TOKEN/HUGGINGFACE_TOKEN 환경변수를 사용하세요."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = audio_path.stem
        artifact_dir = output_dir / f"{stem}.artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        expected_outputs = [output_dir / f"{stem}.{fmt}" for fmt in formats]
        use_word_timestamps = args.diarize or "json" in formats
        log("출력 파일:")
        for expected in expected_outputs:
            log(f"- {expected}")
        log(f"- artifact: {artifact_dir}")

        work_audio_path = (
            prepare_work_audio(audio_path, artifact_dir, force=args.force)
            if args.diarize
            else audio_path
        )
        duration, chunks = plan_chunks(audio_path, artifact_dir, args)
        if len(chunks) > 1:
            log(
                f"긴 오디오 감지: {len(chunks)}개 chunk로 처리합니다 "
                f"({format_clock(duration)} 전체)."
            )

        merged_transcription_segments = []
        merged_intervals = []
        detected_language = None
        current_segments: List[Dict] = []
        current_payload: Dict = {}
        written_outputs: List[Path] = []

        for chunk in chunks:
            source_audio = materialize_chunk(work_audio_path, chunk, force=args.force)
            chunk_dir = artifact_dir / f"chunk_{chunk.index:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            chunk_started = time.perf_counter()
            log(
                f"[chunk {chunk.index + 1}/{len(chunks)}] "
                f"전사 중 ({format_clock(chunk.logical_start)} - {format_clock(chunk.logical_end)})..."
            )
            transcribe_started = time.perf_counter()
            mlx_result = normalize_result(
                load_or_run_mlx(
                    source_audio,
                    chunk_dir,
                    args,
                    word_timestamps=use_word_timestamps,
                )
            )
            log(
                f"[chunk {chunk.index + 1}/{len(chunks)}] 전사 완료 "
                f"({format_elapsed(time.perf_counter() - transcribe_started)})"
            )
            detected_language = detected_language or mlx_result.get("language")
            shifted_segments = shift_segments(mlx_result["segments"], chunk.extract_start)
            trimmed_segments = trim_segments_to_window(
                shifted_segments,
                chunk.logical_start,
                chunk.logical_end,
            )
            merged_transcription_segments.extend(trimmed_segments)

            if args.diarize:
                preview_segments, preview_payload, preview_outputs = write_progress_outputs(
                    output_dir=output_dir,
                    stem=stem,
                    formats=formats,
                    audio_path=audio_path,
                    detected_language=detected_language,
                    diarize=False,
                    args=args,
                    duration=duration,
                    chunks=chunks,
                    transcription_segments=merged_transcription_segments,
                    speaker_intervals=merged_intervals,
                )
                write_json(artifact_dir / "transcription.partial.json", preview_payload)
                log(
                    f"[chunk {chunk.index + 1}/{len(chunks)}] 전사 중간 저장 완료: "
                    f"{', '.join(str(path) for path in preview_outputs)}"
                )

            if args.diarize:
                diarize_audio_path = (
                    materialize_chunk(work_audio_path, chunk, force=args.force)
                    if chunk.is_chunked
                    else work_audio_path
                )
                log(
                    f"[chunk {chunk.index + 1}/{len(chunks)}] "
                    f"화자 분리 중 ({format_clock(chunk.logical_start)} - {format_clock(chunk.logical_end)})..."
                )
                diarize_started = time.perf_counter()
                diarized_intervals = load_or_run_diarization(
                    diarize_audio_path,
                    chunk_dir,
                    hf_token,
                    args,
                )
                log(
                    f"[chunk {chunk.index + 1}/{len(chunks)}] 화자 분리 완료 "
                    f"({format_elapsed(time.perf_counter() - diarize_started)})"
                )
                shifted_intervals = [
                    {
                        "start": interval["start"] + chunk.extract_start,
                        "end": interval["end"] + chunk.extract_start,
                        "speaker": interval["speaker"],
                    }
                    for interval in diarized_intervals
                ]
                if not chunk.is_chunked:
                    shifted_intervals = [dict(interval) for interval in diarized_intervals]
                trimmed_intervals = trim_intervals_to_window(
                    shifted_intervals,
                    chunk.logical_start,
                    chunk.logical_end,
                )
                merged_intervals = extend_merged_intervals(
                    merged_intervals,
                    trimmed_intervals,
                )

            current_segments, current_payload, written_outputs = write_progress_outputs(
                output_dir=output_dir,
                stem=stem,
                formats=formats,
                audio_path=audio_path,
                detected_language=detected_language,
                diarize=args.diarize,
                args=args,
                duration=duration,
                chunks=chunks,
                transcription_segments=merged_transcription_segments,
                speaker_intervals=merged_intervals,
            )
            write_json(artifact_dir / "final.partial.json", current_payload)
            log(
                f"[chunk {chunk.index + 1}/{len(chunks)}] 중간 저장 완료: "
                f"{', '.join(str(path) for path in written_outputs)} "
                f"({format_elapsed(time.perf_counter() - chunk_started)})"
            )

        merged_transcription_payload = {
            "language": detected_language,
            "segments": merged_transcription_segments,
        }
        write_json(artifact_dir / "mlx_merged.json", merged_transcription_payload)

        if args.diarize:
            write_json(artifact_dir / "speaker_intervals.json", {"intervals": merged_intervals})

        final_segments, payload, written_outputs = write_progress_outputs(
            output_dir=output_dir,
            stem=stem,
            formats=formats,
            audio_path=audio_path,
            detected_language=detected_language,
            diarize=args.diarize,
            args=args,
            duration=duration,
            chunks=chunks,
            transcription_segments=merged_transcription_segments,
            speaker_intervals=merged_intervals,
        )
        write_json(artifact_dir / "final.json", payload)

        log("\n완료:")
        for path in written_outputs:
            log(f"- {path}")
        log(f"- artifact: {artifact_dir}")

    except Exception as exc:
        print(f"에러: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
