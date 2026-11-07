#!/usr/bin/env python3
"""Create the gameplay media manifest from captured files and measured metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCREENSHOTS = (
    {
        "file": "docs/assets/gameplay/screenshots/arena-active-gemini-argon-medium-seed42.jpg",
        "description": "Browser screenshot of the actual deployed Arena with Gemini Argon 4 Medium selected, seed 42.",
        "source": "Screenshot of the live GitHub Pages Arena in a browser.",
        "model_label": "Gemini Argon 4 Medium",
        "seed": 42,
    },
    {
        "file": "docs/assets/gameplay/screenshots/arena-frame-gemini-argon-4-medium-seed42.png",
        "description": "Arena-generated PNG of the actual game frame and current policy inspector, Gemini Argon 4 Medium, seed 42.",
        "source": "PNG downloaded from the Arena capture control; game state, observation and inference readouts are from the live browser implementation.",
        "model_label": "Gemini Argon 4 Medium",
        "seed": 42,
    },
    {
        "file": "docs/assets/gameplay/screenshots/website-corrected-results-viewport.jpg",
        "description": "Screenshot of the deployed Results section showing the corrected nine-condition episode summary.",
        "source": "Screenshot of the published GitHub Pages site in the in-app browser; viewport dimensions are recorded below.",
    },
    {
        "file": "docs/assets/gameplay/screenshots/website-visual-results-gallery.jpg",
        "description": "Screenshot of the deployed Visual Results & Gameplay gallery with scientific figures and native video players.",
        "source": "Screenshot of the published GitHub Pages site in the in-app browser; viewport dimensions are recorded below.",
    },
    {
        "file": "docs/assets/gameplay/screenshots/website-methodology-viewport.jpg",
        "description": "Screenshot of the deployed Methodology section describing the research question and MinAtar Breakout implementation.",
        "source": "Screenshot of the published GitHub Pages site in the in-app browser; viewport dimensions are recorded below.",
    },
)
VIDEO_PAIRS = (
    {
        "label": "Gemini Argon 4 Medium",
        "classification": "Recorded primary zero-shot condition label; provider/model provenance limitations remain as documented.",
        "webm": "docs/assets/gameplay/videos/gemini_argon_4_medium_seed42_demo.webm",
        "mp4": "docs/assets/gameplay/videos/gemini_argon_4_medium_seed42_demo.mp4",
    },
    {
        "label": "Claude Sonnet 5.5",
        "classification": "Recorded primary zero-shot condition label; provider/model provenance limitations remain as documented.",
        "webm": "docs/assets/gameplay/videos/claude_sonnet_55_seed42_demo.webm",
        "mp4": "docs/assets/gameplay/videos/claude_sonnet_55_seed42_demo.mp4",
    },
    {
        "label": "Astra reference (local reevaluation)",
        "classification": "Local reference controller; not an LLM-generated zero-shot policy.",
        "webm": "docs/assets/gameplay/videos/astra_reference_seed42_demo.webm",
        "mp4": "docs/assets/gameplay/videos/astra_reference_seed42_demo.mp4",
    },
)
PRESENTATION_VIDEOS = (
    {
        "file": "docs/assets/gameplay/videos/weight_space_programming_intro.mp4",
        "description": "Thirty-second unvoiced introduction combining methodology title cards with actual Gemini Argon 4 Medium Arena gameplay and live inference readouts.",
        "source_files": ["docs/assets/gameplay/videos/gemini_argon_4_medium_seed42_demo.mp4"],
        "policy_classification": "Uses a captured browser demonstration of the recorded Gemini Argon 4 Medium zero-shot condition label; provider identity is not independently authenticated.",
        "derivation": "Four 5/5/15/5-second segments; title cards and documented facts are composed with the first 15 seconds of the retained Arena capture at 1x and 30 fps.",
    },
    {
        "file": "docs/assets/gameplay/videos/zero_shot_side_by_side_comparison.mp4",
        "description": "Side-by-side view of Gemini Argon 4 Medium and Claude Sonnet 5.5 browser Arena captures, aligned from the beginning and trimmed to 571 frames at native 30 fps.",
        "source_files": [
            "docs/assets/gameplay/videos/gemini_argon_4_medium_seed42_demo.mp4",
            "docs/assets/gameplay/videos/claude_sonnet_55_seed42_demo.mp4",
        ],
        "policy_classification": "Two recorded primary zero-shot condition labels; provider/model identity is not independently authenticated.",
        "derivation": "The two preserved 30 fps captures are placed side by side at native playback speed and trimmed to their common frame count; same seed 42 was selected, but stochastic stream matching was not verified.",
    },
    {
        "file": "docs/assets/gameplay/videos/gemini_vs_astra_reference_comparison.mp4",
        "description": "Side-by-side view of the Gemini Argon 4 Medium browser capture and the Astra handcrafted/reference-controller browser capture, aligned from the beginning and trimmed to 464 frames at native 30 fps.",
        "source_files": [
            "docs/assets/gameplay/videos/gemini_argon_4_medium_seed42_demo.mp4",
            "docs/assets/gameplay/videos/astra_reference_seed42_demo.mp4",
        ],
        "policy_classification": "Gemini uses a recorded zero-shot condition label; Astra is a handcrafted/reference controller, not an LLM-generated policy.",
        "derivation": "The two preserved 30 fps captures are placed side by side at native playback speed and trimmed to their common frame count; same seed 42 was selected, but stochastic stream matching was not verified.",
    },
)
OUTPUT = ROOT / "docs/assets/gameplay/media_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as source:
        header = source.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG with an IHDR header: {path}")
    return struct.unpack(">II", header[16:24])


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"Not a JPEG: {path}")
    position = 2
    frame_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    standalone = {0xD8, 0xD9, 0x01, *range(0xD0, 0xD8)}
    while position < len(data):
        while position < len(data) and data[position] != 0xFF:
            position += 1
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in standalone:
            continue
        if position + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[position : position + 2])[0]
        if marker in frame_markers:
            height, width = struct.unpack(">HH", data[position + 3 : position + 7])
            return width, height
        if segment_length < 2:
            break
        position += segment_length
    raise ValueError(f"No JPEG frame header found: {path}")


def probe(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to measure video files")
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "format=duration,size:stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        text=True,
    )
    result = json.loads(raw)
    if not result.get("streams"):
        raise ValueError(f"No video stream found: {path}")
    stream = result["streams"][0]
    info: dict[str, Any] = {
        "codec": stream["codec_name"],
        "resolution": {"width": int(stream["width"]), "height": int(stream["height"])},
        "reported_frame_rate": stream.get("avg_frame_rate"),
        "frame_count": int(stream["nb_read_frames"]),
        "file_duration_seconds": float(result["format"]["duration"]) if result.get("format", {}).get("duration") else None,
    }
    if path.suffix.lower() == ".webm":
        packets = json.loads(
            subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets", "-show_entries", "packet=pts_time", "-of", "json", str(path)],
                text=True,
            )
        )["packets"]
        pts = [float(packet["pts_time"]) for packet in packets if packet.get("pts_time") is not None]
        if not pts:
            raise ValueError(f"No presentation timestamps found: {path}")
        info["packet_timestamp_span_seconds"] = round(max(pts) - min(pts), 6)
        info["frames_per_timestamp_span_estimate"] = round(len(pts) / (max(pts) - min(pts)), 3)
    return info


def entry(relative: str, description: str, source: str, **metadata: Any) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    common = {
        "file": relative,
        "description": description,
        "source": source,
        "format": path.suffix.lstrip(".").lower(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    common.update(metadata)
    return common


def main() -> None:
    assets: list[dict[str, Any]] = []
    for screenshot in SCREENSHOTS:
        relative = screenshot["file"]
        image_path = ROOT / relative
        width, height = jpeg_dimensions(image_path) if image_path.suffix.lower() in {".jpg", ".jpeg"} else png_dimensions(image_path)
        metadata = {"resolution": {"width": width, "height": height}}
        for key in ("model_label", "seed"):
            if key in screenshot:
                metadata[key] = screenshot[key]
        assets.append(entry(relative, screenshot["description"], screenshot["source"], **metadata))

    for version in VIDEO_PAIRS:
        webm, mp4 = version["webm"], version["mp4"]
        model = version["label"]
        webm_info = probe(ROOT / webm)
        assets.append(
            entry(
                webm,
                f"Original Canvas MediaRecorder gameplay with the Arena decision inspector, {model}, seed 42.",
                "Captured from the deployed browser Arena at 1x playback; MediaRecorder canvas capture configured for 30 fps. No audio. Separate run; no synchronized or RNG-matched comparison is claimed.",
                model_label=model,
                policy_classification=version["classification"],
                seed=42,
                audio=False,
                capture_rate_fps=30,
                measured=webm_info,
                benchmark_equivalent=False,
            )
        )
        mp4_info = probe(ROOT / mp4)
        assets.append(
            entry(
                mp4,
                f"H.264 browser gameplay copy of the {model} seed 42 Arena capture.",
                "Transcoded from the retained WebM using ffmpeg libx264, CRF 18, yuv420p, faststart, constant 30 fps; no audio, trimming, or playback-speed increase.",
                model_label=model,
                policy_classification=version["classification"],
                seed=42,
                audio=False,
                derived_from=webm,
                conversion_command="ffmpeg -i input.webm -map 0:v:0 -an -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart -fps_mode cfr -r 30 output.mp4",
                measured=mp4_info,
                benchmark_equivalent=False,
            )
        )

    for video in PRESENTATION_VIDEOS:
        derived_from = [
            {
                "file": relative,
                "bytes": (ROOT / relative).stat().st_size,
                "sha256": sha256(ROOT / relative),
            }
            for relative in video["source_files"]
        ]
        assets.append(
            entry(
                video["file"],
                video["description"],
                f"Generated by docs/scripts/gameplay/compose_showcase_videos.py from preserved source capture(s): {', '.join(video['source_files'])}.",
                policy_classification=video["policy_classification"],
                derived_from=derived_from,
                derivation=video["derivation"],
                composition_script="docs/scripts/gameplay/compose_showcase_videos.py",
                audio=False,
                measured=probe(ROOT / video["file"]),
                benchmark_equivalent=False,
            )
        )

    manifest = {
        "schema_version": 1,
        "generated_by": "docs/scripts/gameplay/generate_media_manifest.py",
        "note": "Gameplay recordings and deployed-site screenshots are recorded separately. Browser Arena media are demonstrations and do not reproduce the NumPy benchmark RNG streams; same seed labels do not assert synchronized trajectories.",
        "assets": assets,
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Recorded metadata and SHA-256 for {len(assets)} gameplay media assets: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
