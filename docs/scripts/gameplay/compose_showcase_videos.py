#!/usr/bin/env python3
"""Compose short, labeled presentations from preserved browser Arena captures.

Only new presentation videos are written. The original Arena captures, model
weights, benchmark data, and experiment configuration are never modified.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
VIDEO_DIR = ROOT / "docs/assets/gameplay/videos"
OUTPUTS = {
    "intro": VIDEO_DIR / "weight_space_programming_intro.mp4",
    "zero_shot_comparison": VIDEO_DIR / "zero_shot_side_by_side_comparison.mp4",
    "reference_comparison": VIDEO_DIR / "gemini_vs_astra_reference_comparison.mp4",
}
SOURCES = {
    "gemini": VIDEO_DIR / "gemini_argon_4_medium_seed42_demo.mp4",
    "claude": VIDEO_DIR / "claude_sonnet_55_seed42_demo.mp4",
    "astra": VIDEO_DIR / "astra_reference_seed42_demo.mp4",
}
FPS = 30
WIDTH, HEIGHT = 1920, 1080
BACKGROUND = "#f4f7fb"
INK = "#17263c"
MUTED = "#52647a"
BLUE = "#1769d2"
REGULAR_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Verdana.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
)
BOLD_FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Verdana Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
)


def find_font(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a supported font from: {candidates}")


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg is required to compose the showcase videos")
    return path


def frame_count(path: Path) -> int:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required to inspect the source videos")
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "json", str(path),
        ],
        text=True,
    )
    streams = json.loads(raw).get("streams", [])
    if not streams or not streams[0].get("nb_read_frames"):
        raise ValueError(f"Could not count video frames: {path}")
    return int(streams[0]["nb_read_frames"])


def load_fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    regular = ImageFont.truetype(find_font(REGULAR_FONT_CANDIDATES), 30)
    bold = ImageFont.truetype(find_font(BOLD_FONT_CANDIDATES), 32)
    return regular, bold


def fit_font(text: str, font_path: Path, max_width: int, start_size: int) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > 18:
        font = ImageFont.truetype(font_path, size)
        if ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)[2] <= max_width:
            return font
        size -= 2
    return ImageFont.truetype(font_path, size)


def text_centered(draw: ImageDraw.ImageDraw, y: int, text: str, font: ImageFont.FreeTypeFont, fill: str) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    x = (WIDTH - (bounds[2] - bounds[0])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def intro_card(path: Path, regular_path: Path, bold_path: Path, eyebrow: str, title: str, lines: list[str], footer: str) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((56, 56, WIDTH - 56, HEIGHT - 56), radius=26, fill="#ffffff", outline="#d8e1ec", width=2)
    draw.rectangle((56, 56, WIDTH - 56, 70), fill=BLUE)
    small = ImageFont.truetype(regular_path, 25)
    title_font = fit_font(title, bold_path, 1670, 58)
    body_font = ImageFont.truetype(regular_path, 34)
    footer_font = fit_font(footer, regular_path, 1680, 27)
    text_centered(draw, 210, eyebrow.upper(), small, BLUE)
    text_centered(draw, 302, title, title_font, INK)
    draw.rounded_rectangle((720, 396, 1200, 404), radius=4, fill=BLUE)
    for index, line in enumerate(lines):
        line_font = fit_font(line, regular_path, 1680, body_font.size)
        text_centered(draw, 470 + index * 90, line, line_font, MUTED if index else INK)
    text_centered(draw, 875, footer, footer_font, MUTED)
    image.save(path, format="PNG", optimize=True)


def comparison_card(path: Path, regular_path: Path, bold_path: Path, title: str, left_label: str, right_label: str) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    header_font = fit_font(title, bold_path, 1760, 42)
    subtitle_font = ImageFont.truetype(regular_path, 25)
    model_font_left = fit_font(left_label, bold_path, 870, 29)
    model_font_right = fit_font(right_label, bold_path, 870, 29)
    footer = "Separate browser runs start with numeric seed 42; stochastic stream matching was not verified."
    footer_font = fit_font(footer, regular_path, 1780, 24)
    note = "Browser Arena uses JavaScript random streams and is not the NumPy benchmark evaluator."
    note_font = fit_font(note, regular_path, 1780, 24)
    draw.text((60, 34), title, font=header_font, fill=INK)
    draw.text((60, 103), "Actual Arena recordings at the captured playback rate; no trajectories are fabricated.", font=subtitle_font, fill=MUTED)
    draw.rounded_rectangle((35, 151, 945, 205), radius=8, fill="#ffffff", outline="#d8e1ec", width=1)
    draw.rounded_rectangle((975, 151, 1885, 205), radius=8, fill="#ffffff", outline="#d8e1ec", width=1)
    draw.text((50, 164), left_label, font=model_font_left, fill=BLUE)
    draw.text((990, 164), right_label, font=model_font_right, fill=BLUE)
    draw.rounded_rectangle((60, 815, 1860, 904), radius=12, fill="#ffffff", outline="#d8e1ec", width=1)
    draw.text((82, 832), footer, font=footer_font, fill=INK)
    draw.text((82, 871), note, font=note_font, fill=MUTED)
    image.save(path, format="PNG", optimize=True)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def encode_intro(executable: str, regular_path: Path, bold_path: Path, temp_dir: Path) -> None:
    cards = [
        temp_dir / "intro-title.png",
        temp_dir / "intro-method.png",
        temp_dir / "intro-limits.png",
    ]
    intro_card(
        cards[0], regular_path, bold_path, "Research snapshot", "Weight-Space Programming",
        ["Can a language model write useful neural-network weights directly?", "Experiment 5 uses MinAtar Breakout to study this question."],
        "The networks are evaluated after generation, without conventional training.",
    )
    intro_card(
        cards[1], regular_path, bold_path, "From observation to action", "A fixed network plays Breakout",
        ["10 x 10 board x 6 channels = 600 binary inputs", "600 -> 32 -> 32 -> 6; tanh hidden layers and six action logits", "Softmax probabilities are used to sample the next action."],
        "The recorded Gemini clip shows the live observation and inference readouts.",
    )
    intro_card(
        cards[2], regular_path, bold_path, "Evidence and limits", "Results come from the Python benchmark",
        ["Five primary zero-shot conditions were evaluated for 3,000 episodes each.", "Astra, PPO, and simple controls have separate scientific roles.", "The browser Arena is a demonstration, not a benchmark replay."],
        "Provider identity and some generation settings remain unauthenticated.",
    )
    filters = [
        "[0:v]fps=30,format=yuv420p[title]",
        "[1:v]fps=30,format=yuv420p[method]",
        f"[2:v]trim=end_frame={15 * FPS},setpts=PTS-STARTPTS,scale={WIDTH}:{HEIGHT}:flags=lanczos,fps={FPS},setsar=1,format=yuv420p[game]",
        "[3:v]fps=30,format=yuv420p[limits]",
        "[title][method][game][limits]concat=n=4:v=1:a=0,format=yuv420p[out]",
    ]
    command = [executable, "-hide_banner", "-loglevel", "error", "-y"]
    for card in cards[:2]:
        command.extend(["-loop", "1", "-framerate", str(FPS), "-t", "5", "-i", str(card)])
    command.extend(["-i", str(SOURCES["gemini"])])
    command.extend(["-loop", "1", "-framerate", str(FPS), "-t", "5", "-i", str(cards[2])])
    command.extend(
        [
            "-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264",
            "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-r", str(FPS), str(OUTPUTS["intro"]),
        ]
    )
    run(command)


def encode_comparison(
    executable: str,
    regular_path: Path,
    bold_path: Path,
    temp_dir: Path,
    output: Path,
    left: Path,
    right: Path,
    title: str,
    left_label: str,
    right_label: str,
) -> int:
    frames = min(frame_count(left), frame_count(right))
    card = temp_dir / (output.stem + "-header.png")
    comparison_card(card, regular_path, bold_path, title, left_label, right_label)
    duration = frames / FPS
    filters = [
        "[0:v]fps=30,format=yuv420p[base]",
        f"[1:v]trim=end_frame={frames},setpts=PTS-STARTPTS,scale=960:540:flags=lanczos,fps={FPS},setsar=1,format=yuv420p[left]",
        f"[2:v]trim=end_frame={frames},setpts=PTS-STARTPTS,scale=960:540:flags=lanczos,fps={FPS},setsar=1,format=yuv420p[right]",
        "[base][left]overlay=x=0:y=220:shortest=1[withleft]",
        "[withleft][right]overlay=x=960:y=220:shortest=1,format=yuv420p[out]",
    ]
    command = [
        executable, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS),
        "-t", f"{duration:.9f}", "-i", str(card), "-i", str(left), "-i", str(right),
        "-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264",
        "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-r", str(FPS), str(output),
    ]
    run(command)
    return frames


def main() -> None:
    for source in SOURCES.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    regular_path = find_font(REGULAR_FONT_CANDIDATES)
    bold_path = find_font(BOLD_FONT_CANDIDATES)
    executable = ffmpeg_path()
    with tempfile.TemporaryDirectory(prefix="wsp-showcase-") as temp_name:
        temp_dir = Path(temp_name)
        encode_intro(executable, regular_path, bold_path, temp_dir)
        primary_frames = encode_comparison(
            executable, regular_path, bold_path, temp_dir, OUTPUTS["zero_shot_comparison"],
            SOURCES["gemini"], SOURCES["claude"], "PRIMARY ZERO-SHOT POLICY COMPARISON",
            "Gemini Argon 4 Medium | recorded zero-shot label",
            "Claude Sonnet 5.5 | recorded zero-shot label",
        )
        reference_frames = encode_comparison(
            executable, regular_path, bold_path, temp_dir, OUTPUTS["reference_comparison"],
            SOURCES["gemini"], SOURCES["astra"], "LLM POLICY AND REFERENCE CONTROLLER",
            "Gemini Argon 4 Medium | recorded zero-shot label",
            "Astra | handcrafted/reference controller",
        )
    print("Wrote three 1920x1080, 30 fps H.264 presentation videos.")
    print(f"Introduction: {frame_count(OUTPUTS['intro'])} frames, {frame_count(OUTPUTS['intro']) / FPS:.3f} seconds.")
    print(f"Zero-shot comparison: {primary_frames} frames, {primary_frames / FPS:.3f} seconds.")
    print(f"Gemini vs Astra comparison: {reference_frames} frames, {reference_frames / FPS:.3f} seconds.")
    print("Sources are preserved Arena captures; no model weights or benchmark records were changed.")


if __name__ == "__main__":
    main()
