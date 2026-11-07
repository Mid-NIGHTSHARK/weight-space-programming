# Capture the browser Arena

The Arena recorder captures the actual browser game canvas and a surrounding panel containing the selected model label, episode state, current observation, hidden-layer activity counts, logits, and action probabilities. It does not record the desktop or add audio. This is a browser demonstration; its random streams and implementation are not the NumPy benchmark evaluator.

## Record a clip

1. Open the deployed site and choose **Live Arena**.
2. Select a primary zero-shot model, set a seed, and confirm the Arena displays the expected model.
3. Select **Record gameplay**. The recorder starts a new episode from that seed, locks the speed to 1×, and records at the Canvas capture rate of 30 fps. The normal Arena loop continues at 1×; recording does not accelerate the policy.
4. Select **Stop & download recording** to save the browser-generated WebM. The Arena frame button downloads one still PNG from the current state.
5. Play the downloaded WebM and inspect the beginning and end before publishing it. Record its actual duration, dimensions, and frame rate; do not infer these from the recorder settings alone.

The model and seed cannot be changed while recording. Recording does not alter the model weights. The browser's environment and policy PRNGs are separate from the benchmark's NumPy streams, so this capture is not a benchmark replay. The inspector exposes the current observation, counts of hidden units over the activity threshold, output logits, and action probabilities. It does not display every hidden activation value.

## Optional H.264 MP4 conversion

If a platform needs MP4, normalize the captured WebM to constant 30 fps without trimming the episode. The source stream has variable presentation timestamps; confirm the resulting frame count and duration rather than assuming the output matches the capture exactly:

```sh
ffmpeg -i arena-capture.webm -map 0:v:0 -an -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart -fps_mode cfr -r 30 arena-capture.mp4
```

Then inspect the actual file metadata:

```sh
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,avg_frame_rate \
  -show_entries format=duration -of default=noprint_wrappers=1 arena-capture.mp4
```

Put reviewed public clips in `docs/assets/gameplay/videos/` and stills in `docs/assets/gameplay/screenshots/`. Rebuild the manifest after adding or replacing media:

```sh
python3 docs/scripts/gameplay/generate_media_manifest.py
```

The script records each media file's size and SHA-256, reads PNG/JPEG dimensions, and queries video dimensions, frame count, frame rate, and duration with `ffprobe`. It also records the capture source and any conversion. The three H.264 copies were normalized to 30 fps from their retained browser WebM sources; original timestamp spans and converted clip durations are recorded separately. Both formats are kept so the source captures remain inspectable. Never label a browser recording as benchmark-equivalent or describe separate runs as matched stochastic trajectories.

## Compose the presentation videos

The repository includes a 30-second, unvoiced project introduction and two side-by-side comparison edits. They are composed from the preserved Arena captures; the title cards use facts already documented by the project. The two source clips in each comparison are aligned at their first frame, played at their recorded 30 fps, and trimmed to the shorter source frame count. Both clips selected seed 42, but the browser random streams were not verified as matched. The comparison is therefore visual and descriptive, not a paired benchmark or synchronized stochastic replay.

With Pillow, ffmpeg, ffprobe, and a supported sans-serif font available, regenerate the three MP4 files with:

```sh
python3 docs/scripts/gameplay/compose_showcase_videos.py
```

The script reads only the preserved Gemini Argon 4 Medium, Claude Sonnet 5.5, and Astra reference captures. It writes new H.264, 1920 × 1080, 30 fps files under `docs/assets/gameplay/videos/`, without changing source recordings, weights, or benchmark data. It prints measured frame counts and durations. Update the media manifest after regeneration with the command above in the previous section.

## Current capture limitations

The browser recorder writes WebM through the browser's MediaRecorder implementation. H.264 MP4 is not captured directly. Three separate captures are preserved: Gemini Argon 4 Medium, Claude Sonnet 5.5 and the Astra reference controller, all started with seed 42. Two comparison edits align separate captures from their first frame; environment and policy random-stream matching is unverified. The project introduction is unvoiced and uses title cards plus actual gameplay. The repository's native Python visualizer screenshots require a functioning Tk/Cocoa display and fail in environments where Tk cannot initialize; that failure does not affect the browser Arena's PNG frame control.
