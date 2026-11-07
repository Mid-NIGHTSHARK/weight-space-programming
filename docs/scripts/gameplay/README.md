# Capture the browser Arena

The Arena recorder captures the actual browser game canvas and a surrounding panel containing the selected model label, episode state, current observation, hidden-layer activity counts, logits, and action probabilities. It does not record the desktop or add audio. This is a browser demonstration; its random streams and implementation are not the NumPy benchmark evaluator.

## Record a clip

1. Open the deployed site and choose **Live Arena**.
2. Select a primary zero-shot model, set a seed, and confirm the Arena displays the expected model.
3. Select **Record gameplay**. The recorder starts a new episode from that seed, locks the speed to 1×, and records at the Canvas capture rate of 30 fps. The normal Arena loop continues at 1×; recording does not accelerate the policy.
4. Select **Stop & download recording** to save the browser-generated WebM. The Arena frame button downloads one still PNG from the current state.
5. Play the downloaded WebM and inspect the beginning and end before publishing it. Record its actual duration, dimensions, and frame rate; do not infer these from the recorder settings alone.

The model and seed cannot be changed while recording. Recording does not alter the model weights. The browser's environment and policy PRNGs are separate from the benchmark's NumPy streams, so this capture is not a benchmark replay.

## Optional H.264 MP4 conversion

If a platform needs MP4, convert the captured WebM without changing its playback rate or trimming the episode:

```sh
ffmpeg -i arena-capture.webm -map 0:v:0 -an -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart arena-capture.mp4
```

Then inspect the actual file metadata:

```sh
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,avg_frame_rate \
  -show_entries format=duration -of default=noprint_wrappers=1 arena-capture.mp4
```

Put reviewed public clips in `docs/assets/gameplay/videos/` and stills in `docs/assets/gameplay/screenshots/`. Add each published item to a gameplay media manifest with a description, selected model and seed where known, source (Arena capture), generation/conversion details, actual dimensions and frame rate, duration for video, format, and SHA-256. Keep the original browser WebM when an MP4 conversion is published; record both hashes and the conversion command. Never label a browser recording as benchmark-equivalent or describe different seeds as matched stochastic trajectories.

## Current capture limitations

The browser recorder writes WebM through the browser's MediaRecorder implementation. H.264 MP4 is not captured directly. A clip must be captured, inspected, and verified before it is shown as a published video. The repository's native Python visualizer screenshots require a functioning Tk/Cocoa display and fail in environments where Tk cannot initialize; that failure does not affect the browser Arena's PNG frame control.
