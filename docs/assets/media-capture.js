export const CAPTURE_WIDTH = 1920;
export const CAPTURE_HEIGHT = 1080;
export const CAPTURE_FPS = 30;

const escapeName = value => String(value ?? "policy").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 48) || "policy";

export function renderCaptureFrame(context, gameCanvas, frame) {
  const width = CAPTURE_WIDTH;
  const height = CAPTURE_HEIGHT;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f3f6fa";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, 112);
  context.fillStyle = "#1764d5";
  context.fillRect(0, 109, width, 3);
  context.textAlign = "left";
  context.fillStyle = "#152238";
  context.font = "700 24px Inter, Arial, sans-serif";
  context.fillText("WEIGHT-SPACE PROGRAMMING", 64, 48);
  context.fillStyle = "#53637a";
  context.font = "16px Inter, Arial, sans-serif";
  context.fillText("MINATAR BREAKOUT · BROWSER ARENA DEMONSTRATION", 64, 78);

  // The game image is copied from the Arena's actual, live canvas.
  context.fillStyle = "#ffffff";
  context.fillRect(56, 145, 850, 850);
  context.fillStyle = "#142033";
  context.fillRect(56, 145, 850, 58);
  context.fillStyle = "#ffffff";
  context.font = "600 19px Inter, Arial, sans-serif";
  context.fillText(`Score  ${frame.score}`, 80, 181);
  context.fillText(`Step  ${frame.step.toLocaleString("en-US")}`, 275, 181);
  context.fillText(`Seed  ${frame.seed}`, 510, 181);
  context.fillStyle = frame.terminal ? "#fde7df" : "#ddf4e7";
  context.fillRect(735, 158, 148, 32);
  context.fillStyle = frame.terminal ? "#a33c23" : "#176443";
  context.font = "700 15px Inter, Arial, sans-serif";
  context.textAlign = "center";
  context.fillText(frame.terminal ? "EPISODE OVER" : frame.running ? "PLAYING" : "PAUSED", 809, 180);
  context.drawImage(gameCanvas, 106, 218, 750, 750);

  // This card is populated from the policy and inspector values already used by the Arena.
  context.fillStyle = "#ffffff";
  context.fillRect(932, 145, 932, 850);
  context.textAlign = "left";
  context.fillStyle = "#152238";
  context.font = "700 30px Inter, Arial, sans-serif";
  context.fillText(frame.model, 974, 200, 840);
  context.fillStyle = "#1764d5";
  context.font = "600 16px Inter, Arial, sans-serif";
  context.fillText(frame.category.toUpperCase(), 974, 235);
  context.fillStyle = "#eaf2ff";
  context.fillRect(974, 266, 850, 86);
  context.fillStyle = "#53637a";
  context.font = "14px Inter, Arial, sans-serif";
  context.fillText("MOST RECENT SELECTED ACTION", 996, 294);
  context.fillStyle = "#152238";
  context.font = "700 24px Inter, Arial, sans-serif";
  context.fillText(frame.lastAction ?? "No action sampled yet", 996, 330);
  context.fillStyle = "#53637a";
  context.font = "14px Inter, Arial, sans-serif";
  context.fillText(`Network: 600 → 32 → 32 → 6  ·  ${frame.parameters.toLocaleString("en-US")} parameters  ·  ${frame.running ? "live episode" : "paused state"}`, 1375, 315, 425);

  context.fillStyle = "#152238";
  context.font = "700 20px Inter, Arial, sans-serif";
  context.fillText("Current observation → action probabilities", 974, 399);
  const maxProbability = Math.max(...frame.probabilities, 1e-12);
  frame.probabilities.forEach((probability, index) => {
    const y = 438 + index * 43;
    context.fillStyle = "#34445b";
    context.font = "15px Inter, Arial, sans-serif";
    context.fillText(frame.actionNames[index], 974, y);
    context.fillStyle = "#e5ebf2";
    context.fillRect(1100, y - 12, 555, 12);
    context.fillStyle = frame.highlightAction === frame.actionNames[index] ? "#1764d5" : "#5793df";
    context.fillRect(1100, y - 12, 555 * probability / maxProbability, 12);
    context.fillStyle = "#253247";
    context.textAlign = "right";
    context.fillText(`${(probability * 100).toFixed(2)}%`, 1815, y);
    context.textAlign = "left";
  });

  context.fillStyle = "#152238";
  context.font = "700 17px Inter, Arial, sans-serif";
  context.fillText(`Current observation input · ${frame.selectedChannel}`, 974, 710);
  const cell = 17;
  const gridColors = ["#2185df", "#dbe3ee", "#e86750", "#f0cd57", "#ffb869", "#dbe3ee"];
  frame.observationGrid.forEach((value, index) => {
    const x = 974 + (index % 10) * cell;
    const y = 730 + Math.floor(index / 10) * cell;
    context.fillStyle = value ? gridColors[frame.selectedChannelIndex] : "#0b1119";
    context.fillRect(x, y, cell - 1, cell - 1);
    context.strokeStyle = "rgba(145,164,188,.30)";
    context.strokeRect(x + 0.5, y + 0.5, cell - 2, cell - 2);
  });
  context.fillStyle = "#53637a";
  context.font = "12px Inter, Arial, sans-serif";
  context.fillText("10 × 10 · selected channel", 974, 913);
  context.fillStyle = "#152238";
  context.font = "700 17px Inter, Arial, sans-serif";
  context.fillText("Network inference", 1170, 748);
  context.fillStyle = "#53637a";
  context.font = "14px Inter, Arial, sans-serif";
  context.fillText(`Input active cells: ${frame.activeByChannel.map((value, i) => `${frame.channelNames[i]} ${value}`).join(" · ")}`, 1170, 783, 640);
  context.fillText(frame.randomPolicy ? "Random-action control: no network activations or logits." : `Hidden |h| > 0.5: layer 1 ${frame.h1Active}/32 · layer 2 ${frame.h2Active}/32`, 1170, 816, 640);
  context.fillText(frame.logits ? `Output logits: ${frame.logits.map(value => value.toFixed(4)).join("  ·  ")}` : "Logits are not used by this control.", 1170, 849, 640);
  context.fillText(`Input size 600 · architecture 600 → 32 → 32 → 6`, 1170, 882, 640);
  context.fillText("Stable softmax maps logits to the displayed action probabilities.", 1170, 915, 640);
  context.fillStyle = "#6b7788";
  context.font = "13px Inter, Arial, sans-serif";
  context.fillText("Fixed archived weights · no training or weight changes during playback", 974, 943);
  context.fillText("Browser demonstration only; its JavaScript random streams do not reproduce the NumPy benchmark streams.", 64, 1044);
  context.textAlign = "right";
  context.fillText("30 fps canvas capture · live step timing preserved", 1854, 1044);
  context.textAlign = "left";
}

export function downloadCanvasPng(canvas, filename) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => {
      if (!blob) { reject(new Error("The browser could not encode this Arena frame as PNG.")); return; }
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      resolve(blob.size);
    }, "image/png");
  });
}

export function createArenaRecorder({ canvas, gameCanvas, getFrame, onStatus, MediaRecorderImpl = globalThis.MediaRecorder }) {
  let recorder = null;
  let animation = null;
  let stream = null;
  let chunks = [];
  let activeModel = "policy";

  const render = () => {
    renderCaptureFrame(canvas.getContext("2d"), gameCanvas, getFrame());
  };
  const draw = () => {
    animation = null;
    render();
    if (recorder?.state === "recording") animation = requestAnimationFrame(draw);
  };

  const stopTracks = () => {
    if (animation !== null) cancelAnimationFrame(animation);
    animation = null;
    stream?.getTracks().forEach(track => track.stop());
    stream = null;
  };

  const downloadRecording = () => {
    const blob = new Blob(chunks, { type: recorder.mimeType || "video/webm" });
    chunks = [];
    if (blob.size === 0) {
      onStatus("Recording was empty; no video was downloaded.", false);
      return;
    }
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `arena-${escapeName(activeModel)}-${new Date().toISOString().replace(/[:.]/g, "-")}.webm`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    onStatus("WebM recording downloaded at the Arena's live speed (30 fps capture).", false);
  };

  return {
    snapshot() {
      render();
      const model = escapeName(getFrame().model);
      return downloadCanvasPng(canvas, `arena-${model}-frame.png`)
        .then(() => onStatus("PNG frame of the actual Arena game and current network decision downloaded.", false))
        .catch(error => onStatus(`Could not export Arena frame: ${error.message}`, false));
    },
    start() {
      if (!MediaRecorderImpl || typeof canvas.captureStream !== "function") {
        onStatus("Recording is unavailable in this browser; the Arena itself remains usable.", false);
        return false;
      }
      try {
        activeModel = getFrame().model;
        const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(type => !MediaRecorderImpl.isTypeSupported || MediaRecorderImpl.isTypeSupported(type));
        stream = canvas.captureStream(CAPTURE_FPS);
        recorder = new MediaRecorderImpl(stream, mimeType ? { mimeType, videoBitsPerSecond: 6_000_000 } : { videoBitsPerSecond: 6_000_000 });
        chunks = [];
        recorder.ondataavailable = event => { if (event.data?.size) chunks.push(event.data); };
        recorder.onstop = () => { stopTracks(); downloadRecording(); };
        recorder.start(1000);
        draw();
        onStatus("Recording actual Arena state. Stop to download the WebM; gameplay speed remains controlled by the Arena.", true);
        return true;
      } catch (error) {
        stopTracks();
        recorder = null;
        onStatus(`Could not start recording: ${error.message}`, false);
        return false;
      }
    },
    stop() {
      if (recorder?.state === "recording") recorder.stop();
    },
    isRecording() {
      return recorder?.state === "recording";
    },
  };
}
