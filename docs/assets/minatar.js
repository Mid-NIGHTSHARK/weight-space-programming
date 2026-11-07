export const ACTION_NAMES = ["no-op", "left", "up", "right", "down", "fire"];
export const ACTION_CHARS = ["n", "l", "u", "r", "d", "f"];
export const CHANNELS = ["Paddle", "Danger", "Bricks", "Ball", "Trail", "Gauge"];
const dx = [-1, 1, 1, -1];
const dy = [-1, -1, 1, 1];
const wallReflect = [1, 0, 3, 2];
const verticalReflect = [3, 2, 1, 0];
const paddleReflect = [2, 3, 0, 1];

function mix32(value) {
  let x = value >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x21f0aaad);
  x = Math.imul(x ^ (x >>> 15), 0x735a2d97);
  return (x ^ (x >>> 15)) >>> 0;
}

export function makeDemoRandom(seed, domain = 0) {
  let state = mix32((Number(seed) >>> 0) ^ (domain >>> 0) ^ 0x9e3779b9);
  if (state === 0) state = 0x6d2b79f5;
  return {
    nextUint32() {
      state ^= state << 13;
      state ^= state >>> 17;
      state ^= state << 5;
      return state >>> 0;
    },
    next() { return this.nextUint32() / 0x100000000; },
  };
}

export function makeDemoStreams(seed) {
  const normalized = Number(seed) >>> 0;
  return {
    environment: makeDemoRandom(normalized, 0xa341316c),
    policy: makeDemoRandom(normalized, 0xc8013ea4),
  };
}

export function createInitialState(ballStart = 0) {
  const side = Number(ballStart) === 1 ? 1 : 0;
  const ballX = side === 0 ? 0 : 9;
  const ballDir = side === 0 ? 2 : 3;
  const bricks = new Uint8Array(100);
  for (let row = 1; row <= 3; row += 1) {
    bricks.fill(1, row * 10, row * 10 + 10);
  }
  return {
    ballX,
    ballY: 3,
    ballDir,
    paddle: 4,
    lastX: ballX,
    lastY: 3,
    bricks,
    strike: false,
    terminal: false,
    step: 0,
    score: 0,
    lastAction: 0,
    terminalReason: null,
  };
}

export function resetEpisode(seed, streams = makeDemoStreams(seed)) {
  const side = Math.floor(streams.environment.next() * 2);
  return { state: createInitialState(side), streams, seed: Number(seed) };
}

export function advanceState(current, selectedAction, stickyDraw, stickyProbability = 0.1, maxSteps = 10000) {
  const selected = Number(selectedAction);
  if (!Number.isInteger(selected) || selected < 0 || selected > 5) throw new RangeError("Action must be an integer in [0, 5]");
  if (current.terminal) throw new Error("Cannot step a terminal episode; reset first");
  if (!(stickyProbability >= 0 && stickyProbability <= 1)) throw new RangeError("Sticky-action probability must be in [0, 1]");
  const state = { ...current, bricks: new Uint8Array(current.bricks) };
  const stickyOverride = Number(stickyDraw) < stickyProbability;
  const executed = stickyOverride ? state.lastAction : selected;
  state.lastAction = executed;
  if (executed === 1) state.paddle = Math.max(0, state.paddle - 1);
  else if (executed === 3) state.paddle = Math.min(9, state.paddle + 1);

  const oldX = state.ballX;
  const oldY = state.ballY;
  let nextX = oldX + dx[state.ballDir];
  let nextY = oldY + dy[state.ballDir];
  if (nextX < 0 || nextX > 9) {
    nextX = Math.max(0, Math.min(9, nextX));
    state.ballDir = wallReflect[state.ballDir];
  }
  let strikeToggle = false;
  if (nextY < 0) {
    nextY = 0;
    state.ballDir = verticalReflect[state.ballDir];
  } else if (state.bricks[nextY * 10 + nextX] === 1) {
    strikeToggle = true;
    if (!state.strike) {
      state.score += 1;
      state.bricks[nextY * 10 + nextX] = 0;
      nextY = oldY;
      state.ballDir = verticalReflect[state.ballDir];
      state.strike = true;
    }
  } else if (nextY === 9) {
    if (state.bricks.reduce((sum, value) => sum + value, 0) === 0) {
      for (let row = 1; row <= 3; row += 1) state.bricks.fill(1, row * 10, row * 10 + 10);
    }
    if (oldX === state.paddle) {
      state.ballDir = verticalReflect[state.ballDir];
      nextY = oldY;
    } else if (nextX === state.paddle) {
      state.ballDir = paddleReflect[state.ballDir];
      nextY = oldY;
    } else {
      state.terminal = true;
      state.terminalReason = "ball_lost";
    }
  }
  if (!strikeToggle) state.strike = false;
  state.lastX = oldX;
  state.lastY = oldY;
  state.ballX = nextX;
  state.ballY = nextY;
  state.step += 1;
  if (!state.terminal && state.step >= maxSteps) {
    state.terminal = true;
    state.terminalReason = "step_cutoff";
  }
  return {
    state,
    reward: state.score - current.score,
    done: state.terminal,
    info: {
      step: state.step,
      totalReward: state.score,
      ballLost: state.terminalReason === "ball_lost",
      stepCutoff: state.terminalReason === "step_cutoff",
      paddle: state.paddle,
      ballX: state.ballX,
      ballY: state.ballY,
      ballDir: state.ballDir,
      bricksRemaining: state.bricks.reduce((sum, value) => sum + value, 0),
      selectedAction: selected,
      executedAction: executed,
      stickyOverride,
    },
  };
}

export function encodeObservation(state) {
  const result = new Float32Array(600);
  result[(9 * 10 + state.paddle) * 6] = 1;
  for (let i = 0; i < 100; i += 1) {
    if (state.bricks[i]) result[i * 6 + 2] = 1;
  }
  result[(state.ballY * 10 + state.ballX) * 6 + 3] = 1;
  result[(state.lastY * 10 + state.lastX) * 6 + 4] = 1;
  return result;
}

export function encodeNativeState(native) {
  if (!Array.isArray(native) || native.length !== 100) throw new TypeError("Native state must contain 100 cells");
  const observation = new Float32Array(600);
  for (let cell = 0; cell < 100; cell += 1) {
    const channels = native[cell];
    if (!Array.isArray(channels) || channels.length !== 4) throw new TypeError("Native cells must contain paddle, ball, trail, brick channels");
    const start = cell * 6;
    observation[start] = channels[0];
    observation[start + 2] = channels[3];
    observation[start + 3] = channels[1];
    observation[start + 4] = channels[2];
  }
  return observation;
}

export function channelGrid(observation, channel) {
  const grid = new Uint8Array(100);
  for (let cell = 0; cell < 100; cell += 1) grid[cell] = observation[cell * 6 + Number(channel)] ? 1 : 0;
  return grid;
}
