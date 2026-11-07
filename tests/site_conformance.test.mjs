import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ACTION_CHARS, advanceState, channelGrid, createInitialState, encodeNativeState, encodeObservation, makeDemoStreams, resetEpisode } from "../docs/assets/minatar.js";
import { forward, sampleAction, stableSoftmax } from "../docs/assets/network.js";
import { renderIterativeComparisonRows, renderIterativeHistograms, renderIterativeMeanChart, renderIterativeProfiles, renderIterativeResultRows } from "../docs/assets/iterative.js";

test("action identifiers preserve the MinAtar action map", () => {
  assert.deepEqual(ACTION_CHARS, ["n", "l", "u", "r", "d", "f"]);
});

test("native channels map into the six input channels in row-major order", () => {
  const native = Array.from({ length: 100 }, () => [0, 0, 0, 0]);
  native[94] = [1, 0, 0, 0];
  native[32] = [0, 1, 0, 0];
  native[43] = [0, 0, 1, 0];
  native[27] = [0, 0, 0, 1];
  const observation = encodeNativeState(native);
  assert.equal(observation[(9 * 10 + 4) * 6], 1);
  assert.equal(observation[(3 * 10 + 2) * 6 + 3], 1);
  assert.equal(observation[(4 * 10 + 3) * 6 + 4], 1);
  assert.equal(observation[(2 * 10 + 7) * 6 + 2], 1);
  assert.equal(observation.length, 600);
});

test("Breakout resets with the one-cell paddle, thirty bricks, and selected side", () => {
  const left = createInitialState(0);
  const right = createInitialState(1);
  assert.deepEqual([left.ballX, left.ballY, left.ballDir, left.paddle], [0, 3, 2, 4]);
  assert.deepEqual([right.ballX, right.ballY, right.ballDir, right.paddle], [9, 3, 3, 4]);
  assert.equal(left.bricks.reduce((sum, v) => sum + v, 0), 30);
  const observation = encodeObservation(left);
  assert.equal(channelGrid(observation, 0).reduce((a, b) => a + b, 0), 1);
  assert.equal(channelGrid(observation, 2).reduce((a, b) => a + b, 0), 30);
});

test("paddle moves before ball collision and current-x capture is preserved", () => {
  const state = createInitialState(0);
  state.ballX = 5; state.ballY = 8; state.ballDir = 3;
  state.paddle = 6; state.lastAction = 0;
  const result = advanceState(state, 1, 0.99, 0.1);
  assert.equal(result.state.paddle, 5);
  assert.equal(result.state.ballX, 4);
  assert.equal(result.state.ballY, 8);
  assert.equal(result.done, false);
});

test("sticky action can override a selected action", () => {
  const state = createInitialState(0);
  state.ballX = 5; state.ballY = 5; state.ballDir = 2;
  state.lastAction = 1;
  const result = advanceState(state, 3, 0.05, 0.1);
  assert.equal(result.info.selectedAction, 3);
  assert.equal(result.info.executedAction, 1);
  assert.equal(result.info.stickyOverride, true);
  assert.equal(result.state.paddle, 3);
});

test("an uncatchable ball ends the episode and a max-step cutoff is distinct", () => {
  const miss = createInitialState(0);
  miss.ballX = 5; miss.ballY = 8; miss.ballDir = 2; miss.paddle = 4;
  const lost = advanceState(miss, 0, 0.99);
  assert.equal(lost.done, true);
  assert.equal(lost.info.ballLost, true);
  assert.equal(lost.info.stepCutoff, false);

  const cutoff = createInitialState(0);
  cutoff.ballX = 5; cutoff.ballY = 5; cutoff.ballDir = 2; cutoff.step = 9;
  const limited = advanceState(cutoff, 0, 0.99, 0.1, 10);
  assert.equal(limited.done, true);
  assert.equal(limited.info.ballLost, false);
  assert.equal(limited.info.stepCutoff, true);
});

test("the corrected game channels leave danger and gauge empty", () => {
  const state = resetEpisode(42);
  const observation = encodeObservation(state.state);
  assert.equal(channelGrid(observation, 1).reduce((a, b) => a + b, 0), 0);
  assert.equal(channelGrid(observation, 5).reduce((a, b) => a + b, 0), 0);
});

test("environment and policy demo streams do not share state", () => {
  const first = makeDemoStreams(123);
  const second = makeDemoStreams(123);
  for (let i = 0; i < 40; i += 1) first.policy.next();
  assert.equal(first.environment.next(), second.environment.next());
  const third = makeDemoStreams(456);
  const fourth = makeDemoStreams(456);
  for (let i = 0; i < 40; i += 1) third.environment.next();
  assert.equal(third.policy.next(), fourth.policy.next());
});

test("softmax is stable and categorical selection respects cumulative probability", () => {
  const probabilities = stableSoftmax([1200, 1199, -1200]);
  assert.ok(probabilities.every(Number.isFinite));
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.equal(sampleAction([0.2, 0.3, 0.5], 0.19), 0);
  assert.equal(sampleAction([0.2, 0.3, 0.5], 0.2), 1);
  assert.equal(sampleAction([0.2, 0.3, 0.5], 0.99), 2);
});

test("network forward pass uses two tanh layers and six output logits", () => {
  const zeros = (length) => new Float32Array(length);
  const weights = {
    W1: { shape: [600, 32], values: zeros(600 * 32) }, b1: { shape: [32], values: zeros(32) },
    W2: { shape: [32, 32], values: zeros(32 * 32) }, b2: { shape: [32], values: zeros(32) },
    W3: { shape: [32, 6], values: zeros(32 * 6) }, b3: { shape: [6], values: zeros(6) },
  };
  const result = forward(weights, new Float32Array(600));
  assert.equal(result.h1.length, 32);
  assert.equal(result.h2.length, 32);
  assert.deepEqual(Array.from(result.logits), [0, 0, 0, 0, 0, 0]);
  assert.deepEqual(result.probabilities, new Array(6).fill(1 / 6));
});

test("iterative-development views render three source-backed versions separately from the primary model set", () => {
  const group = JSON.parse(readFileSync(new URL("../docs/data/iterative_development.json", import.meta.url), "utf8"));
  const primaryModels = JSON.parse(readFileSync(new URL("../docs/data/models.json", import.meta.url), "utf8"));
  assert.equal(group.separate_from_primary_zero_shot_ranking, true);
  assert.deepEqual(group.models.map(model => model.id), ["argon_high_original", "argon_high_followup_1", "argon_high_followup_2"]);
  assert.equal(primaryModels.primary_zero_shot_order.length, 5);
  assert.equal(primaryModels.models.some(model => model.id.startsWith("argon_high_followup")), false);

  const profiles = renderIterativeProfiles(group);
  const means = renderIterativeMeanChart(group);
  const histograms = renderIterativeHistograms(group);
  const results = renderIterativeResultRows(group);
  const comparisons = renderIterativeComparisonRows(group);
  assert.match(profiles, /59\.998/);
  assert.match(profiles, /62\.368/);
  assert.match(profiles, /62\.188/);
  assert.match(profiles, /argon_high_followup_2/);
  assert.match(means, /59\.998/);
  assert.match(means, /62\.368/);
  assert.match(means, /62\.188/);
  assert.match(histograms, /Score distribution/);
  assert.equal((histograms.match(/iterative-histogram-bin/g) ?? []).length, 60);
  assert.match(results, /3,000 \(100%\)/);
  assert.match(results, /62\.368/);
  assert.match(comparisons, /-2\.3693/);
  assert.match(comparisons, /-2\.1897/);
  assert.match(comparisons, /0\.1797/);
  assert.match(comparisons, /7\.25e-1/);
});
