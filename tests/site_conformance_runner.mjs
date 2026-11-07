import { readFileSync } from "node:fs";
import { advanceState, encodeNativeState } from "../docs/assets/minatar.js";
import { forward } from "../docs/assets/network.js";

const input = JSON.parse(readFileSync(0, "utf8"));

if (input.operation === "transitions") {
  const results = input.cases.map(item => {
    const result = advanceState({ ...item.state, bricks: Uint8Array.from(item.state.bricks) }, item.action, item.sticky_draw, item.sticky_probability, item.max_steps);
    return {
      state: { ...result.state, bricks: Array.from(result.state.bricks) },
      reward: result.reward,
      done: result.done,
      info: result.info,
      observation: Array.from((awaitObservation(result.state))),
    };
  });
  process.stdout.write(JSON.stringify(results));
} else if (input.operation === "observations") {
  process.stdout.write(JSON.stringify(input.states.map(native => Array.from(encodeNativeState(native)))));
} else if (input.operation === "networks") {
  const results = input.cases.map(item => {
    const weights = Object.fromEntries(Object.entries(item.weights).map(([name, tensor]) => [name, { shape: tensor.shape, values: Float32Array.from(tensor.values) }]));
    const result = forward(weights, Float32Array.from(item.observation));
    return { h1: Array.from(result.h1), h2: Array.from(result.h2), logits: Array.from(result.logits), probabilities: result.probabilities };
  });
  process.stdout.write(JSON.stringify(results));
} else {
  throw new Error(`Unknown test operation: ${input.operation}`);
}

function awaitObservation(state) {
  const values = new Float32Array(600);
  values[(9 * 10 + state.paddle) * 6] = 1;
  for (let i = 0; i < 100; i += 1) if (state.bricks[i]) values[i * 6 + 2] = 1;
  values[(state.ballY * 10 + state.ballX) * 6 + 3] = 1;
  values[(state.lastY * 10 + state.lastX) * 6 + 4] = 1;
  return values;
}
