export function decodeModelWeights(model) {
  if (!model?.weights?.tensors) throw new Error("This condition has no neural-network weight archive");
  const tensors = {};
  for (const [name, tensor] of Object.entries(model.weights.tensors)) {
    const raw = decodeBase64(tensor.data_base64);
    if (raw.byteLength !== tensor.shape.reduce((a, b) => a * b, 1) * 4) throw new Error(`Invalid byte length for ${name}`);
    const view = new DataView(raw);
    const values = new Float32Array(raw.byteLength / 4);
    for (let i = 0; i < values.length; i += 1) values[i] = view.getFloat32(i * 4, true);
    tensors[name] = { shape: tensor.shape, values };
  }
  return tensors;
}

function decodeBase64(encoded) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function affine(input, weights, bias) {
  const [inSize, outSize] = weights.shape;
  const output = new Float32Array(outSize);
  for (let out = 0; out < outSize; out += 1) {
    let sum = 0;
    for (let i = 0; i < inSize; i += 1) sum += input[i] * weights.values[i * outSize + out];
    output[out] = Math.fround(Math.fround(sum) + bias.values[out]);
  }
  return output;
}

export function stableSoftmax(logits) {
  let max = -Infinity;
  for (const value of logits) if (value > max) max = value;
  const exp = new Float64Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i += 1) {
    exp[i] = Math.exp(logits[i] - max);
    total += exp[i];
  }
  return Array.from(exp, value => value / total);
}

export function forward(weights, observation) {
  if (!weights?.W1 || !observation || observation.length !== 600) throw new Error("Expected six tensors and a 600-value observation");
  const h1 = affine(observation, weights.W1, weights.b1);
  for (let i = 0; i < h1.length; i += 1) h1[i] = Math.fround(Math.tanh(h1[i]));
  const h2 = affine(h1, weights.W2, weights.b2);
  for (let i = 0; i < h2.length; i += 1) h2[i] = Math.fround(Math.tanh(h2[i]));
  const logits = affine(h2, weights.W3, weights.b3);
  return { h1, h2, logits, probabilities: stableSoftmax(logits) };
}

export function sampleAction(probabilities, randomValue) {
  let cumulative = 0;
  const draw = Number(randomValue);
  for (let i = 0; i < probabilities.length; i += 1) {
    cumulative += probabilities[i];
    if (draw < cumulative || i === probabilities.length - 1) return i;
  }
  return probabilities.length - 1;
}
