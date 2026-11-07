import { ACTION_NAMES, CHANNELS, advanceState, channelGrid, encodeObservation, makeDemoStreams, resetEpisode } from "./minatar.js";
import { decodeModelWeights, forward, sampleAction } from "./network.js";
import { renderIterativeComparisonRows, renderIterativeDownloads, renderIterativeHistograms, renderIterativeMeanChart, renderIterativeProfiles, renderIterativeResultRows } from "./iterative.js";
import { CAPTURE_HEIGHT, CAPTURE_WIDTH, createArenaRecorder } from "./media-capture.js";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const nf = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });
const fmt = (value, places = 3) => Number(value).toLocaleString("en-US", { maximumFractionDigits: places, minimumFractionDigits: 0 });
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
const categoryLabel = value => ({
  primary_zero_shot: "Zero-shot generation",
  iterative_development: "Iterative-development version",
  reference: "Reference controller",
  trained_baseline: "Trained baseline",
  random_control: "Random control",
  zero_weight_control: "Zero-weight control",
})[value] ?? value;
const ciText = ci => `[${fmt(ci[0])}, ${fmt(ci[1])}]`;

let research;
let modelData;
let iterativeData;
let conditionsById;
let modelsById;
let weightCache = new Map();
let arena = null;

async function loadData() {
  const [researchResponse, modelResponse, iterativeResponse] = await Promise.all([
    fetch("data/research.json"),
    fetch("data/models.json"),
    fetch("data/iterative_development.json"),
  ]);
  if (!researchResponse.ok || !modelResponse.ok || !iterativeResponse.ok) throw new Error("Could not load the published research data.");
  [research, modelData, iterativeData] = await Promise.all([researchResponse.json(), modelResponse.json(), iterativeResponse.json()]);
  conditionsById = new Map(research.conditions.map(item => [item.id, item]));
  modelsById = new Map(modelData.models.map(item => [item.id, item]));
  for (const model of iterativeData.models) {
    modelsById.set(model.id, {
      id: model.id,
      label: model.name,
      condition_class: "iterative_development",
      metrics: model.metrics,
      weights: {
        source_sha256: model.weights.sha256,
        total_parameters: model.weights.total_parameters,
        nonzero_parameters: model.weights.nonzero_parameters,
        sparsity_percent: model.weights.sparsity_percent,
        tensors: model.weights.tensors,
      },
    });
  }
  renderOverview();
  renderModels();
  renderResults();
  renderIterativeDevelopment();
  initializeArena();
  $(`#data-status`).textContent = `Loaded ${research.conditions.length} corrected conditions and ${iterativeData.models.length} separate iterative versions · ${research.dataset_label} · source files hash-checked by the exporter`;
  $(`#data-status`).dataset.state = "ready";
}

function routeFromHash() {
  const hash = decodeURIComponent(location.hash.slice(1) || "overview");
  const anchorRoute = hash.startsWith("method-");
  const page = anchorRoute ? "methodology" : (["overview", "arena", "models", "results", "methodology"].includes(hash) ? hash : "overview");
  for (const section of $$(".page")) section.hidden = section.dataset.page !== page;
  for (const link of $$(".primary-nav a")) {
    if (link.dataset.route === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("#primary-nav").classList.remove("is-open");
  $("#menu-toggle").setAttribute("aria-expanded", "false");
  if (anchorRoute) requestAnimationFrame(() => document.getElementById(hash)?.scrollIntoView());
  else window.scrollTo({ top: 0, behavior: "instant" });
}

function setupNavigation() {
  window.addEventListener("hashchange", routeFromHash);
  $("#menu-toggle").addEventListener("click", () => {
    const nav = $("#primary-nav");
    const open = nav.classList.toggle("is-open");
    $("#menu-toggle").setAttribute("aria-expanded", String(open));
  });
  routeFromHash();
}

function getPrimary() {
  return research.conditions.filter(item => item.is_primary_zero_shot).sort((a, b) => b.metrics.mean_score - a.metrics.mean_score);
}

function renderOverview() {
  const stats = [
    ["5", "original zero-shot model conditions"],
    [nf.format(modelData.architecture.total_parameters), "parameters in each network"],
    ["3,000", "episodes per corrected condition"],
    ["600 → 32 → 32 → 6", "network architecture"],
  ];
  $("#overview-stats").innerHTML = stats.map(([value, label]) => `<div class="stat-card"><strong class="stat-value">${escapeHtml(value)}</strong><span class="stat-label">${escapeHtml(label)}</span></div>`).join("");
  const selected = [
    ...getPrimary(),
    ...research.conditions.filter(item => ["reference", "trained_baseline", "zero_weight_control", "random_control"].includes(item.condition_class)).sort((a, b) => ["reference", "trained_baseline", "zero_weight_control", "random_control"].indexOf(a.condition_class) - ["reference", "trained_baseline", "zero_weight_control", "random_control"].indexOf(b.condition_class)),
  ];
  $("#overview-ranking").innerHTML = selected.map(item => `<div class="ranking-row"><span>${escapeHtml(item.label)}<small class="class-label">${escapeHtml(categoryLabel(item.condition_class))}</small></span><strong>${fmt(item.metrics.mean_score)}</strong></div>`).join("");
}

function nonzeroActionDescription(model) {
  const counts = model.metrics?.selected_action_counts ?? {};
  const n = Object.entries(counts).filter(([, count]) => Number(count) > 0).map(([action]) => ACTION_NAMES[Number(action)]);
  if (model.id === "deepseek_original") return "The recorded evaluation mostly selected Up, Down, and Fire. Those actions do not move the paddle in Breakout.";
  if (model.condition_class === "reference") return "The reference policy selected no-op, left, and right during the recorded evaluation.";
  if (model.condition_class === "trained_baseline") return "The trained baseline used all six action labels in the recorded evaluation.";
  if (!n.length) return "Action distribution is not available for this condition.";
  return `The recorded evaluation selected ${n.join(", ")}. This summarizes actions; it does not prove why the policy performed as it did.`;
}

function recordedFacts(model) {
  const recorded = model.generation?.recorded ?? {};
  const identifier = recorded.model_identifier ?? recorded.model_version ?? "Not recorded in the inspected metadata";
  const provider = recorded.provider ? `${recorded.provider} (recorded; not independently authenticated)` : "Not recorded locally; provider not independently authenticated";
  const shot = recorded.one_shot === true ? "Recorded as one-shot" : recorded.one_shot === false ? "Recorded as multi-turn" : "Not recorded";
  const tools = recorded.tools_enabled === false ? "Recorded as tools disabled" : recorded.tools_enabled === true ? "Recorded as tools enabled" : "Not recorded";
  const effort = recorded.reasoning_effort ?? "Not recorded";
  return { identifier, provider, shot, tools, effort };
}

function profileCard(model, primary) {
  const metrics = model.metrics ?? {};
  const weights = model.weights;
  const facts = recordedFacts(model);
  const nonzero = weights ? `${nf.format(weights.nonzero_parameters)} / ${nf.format(weights.total_parameters)}` : "No network archive";
  const sparsity = weights ? `${fmt(weights.sparsity_percent, 2)}%` : "n/a";
  const mean = metrics.mean_score === undefined ? "—" : fmt(metrics.mean_score);
  const median = metrics.median_score === undefined ? "—" : fmt(metrics.median_score);
  const max = metrics.maximum_score === undefined ? "—" : fmt(metrics.maximum_score);
  const layers = weights?.tensor_stats ? Object.entries(weights.tensor_stats).map(([name, stat]) => `<tr><th scope="row"><code>${name}</code></th><td>${weights.tensors[name].shape.join(" × ")}</td><td>${nf.format(stat.nonzero_parameters)}</td><td>${fmt(stat.minimum, 5)} … ${fmt(stat.maximum, 5)}</td><td>${fmt(stat.mean_absolute_value, 5)}</td></tr>`).join("") : "";
  const weightSource = weights?.source_sha256 ? `<span class="hash-text">${weights.source_sha256}</span>` : "No canonical parameter archive";
  const weightDownload = weights?.download_file ? `<p><a class="button button-secondary" href="data/${escapeHtml(weights.download_file)}" download>Download original NPZ archive</a></p>` : "";
  const response = model.generation?.response_file ? `<details class="response-details" data-response-url="data/${escapeHtml(model.generation.response_file)}"><summary>View original captured response</summary><p class="small-note"><a href="data/${escapeHtml(model.generation.response_file)}" download>Download raw response</a></p><pre class="response-text">Response text loads when opened.</pre></details>` : "";
  return `<details class="panel model-card" data-model-id="${escapeHtml(model.id)}"><summary>
    <span class="profile-title"><strong>${escapeHtml(model.label)}</strong><small><span class="badge ${primary ? "badge-blue" : "badge-neutral"}">${escapeHtml(categoryLabel(model.condition_class))}</span></small></span>
    <span class="profile-metric"><span>Mean</span><strong>${mean}</strong></span>
    <span class="profile-metric"><span>Median</span><strong>${median}</strong></span>
    <span class="profile-metric"><span>Maximum</span><strong>${max}</strong></span>
    <span class="profile-metric"><span>Sparsity</span><strong>${sparsity}</strong></span>
    <span class="profile-chevron" aria-hidden="true">⌄</span>
  </summary><div class="profile-body">
    <div><h3>Recorded generation condition</h3><dl class="profile-definition"><dt>Model identifier</dt><dd>${escapeHtml(facts.identifier)}</dd><dt>Provider label</dt><dd>${escapeHtml(facts.provider)}</dd><dt>One-shot status</dt><dd>${escapeHtml(facts.shot)}</dd><dt>Tool setting</dt><dd>${escapeHtml(facts.tools)}</dd><dt>Reasoning effort</dt><dd>${escapeHtml(facts.effort)}</dd><dt>Provider evidence</dt><dd>Not independently authenticated</dd></dl>
      <h3 style="margin-top:18px">Observed behavior</h3><p>${escapeHtml(nonzeroActionDescription(model))}</p>
      <p class="small-note">The profile uses one archived generation for this recorded condition. The model/provider labels are project metadata, not provider verification.</p>
    </div>
    <div><h3>Network and evaluation</h3><dl class="profile-definition"><dt>Architecture</dt><dd>600 → 32 → 32 → 6; tanh hidden layers</dd><dt>Parameters</dt><dd>${weights ? nf.format(weights.total_parameters) : "—"}</dd><dt>Nonzero parameters</dt><dd>${nonzero}</dd><dt>Sparsity</dt><dd>${sparsity}</dd><dt>Corrected episodes</dt><dd>${metrics.episodes ? nf.format(metrics.episodes) : "—"}</dd><dt>Mean ± SE</dt><dd>${metrics.mean_score === undefined ? "—" : `${fmt(metrics.mean_score)} ± ${fmt(metrics.standard_error_score)}`}</dd><dt>Median · range</dt><dd>${metrics.median_score === undefined ? "—" : `${fmt(metrics.median_score)} · ${fmt(metrics.minimum_score)}–${fmt(metrics.maximum_score)}`}</dd><dt>Weight SHA-256</dt><dd>${weightSource}</dd></dl>
      ${weightDownload}${weights?.tensor_stats ? `<details class="response-details"><summary>Inspect parameter tensors</summary><div class="table-wrap"><table><thead><tr><th>Tensor</th><th>Shape</th><th>Nonzero</th><th>Min … max</th><th>Mean |weight|</th></tr></thead><tbody>${layers}</tbody></table></div></details>` : ""}${response}
    </div>
  </div></details>`;
}

function renderModels() {
  const byId = new Map(modelData.models.map(model => [model.id, model]));
  const primary = getPrimary().map(condition => byId.get(condition.id)).filter(Boolean);
  $("#model-profiles").innerHTML = primary.map(model => profileCard(model, true)).join("");
  const comparison = modelData.models.filter(model => ["reference", "trained_baseline", "zero_weight_control", "random_control"].includes(model.condition_class));
  $("#comparison-profiles").innerHTML = comparison.map(model => profileCard(model, false)).join("");
  const intervention = modelData.separate_intervention;
  const parameterMetrics = intervention.parameter_metrics ?? {};
  const nonzeroSummary = parameterMetrics.base_nonzero_parameters === undefined ? "not recorded" : `${nf.format(parameterMetrics.base_nonzero_parameters)} → ${nf.format(parameterMetrics.corrected_nonzero_parameters)} nonzero values`;
  $("#intervention-summary").innerHTML = `<span><strong>Classification:</strong> ${escapeHtml(intervention.classification ?? "researcher intervention")}</span><span><strong>Recorded nonzero count:</strong> ${escapeHtml(nonzeroSummary)}</span><span><strong>Primary ranking:</strong> excluded</span><span><strong>Weight SHA-256:</strong> <code>${escapeHtml(intervention.weight_sha256 ?? "not recorded")}</code></span>`;
  for (const details of $$("details[data-response-url]")) {
    details.addEventListener("toggle", async () => {
      if (!details.open || details.dataset.loaded) return;
      const pre = $("pre", details);
      try {
        const response = await fetch(details.dataset.responseUrl);
        if (!response.ok) throw new Error("Response file is not available");
        pre.textContent = await response.text();
      } catch (error) { pre.textContent = `Could not load this captured response: ${error.message}`; }
      details.dataset.loaded = "true";
    });
  }
}

function renderIterativeDevelopment() {
  const group = iterativeData;
  $("#iterative-model-profiles").innerHTML = renderIterativeProfiles(group);
  $("#iterative-mean-chart").innerHTML = renderIterativeMeanChart(group);
  $("#iterative-score-distribution").innerHTML = renderIterativeHistograms(group);
  $("#iterative-results-table tbody").innerHTML = renderIterativeResultRows(group);
  $("#iterative-comparison-table tbody").innerHTML = renderIterativeComparisonRows(group);
  const byId = new Map(group.paired_comparisons.map(item => [`${item.condition_a_id}:${item.condition_b_id}`, item]));
  const originalFollowup1 = byId.get("argon_high_original:argon_high_followup_1");
  const originalFollowup2 = byId.get("argon_high_original:argon_high_followup_2");
  const followupComparison = byId.get("argon_high_followup_1:argon_high_followup_2");
  $("#iterative-comparison-note").textContent = `The two follow-up mean scores were ${nf.format(Math.abs(originalFollowup1.mean_paired_difference))} and ${nf.format(Math.abs(originalFollowup2.mean_paired_difference))} points higher than the supplied original on matched episodes. Their paired Cohen's dz values are small in magnitude (${nf.format(Math.abs(originalFollowup1.effect_size_cohen_dz))} and ${nf.format(Math.abs(originalFollowup2.effect_size_cohen_dz))}). Follow-up 1 minus Follow-up 2 was ${nf.format(followupComparison.mean_paired_difference)} points (Holm-adjusted p=${Number(followupComparison.p_value_holm_adjusted_all_pairwise_conditions).toFixed(5)}); this nonsignificant result does not establish equivalence.`;
  $("#iterative-download-links").innerHTML = renderIterativeDownloads(group);
  const registry = group.artifact_registry;
  $("#iterative-data-note").textContent = `The three versions are a separate iterative-development experiment, not part of the five-condition zero-shot ranking. ${nf.format(registry.registered_file_count)} source artifacts were checked against the experiment registry; source bundle SHA-256: ${registry.sha256}.`;
}

function renderResults() {
  const conditions = research.conditions;
  const tbody = $("#results-table tbody");
  tbody.innerHTML = conditions.map(item => {
    const m = item.metrics;
    const ci = ciText(m.score_mean_ci_95);
    return `<tr><td class="condition-cell"><strong>${escapeHtml(item.label)}</strong><small>${escapeHtml(item.id)}</small></td><td><span class="class-chip ${escapeHtml(item.condition_class)}">${escapeHtml(categoryLabel(item.condition_class))}</span></td><td>${nf.format(m.episodes)}</td><td>${fmt(m.mean_score)} ± ${fmt(m.standard_error_score)}</td><td>${escapeHtml(ci)}</td><td>${fmt(m.median_score)}</td><td>${fmt(m.minimum_score)}–${fmt(m.maximum_score)}</td><td>${fmt(m.mean_episode_length, 2)}</td><td>${fmt(m.median_episode_length, 2)}</td><td>${nf.format(m.cutoff_count)} (${fmt(m.cutoff_percentage, 2)}%)</td><td><a href="data/${escapeHtml(item.csv_download)}" download>CSV</a></td></tr>`;
  }).join("");
  const distSelect = $("#distribution-model");
  distSelect.innerHTML = conditions.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join("");
  distSelect.addEventListener("change", drawDistribution);
  drawMeanChart();
  drawDistribution();
  renderActions();
  renderComparisons();
  renderHistorical();
  renderDownloads();
  $("#comparison-search").addEventListener("input", renderComparisons);
}

function drawMeanChart() {
  const maxMean = Math.max(...research.conditions.map(item => item.metrics.mean_score));
  $("#mean-chart").innerHTML = research.conditions.map(item => {
    const percent = Math.max(0.5, 100 * item.metrics.mean_score / maxMean);
    return `<div class="mean-row" data-class="${escapeHtml(item.condition_class)}"><span class="mean-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</span><span class="mean-track"><span class="mean-bar" style="display:block;width:${percent}%"></span></span><span class="mean-value">${fmt(item.metrics.mean_score)}</span></div>`;
  }).join("");
}

function drawDistribution() {
  const condition = conditionsById.get($("#distribution-model").value) ?? research.conditions[0];
  const canvas = $("#distribution-chart");
  const context = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const height = 300;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const values = condition.metrics.episode_records.map(row => Number(row[0]));
  const max = Math.max(1, condition.metrics.maximum_score);
  const bins = 20;
  const counts = new Array(bins).fill(0);
  for (const score of values) counts[Math.min(bins - 1, Math.floor(score / (max + Number.EPSILON) * bins))] += 1;
  const pad = { left: 43, right: 13, top: 13, bottom: 33 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const maxCount = Math.max(1, ...counts);
  context.strokeStyle = "#cbd5e1";
  context.lineWidth = 1;
  context.beginPath(); context.moveTo(pad.left, pad.top); context.lineTo(pad.left, pad.top + chartH); context.lineTo(pad.left + chartW, pad.top + chartH); context.stroke();
  const barGap = 3;
  const barW = chartW / bins - barGap;
  context.fillStyle = "#347bce";
  counts.forEach((count, index) => {
    const barH = count / maxCount * chartH;
    context.fillRect(pad.left + index * chartW / bins + barGap / 2, pad.top + chartH - barH, Math.max(1, barW), barH);
  });
  context.fillStyle = "#58677e";
  context.font = "12px system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText("Score per episode", pad.left + chartW / 2, height - 4);
  context.textAlign = "left";
  context.fillText("0", pad.left, height - 17);
  context.textAlign = "right";
  context.fillText(fmt(max, 0), pad.left + chartW, height - 17);
  context.save(); context.translate(13, pad.top + chartH / 2); context.rotate(-Math.PI / 2); context.textAlign = "center"; context.fillText("Episode count", 0, 0); context.restore();
  canvas.setAttribute("aria-label", `Score histogram for ${condition.label}, based on ${condition.metrics.episode_records.length} episode scores. Minimum ${condition.metrics.minimum_score}, maximum ${condition.metrics.maximum_score}.`);
  $("#distribution-caption").textContent = `${condition.label}: n=${nf.format(values.length)}, median ${fmt(condition.metrics.median_score)}, range ${fmt(condition.metrics.minimum_score)}–${fmt(condition.metrics.maximum_score)}. Bins span this condition’s observed score range.`;
}

function renderActions() {
  const head = ACTION_NAMES.map(action => `<th scope="col">${escapeHtml(action)}</th>`).join("");
  const body = research.conditions.map(item => `<tr><th scope="row">${escapeHtml(item.label)}</th>${ACTION_NAMES.map((_, action) => `<td>${nf.format(item.metrics.selected_action_counts[String(action)] ?? 0)}</td>`).join("")}</tr>`).join("");
  $("#action-table").innerHTML = `<table class="action-table"><thead><tr><th scope="col">Condition</th>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderComparisons() {
  const filter = $("#comparison-search").value.trim().toLowerCase();
  const rows = research.pairwise_analysis.filter(item => `${item.condition_a_label} ${item.condition_b_label}`.toLowerCase().includes(filter));
  $("#comparison-table tbody").innerHTML = rows.map(item => {
    const ci = ciText(item.paired_difference_ci_95_percentile_bootstrap);
    const rawP = item.paired_sign_flip_p_value;
    const holm = item.p_value_holm_adjusted_all_pairwise_conditions;
    const nonsig = Number(holm) >= 0.05 ? "comparison-row-ns" : "";
    return `<tr class="${nonsig}"><td>${escapeHtml(item.condition_a_label)} − ${escapeHtml(item.condition_b_label)}</td><td>${fmt(item.mean_paired_difference)}</td><td>${escapeHtml(ci)}</td><td>${fmt(item.effect_size_cohen_dz, 4)}</td><td>${Number(rawP).toExponential(2)}</td><td>${Number(holm).toExponential(2)}</td></tr>`;
  }).join("");
}

function renderHistorical() {
  const idLabel = new Map(research.conditions.map(item => [item.id, item.label]));
  const items = Object.entries(research.corrected_vs_original).filter(([, item]) => typeof item === "object" && item.mean_paired_score_change_corrected_minus_original !== undefined);
  $("#historical-table tbody").innerHTML = items.map(([id, item]) => `<tr><th scope="row">${escapeHtml(idLabel.get(id) ?? id)}</th><td>${nf.format(item.matched_episode_count)}</td><td>${fmt(item.mean_paired_score_change_corrected_minus_original)}</td><td>${escapeHtml(ciText(item.paired_bootstrap_ci_95_for_change))}</td></tr>`).join("");
}

function renderDownloads() {
  $("#download-links").innerHTML = research.conditions.map(item => `<a class="download-link" href="data/${escapeHtml(item.csv_download)}" download>${escapeHtml(item.label)} CSV</a>`).join("");
}

function initModelSelect() {
  const select = $("#arena-model");
  const primary = getPrimary();
  const iterative = iterativeData.models.map(model => ({ id: model.id, label: model.name }));
  const reference = research.conditions.filter(item => item.condition_class === "reference");
  const trained = research.conditions.filter(item => item.condition_class === "trained_baseline");
  const controls = research.conditions.filter(item => ["zero_weight_control", "random_control"].includes(item.condition_class));
  const options = (items, label) => `<optgroup label="${label}">${items.map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.label)}</option>`).join("")}</optgroup>`;
  select.innerHTML = options(primary, "Original zero-shot generations") + options(iterative, "Argon High iterative-development stages") + options(reference, "Reference") + options(trained, "Trained baseline") + options(controls, "Controls");
  select.value = "gemini_argon_4_medium";
}

function getPolicyModel(id) {
  return modelsById.get(id);
}

function policyOutput(model, observation, rng, drawAction = true) {
  if (model.id === "random_action_control") {
    const probabilities = new Array(6).fill(1 / 6);
    return { action: drawAction ? Math.floor(rng.next() * 6) : null, probabilities, logits: null, h1: null, h2: null, random: true };
  }
  if (!model.weights) throw new Error(`Weights are unavailable for ${model.label}`);
  if (!weightCache.has(model.id)) weightCache.set(model.id, decodeModelWeights(model));
  const result = forward(weightCache.get(model.id), observation);
  return { ...result, action: drawAction ? sampleAction(result.probabilities, rng.next()) : null, random: false };
}

function initializeArena() {
  initModelSelect();
  const seedInput = $("#arena-seed");
  const modelSelect = $("#arena-model");
  const canvas = $("#game-canvas");
  arena = { running: false, timer: null, episode: null, model: null, decision: null, lastTransition: null, createdAt: null };
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = CAPTURE_WIDTH;
  captureCanvas.height = CAPTURE_HEIGHT;
  const captureStatus = $("#arena-capture-status");
  const captureRecorder = createArenaRecorder({
    canvas: captureCanvas,
    gameCanvas: canvas,
    getFrame: () => {
      const state = arena.episode.state;
      const observation = encodeObservation(state);
      // Forward the current board without drawing a categorical sample or consuming policy RNG.
      const currentInference = policyOutput(arena.model, observation, arena.streams.policy, false);
      const activeByChannel = CHANNELS.map((_, channel) => {
        let sum = 0;
        for (let i = 0; i < 100; i += 1) sum += observation[i * 6 + channel];
        return sum;
      });
      return {
        model: arena.model.label,
        category: categoryLabel(arena.model.condition_class),
        score: state.score,
        step: state.step,
        seed: arena.episode.seed,
        terminal: state.terminal,
        running: arena.running,
        lastAction: arena.lastTransition ? ACTION_NAMES[arena.lastTransition.selectedAction] : null,
        highlightAction: null,
        actionNames: ACTION_NAMES,
        probabilities: Array.from(currentInference.probabilities),
        logits: currentInference.logits ? Array.from(currentInference.logits) : null,
        h1Active: currentInference.h1 ? currentInference.h1.filter(value => Math.abs(value) > 0.5).length : 0,
        h2Active: currentInference.h2 ? currentInference.h2.filter(value => Math.abs(value) > 0.5).length : 0,
        randomPolicy: currentInference.random,
        activeByChannel,
        channelNames: CHANNELS,
        observationGrid: Array.from(channelGrid(observation, Number($("#observation-channel").value))),
        selectedChannel: CHANNELS[Number($("#observation-channel").value)],
        selectedChannelIndex: Number($("#observation-channel").value),
        parameters: arena.model.weights?.total_parameters ?? 0,
      };
    },
    onStatus: (message, recording) => {
      captureStatus.textContent = message;
      captureStatus.dataset.state = recording ? "recording" : "ready";
    },
  });
  const recordButton = $("#arena-record");
  const setCaptureControls = recording => {
    recordButton.textContent = recording ? "■ Stop & download recording" : "● Record gameplay";
    recordButton.setAttribute("aria-label", recording ? "Stop and download the Arena recording" : "Record the Arena gameplay at its current speed");
    modelSelect.disabled = recording;
    seedInput.disabled = recording;
    $("#arena-speed").disabled = recording;
    $("#arena-reset").disabled = recording;
    $("#arena-step-button").disabled = recording;
    $("#arena-random-seed").disabled = recording;
  };
  const reset = () => {
    stopArena();
    const seed = Math.max(0, Math.min(2147483647, Math.floor(Number(seedInput.value) || 0)));
    seedInput.value = String(seed);
    arena.model = getPolicyModel(modelSelect.value);
    arena.episode = resetEpisode(seed);
    arena.streams = arena.episode.streams;
    arena.decision = policyOutput(arena.model, encodeObservation(arena.episode.state), arena.streams.policy, false);
    arena.lastTransition = null;
    arena.createdAt = new Date().toISOString();
    drawArena();
  };
  const oneStep = () => {
    if (arena.episode.state.terminal) return;
    const observation = encodeObservation(arena.episode.state);
    const decision = policyOutput(arena.model, observation, arena.streams.policy, true);
    const transition = advanceState(arena.episode.state, decision.action, arena.streams.environment.next());
    arena.episode.state = transition.state;
    arena.decision = { ...decision, action: decision.action };
    arena.lastTransition = transition.info;
    drawArena();
    if (transition.done) stopArena();
  };
  const start = () => {
    if (arena.episode.state.terminal || arena.running) return;
    arena.running = true;
    updateRunControls();
    const loop = () => {
      if (!arena.running) return;
      const speed = Number($("#arena-speed").value) || 1;
      for (let i = 0; i < speed && !arena.episode.state.terminal; i += 1) oneStep();
      if (arena.running && !arena.episode.state.terminal) arena.timer = window.setTimeout(loop, 90);
    };
    arena.timer = window.setTimeout(loop, 0);
  };
  $("#arena-start").addEventListener("click", start);
  $("#arena-pause").addEventListener("click", stopArena);
  $("#arena-reset").addEventListener("click", reset);
  $("#arena-step-button").addEventListener("click", () => { if (arena.episode.state.terminal) reset(); oneStep(); });
  $("#arena-random-seed").addEventListener("click", () => { seedInput.value = String(Math.floor(Math.random() * 2147483648)); reset(); });
  recordButton.addEventListener("click", () => {
    if (captureRecorder.isRecording()) {
      stopArena();
      setCaptureControls(false);
      captureRecorder.stop();
      return;
    }
    // Start every capture from the selected seed at the Arena's normal 1× speed.
    $("#arena-speed").value = "1";
    reset();
    if (captureRecorder.start()) {
      setCaptureControls(true);
      start();
    }
  });
  $("#arena-capture-frame").addEventListener("click", () => captureRecorder.snapshot());
  seedInput.addEventListener("change", reset);
  modelSelect.addEventListener("change", reset);
  $("#observation-channel").addEventListener("change", drawObservation);
  window.addEventListener("keydown", event => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (event.code === "Space" && !$("[data-page='arena']").hidden) {
      event.preventDefault();
      arena.running ? stopArena() : start();
    } else if (event.key.toLowerCase() === "r" && !$("[data-page='arena']").hidden) reset();
    else if (event.key === "Enter" && !$("[data-page='arena']").hidden) { event.preventDefault(); if (arena.episode.state.terminal) reset(); oneStep(); }
  });
  window.addEventListener("resize", () => drawArena());
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser does not support a 2D canvas context.");
  reset();
}

function stopArena() {
  if (!arena) return;
  arena.running = false;
  if (arena.timer !== null) window.clearTimeout(arena.timer);
  arena.timer = null;
  updateRunControls();
  if (arena.episode) drawEpisodeSummary();
}

function updateRunControls() {
  $("#arena-start").disabled = arena.running || Boolean(arena.episode?.state.terminal);
  $("#arena-pause").disabled = !arena.running;
  const pill = $("#arena-state");
  const terminal = Boolean(arena.episode?.state.terminal);
  pill.textContent = arena.running ? "Playing" : terminal ? (arena.episode.state.terminalReason === "step_cutoff" ? "Time limit" : "Episode over") : "Paused";
  pill.dataset.state = arena.running ? "playing" : terminal ? "over" : "ready";
}

function drawArena() {
  if (!arena?.episode) return;
  const state = arena.episode.state;
  const canvas = $("#game-canvas");
  const context = canvas.getContext("2d");
  const cell = canvas.width / 10;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#080b10";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < 10; row += 1) {
    for (let col = 0; col < 10; col += 1) {
      context.strokeStyle = "rgba(145,164,188,.15)";
      context.strokeRect(col * cell + .5, row * cell + .5, cell, cell);
      if (state.bricks[row * 10 + col]) {
        context.fillStyle = ["#e65047", "#ec9b3a", "#45a36f"][(row - 1) % 3];
        context.fillRect(col * cell + 3, row * cell + 3, cell - 6, cell - 6);
      }
    }
  }
  context.fillStyle = "#ffbd69";
  context.beginPath(); context.arc((state.lastX + .5) * cell, (state.lastY + .5) * cell, cell * .075, 0, Math.PI * 2); context.fill();
  context.fillStyle = "#ffffff";
  context.beginPath(); context.arc((state.ballX + .5) * cell, (state.ballY + .5) * cell, cell * .13, 0, Math.PI * 2); context.fill();
  context.fillStyle = "#e8f2ff";
  context.fillRect(state.paddle * cell + 5, 9 * cell + 13, cell - 10, cell - 26);
  $("#arena-score").textContent = nf.format(state.score);
  $("#arena-step").textContent = nf.format(state.step);
  $("#game-canvas").setAttribute("aria-label", `Breakout board: score ${state.score}, step ${state.step}, paddle column ${state.paddle}, ball row ${state.ballY} column ${state.ballX}, ${state.bricks.reduce((sum, value) => sum + value, 0)} bricks remaining.`);
  updateRunControls();
  drawObservation();
  drawInspector();
  drawModelSummary();
  drawEpisodeSummary();
}

function drawObservation() {
  if (!arena?.episode) return;
  const state = arena.episode.state;
  const observation = encodeObservation(state);
  const channel = Number($("#observation-channel").value);
  const grid = channelGrid(observation, channel);
  $("#observation-grid").innerHTML = Array.from(grid, (value, index) => {
    const classes = ["observation-cell", value ? "on" : "", channel === 2 ? "brick" : "", channel === 3 ? "ball" : "", channel === 4 ? "trail" : "", channel === 0 ? "paddle" : ""].filter(Boolean).join(" ");
    return `<span class="${classes}" aria-hidden="true" title="row ${Math.floor(index / 10)}, column ${index % 10}: ${value}"></span>`;
  }).join("");
  $("#observation-grid").setAttribute("aria-label", `${CHANNELS[channel]} channel, 10 by 10. ${grid.reduce((sum, value) => sum + value, 0)} active cells.`);
  $("#observation-description").textContent = `${CHANNELS[channel]} channel · ${grid.reduce((sum, value) => sum + value, 0)} active cells. Input index is (row × 10 + column) × 6 + channel; the full network input contains all six channels.`;
}

function drawInspector() {
  const decision = arena.decision;
  const target = $("#decision-inspector");
  if (!decision) { target.innerHTML = `<p class="muted">Choose a policy with an available network to inspect it.</p>`; return; }
  const action = decision.action === null ? "waiting for next sampled action" : `selected ${ACTION_NAMES[decision.action]}`;
  const probabilities = decision.probabilities;
  const maxProb = Math.max(...probabilities, 1e-9);
  const bars = probabilities.map((prob, index) => `<div class="logit-row"><span>${escapeHtml(ACTION_NAMES[index])}</span><span class="logit-track"><span class="logit-fill ${decision.action === index ? "selected" : ""}" style="display:block;width:${Math.max(1, 100 * prob / maxProb)}%"></span></span><output>${(100 * prob).toFixed(1)}%</output></div>`).join("");
  const obs = encodeObservation(arena.episode.state);
  const activeByChannel = CHANNELS.map((_, channel) => {
    let sum = 0; for (let i = 0; i < 100; i += 1) sum += obs[i * 6 + channel];
    return sum;
  });
  const hiddenText = decision.random ? "The random-action control has no neural network; it samples all six actions uniformly." : `Layer 1 active (|activation| > 0.5): ${decision.h1.filter(v => Math.abs(v) > .5).length}/32 · Layer 2: ${decision.h2.filter(v => Math.abs(v) > .5).length}/32.`;
  const logits = decision.logits ? `Logits: ${Array.from(decision.logits, value => fmt(value, 4)).join(" · ")}` : "Logits are not used by the random-action control.";
  target.innerHTML = `<div class="layer-summary"><strong>${escapeHtml(action)}</strong><br>Input active cells by channel: ${activeByChannel.map((n, i) => `${CHANNELS[i]} ${n}`).join(" · ")}<br>${escapeHtml(hiddenText)}<br>${escapeHtml(logits)}</div>${bars}`;
}

function drawModelSummary() {
  const model = arena.model;
  const weights = model.weights;
  const m = model.metrics;
  const classBadge = categoryLabel(model.condition_class);
  $("#arena-model-summary").innerHTML = `<span class="badge ${model.condition_class === "primary_zero_shot" ? "badge-blue" : "badge-neutral"}">${escapeHtml(classBadge)}</span><dl><dt>Parameters</dt><dd>${weights ? nf.format(weights.total_parameters) : "No network"}</dd><dt>Nonzero</dt><dd>${weights ? nf.format(weights.nonzero_parameters) : "—"}</dd><dt>Sparsity</dt><dd>${weights ? `${fmt(weights.sparsity_percent, 2)}%` : "—"}</dd><dt>Corrected mean</dt><dd>${m?.mean_score === undefined ? "—" : fmt(m.mean_score)}</dd><dt>Weight SHA-256</dt><dd class="hash-text">${escapeHtml(weights?.source_sha256 ?? "not applicable")}</dd></dl>`;
}

function drawEpisodeSummary() {
  const state = arena.episode.state;
  const info = arena.lastTransition;
  const rows = [
    ["Policy", arena.model.label],
    ["Browser demo seed", arena.episode.seed],
    ["Outcome", state.terminal ? (state.terminalReason === "step_cutoff" ? "10,000-step limit" : "Ball missed") : arena.running ? "In progress" : "Ready / paused"],
    ["Score", state.score],
    ["Episode length", `${state.step} steps`],
    ["Bricks remaining", state.bricks.reduce((sum, value) => sum + value, 0)],
    ["Last selected action", info ? ACTION_NAMES[info.selectedAction] : "No action yet"],
    ["Last executed action", info ? ACTION_NAMES[info.executedAction] : "No action yet"],
  ];
  $("#episode-summary-content").innerHTML = rows.map(([label, value]) => `<div class="summary-item"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`).join("");
}

async function startApp() {
  setupNavigation();
  try {
    await loadData();
    routeFromHash();
  } catch (error) {
    const status = $("#data-status");
    status.textContent = `Research data could not be loaded: ${error.message}. Try serving the docs directory over HTTP.`;
    status.dataset.state = "error";
  }
}

startApp();
