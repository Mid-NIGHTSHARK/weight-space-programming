const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });
const fmt = (value, places = 3) => Number(value).toLocaleString("en-US", { maximumFractionDigits: places, minimumFractionDigits: 0 });
const meanMax = group => Math.max(1, ...group.models.map(model => model.metrics.mean_score));

export function renderIterativeProfiles(group) {
  return group.models.map(model => {
    const m = model.metrics;
    const isOriginal = model.id === "argon_high_original";
    const stageLabel = isOriginal ? "Supplied original excerpt" : model.id.endsWith("1") ? "Researcher-directed follow-up 1" : "Researcher-directed follow-up 2";
    return `<article class="panel iterative-profile" data-model-id="${escapeHtml(model.id)}">
      <p class="eyebrow">${escapeHtml(stageLabel)}</p>
      <h3>${escapeHtml(model.name)}</h3>
      <dl class="iterative-metrics">
        <div><dt>Mean ± SE</dt><dd>${fmt(m.mean_score)} ± ${fmt(m.standard_error_score)}</dd></div>
        <div><dt>Median</dt><dd>${fmt(m.median_score)}</dd></div>
        <div><dt>Episodes</dt><dd>${numberFormat.format(m.episodes)}</dd></div>
        <div><dt>Range</dt><dd>${fmt(m.minimum_score)}–${fmt(m.maximum_score)}</dd></div>
        <div><dt>Episode length, mean</dt><dd>${fmt(m.mean_episode_length, 2)}</dd></div>
        <div><dt>Time-limit cutoffs</dt><dd>${numberFormat.format(m.cutoff_count)}</dd></div>
      </dl>
      <p class="small-note">Network: ${model.architecture.join(" → ")} · ${numberFormat.format(model.parameter_count)} parameters · ${numberFormat.format(model.nonzero_parameter_count)} nonzero (${fmt(model.sparsity_percent, 2)}% sparse).</p>
      <p class="small-note">Recorded attribution: ${escapeHtml(model.attribution)}. ${escapeHtml(model.generation_provenance_limit)} Provider identity is not independently authenticated.</p>
      <div class="download-list">
        <a class="download-link" href="data/${escapeHtml(model.csv_download)}" download>Episode CSV</a>
        <a class="download-link" href="data/${escapeHtml(model.source.download_file)}" download>Supplied ${isOriginal ? "assignment excerpt" : "follow-up script"}</a>
        <a class="download-link" href="data/${escapeHtml(model.weights.download_file)}" download>Weight archive</a>
      </div>
      <p class="hash-line">Source SHA-256 <code>${escapeHtml(model.source.sha256)}</code><br>Weight SHA-256 <code>${escapeHtml(model.weights.sha256)}</code></p>
    </article>`;
  }).join("");
}

export function renderIterativeMeanChart(group) {
  const max = meanMax(group);
  return group.models.map(model => {
    const m = model.metrics;
    const percent = 100 * m.mean_score / max;
    const errorWidth = Math.min(100 - percent, 100 * m.standard_error_score / max);
    return `<div class="iterative-mean-row">
      <span class="iterative-chart-label">${escapeHtml(model.name)}</span>
      <div class="iterative-score-track" role="meter" aria-label="Mean score for ${escapeHtml(model.name)}" aria-valuemin="0" aria-valuemax="${fmt(max, 3)}" aria-valuenow="${fmt(m.mean_score, 3)}">
        <span class="iterative-score-bar" style="width:${percent}%"></span><span class="iterative-score-error" style="left:${percent}%;width:${errorWidth}%" aria-hidden="true"></span>
      </div><strong>${fmt(m.mean_score)} ± ${fmt(m.standard_error_score)}</strong>
    </div>`;
  }).join("");
}

export function renderIterativeHistograms(group) {
  return group.models.map(model => {
    const maxCount = Math.max(1, ...model.score_histogram.map(bin => bin.count));
    const bins = model.score_histogram.map(bin => {
      const height = Math.max(1, 100 * bin.count / maxCount);
      return `<span class="iterative-histogram-bin" style="height:${height}%" title="${fmt(bin.lower, 1)}–${fmt(bin.upper, 1)} points: ${numberFormat.format(bin.count)} episodes"></span>`;
    }).join("");
    return `<div class="iterative-histogram-row"><strong>${escapeHtml(model.name)}</strong><div class="iterative-histogram" role="img" aria-label="Score distribution for ${escapeHtml(model.name)} across ${numberFormat.format(model.metrics.episodes)} episodes">${bins}</div></div>`;
  }).join("");
}

export function renderIterativeResultRows(group) {
  return group.models.map(model => {
    const m = model.metrics;
    return `<tr><th scope="row">${escapeHtml(model.name)}</th><td>${numberFormat.format(m.episodes)}</td><td>${fmt(m.mean_score)} ± ${fmt(m.standard_error_score)}</td><td>${fmt(m.median_score)}</td><td>${fmt(m.minimum_score)}–${fmt(m.maximum_score)}</td><td>${numberFormat.format(m.ball_lost_count)} (${fmt(100 * m.ball_lost_count / m.episodes, 1)}%)</td><td>${numberFormat.format(m.cutoff_count)}</td><td><a href="data/${escapeHtml(model.csv_download)}" download>CSV</a></td></tr>`;
  }).join("");
}

export function renderIterativeComparisonRows(group) {
  return group.paired_comparisons.map(comparison => {
    const ci = comparison.paired_difference_ci_95_percentile_bootstrap;
    const nonsignificant = Number(comparison.p_value_holm_adjusted_all_pairwise_conditions) >= 0.05;
    return `<tr class="${nonsignificant ? "comparison-row-ns" : ""}"><th scope="row">${escapeHtml(comparison.condition_a_label)} − ${escapeHtml(comparison.condition_b_label)}</th><td>${fmt(comparison.mean_paired_difference, 4)}</td><td>[${fmt(ci[0], 4)}, ${fmt(ci[1], 4)}]</td><td>${fmt(comparison.effect_size_cohen_dz, 4)}</td><td>${Number(comparison.paired_sign_flip_p_value).toExponential(2)}</td><td>${Number(comparison.p_value_holm_adjusted_all_pairwise_conditions).toExponential(2)}</td></tr>`;
  }).join("");
}

export function renderIterativeDownloads(group) {
  return Object.entries(group.downloads).map(([label, item]) => {
    const linkLabel = ({ report: "Detailed experiment report", analysis_report: "Analysis report", analysis_json: "Analysis JSON", episode_seed_schedule: "Matched episode seed schedule" })[label] ?? label;
    return `<a class="download-link" href="data/${escapeHtml(item.download_file)}" download>${escapeHtml(linkLabel)}</a>`;
  }).join("");
}
