# Gemini Argon High — Iterative Development

This is a separate three-stage experiment. It does not alter the Phase 2 primary condition registry, its five-model zero-shot ranking, any historical result, or the archived Gemini run under `gemini/`.

## Contents

- `source/` preserves the user-supplied original assignment reconstruction and both follow-up scripts. The two follow-up scripts are byte-identical to the supplied text files.
- `weights/` contains reconstructed 600-32-32-6 weight archives.
- `evaluation/pilot/` contains the configured 10-episode-per-version software pilot and replay checks.
- `evaluation/full_run/` contains the 9,000 episode records, exact seed schedule, run manifest, raw-data checks, metrics, paired tests, and report.
- `evaluation/representative_trajectories.json` contains six complete MinAtar episode replays, covering each version on both initial ball directions. Each trajectory digest matches its corresponding full-run CSV row.
- `website_model_records.json` is data for later site integration. No website files were edited in this task.
- `report.md` is the full technical and research report.
- `artifact_registry.json` records hashes and byte sizes for the files in this experiment directory (it excludes its own hash).

## Reproduce

Run commands from the repository root in the project virtual environment.

1. Reconstruct the tensors into a new directory. The script refuses to overwrite existing weight or manifest files:

   ```sh
   .venv/bin/python experiments/argon_high_iterative_development/reconstruct_networks.py --output-dir weights_rebuild_20261107
   ```

2. Run the unchanged supplied follow-up-1 assertions:

   ```sh
   .venv/bin/python experiments/argon_high_iterative_development/source/argon_high_followup_1.py
   ```

   Run the second supplied verifier as provided to reproduce its documented failure:

   ```sh
   .venv/bin/python experiments/argon_high_iterative_development/source/argon_high_followup_2.py
   ```

   It stops at its floating-point assertion. The source is preserved unchanged; its output and the independent exhaustive mathematical check are documented in `report.md`.

3. Run a fresh pilot and full benchmark into new output paths. The runner refuses to overwrite an existing run directory:

   ```sh
   .venv/bin/python experiments/argon_high_iterative_development/run_evaluation.py --pilot --output-dir evaluation/pilot_rerun_20261107
   .venv/bin/python experiments/argon_high_iterative_development/run_evaluation.py --output-dir evaluation/full_run_rerun_20261107
   ```

4. Analyze the new full run by passing its path. Analysis refuses to overwrite a pre-existing `analysis/` directory unless `--overwrite-analysis` is explicitly provided. The raw episode CSV hashes are rechecked before analysis writes.

   ```sh
   .venv/bin/python experiments/argon_high_iterative_development/analyze_evaluation.py experiments/argon_high_iterative_development/evaluation/full_run_rerun_20261107
   ```

5. Generate representative replay traces and future site data at new paths:

   ```sh
   .venv/bin/python experiments/argon_high_iterative_development/trace_representative_episodes.py --output evaluation/representative_trajectories_rerun_20261107.json
   .venv/bin/python experiments/argon_high_iterative_development/export_website_data.py --output website_model_records_rerun_20261107.json
   ```

The recorded full run used commit `e148ea353c3cf0307c69de43f5e49201d2c4478f`, the exact copied Phase 2 config (`evaluation/evaluation_config.json`), MinAtar 1.0.15, independent environment/policy streams, categorical softmax sampling, sticky-action probability 0.1, and the 10,000-step cap.
