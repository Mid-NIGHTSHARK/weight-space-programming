# Research References

## Primary Research Publications and Source Code

1. **Original Publication:**
   - **Title:** *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*
   - **Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL)
   - **Publication Date:** November 2, 2026
   - **URL:** [https://rikiyat.github.io/blog/weight-space-programming/](https://rikiyat.github.io/blog/weight-space-programming/)
   - **Focus:** Demonstration that frontier LLMs (specifically GPT-6 Astra) can natively read, audit, and write functional neural network weights directly without training loops, gradients, backpropagation, or environment feedback.

2. **Original Experiment Logs & Artifacts Repository:**
   - **Repository:** `RikiyaT/lm_understand-logs`
   - **URL:** [https://github.com/RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)
   - **Commit Audited:** `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` (Oct 29, 2026)
   - **Key Artifacts:**
     - `exp5_write_from_scratch/runs/breakout/reply/prompt.txt`: Full raw prompt sent to Astra.
     - `exp5_write_from_scratch/runs/breakout/reply/answer.md`: Full raw response received from Astra.
     - `exp5_write_from_scratch/runs/breakout/reply/run.json`: Execution metadata, token usage, tool-free verification.
     - `exp5_write_from_scratch/networks/breakout/hand.npz`: Original weights array saved from Astra's response.
       - SHA-256: `6e8d42658aed3704c7bf6fb67015e9468cc03f1d67cb8a4fd192e6534fcccb4a`
     - `exp5_write_from_scratch/eval/minatar_eval.json`: Official evaluation scores (Breakout Hand: mean 517.411, se 7.417, episodes 1000).

3. **Independent Replication Repository:**
   - **Repository:** `alan-turing-institute/weights_editing`
   - **URL:** [https://github.com/alan-turing-institute/weights_editing](https://github.com/alan-turing-institute/weights_editing)
   - **Status as of Audit (Nov 2026):** HTTP 404 (private / inaccessible). Documented in `research_audit.md`.

4. **Environment Specification (MinAtar):**
   - **Paper:** *MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning*
   - **Authors:** Kenny Young, Tian Tian (University of Alberta)
   - **arXiv:** [arXiv:1903.03176](https://arxiv.org/abs/1903.03176) (2019)
   - **Package Version:** `minatar==1.0.15`
   - **Repository:** [https://github.com/kenjyoung/MinAtar](https://github.com/kenjyoung/MinAtar)
   - **License:** GNU General Public License v3.0 (`LICENSE-MinAtar`)

5. **Model Subject Execution Harness:**
   - **Platform:** OpenAI Codex CLI (`codex exec`)
   - **Model:** `gpt-6-astra`
   - **Reasoning Effort:** `xhigh`
   - **Reasoning Summaries:** `detailed`
   - **Execution Configuration:** `--strict-config`, `--skip-git-repo-check`, tool-free (`"tool_free": true`), 0 tool calls.
