# Rebuild the research figures

Run this command from the repository root after installing the project dependencies:

```sh
.venv/bin/python docs/scripts/visualization/generate_figures.py
```

The script reads the nine corrected Phase 2 episode CSVs, the three separate Argon High iterative-development CSVs, their checked condition-analysis JSON, the Phase 2 paired-comparisons JSON, and the preserved diagnostic trace JSON. It independently recomputes the reported means, standard errors, medians, score extrema, episode-length means/medians, cutoff counts, action totals, matched episode means, paired standard deviations, and paired Cohen's *dz* from raw records. For the ten primary-model pairs it also repeats the stored paired-bootstrap confidence intervals and two-sided sign-flip p-values; it recomputes the Holm step-down adjustment over all 36 recorded p-values. A mismatch stops figure generation.

It writes nine figure families under `docs/assets/figures/`, each as SVG, 300-DPI PNG, and PDF. It writes `media_manifest.json` with input hashes and output file hashes, formats, resolutions, source paths, and figure descriptions. The command never writes to the experiment data or analysis directories.

The comparison figure keeps the handcrafted Astra reference in a separate linear panel. The paired-comparison figure is restricted to the five original zero-shot conditions, but Holm-adjusted p-values shown there come from the full 36-comparison family. The iterative-development figure is a separate experiment. The diagnostic trace figure shows one preserved historical diagnostic replay per primary condition and is not a representative sample.

The project environment declares NumPy and Matplotlib. The script uses those libraries and the Python standard library, including its PNG-header parser for output resolution checks.

The browser Arena's gameplay recorder and still-frame control are documented in [`../gameplay/README.md`](../gameplay/README.md). Browser recordings are demonstration captures with browser RNG streams; they are not benchmark runs.
