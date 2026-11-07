# Weight-Space Programming research site

The site is a static, dependency-free GitHub Pages project. Pages and browser-side inference are plain HTML, CSS, and JavaScript. The Arena is an educational MinAtar port; it uses independent browser PRNG streams and does not claim to reproduce the corrected NumPy evaluation random draws.

## Rebuild the published data

From the repository root, with Python and NumPy available:

```bash
python scripts/export_research_site_data.py
```

The exporter reads the corrected episode CSVs, Phase 2 configuration, registries, metrics, model metadata, captured responses, and canonical weight archives. It also verifies the separate Argon High iterative-development artifact registry, supplied source scripts and weights, full-run episode CSVs, and paired analysis. It recomputes each iterative version's episode summary from raw CSV data before exporting it. Its outputs are confined to `docs/data/`; it does not rewrite experiment artifacts.

To check that the checked-in publication bundle still matches its sources without writing files:

```bash
python scripts/export_research_site_data.py --check
```

## Run locally

```bash
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. The browser must load the site over HTTP so its ES modules and data files can be fetched.

## Tests

```bash
node --test tests/site_conformance.test.mjs
pytest -q tests/test_site_export.py tests/test_browser_conformance.py
```

The Python-backed conformance checks compare the browser environment and network with the installed MinAtar 1.0.15 implementation and this repository's NumPy network. Passing checks demonstrate the tested fixtures and transitions; they do not verify every possible browser execution or establish parity of the browser demo RNG with NumPy.

The Argon High iterative-development results are exported as a separate dataset and are not included in the five-condition zero-shot ranking. The raw episode CSVs, exact supplied source files, weight archives, seed schedule, experiment report, and analysis JSON are included in the downloadable data bundle. The Follow-up 2 verifier discrepancy and provider-provenance limitations are documented on the Results page.

## GitHub Pages

The workflow in `.github/workflows/pages.yml` validates the generated bundle and deploys `docs/` through GitHub Pages on pushes to `main` or `phase-2-corrected-evaluation`. It also runs when the iterative-development experiment sources change. In repository settings, set **Pages → Build and deployment → Source** to **GitHub Actions**. A successful workflow run publishes the site at the repository's configured Pages URL.
