# Research website

This directory contains the dependency-free static research site published at <https://mid-nightshark.github.io/weight-space-programming/> through GitHub Pages. The site includes an educational browser Arena; the Arena does not claim to reproduce the corrected evaluator's NumPy random streams.

For the research question, current benchmark results, conditions, and limitations, start at the [repository README](../README.md). For implementation details, data provenance, and reproduction steps, see the [developer guide](DEVELOPMENT.md).

## Data flow

`docs/index.html` and `docs/assets/` provide the page and browser code. The generated JSON, CSV, model records, and downloadable artifacts under `docs/data/` are exported from the research records. Do not hand-edit generated data to change a published score.

From the repository root, rebuild generated site data from the authoritative sources with:

```bash
python scripts/export_research_site_data.py
```

The exporter writes only beneath `docs/data/`. To check the existing site bundle and source hashes without writing files, run:

```bash
python scripts/export_research_site_data.py --check
```

Run the site's non-GUI tests with:

```bash
pytest -q tests/test_site_export.py tests/test_browser_conformance.py
node --test tests/site_conformance.test.mjs
```

These checks cover the tested data bundle, fixtures, and browser/Python conformance cases. They do not prove every browser behavior or benchmark equivalence of the Arena.

## Preview and deployment

To preview locally, serve the static files over HTTP:

```bash
python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>. The [Pages workflow](../.github/workflows/pages.yml) runs validation and deploys the contents of `docs/` through GitHub Pages on pushes to `main` and `phase-2-corrected-evaluation`. The configured repository Pages URL is the public website above.
