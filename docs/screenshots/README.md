# Screenshots

Capture these from the running application (`npm run dev` + uvicorn) and save
them here with the exact file names below so the references in
`README.md` and `PROJECT_REPORT.md` resolve.

| File name | What to capture |
|---|---|
| `dashboard.png` | `/` — stat cards, confusion matrix, risk pie, model comparison |
| `analyzer.png` | `/analyzer` — result panel for a real held-out fraud sample |
| `batch.png` | `/batch` — summary, flagged rows and the download button |
| `performance.png` | `/performance` — validation/test curves and metrics table |
| `swagger.png` | `http://localhost:8000/docs` — the Swagger UI |

Screenshot tips for the report:
- Use a window wide enough for the two-column layout (≥ 1280 px).
- Capture the Analyzer twice — once for an actual-fraud sample (high risk) and
  once for a legitimate sample (low risk) — to demonstrate both outcomes.
- Do not edit the numbers in the screenshots; every figure in the UI is
  produced by the training run and must match `PROJECT_REPORT.md` §14.
