# Notebooks

The notebook in this folder is a thin, document-friendly wrapper around the same
logic used by `scripts/train.py`. The script is the **source of truth** for
reproducible training; notebooks are provided for interactive exploration in an
academic setting (e.g. showing EDA step by step during a viva).

| Notebook | Purpose |
|---|---|
| `01_exploratory_analysis.ipynb` | Loads the dataset with the project's validated loader and reproduces the EDA figures printed in the report. |

To run a notebook:

```bash
pip install jupyter
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

Notebooks are intentionally **not** used for model training or evaluation:
keeping training in `scripts/train.py` guarantees deterministic seeds, a single
documented split and a single test-set evaluation, which notebooks make easy to
break accidentally.
