# FraudShield AI — Quick Run Guide

Every command below is run from the project root
(`FraudShield-AI/`), in **two terminals**.

---

## One-time setup

```bash
# 1. Python environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt

# 2. Dataset (real Kaggle data)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
#    ...or download manually from
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    and place creditcard.csv in data/raw/

# 3. Frontend dependencies
cd frontend
npm install
cd ..
```

## Train the models (~3 minutes)

```bash
python scripts/train.py
# No dataset? Use the clearly-labelled synthetic demo:
python scripts/train.py --demo
```

Expected output ends with:

```
Selected model by validation PR-AUC: random_forest
Tuned threshold: 0.2997 -> val F1 0.8804 ...
TEST (random_forest @ 0.2997): precision 0.9342 recall 0.7474 F1 0.8304 PR-AUC 0.8110 ROC-AUC 0.9437
All artifacts saved to .../artifacts
```

## Run the application (two terminals)

```bash
# Terminal 1 — FastAPI backend (port 8000)
python -m uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — React frontend (port 5173)
cd frontend
npm run dev
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:5173 |
| API + Swagger | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

## Verify everything works

```bash
# Backend tests (56 tests, ~10 s, no dataset needed)
python -m pytest backend/tests -q

# Frontend type-check + production build
cd frontend
npm run build
```

## Demo script for the evaluator (3 minutes)

1. Open http://localhost:5173 → **Dashboard**: real dataset stats, confusion
   matrix, model comparison — all from the saved training artifacts.
2. **Transaction Analyzer** → click "Load a real held-out sample #1" (actual
   fraud) → **Predict** → shows high probability, risk badge and explanation.
3. Load a legitimate sample → Predict → low risk. Explain the threshold line.
4. **Batch Analysis** → upload a CSV of transactions → review flagged rows →
   download the predictions CSV.
5. **Model Performance** → switch validation/test tabs on the curves; explain
   why PR-AUC beats accuracy for a 0.17% fraud rate.
6. Open http://localhost:8000/docs → run `POST /api/predict` live.
7. Be ready for: "why is recall below 1?", "what is a false negative here?",
   "why not use accuracy?" → see `docs/VIVA_QUESTIONS.md`.
