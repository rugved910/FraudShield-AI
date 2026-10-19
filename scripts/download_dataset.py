"""Download helper for the Kaggle credit-card fraud dataset.

Two options are supported:

1. Kaggle CLI (recommended if you have an API token):
       python scripts/download_dataset.py --kaggle
   Requires ~/.kaggle/kaggle.json  (create it at https://www.kaggle.com/settings → API)

2. Manual download:
       python scripts/download_dataset.py --instructions
   prints step-by-step instructions for downloading from the website.

The script NEVER downloads anything from unofficial mirrors and NEVER
substitutes synthetic data for the real dataset.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = REPO_ROOT / "data" / "raw"
TARGET_CSV = TARGET_DIR / "creditcard.csv"
KAGGLE_SLUG = "mlg-ulb/creditcardfraud"

INSTRUCTIONS = f"""
Downloading the Credit Card Fraud Detection dataset (Kaggle MLG-ULB)
====================================================================

1. Open https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   (free Kaggle account required)
2. Click "Download" — you will get  creditcard.zip
3. Extract it; you should see  creditcard.csv  (~150 MB)
4. Place the file at:
       {TARGET_CSV}
5. Verify:
       python -c "import pandas as pd; df = pd.read_csv(r'{TARGET_CSV}'); print(df.shape, df['Class'].sum())"
   Expected: (284807, 31) 492

Alternatively with the Kaggle CLI:
    pip install kaggle
    # create API token at https://www.kaggle.com/settings -> "Create New Token"
    # place kaggle.json in  %USERPROFILE%\\.kaggle\\  (Windows) or ~/.kaggle/ (Linux/macOS)
    kaggle datasets download -d {KAGGLE_SLUG} -p data/raw --unzip

If you cannot access Kaggle at all, train a clearly-labelled synthetic DEMO
model instead (results are NOT real-world fraud performance):
    python scripts/train.py --demo
"""


def download_with_kaggle_cli() -> int:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        print("ERROR: 'kaggle' CLI not found. Install it with:  pip install kaggle",
              file=sys.stderr)
        return 1
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle, "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(TARGET_DIR), "--unzip"]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("kaggle CLI failed — check that %USERPROFILE%\\.kaggle\\kaggle.json exists.",
              file=sys.stderr)
        return result.returncode
    if TARGET_CSV.exists():
        print(f"OK — dataset installed at {TARGET_CSV}")
        return 0
    print("Download finished but creditcard.csv was not found in", TARGET_DIR,
          file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", action="store_true",
                        help="Download using the Kaggle CLI (requires API token).")
    parser.add_argument("--instructions", action="store_true",
                        help="Print manual download instructions.")
    args = parser.parse_args()

    if TARGET_CSV.exists():
        print(f"Dataset already present: {TARGET_CSV}")
        return 0
    if args.kaggle:
        return download_with_kaggle_cli()
    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
