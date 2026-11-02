"""Convert only a locally trusted sklearn .joblib artifact to safer .skops format.

Usage: python scripts/export_trusted_model.py artifacts/churn_model.joblib /tmp/churn.skops
Do not run this on .joblib files received from strangers.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import skops.io as sio

from src.modelops.model_workspace import validate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = joblib.load(args.input)  # trusted local artifacts ONLY
    print("Features:", validate_model(model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, args.output)
    print("Wrote:", args.output)
    unknown = sio.get_untrusted_types(file=args.output)
    if unknown:
        print("Warning: model includes unsupported types:", unknown)
        print("The public upload page will reject this export.")


if __name__ == "__main__":
    main()
