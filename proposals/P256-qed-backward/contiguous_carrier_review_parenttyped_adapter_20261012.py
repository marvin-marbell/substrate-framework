"""Source-selected request adapter for the immutable typed archive consumer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contiguous_carrier_review_parenttyped_consumer_20261012 import run

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    run(ROOT / request['input_archive'], args.output)
