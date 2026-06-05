from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_HF_ROOT
from .sources.hf_prompts import check_hf_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify HF datasets for air_scenario_lab.")
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    args = parser.parse_args(argv)

    status = check_hf_data(args.hf_root)
    print(json.dumps(status, indent=2))
    if not status["ok"]:
        print("\nMissing datasets. Run scripts/check_hf_data.sh for download hints.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
