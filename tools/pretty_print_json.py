"""Pretty-print detector JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print a JSON file.")
    parser.add_argument("input_file", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input_file.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
