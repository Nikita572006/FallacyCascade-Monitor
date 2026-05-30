"""Export graph edges from a detector analysis JSON file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export detector graph edges to CSV.")
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    edges = payload.get("edges", [])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "target", "kind", "weight", "overlap"],
        )
        writer.writeheader()
        for edge in edges:
            writer.writerow({
                "source": edge.get("source"),
                "target": edge.get("target"),
                "kind": edge.get("kind"),
                "weight": edge.get("weight"),
                "overlap": ";".join(edge.get("overlap", [])),
            })


if __name__ == "__main__":
    main()
