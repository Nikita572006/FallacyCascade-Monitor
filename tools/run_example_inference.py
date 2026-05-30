"""Run detector inference on example text files."""

from __future__ import annotations

import json
from pathlib import Path

from hallucination_cascade_detector import HallucinationCascadeDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "artifacts" / "cascade_detector"
GENERATION_FILE = PROJECT_ROOT / "examples" / "sample_generation.txt"
EVIDENCE_FILE = PROJECT_ROOT / "examples" / "sample_evidence.txt"


def main() -> None:
    detector = HallucinationCascadeDetector.load(MODEL_DIR)

    result = detector.analyze(
        GENERATION_FILE.read_text(encoding="utf-8"),
        evidence=EVIDENCE_FILE.read_text(encoding="utf-8"),
        alert_threshold=0.65,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
