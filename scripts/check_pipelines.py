"""Run the original synthetic checks without GPU/model inference."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import check_h29_fixtures
    import h19_pipeline
    import h25_pipeline
    import h31_pipeline

    check_h29_fixtures.main()
    for module in [h19_pipeline, h25_pipeline, h31_pipeline]:
        print(json.dumps({"module": module.__name__, "result": module.self_check()}))


if __name__ == "__main__":
    main()
