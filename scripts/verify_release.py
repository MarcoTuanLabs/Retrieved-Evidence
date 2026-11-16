"""Verify the extracted original files without model execution."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "provenance/original_files.json").read_text())
    for name, record in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == record["sha256"], f"Original file changed: {name}"
    sources = sorted((ROOT / "src").glob("*.py"))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    print(json.dumps({"status": "passed", "original_files": len(manifest),
                      "python_sources": len(sources), "new_model_calls": 0}))


if __name__ == "__main__":
    main()
