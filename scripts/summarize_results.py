"""Recompute headline aggregates from published per-question scored outcomes.

This does not rerun models, rescore predictions against withheld labels, or
independently validate the raw token-level capture receipts.
"""
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def summarize(name, table_key, n):
    folder = ROOT / "experiments" / name / "results"
    summary = json.loads((folder / "summary.json").read_text())
    rows = [json.loads(line) for line in (folder / "scored.jsonl").read_text().splitlines()]
    methods = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["id"], row["method"])
        assert key not in seen, f"Duplicate scored question/method: {key}"
        seen.add(key)
        methods[row["method"]].append(row)
    expected = summary[table_key]
    if isinstance(expected, list):
        expected = {r["method"]: r for r in expected}
    assert set(methods) == set(expected)
    result = {}
    for method, records in methods.items():
        assert len(records) == n
        metrics = {key: mean(r[key] for r in records) for key in ["f1", "em", "work_TF", "seconds"]}
        for key, value in metrics.items():
            assert math.isclose(value, expected[method][key], rel_tol=1e-10, abs_tol=1e-12), (method, key)
        metrics["exact_answers"] = sum(int(r["em"]) for r in records)
        result[method] = metrics
    return result


def main():
    confirmation = summarize("h31_fixed_policy_confirmation", "methods", 800)
    assert [confirmation[m]["exact_answers"] for m in
            ["bm25_direct", "bm25_rewrite", "prefix_only", "relation_following"]] == [276, 356, 375, 384]
    draft = summarize("h19_draft_free_baseline", "table", 200)
    assert all(r["exact_answers"] == 43 for r in draft.values())
    print(json.dumps({"status": "passed", "confirmation": confirmation, "draft_removal": draft,
                      "scope": "Recomputed aggregates from recorded scored outcomes; no new inference"}, indent=2))


if __name__ == "__main__":
    main()
