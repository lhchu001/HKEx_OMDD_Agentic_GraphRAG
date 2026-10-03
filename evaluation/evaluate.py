"""evaluation/evaluate.py — Offline benchmark runner for HKEX OMDD RAG.

Usage from project root:
    python evaluation/evaluate.py

The evaluator calls retrieving.ask() for each test case and records:
- answer text
- whether expected source filenames were retrieved
- whether expected terms occur in the answer
- answer latency
- graph route usage (only expected for cross-document cases)

This is a deterministic regression harness, not a substitute for human review.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from retrieving import ask  # noqa: E402

TEST_CASES_PATH = PROJECT_ROOT / "evaluation" / "test_cases.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.json"


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def contains_all_terms(text: str, terms: List[str]) -> bool:
    normalized = normalize(text)
    return all(normalize(term) in normalized for term in terms)


def retrieved_sources(result: Dict[str, Any]) -> set[str]:
    return {
        str(doc.metadata.get("source", ""))
        for doc in result.get("reranked_docs", [])
    }


def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    result = ask(case["question"])
    latency_seconds = round(time.perf_counter() - started, 3)

    answer = result.get("answer", "")
    sources = retrieved_sources(result)
    expected_sources = set(case.get("expected_sources", []))
    required_terms = case.get("required_answer_terms", [])

    source_recall = (
        len(expected_sources & sources) / len(expected_sources)
        if expected_sources else 1.0
    )
    answer_has_terms = contains_all_terms(answer, required_terms)
    abstained = normalize(answer) == "do not know"
    graph_used = bool(result.get("graph_chunk_ids", []))

    passed = (
        source_recall == 1.0
        and answer_has_terms
        and (not case.get("must_not_abstain", True) or not abstained)
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "passed": passed,
        "latency_seconds": latency_seconds,
        "answer": answer,
        "retrieved_sources": sorted(sources),
        "expected_sources": sorted(expected_sources),
        "source_recall": round(source_recall, 3),
        "required_answer_terms": required_terms,
        "answer_has_required_terms": answer_has_terms,
        "abstained": abstained,
        "graph_used": graph_used,
        "expected_graph_route": case.get("expect_graph_route", False),
        "graph_route_matches_expectation": graph_used == case.get("expect_graph_route", False),
    }


def main() -> None:
    with TEST_CASES_PATH.open("r", encoding="utf-8") as handle:
        test_cases = json.load(handle)

    results = []
    for case in test_cases:
        print(f"Running {case['id']}: {case['question']}")
        outcome = evaluate_case(case)
        results.append(outcome)
        status = "PASS" if outcome["passed"] else "FAIL"
        print(f"  {status}; source recall={outcome['source_recall']}; latency={outcome['latency_seconds']}s")

    total = len(results)
    passed = sum(item["passed"] for item in results)
    mean_latency = round(sum(item["latency_seconds"] for item in results) / max(total, 1), 3)
    source_recall = round(sum(item["source_recall"] for item in results) / max(total, 1), 3)
    graph_expectation = round(
        sum(item["graph_route_matches_expectation"] for item in results) / max(total, 1), 3
    )

    report = {
        "summary": {
            "total_cases": total,
            "passed_cases": passed,
            "pass_rate": round(passed / max(total, 1), 3),
            "mean_latency_seconds": mean_latency,
            "mean_source_recall": source_recall,
            "graph_route_expectation_accuracy": graph_expectation,
        },
        "cases": results,
    }
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\nSummary")
    print("-------")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")
    print(f"\nDetailed report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()