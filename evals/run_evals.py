"""
Continuous Evaluation script for the RAG system.

WHY EVALS?
Unlike traditional software, RAG systems can silently degrade —
a model swap, a chunking change, or a threshold tweak can quietly
break retrieval quality without throwing any errors.

This script runs a fixed set of test questions against known-ingested
documents and checks:
1. Did retrieval find evidence (not a hallucination fallback)?
2. Is the retrieval score above a healthy bar?
3. Does the answer contain expected keywords?

Run manually: python3 run_evals.py --url https://farz-rag.duckdns.org --token YOUR_TOKEN
Run in CI: exit code is 1 if any eval fails, 0 if all pass.
"""

import argparse
import sys
import httpx

# Each eval case: a question, expected keywords in the answer,
# and whether we expect evidence to be found at all.
EVAL_CASES = [
    {
        "question": "What is the RAG system built on?",
        "expect_keywords": ["aws", "kubernetes", "chromadb"],
        "expect_evidence": True,
    },
    {
        "question": "What retrieval techniques does this system use?",
        "expect_keywords": ["bm25", "vector", "rerank"],
        "expect_evidence": True,
    },
    {
        "question": "What is the capital of France?",
        "expect_keywords": [],
        "expect_evidence": False,  # should trigger hallucination fallback
    },
]


def run_eval(client: httpx.Client, base_url: str, case: dict) -> dict:
    resp = client.post(
        f"{base_url}/query",
        json={"question": case["question"]},
        timeout=30,
    )
    result = {"question": case["question"], "passed": True, "reasons": []}

    if resp.status_code != 200:
        result["passed"] = False
        result["reasons"].append(f"HTTP {resp.status_code}")
        return result

    data = resp.json()
    evidence_sufficient = data.get("evidence_sufficient", False)
    answer = data.get("answer", "").lower()

    if case["expect_evidence"] and not evidence_sufficient:
        result["passed"] = False
        result["reasons"].append("expected evidence found, but got fallback")

    if not case["expect_evidence"] and evidence_sufficient:
        result["passed"] = False
        result["reasons"].append("expected fallback, but system answered confidently (possible hallucination)")

    for kw in case["expect_keywords"]:
        if kw.lower() not in answer:
            result["passed"] = False
            result["reasons"].append(f"missing expected keyword: '{kw}'")

    result["evidence_sufficient"] = evidence_sufficient
    result["latency_ms"] = data.get("latency_ms")
    result["scores"] = data.get("scores")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base URL of the RAG API")
    parser.add_argument("--token", required=True, help="Bearer token for auth")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {args.token}"}
    all_passed = True

    print(f"Running {len(EVAL_CASES)} eval cases against {args.url}\n")

    with httpx.Client(headers=headers) as client:
        for case in EVAL_CASES:
            result = run_eval(client, args.url, case)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{status}] {result['question']}")
            if result.get("evidence_sufficient") is not None:
                print(f"       evidence_sufficient={result['evidence_sufficient']} latency={result.get('latency_ms')}ms scores={result.get('scores')}")
            for reason in result["reasons"]:
                print(f"       - {reason}")
            print()
            if not result["passed"]:
                all_passed = False

    if all_passed:
        print("All eval cases passed.")
        sys.exit(0)
    else:
        print("Some eval cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
