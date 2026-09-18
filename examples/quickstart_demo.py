"""
Concrete SynapseRAG demo: point the engine at its own source code.

Ingests a handful of SynapseRAG's own modules (the circuit breaker and
fusion logic from Faza 7/10, plus the README) into a fresh SynapseEngine,
then asks a few real questions and prints the answers with their
file:line citations.

Run from the repo root:
    python examples/quickstart_demo.py
"""

import shutil
from pathlib import Path

from synapserag import SynapseEngine, SynapseConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
STORAGE_DIR = REPO_ROOT / "data" / "quickstart_demo_db"

FILES_TO_INGEST = [
    "synapserag/verify/circuit_breaker.py",
    "synapserag/retrieval/fusion.py",
    "README.md",
]

QUESTIONS = [
    "How does the circuit breaker guard against small-corpus false consensus?",
    "What does Reciprocal Rank Fusion do to combine scores from different channels?",
    "How do I install synapserag with the MCP server extra?",
]


def main() -> None:
    # Start from a clean slate so re-running the demo gives consistent results.
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)

    engine = SynapseEngine(config=SynapseConfig(storage_dir=str(STORAGE_DIR)))

    print("Ingesting files:")
    for rel_path in FILES_TO_INGEST:
        full_path = REPO_ROOT / rel_path
        engine.ingest_file(str(full_path))
        print(f"  + {rel_path}")

    for question in QUESTIONS:
        print(f"\nQ: {question}")
        results = engine.query(question, mode="hybrid", top_k=3)
        if results.circuit_breaker_triggered:
            print(f"  (circuit breaker triggered: {results.warning})")
            continue
        if not results.matches:
            print("  (no matches)")
            continue
        for match, citation in zip(results.matches, results.citations):
            snippet = citation.exact_quote.strip().replace("\n", " ")[:160]
            print(f"  [{match.score:.3f}] {citation.uri}:{citation.start_line} -> {snippet}")


if __name__ == "__main__":
    main()
