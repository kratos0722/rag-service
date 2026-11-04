"""CLI: ask questions over a document folder, run evals, or serve the API."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evals import EvalCase, evaluate
from .factory import build_pipeline, build_triage_service
from .triage import Alert

_DEFAULT_DOCS = Path(__file__).resolve().parents[2] / "data" / "sample_docs"
_DEFAULT_RUNBOOKS = Path(__file__).resolve().parents[2] / "data" / "sample_runbooks"

# Evals tied to the bundled sample docs.
_DATASET = [
    EvalCase("What time is the daily standup?", ["9:30"], "onboarding.md"),
    EvalCase("Where are production secrets stored?", ["vault"], "security.md"),
    EvalCase("Is MFA required for admin accounts?", ["MFA"], "security.md"),
]


def _ingest_dir(pipeline, docs_dir: Path) -> int:
    total = 0
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            total += pipeline.retriever.add_document(path.name, path.read_text(encoding="utf-8"))
    return total


def cmd_ask(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(k=args.k)
    _ingest_dir(pipeline, Path(args.docs))
    result = pipeline.answer(args.question)
    print(f"Answer: {result.answer}")
    print(f"Sources: {', '.join(result.sources) or '(none)'}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    pipeline = build_pipeline(k=args.k)
    _ingest_dir(pipeline, Path(args.docs))
    report = evaluate(pipeline, _DATASET)
    print(f"cases={report.total} relevant={report.relevant} grounded={report.grounded} "
          f"source_correct={report.source_correct} passed={report.passed} "
          f"pass_rate={report.pass_rate:.0%}")
    return 0 if report.passed == report.total else 1


def cmd_triage(args: argparse.Namespace) -> int:
    service = build_triage_service(k=args.k)
    runbooks = Path(args.runbooks)
    for path in sorted(runbooks.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            service.ingest_runbook(path.name, path.read_text(encoding="utf-8"))
    result = service.triage(Alert(message=args.message, service=args.service, severity=args.severity))
    print(f"Priority: {result.priority}  (matched={result.matched}, confidence={result.confidence})")
    print(f"Route:    {result.route}")
    print(f"Runbooks: {', '.join(result.matched_runbooks) or '(none)'}")
    print(f"Steps:    {result.suggested_steps}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("rag.app:app", host="0.0.0.0", port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Answer a question over a document folder")
    ask.add_argument("question")
    ask.add_argument("--docs", default=str(_DEFAULT_DOCS))
    ask.add_argument("--k", type=int, default=3)
    ask.set_defaults(func=cmd_ask)

    ev = sub.add_parser("eval", help="Run output evals over the bundled dataset")
    ev.add_argument("--docs", default=str(_DEFAULT_DOCS))
    ev.add_argument("--k", type=int, default=3)
    ev.set_defaults(func=cmd_eval)

    tri = sub.add_parser("triage", help="Triage an alert against a runbook folder")
    tri.add_argument("message", help="The alert message/summary")
    tri.add_argument("--service", default=None, help="Affected service name")
    tri.add_argument("--severity", default="medium", choices=["critical", "high", "medium", "low"])
    tri.add_argument("--runbooks", default=str(_DEFAULT_RUNBOOKS))
    tri.add_argument("--k", type=int, default=3)
    tri.set_defaults(func=cmd_triage)

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
