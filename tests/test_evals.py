from rag.embeddings import HashingEmbedder
from rag.evals import EvalCase, evaluate
from rag.llm import ExtractiveLLM
from rag.pipeline import RagPipeline
from rag.retriever import Retriever

ONBOARDING = "The daily standup is at 9:30 AM. New engineers get a laptop on day one."
SECURITY = "Production secrets are stored in the vault. MFA is required for all admin accounts."


def _pipeline():
    retriever = Retriever(HashingEmbedder())
    retriever.add_document("onboarding.md", ONBOARDING)
    retriever.add_document("security.md", SECURITY)
    return RagPipeline(retriever, ExtractiveLLM(), k=3)


def test_eval_dataset_passes_fully():
    cases = [
        EvalCase("What time is the daily standup?", ["9:30"], "onboarding.md"),
        EvalCase("Where are production secrets stored?", ["vault"], "security.md"),
        EvalCase("Is MFA required for admin accounts?", ["MFA"], "security.md"),
    ]
    report = evaluate(_pipeline(), cases)
    assert report.total == 3
    assert report.pass_rate == 1.0
    assert report.grounded == 3


def test_eval_flags_missing_keyword():
    # A keyword that does not appear anywhere should fail relevance (not pass).
    cases = [EvalCase("What time is the daily standup?", ["midnight"], "onboarding.md")]
    report = evaluate(_pipeline(), cases)
    assert report.passed == 0
