from rag.embeddings import HashingEmbedder
from rag.llm import ExtractiveLLM
from rag.pipeline import RagPipeline
from rag.retriever import Retriever

ONBOARDING = (
    "New engineers receive their laptop on day one. The daily standup is at "
    "9:30 AM in the team channel. Pull requests require one review before merge."
)
SECURITY = (
    "Production secrets are stored in the vault and never committed to source "
    "control. MFA is required for all admin accounts."
)


def _pipeline():
    retriever = Retriever(HashingEmbedder())
    retriever.add_document("onboarding.md", ONBOARDING)
    retriever.add_document("security.md", SECURITY)
    return RagPipeline(retriever, ExtractiveLLM(), k=3)


def test_answers_from_correct_source():
    result = _pipeline().answer("Where are production secrets stored?")
    assert "vault" in result.answer.lower()
    assert "security.md" in result.sources


def test_answer_is_grounded_in_context():
    result = _pipeline().answer("What time is the daily standup?")
    assert "9:30" in result.answer
    # The answer text must come from the retrieved context (extractive).
    assert any(result.answer in ctx or result.answer.strip(".") in ctx for ctx in result.contexts)


def test_unknown_question_declines():
    result = _pipeline().answer("What is the airspeed velocity of an unladen swallow?")
    assert "enough information" in result.answer.lower()
