"""
evals/run_ragas.py — runs the golden dataset through the real retrieval +
generation pipeline, then scores it with RAGAS.

Run with:
    python evals/run_ragas.py

Uses JUDGE_GROQ (a separate key from the live app's GROQ_API_KEY) as the
judge model — see our RAGAS notes on why this separation matters: a full
eval run makes many extra LLM calls, and sharing a key with the live app
risks rate-limiting real users mid-conversation.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- Shim for a known upstream ragas bug ---------------------------
# ragas (both 0.3.x and 0.4.x, as of testing) unconditionally imports
# ChatVertexAI from a langchain_community path that was removed in
# current langchain-community versions — see
# github.com/vibrantlabsai/ragas/issues/2745. We don't use Google
# VertexAI anywhere in this project, so instead of waiting for ragas to
# patch this, we register a harmless fake module in sys.modules BEFORE
# ragas is imported. Python's import system finds our fake module
# already present and uses it, instead of trying (and failing) to find
# the real one — ragas never actually calls anything on it.
import types

_fake_vertexai_module = types.ModuleType("langchain_community.chat_models.vertexai")


class _FakeChatVertexAI:
    pass


_fake_vertexai_module.ChatVertexAI = _FakeChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _fake_vertexai_module
# --- End shim --------------------------------------------------------

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.metrics import AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.services.retrieval.vector_search import search
from app.services.retrieval.reranker import rerank
from app.gateway.portkey_client import call_llm
from evals.golden_dataset import GOLDEN_DATASET
from app.security.access_control import DEMO_USERS


def run_pipeline_for_question(
    question: str,
    active_document_id: str,
) -> tuple[str, list[str]]:
    """
    Runs ONLY the retrieval + generation pipeline directly — bypassing
    Guardrails and the LangGraph agent wrapper, since evaluation cares
    about retrieval/generation quality, not the safety layer. Returns
    the answer plus the raw chunk texts (RAGAS needs these separately as
    "contexts" to compute Faithfulness/Precision/Recall against).
    """
    candidates = search(
        query=question,
        user=DEMO_USERS["carol"],
        top_k=20,
        document_id=active_document_id,
    )
    reranked = rerank(
        original_query=question,
        candidates=candidates,
        active_document_id=active_document_id,
        top_n=5,
    )
    answer = call_llm(user_query=question, context_chunks=reranked, conversation_history=[])
    contexts = [chunk["text"] for chunk in reranked]
    return answer, contexts


def main():
    active_document_id = os.getenv("EVAL_ACTIVE_DOCUMENT_ID", "")
    if not active_document_id:
        raise ValueError(
            "Set EVAL_ACTIVE_DOCUMENT_ID to the explicitly selected document's "
            "64-character SHA-256 ID before running the scoped evaluation."
        )

    print(f"Running {len(GOLDEN_DATASET)} golden questions through the pipeline...\n")

    # The evaluation harness records one output row per question:
    # the question itself, the model answer, the actual retrieved contexts,
    # and the known-good reference answer used as the benchmark.
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in GOLDEN_DATASET:
        print(f"  -> {item['question']}")
        answer, contexts = run_pipeline_for_question(item["question"], active_document_id)
        questions.append(item["question"])
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["reference_answer"])

    # RAGAS expects this specific column shape: question, the generated
    # answer, the list of retrieved context strings, and the reference
    # (ground truth) answer for Answer Correctness-style comparison.
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    print("\nScoring with RAGAS (this makes many judge LLM calls, may take a few minutes)...\n")

    # The judge model uses JUDGE_GROQ — a separate key from the live
    # app's GROQ_API_KEY, exactly as covered in our RAGAS notes.
    judge_llm = LangchainLLMWrapper(
        ChatGroq(api_key=settings.judge_groq, model="openai/gpt-oss-120b")
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key=settings.gemini_api_key,
        )
    )

    # The default run fired off many parallel judge calls at once, which
    # blew straight through Groq's free-tier rate limit (30 requests/min)
    # — causing a storm of 429s and, eventually, several outright
    # TimeoutErrors once retries piled up. max_workers=1 forces RAGAS to
    # run one judge call at a time instead of in parallel, trading total
    # runtime (slower) for a clean, fully-completed run (no dropped
    # jobs) — the right tradeoff on a free-tier key.
    run_config = RunConfig(max_workers=1, timeout=180)

    # answer_relevancy's default strictness=3 asks the judge for 3
    # generations in a single call, which Groq's API rejects outright
    # ('n' must be at most 1) — every job for this metric failed with
    # that error, which is why it came back as NaN last run. strictness=1
    # asks for exactly 1 generation, matching Groq's limit, at the cost
    # of slightly less internal self-consistency for this one metric.
    answer_relevancy_metric = AnswerRelevancy(strictness=1)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy_metric],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()

    output_path = Path(__file__).parent / "eval_results.csv"
    df.to_csv(output_path, index=False)

    print(f"\nFull per-question results saved to {output_path}\n")
    print("=== Summary (average scores across all questions) ===")
    for metric in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
        if metric in df.columns:
            print(f"  {metric}: {df[metric].mean():.3f}")


if __name__ == "__main__":
    main()