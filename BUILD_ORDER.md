# BUILD_ORDER.md — implementation sequence

This skeleton is organizational only — no real logic yet. Fill in files in
this order once the matching concept has been taught/re-confirmed:

## Already covered — safe to implement now
1. `app/config.py` — straightforward, no new concept needed
2. `app/ingestion/loaders/` + `app/ingestion/chunking/` — largely portable
   from the original v1 project
3. `app/services/retrieval/vector_search.py` — Qdrant + cosine similarity
   (Concepts 2, 6 — done)
4. `app/services/retrieval/reranker.py` — FlashRank (Concept 5 — done)
5. `app/gateway/portkey_client.py` — Portkey fallback config (Concept 7 — done)
6. `app/guardrails/guardrail_gate.py` — NeMo Guardrails (Concept 3 — done)
7. `app/agents/nodes/*.py` + `app/agents/graph.py` — LangGraph (Concept 4 — done)
8. `app/main.py` — wires everything above together
9. `ui/app.py` — Streamlit UI (mostly ports from v1)

## Hold off until taught
10. `app/services/retrieval/hyde.py` — needs HyDE concept session
11. `app/services/retrieval/multi_query.py` — needs Multi-Query concept session
12. `evals/` full implementation — RAGAS is covered conceptually (Concept 8),
    but wiring it with LangSmith tracing benefits from Concept 9 first
13. Tracing hooks throughout (`logfire`, `langsmith` calls) — needs Concept 9
    (LangSmith + Pydantic Logfire)
14. `Dockerfile` + deployment scripts — needs the Docker/Deployment concept

## Suggested order to finish the syllabus before finishing the build
- Concept 9: LangSmith + Pydantic Logfire
- HyDE (new)
- Multi-Query Expansion (new)
- SLMs vs LLMs
- Fine-tuning
- Docker + Cloud Deployment
