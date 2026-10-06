"""
app/agents/graph.py — assembles the full LangGraph agent.

This is where every piece we've built so far gets wired together into
one runnable graph, matching the intelligence flow diagram:

    START -> planner -> (conditional) -> retriever -> responder -> END
                      \\-> responder (conversational path) -----^

The compiled `compiled_graph` object at the bottom is what main.py
actually calls to run a query end to end.
"""

from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.responder import responder_node


# The graph state is the shared memory for the whole agent. Each node
# reads only the fields it needs and emits a partial update; LangGraph
# merges those updates back into the checkpointed conversation state.
# This keeps the flow predictable and makes multi-turn memory possible
# without manually passing a huge object between every function call.
class AgentState(TypedDict):
    """
    The formal shape of shared state (Concept 4, Piece 3). Every node
    function we wrote reads from and writes partial updates to a dict
    matching this shape.
    """
    user_query: str
    user_context: dict
    active_document_id: str | None
    answer_length: str | None
    intent: str                      # active-document request route: "technical"
    reranked_chunks: list             # set by retriever_node (only on the technical path)
    restricted_evidence_found: bool   # boolean only; restricted text never enters graph state
    final_answer: str                 # set by responder_node
    # Annotated + operator.add tells LangGraph HOW to merge updates to
    # this specific field: instead of overwriting the list each time,
    # APPEND new messages to it. This is what actually builds up
    # multi-turn conversation history across checkpointed runs.
    conversation_history: Annotated[list, operator.add]
    # Trace is per request and is reset by the API input on every invoke.
    # Nodes explicitly carry forward the current turn's entries.
    trace: list[str]

def _route_by_intent(state: AgentState) -> str:
    """
    The conditional edge function (Concept 4, Piece 2). LangGraph calls
    this after planner_node runs, passing it the current state, and
    expects back the STRING NAME of whichever node should run next.
    This is the literal, concrete version of the intelligence flow
    diagram's Planner branching logic.
    """
    if state["intent"] == "technical":
        return "retriever"
    return "responder"


def _build_graph():
    graph = StateGraph(AgentState)

    # Register each node function under a name — this name is what
    # add_edge / conditional routing refers to.
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", responder_node)

    # Every graph needs an entry point. START is LangGraph's special
    # built-in marker for "this is where execution begins".
    graph.add_edge(START, "planner")

    # This is the conditional edge itself: after "planner" runs, call
    # _route_by_intent(state) and go to WHICHEVER node name it returns.
    # The dict maps possible return values to actual registered node
    # names (here they happen to match, but LangGraph requires this
    # explicit mapping regardless).
    graph.add_conditional_edges(
        "planner",
        _route_by_intent,
        {
            "retriever": "retriever",
            "responder": "responder",
        },
    )

    # The technical path continues: after retriever, always go to responder.
    # (This is a plain, unconditional edge — not every edge needs to be
    # conditional, only the ones that actually branch.)
    graph.add_edge("retriever", "responder")

    # Both paths converge here: after responder, the graph is done.
    # END is LangGraph's built-in marker for "execution may stop here".
    graph.add_edge("responder", END)

    # MemorySaver is the checkpointer (Concept 4, Piece "MemorySaver").
    # It's what lets conversation_history persist across separate calls
    # to the graph, as long as they share the same thread_id.
    #
    # NOTE: MemorySaver stores checkpoints in-process memory only — lost
    # on restart. A production deployment would swap this for a
    # Postgres- or Redis-backed checkpointer instead (see our notes —
    # Redis is already in requirements.txt for exactly this reason).
    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


# Built once at import time. main.py imports THIS object directly and
# calls .invoke(...) on it per request.
compiled_graph = _build_graph()