"""Route every request through active-document retrieval."""


def planner_node(state: dict) -> dict:
    return {
        "intent": "technical",
        "trace": [
            *state.get("trace", []),
            "Planner: routing through active-document grounded retrieval",
        ],
    }
