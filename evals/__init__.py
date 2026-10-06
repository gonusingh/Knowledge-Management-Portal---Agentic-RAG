"""
evals/ — RAGAS evaluation suite + Streamlit 3-tab demo.

Responsibilities:
- golden_dataset.json: reference questions + verified correct answers
- run_ragas.py: computes Faithfulness, Context Precision, Context Recall,
  Answer Relevancy, Answer Correctness against the golden dataset,
  using JUDGE_GROQ (separate key from the live app)
- tool_correctness.py: Jaccard similarity check on agent action sequences
- eval_ui.py: Streamlit demo showing scores per test case

Status: SKELETON ONLY — build after RAGAS concept is re-confirmed solid
(already covered in depth).
"""
