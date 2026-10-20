import unittest

from app.services.source_evidence import (
    distinct_source_count,
    extract_selected_sources,
)


def trace_source(
    source: str,
    document_id: str,
    chunk_index: int,
    score: str = "0.42",
) -> str:
    return (
        f"Selected chunk: source={source}, document_id={document_id}, "
        f"chunk_index={chunk_index}, rerank_score={score}"
    )


class SourceEvidenceTests(unittest.TestCase):
    def test_no_sources(self):
        self.assertEqual(distinct_source_count([]), 0)
        self.assertEqual(extract_selected_sources(["No chunks selected."]), [])

    def test_one_source(self):
        sources = extract_selected_sources(
            [trace_source("resume.pdf", "a" * 64, 0)]
        )
        self.assertEqual(distinct_source_count(sources), 1)

    def test_three_distinct_sources(self):
        sources = extract_selected_sources(
            [
                trace_source("one.pdf", "a" * 64, 0),
                trace_source("two.docx", "b" * 64, 0),
                trace_source("three.txt", "c" * 64, 0),
            ]
        )
        self.assertEqual(distinct_source_count(sources), 3)

    def test_multiple_chunks_from_one_document_count_once(self):
        sources = extract_selected_sources(
            [
                trace_source("resume.pdf", "a" * 64, 0),
                trace_source("resume.pdf", "a" * 64, 1),
            ]
        )
        self.assertEqual(len(sources), 2)
        self.assertEqual(distinct_source_count(sources), 1)

    def test_duplicate_chunk_trace_is_deduplicated(self):
        line = trace_source("resume.pdf", "a" * 64, 0)
        sources = extract_selected_sources([line, line])
        self.assertEqual(len(sources), 1)
        self.assertEqual(distinct_source_count(sources), 1)

    def test_missing_document_id_uses_normalized_filename(self):
        sources = [
            {"source": " Resume.PDF ", "document_id": "unknown"},
            {"source": "resume.pdf", "document_id": ""},
            {"source": "project.docx", "document_id": ""},
        ]
        self.assertEqual(distinct_source_count(sources), 2)


if __name__ == "__main__":
    unittest.main()
