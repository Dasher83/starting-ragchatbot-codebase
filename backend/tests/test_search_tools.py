"""
Tests for CourseSearchTool.execute() and _format_results() in search_tools.py.

Mocking strategy: MagicMock() replaces VectorStore so no ChromaDB or embedding
model is needed. SearchResults objects are constructed directly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
import pytest

from vector_store import SearchResults
from search_tools import CourseSearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_store():
    store = MagicMock()
    store.get_lesson_link.return_value = None
    return store


def make_results(docs, metas, distances=None):
    if distances is None:
        distances = [0.1] * len(docs)
    return SearchResults(documents=docs, metadata=metas, distances=distances)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:

    # --- output format ---

    def test_execute_formats_result_with_header(self):
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["Lesson content about tool calling."],
            metas=[{"course_title": "MCP Course", "lesson_number": 5}],
        )
        tool = CourseSearchTool(store)

        result = tool.execute(query="tool calling", course_name="MCP", lesson_number=5)

        assert "MCP Course" in result
        assert "Lesson 5" in result
        assert "Lesson content about tool calling." in result

    def test_execute_formats_multiple_results(self):
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["Chunk A.", "Chunk B."],
            metas=[
                {"course_title": "MCP Course", "lesson_number": 1},
                {"course_title": "MCP Course", "lesson_number": 2},
            ],
        )
        tool = CourseSearchTool(store)

        result = tool.execute(query="content")

        assert "Chunk A." in result
        assert "Chunk B." in result
        assert "Lesson 1" in result
        assert "Lesson 2" in result

    # --- store.search() call ---

    def test_search_called_with_correct_parameters(self):
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "MCP Course", "lesson_number": 5}],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="what was covered", course_name="MCP", lesson_number=5)

        store.search.assert_called_once_with(
            query="what was covered",
            course_name="MCP",
            lesson_number=5,
        )

    def test_search_called_with_only_query_when_no_filters(self):
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "Any Course"}],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="some topic")

        store.search.assert_called_once_with(query="some topic", course_name=None, lesson_number=None)

    # --- empty / error results ---

    def test_returns_no_content_found_on_empty_results(self):
        store = make_mock_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)

        result = tool.execute(query="lesson 5 content", course_name="MCP", lesson_number=5)

        assert "No relevant content found" in result

    def test_empty_results_message_includes_course_filter(self):
        store = make_mock_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)

        result = tool.execute(query="content", course_name="MCP")

        assert "MCP" in result

    def test_empty_results_message_includes_lesson_filter(self):
        store = make_mock_store()
        store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)

        result = tool.execute(query="content", lesson_number=5)

        assert "lesson 5" in result

    def test_returns_error_string_when_store_returns_error(self):
        store = make_mock_store()
        store.search.return_value = SearchResults.empty("No course found matching 'Unknown'")
        tool = CourseSearchTool(store)

        result = tool.execute(query="content", course_name="Unknown")

        assert "No course found matching" in result

    # --- last_sources ---

    def test_last_sources_initially_empty(self):
        store = make_mock_store()
        tool = CourseSearchTool(store)
        assert tool.last_sources == []

    def test_last_sources_populated_with_label_and_url(self):
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "MCP Course", "lesson_number": 5}],
        )
        store.get_lesson_link.return_value = "https://example.com/lesson/5"
        tool = CourseSearchTool(store)

        tool.execute(query="content", course_name="MCP", lesson_number=5)

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "MCP Course - Lesson 5"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson/5"

    def test_sources_deduplication_for_multiple_chunks_same_lesson(self):
        """Two chunks from the same lesson → exactly one source entry."""
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["chunk 1", "chunk 2"],
            metas=[
                {"course_title": "MCP Course", "lesson_number": 5},
                {"course_title": "MCP Course", "lesson_number": 5},
            ],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="content")

        assert len(tool.last_sources) == 1

    def test_sources_from_different_lessons_not_deduplicated(self):
        """Two chunks from different lessons → two source entries."""
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["chunk 1", "chunk 2"],
            metas=[
                {"course_title": "MCP Course", "lesson_number": 4},
                {"course_title": "MCP Course", "lesson_number": 5},
            ],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="content")

        assert len(tool.last_sources) == 2

    def test_sources_url_is_none_when_chunk_has_no_lesson_number(self):
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "MCP Course"}],  # no lesson_number key
        )
        tool = CourseSearchTool(store)

        tool.execute(query="content")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["url"] is None

    def test_get_lesson_link_called_with_correct_args(self):
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["content"],
            metas=[{"course_title": "MCP Course", "lesson_number": 5}],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="content")

        store.get_lesson_link.assert_called_once_with("MCP Course", 5)

    def test_get_lesson_link_called_once_per_unique_label(self):
        """get_lesson_link should only be called once per unique label (deduplication)."""
        store = make_mock_store()
        store.search.return_value = make_results(
            docs=["chunk 1", "chunk 2"],
            metas=[
                {"course_title": "MCP Course", "lesson_number": 5},
                {"course_title": "MCP Course", "lesson_number": 5},
            ],
        )
        tool = CourseSearchTool(store)

        tool.execute(query="content")

        assert store.get_lesson_link.call_count == 1
