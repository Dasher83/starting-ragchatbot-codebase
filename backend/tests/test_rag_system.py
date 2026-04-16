"""
Tests for RAGSystem.query() content-query handling in rag_system.py.

Mocking strategy:
- Patch rag_system.VectorStore and rag_system.AIGenerator at class construction.
- Re-wire tool_manager with REAL CourseSearchTool / CourseOutlineTool backed by
  the mock store so source tracking runs through actual code paths.
- SessionManager is not patched — it's lightweight in-memory state.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
import pytest

from rag_system import RAGSystem
from vector_store import SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_config():
    config = MagicMock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "/tmp/test_chroma"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.ANTHROPIC_API_KEY = "test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_HISTORY = 2
    return config


def make_search_results(docs=None, metas=None):
    docs = docs or ["Lesson content here."]
    metas = metas or [{"course_title": "MCP Course", "lesson_number": 5}]
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


class RAGFixture:
    """Builds a RAGSystem with mocked I/O components for each test."""

    def __init__(self):
        with patch("rag_system.VectorStore") as mock_vs_cls, \
             patch("rag_system.AIGenerator") as mock_ai_cls, \
             patch("rag_system.DocumentProcessor"):
            self.mock_vector_store = MagicMock()
            self.mock_ai_generator = MagicMock()
            mock_vs_cls.return_value = self.mock_vector_store
            mock_ai_cls.return_value = self.mock_ai_generator
            self.rag = RAGSystem(make_mock_config())

        # Re-wire tool_manager with real tools backed by the mock store
        self.rag.tool_manager = ToolManager()
        self.rag.search_tool = CourseSearchTool(self.mock_vector_store)
        self.rag.outline_tool = CourseOutlineTool(self.mock_vector_store)
        self.rag.tool_manager.register_tool(self.rag.search_tool)
        self.rag.tool_manager.register_tool(self.rag.outline_tool)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGSystemQuery:

    def setup_method(self):
        self.fx = RAGFixture()

    # --- basic answer retrieval ---

    def test_query_returns_answer_string(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Lesson 5 covered tool calling."

        answer, _ = self.fx.rag.query("What was covered in lesson 5 of the MCP course?")

        assert answer == "Lesson 5 covered tool calling."

    def test_query_returns_two_element_tuple(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Answer."

        result = self.fx.rag.query("Any question?")

        assert isinstance(result, tuple)
        assert len(result) == 2

    # --- tools passed to AIGenerator ---

    def test_query_passes_search_course_content_tool(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Answer."

        self.fx.rag.query("What was covered in lesson 5?")

        call_kwargs = self.fx.mock_ai_generator.generate_response.call_args.kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search_course_content" in tool_names

    def test_query_passes_get_course_outline_tool(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Answer."

        self.fx.rag.query("What is the outline of the MCP course?")

        call_kwargs = self.fx.mock_ai_generator.generate_response.call_args.kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "get_course_outline" in tool_names

    def test_query_passes_tool_manager_to_ai_generator(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Answer."

        self.fx.rag.query("Question?")

        call_kwargs = self.fx.mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["tool_manager"] is self.fx.rag.tool_manager

    # --- sources flow ---

    def test_query_returns_sources_populated_by_search_tool(self):
        self.fx.mock_vector_store.search.return_value = make_search_results()
        self.fx.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/5"

        # Simulate the AI generator invoking search_course_content
        def fake_generate(query, conversation_history=None, tools=None, tool_manager=None):
            if tool_manager:
                tool_manager.execute_tool(
                    "search_course_content",
                    query="lesson 5",
                    course_name="MCP",
                    lesson_number=5,
                )
            return "Lesson 5 covered tool calling."

        self.fx.mock_ai_generator.generate_response.side_effect = fake_generate

        answer, sources = self.fx.rag.query("What was covered in lesson 5 of the MCP course?")

        assert answer == "Lesson 5 covered tool calling."
        assert len(sources) == 1
        assert sources[0]["label"] == "MCP Course - Lesson 5"
        assert sources[0]["url"] == "https://example.com/lesson/5"

    def test_query_returns_empty_sources_when_no_tool_called(self):
        self.fx.mock_ai_generator.generate_response.return_value = "General answer from knowledge."

        _, sources = self.fx.rag.query("What is machine learning?")

        assert sources == []

    def test_query_resets_sources_after_retrieval(self):
        """Sources on the search tool should be cleared after each query."""
        self.fx.mock_vector_store.search.return_value = make_search_results()

        def fake_generate(query, conversation_history=None, tools=None, tool_manager=None):
            if tool_manager:
                tool_manager.execute_tool("search_course_content", query="test", course_name="MCP")
            return "Answer."

        self.fx.mock_ai_generator.generate_response.side_effect = fake_generate

        self.fx.rag.query("First query")

        assert self.fx.rag.search_tool.last_sources == []
        assert self.fx.rag.outline_tool.last_sources == []

    # --- session management ---

    def test_query_saves_exchange_to_session(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Answer about lesson 5."

        session_id = self.fx.rag.session_manager.create_session()
        self.fx.rag.query("What was covered in lesson 5?", session_id=session_id)

        history = self.fx.rag.session_manager.get_conversation_history(session_id)
        assert "What was covered in lesson 5?" in history
        assert "Answer about lesson 5." in history

    def test_query_passes_conversation_history_to_ai_generator(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Answer."

        session_id = self.fx.rag.session_manager.create_session()
        self.fx.rag.session_manager.add_exchange(
            session_id, "Previous question", "Previous answer"
        )

        self.fx.rag.query("Follow-up question", session_id=session_id)

        call_kwargs = self.fx.mock_ai_generator.generate_response.call_args.kwargs
        history = call_kwargs.get("conversation_history")
        assert history is not None
        assert "Previous" in history

    def test_query_without_session_does_not_pass_history(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Answer."

        self.fx.rag.query("Any question?", session_id=None)

        call_kwargs = self.fx.mock_ai_generator.generate_response.call_args.kwargs
        history = call_kwargs.get("conversation_history")
        assert history is None

    # --- prompt construction ---

    def test_query_wraps_question_in_prompt(self):
        self.fx.mock_ai_generator.generate_response.return_value = "Answer."

        self.fx.rag.query("What was covered in lesson 5?")

        call_kwargs = self.fx.mock_ai_generator.generate_response.call_args.kwargs
        query_arg = call_kwargs["query"]
        assert "What was covered in lesson 5?" in query_arg
