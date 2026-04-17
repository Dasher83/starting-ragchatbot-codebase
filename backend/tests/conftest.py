"""
Shared fixtures for the RAG chatbot test suite.

conftest.py is auto-loaded by pytest, making all fixtures here available to
every test module without explicit imports.

Design:
  mock_rag  – a pre-configured MagicMock that stands in for RAGSystem.
  test_app  – a minimal FastAPI app wired to mock_rag.  It mirrors the routes
              in app.py but omits the ChromaDB initialisation, docs-loading
              startup event, and the frontend static-file mount, all of which
              require resources that don't exist in the test environment.
  client    – a synchronous TestClient wrapping test_app.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic models (mirrored from app.py so the test app validates identically)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    label: str
    url: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag():
    """Return a MagicMock pre-configured with sensible defaults for RAGSystem."""
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "test-session-id"
    rag.session_manager.sessions = {}
    rag.query.return_value = ("Test answer.", [])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    return rag


@pytest.fixture
def test_app(mock_rag):
    """
    Return a FastAPI app whose routes mirror app.py but use mock_rag instead
    of a real RAGSystem.  No static files are mounted.
    """
    app = FastAPI(title="Test RAG App")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        mock_rag.session_manager.clear_session(session_id)
        mock_rag.session_manager.sessions.pop(session_id, None)
        return {"status": "ok"}

    return app


@pytest.fixture
def client(test_app):
    """Return a synchronous TestClient wrapping the test FastAPI app."""
    return TestClient(test_app)
