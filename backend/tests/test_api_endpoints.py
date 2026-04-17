"""
Tests for the FastAPI HTTP endpoints defined in app.py.

Uses the `client` and `mock_rag` fixtures from conftest.py.  The test app
mirrors the real routes but substitutes a MagicMock for RAGSystem, so no
ChromaDB, embedding model, or Anthropic API key is required.
"""


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_returns_200_with_valid_request(self, client):
        response = client.post("/api/query", json={"query": "What is RAG?"})
        assert response.status_code == 200

    def test_response_contains_answer_field(self, client):
        response = client.post("/api/query", json={"query": "What is RAG?"})
        assert "answer" in response.json()

    def test_response_contains_session_id_field(self, client):
        response = client.post("/api/query", json={"query": "What is RAG?"})
        assert "session_id" in response.json()

    def test_response_contains_sources_field(self, client):
        response = client.post("/api/query", json={"query": "What is RAG?"})
        assert "sources" in response.json()

    def test_answer_matches_rag_system_return_value(self, client, mock_rag):
        mock_rag.query.return_value = ("RAG stands for retrieval-augmented generation.", [])
        response = client.post("/api/query", json={"query": "What is RAG?"})
        assert response.json()["answer"] == "RAG stands for retrieval-augmented generation."

    def test_auto_creates_session_when_none_provided(self, client, mock_rag):
        client.post("/api/query", json={"query": "Hello?"})
        mock_rag.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(self, client, mock_rag):
        client.post("/api/query", json={"query": "Hello?", "session_id": "existing-session"})
        mock_rag.session_manager.create_session.assert_not_called()

    def test_session_id_in_response_matches_provided_value(self, client):
        response = client.post("/api/query", json={"query": "Hello?", "session_id": "my-session"})
        assert response.json()["session_id"] == "my-session"

    def test_session_id_in_response_is_generated_value_when_not_provided(self, client, mock_rag):
        mock_rag.session_manager.create_session.return_value = "generated-session-abc"
        response = client.post("/api/query", json={"query": "Hello?"})
        assert response.json()["session_id"] == "generated-session-abc"

    def test_sources_returned_with_label_and_url(self, client, mock_rag):
        mock_rag.query.return_value = (
            "Answer with source.",
            [{"label": "MCP Course - Lesson 3", "url": "https://example.com/lesson/3"}],
        )
        response = client.post("/api/query", json={"query": "What is MCP?"})
        sources = response.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["label"] == "MCP Course - Lesson 3"
        assert sources[0]["url"] == "https://example.com/lesson/3"

    def test_sources_empty_list_when_no_tool_called(self, client, mock_rag):
        mock_rag.query.return_value = ("General answer.", [])
        response = client.post("/api/query", json={"query": "General question?"})
        assert response.json()["sources"] == []

    def test_source_url_can_be_null(self, client, mock_rag):
        mock_rag.query.return_value = ("Answer.", [{"label": "Course A - Lesson 1", "url": None}])
        response = client.post("/api/query", json={"query": "Tell me about Course A."})
        assert response.json()["sources"][0]["url"] is None

    def test_rag_query_called_with_correct_query_text(self, client, mock_rag):
        client.post("/api/query", json={"query": "Tell me about lesson 5."})
        call_args = mock_rag.query.call_args
        assert call_args.args[0] == "Tell me about lesson 5."

    def test_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={"session_id": "abc"})
        assert response.status_code == 422

    def test_rag_exception_returns_500(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("DB unavailable")
        response = client.post("/api/query", json={"query": "Crash?"})
        assert response.status_code == 500

    def test_500_response_contains_error_detail(self, client, mock_rag):
        mock_rag.query.side_effect = RuntimeError("DB unavailable")
        response = client.post("/api/query", json={"query": "Crash?"})
        assert "DB unavailable" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_returns_200(self, client):
        response = client.get("/api/courses")
        assert response.status_code == 200

    def test_response_contains_total_courses(self, client):
        response = client.get("/api/courses")
        assert "total_courses" in response.json()

    def test_response_contains_course_titles(self, client):
        response = client.get("/api/courses")
        assert "course_titles" in response.json()

    def test_total_courses_matches_analytics_value(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["A", "B", "C", "D", "E"],
        }
        response = client.get("/api/courses")
        assert response.json()["total_courses"] == 5

    def test_course_titles_match_analytics_value(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["MCP Course", "Python Basics"],
        }
        response = client.get("/api/courses")
        assert response.json()["course_titles"] == ["MCP Course", "Python Basics"]

    def test_empty_catalog_returns_zero_courses(self, client, mock_rag):
        mock_rag.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}
        response = client.get("/api/courses")
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_analytics_exception_returns_500(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("ChromaDB down")
        response = client.get("/api/courses")
        assert response.status_code == 500

    def test_500_response_contains_error_detail(self, client, mock_rag):
        mock_rag.get_course_analytics.side_effect = RuntimeError("ChromaDB down")
        response = client.get("/api/courses")
        assert "ChromaDB down" in response.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:

    def test_returns_200(self, client):
        response = client.delete("/api/session/some-session")
        assert response.status_code == 200

    def test_response_body_is_ok_status(self, client):
        response = client.delete("/api/session/some-session")
        assert response.json() == {"status": "ok"}

    def test_calls_clear_session_with_correct_id(self, client, mock_rag):
        client.delete("/api/session/session-xyz")
        mock_rag.session_manager.clear_session.assert_called_once_with("session-xyz")

    def test_removes_session_from_sessions_dict(self, client, mock_rag):
        mock_rag.session_manager.sessions = {"session-xyz": {"history": []}}
        client.delete("/api/session/session-xyz")
        assert "session-xyz" not in mock_rag.session_manager.sessions

    def test_delete_nonexistent_session_still_returns_200(self, client, mock_rag):
        mock_rag.session_manager.sessions = {}
        response = client.delete("/api/session/nonexistent")
        assert response.status_code == 200
