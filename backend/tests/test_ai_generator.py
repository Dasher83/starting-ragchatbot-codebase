"""
Tests for AIGenerator.generate_response() and _handle_tool_execution()
in ai_generator.py.

Mocking strategy: patch 'ai_generator.anthropic.Anthropic' so the real SDK
client is never constructed. Fake Message/ContentBlock objects are plain
MagicMocks with the required attributes (stop_reason, content, type, text,
id, name, input).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch, call
import pytest

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_text_block(text="The answer."):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(tool_name, tool_input, tool_id="toolu_001"):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = tool_name
    block.input = tool_input
    return block


def make_text_response(text="The answer."):
    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [make_text_block(text)]
    return response


def make_tool_use_response(tool_name, tool_input, tool_id="toolu_001"):
    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [make_tool_use_block(tool_name, tool_input, tool_id)]
    return response


def build_generator():
    """Return (AIGenerator instance, mock_client) without touching the real SDK."""
    with patch("ai_generator.anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
    # The patch is gone but generator.client already holds mock_client
    return generator, mock_client


def make_tool_manager(tool_result="Search results."):
    tm = MagicMock()
    tm.execute_tool.return_value = tool_result
    return tm


SAMPLE_TOOLS = [
    {"name": "search_course_content", "description": "Search content", "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_course_outline", "description": "Get outline", "input_schema": {"type": "object", "properties": {}}},
]


# ---------------------------------------------------------------------------
# Tests: direct-answer path (no tool use)
# ---------------------------------------------------------------------------

class TestDirectAnswerPath:

    def test_returns_text_from_content_block(self):
        generator, mock_client = build_generator()
        mock_client.messages.create.return_value = make_text_response("Direct answer here.")

        result = generator.generate_response(query="What is machine learning?")

        assert result == "Direct answer here."

    def test_only_one_api_call_when_no_tool_used(self):
        generator, mock_client = build_generator()
        mock_client.messages.create.return_value = make_text_response("Answer.")

        generator.generate_response(query="General question?")

        assert mock_client.messages.create.call_count == 1

    def test_query_appears_in_messages(self):
        generator, mock_client = build_generator()
        mock_client.messages.create.return_value = make_text_response("Answer.")

        generator.generate_response(query="What is RAG?")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        messages = call_kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        assert "What is RAG?" in user_message["content"]

    def test_system_prompt_present_in_api_call(self):
        generator, mock_client = build_generator()
        mock_client.messages.create.return_value = make_text_response("Answer.")

        generator.generate_response(query="Hello?")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

    def test_system_prompt_includes_conversation_history(self):
        generator, mock_client = build_generator()
        mock_client.messages.create.return_value = make_text_response("Answer.")

        generator.generate_response(
            query="Follow-up question?",
            conversation_history="User: Hello\nAssistant: Hi!",
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system = call_kwargs["system"]
        assert "Hello" in system
        assert "Hi!" in system

    def test_no_tools_in_api_call_when_tools_not_provided(self):
        generator, mock_client = build_generator()
        mock_client.messages.create.return_value = make_text_response("Answer.")

        generator.generate_response(query="General question?")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    def test_tools_included_when_provided(self):
        generator, mock_client = build_generator()
        mock_client.messages.create.return_value = make_text_response("Answer.")

        generator.generate_response(query="Course question?", tools=SAMPLE_TOOLS)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == {"type": "auto"}


# ---------------------------------------------------------------------------
# Tests: tool-use path (two API calls)
# ---------------------------------------------------------------------------

class TestToolUsePath:

    def test_two_api_calls_made_when_tool_invoked(self):
        generator, mock_client = build_generator()
        first = make_tool_use_response("search_course_content", {"query": "lesson 5"})
        second = make_text_response("Lesson 5 covered tool calling.")
        mock_client.messages.create.side_effect = [first, second]

        generator.generate_response(
            query="What was covered in lesson 5 of MCP?",
            tools=SAMPLE_TOOLS,
            tool_manager=make_tool_manager(),
        )

        assert mock_client.messages.create.call_count == 2

    def test_returns_text_from_second_api_call(self):
        generator, mock_client = build_generator()
        first = make_tool_use_response("search_course_content", {"query": "lesson 5"})
        second = make_text_response("Lesson 5 covered tool calling.")
        mock_client.messages.create.side_effect = [first, second]

        result = generator.generate_response(
            query="What was covered in lesson 5 of MCP?",
            tools=SAMPLE_TOOLS,
            tool_manager=make_tool_manager(),
        )

        assert result == "Lesson 5 covered tool calling."

    def test_calls_tool_manager_execute_for_search_content(self):
        generator, mock_client = build_generator()
        tool_input = {"query": "lesson 5 content", "course_name": "MCP", "lesson_number": 5}
        first = make_tool_use_response("search_course_content", tool_input)
        second = make_text_response("Answer.")
        mock_client.messages.create.side_effect = [first, second]

        tm = make_tool_manager()
        generator.generate_response(
            query="What was covered in lesson 5 of MCP?",
            tools=SAMPLE_TOOLS,
            tool_manager=tm,
        )

        tm.execute_tool.assert_called_once_with(
            "search_course_content",
            query="lesson 5 content",
            course_name="MCP",
            lesson_number=5,
        )

    def test_calls_tool_manager_execute_for_get_outline(self):
        generator, mock_client = build_generator()
        first = make_tool_use_response("get_course_outline", {"course_title": "MCP"})
        second = make_text_response("The MCP course has 5 lessons.")
        mock_client.messages.create.side_effect = [first, second]

        tm = make_tool_manager("Course: MCP\nLessons:\n  Lesson 1: Intro")
        generator.generate_response(
            query="What is the outline of MCP?",
            tools=SAMPLE_TOOLS,
            tool_manager=tm,
        )

        tm.execute_tool.assert_called_once_with("get_course_outline", course_title="MCP")

    def test_tool_result_included_in_second_api_call_messages(self):
        generator, mock_client = build_generator()
        first = make_tool_use_response("search_course_content", {"query": "lesson 5"}, tool_id="toolu_abc")
        second = make_text_response("Final answer.")
        mock_client.messages.create.side_effect = [first, second]

        tm = make_tool_manager("Relevant content from lesson 5.")
        generator.generate_response(
            query="What was covered in lesson 5?",
            tools=SAMPLE_TOOLS,
            tool_manager=tm,
        )

        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        # Find the tool_result entry
        tool_result = None
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        tool_result = item
                        break

        assert tool_result is not None
        assert tool_result["tool_use_id"] == "toolu_abc"
        assert tool_result["content"] == "Relevant content from lesson 5."

    def test_mid_loop_api_call_includes_tools(self):
        """After one tool round the mid-loop call still carries tools so Claude can
        decide whether to invoke another tool or answer directly."""
        generator, mock_client = build_generator()
        first = make_tool_use_response("search_course_content", {"query": "lesson 5"})
        second = make_text_response("Answer.")
        mock_client.messages.create.side_effect = [first, second]

        result = generator.generate_response(
            query="Lesson query?",
            tools=SAMPLE_TOOLS,
            tool_manager=make_tool_manager(),
        )

        assert result == "Answer."
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" in second_call_kwargs
        assert second_call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tool_call_when_tool_manager_is_none(self):
        """If tool_manager is None, tool_use response falls through to content[0].text."""
        generator, mock_client = build_generator()
        # stop_reason is tool_use but no tool_manager → tries to return content[0].text
        # The content[0] is a ToolUseBlock (no .text attribute on our mock)
        tool_block = make_tool_use_block("search_course_content", {"query": "test"})
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [tool_block]
        mock_client.messages.create.return_value = response

        # MagicMock auto-creates .text so this won't raise — just verifies no second call
        generator.generate_response(query="Test?", tools=SAMPLE_TOOLS, tool_manager=None)

        assert mock_client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Tests: edge cases that reveal latent bugs
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_final_response_content_raises_descriptive_error(self):
        """
        When the Claude API returns empty content after tool use, a ValueError
        with a descriptive message is raised (not a bare IndexError).
        """
        generator, mock_client = build_generator()
        first = make_tool_use_response("search_course_content", {"query": "lesson 5"})
        empty_response = MagicMock()
        empty_response.stop_reason = "end_turn"
        empty_response.content = []
        mock_client.messages.create.side_effect = [first, empty_response]

        with pytest.raises(ValueError, match="Unexpected response from Claude API after tool use"):
            generator.generate_response(
                query="What was covered in lesson 5?",
                tools=SAMPLE_TOOLS,
                tool_manager=make_tool_manager(),
            )

    def test_empty_direct_response_content_raises_descriptive_error(self):
        """
        When the Claude API returns empty content on the direct-answer path,
        a ValueError with a descriptive message is raised.
        """
        generator, mock_client = build_generator()
        empty_response = MagicMock()
        empty_response.stop_reason = "end_turn"
        empty_response.content = []
        mock_client.messages.create.return_value = empty_response

        with pytest.raises(ValueError, match="Unexpected response from Claude API"):
            generator.generate_response(query="Hello?")


# ---------------------------------------------------------------------------
# Tests: sequential tool calling (up to 2 rounds)
# ---------------------------------------------------------------------------

class TestSequentialToolCalling:

    def test_two_tool_rounds_three_api_calls(self):
        """Two sequential tool rounds produce 3 API calls and 2 tool executions."""
        generator, mock_client = build_generator()
        first = make_tool_use_response("get_course_outline", {"course_title": "MCP"}, tool_id="toolu_001")
        second = make_tool_use_response("search_course_content", {"query": "tool calling"}, tool_id="toolu_002")
        third = make_text_response("Here is the complete answer.")
        mock_client.messages.create.side_effect = [first, second, third]

        result = generator.generate_response(
            query="Find a course discussing the same topic as lesson 4 of MCP.",
            tools=SAMPLE_TOOLS,
            tool_manager=make_tool_manager(),
        )

        assert mock_client.messages.create.call_count == 3
        assert make_tool_manager().execute_tool.call_count == 0  # sanity — real tm below
        tm = make_tool_manager()
        mock_client.messages.create.side_effect = [first, second, third]
        generator.generate_response(
            query="Find a course discussing the same topic as lesson 4 of MCP.",
            tools=SAMPLE_TOOLS,
            tool_manager=tm,
        )
        assert tm.execute_tool.call_count == 2
        assert result == "Here is the complete answer."

    def test_two_tool_rounds_tools_present_in_mid_calls_absent_in_synthesis(self):
        """Tools are included in the first two API calls, excluded in the final synthesis."""
        generator, mock_client = build_generator()
        first = make_tool_use_response("get_course_outline", {"course_title": "MCP"}, tool_id="toolu_001")
        second = make_tool_use_response("search_course_content", {"query": "tool calling"}, tool_id="toolu_002")
        third = make_text_response("Final answer.")
        mock_client.messages.create.side_effect = [first, second, third]

        generator.generate_response(
            query="Multi-step query.",
            tools=SAMPLE_TOOLS,
            tool_manager=make_tool_manager(),
        )

        call_list = mock_client.messages.create.call_args_list
        assert "tools" in call_list[0].kwargs          # initial call
        assert "tools" in call_list[1].kwargs          # mid-loop round-2 call
        assert "tools" not in call_list[2].kwargs      # final synthesis
        assert "tool_choice" not in call_list[2].kwargs

    def test_single_tool_round_claude_answers_directly(self):
        """When Claude answers directly after 1 tool round, exactly 2 API calls are made."""
        generator, mock_client = build_generator()
        first = make_tool_use_response("search_course_content", {"query": "lesson 5"})
        second = make_text_response("Lesson 5 covered tool calling.")
        mock_client.messages.create.side_effect = [first, second]

        result = generator.generate_response(
            query="What was in lesson 5?",
            tools=SAMPLE_TOOLS,
            tool_manager=make_tool_manager(),
        )

        assert mock_client.messages.create.call_count == 2
        assert result == "Lesson 5 covered tool calling."

    def test_max_rounds_cap_enforced(self):
        """Even if Claude keeps requesting tools, execute_tool is called at most MAX_TOOL_ROUNDS times."""
        generator, mock_client = build_generator()
        # Supply more tool_use responses than MAX_TOOL_ROUNDS would allow
        r1 = make_tool_use_response("get_course_outline", {"course_title": "A"}, tool_id="toolu_001")
        r2 = make_tool_use_response("search_course_content", {"query": "topic"}, tool_id="toolu_002")
        r3 = make_tool_use_response("search_course_content", {"query": "more"}, tool_id="toolu_003")
        synthesis = make_text_response("Capped answer.")
        mock_client.messages.create.side_effect = [r1, r2, r3, synthesis]

        tm = make_tool_manager()
        generator.generate_response(
            query="Multi-step query.",
            tools=SAMPLE_TOOLS,
            tool_manager=tm,
        )

        assert tm.execute_tool.call_count == generator.MAX_TOOL_ROUNDS

    def test_tool_execution_error_terminates_loop_gracefully(self):
        """A tool execution exception ends the loop and a synthesis call is still made."""
        generator, mock_client = build_generator()
        first = make_tool_use_response("search_course_content", {"query": "lesson 5"}, tool_id="toolu_err")
        synthesis = make_text_response("I was unable to retrieve that information.")
        mock_client.messages.create.side_effect = [first, synthesis]

        tm = MagicMock()
        tm.execute_tool.side_effect = RuntimeError("DB unavailable")

        result = generator.generate_response(
            query="What was in lesson 5?",
            tools=SAMPLE_TOOLS,
            tool_manager=tm,
        )

        assert mock_client.messages.create.call_count == 2
        assert result == "I was unable to retrieve that information."

        # Synthesis call messages must contain an is_error tool_result
        synthesis_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        messages = synthesis_kwargs["messages"]
        error_result = None
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("is_error"):
                        error_result = item
        assert error_result is not None
        assert error_result["tool_use_id"] == "toolu_err"

    def test_message_alternation_after_two_rounds(self):
        """After 2 tool rounds the synthesis call receives properly alternating messages."""
        generator, mock_client = build_generator()
        first = make_tool_use_response("get_course_outline", {"course_title": "MCP"}, tool_id="toolu_001")
        second = make_tool_use_response("search_course_content", {"query": "topic"}, tool_id="toolu_002")
        third = make_text_response("Final.")
        mock_client.messages.create.side_effect = [first, second, third]

        generator.generate_response(
            query="Multi-step query.",
            tools=SAMPLE_TOOLS,
            tool_manager=make_tool_manager(),
        )

        synthesis_messages = mock_client.messages.create.call_args_list[2].kwargs["messages"]
        # Expected: [user_query, assistant_round1, user_results1, assistant_round2, user_results2]
        assert len(synthesis_messages) == 5
        assert synthesis_messages[0]["role"] == "user"
        assert synthesis_messages[1]["role"] == "assistant"
        assert synthesis_messages[2]["role"] == "user"
        assert synthesis_messages[3]["role"] == "assistant"
        assert synthesis_messages[4]["role"] == "user"
