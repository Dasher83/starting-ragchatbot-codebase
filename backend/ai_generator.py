import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information.

Search Tool Usage:
- Use `search_course_content` for questions about specific course content or detailed educational materials
- Use `get_course_outline` for questions about a course's structure, lesson list, or overview
- **Up to 2 sequential tool calls per query** — use a second call only when the first result is needed to form the second query
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific content questions**: Use search_course_content, then answer
- **Outline/structure questions**: Use get_course_outline and return the course title, course link, and the number and title of each lesson
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        if not response.content or not hasattr(response.content[0], "text"):
            raise ValueError(
                f"Unexpected response from Claude API (stop_reason={response.stop_reason!r}, "
                f"content_length={len(response.content)})"
            )
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle sequential tool execution up to MAX_TOOL_ROUNDS rounds.

        Each round appends the assistant's tool-use content and the tool results
        to the message history. After all rounds, a final synthesis call is made
        without tools. If Claude answers directly mid-loop (stop_reason != tool_use),
        that response is returned immediately without an extra synthesis call.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = base_params["messages"].copy()
        tools = base_params.get("tools")
        system = base_params["system"]

        current_response = initial_response

        for round_num in range(self.MAX_TOOL_ROUNDS):
            # Append assistant's tool-use content to history
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute every tool_use block in this response
            tool_results = []
            error_occurred = False
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {e}",
                            "is_error": True
                        })
                        error_occurred = True
                        break
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result
                    })

            # Append tool results as a user message
            messages.append({"role": "user", "content": tool_results})

            # On error or last allowed round, fall through to synthesis
            if error_occurred or round_num == self.MAX_TOOL_ROUNDS - 1:
                break

            # Make next API call with tools still available
            next_response = self.client.messages.create(
                **self.base_params,
                messages=messages,
                system=system,
                tools=tools,
                tool_choice={"type": "auto"}
            )

            if next_response.stop_reason != "tool_use":
                # Claude answered directly — return without an extra synthesis call
                if not next_response.content or not hasattr(next_response.content[0], "text"):
                    raise ValueError(
                        f"Unexpected response from Claude API after tool use "
                        f"(stop_reason={next_response.stop_reason!r}, "
                        f"content_length={len(next_response.content)})"
                    )
                return next_response.content[0].text

            current_response = next_response

        # Final synthesis call — tools intentionally excluded
        final_response = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=system
        )
        if not final_response.content or not hasattr(final_response.content[0], "text"):
            raise ValueError(
                f"Unexpected response from Claude API after tool use "
                f"(stop_reason={final_response.stop_reason!r}, "
                f"content_length={len(final_response.content)})"
            )
        return final_response.content[0].text