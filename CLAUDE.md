# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

Always use `uv` to run the server — never use `pip` directly.

All Python commands must be run from the `backend/` directory using `uv run`:

```bash
cd backend
uv run uvicorn app:app --reload --port 8000  # start server
uv run python <script.py>                    # run any Python file
uv add <package>                             # add a dependency
uv sync                                      # sync the environment
```

Or from the project root:

```bash
./run.sh
```

The app is then available at `http://localhost:8000`. FastAPI auto-docs at `http://localhost:8000/docs`.

## Architecture

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials stored as text files in `docs/`.

**Data flow at startup:** `app.py` → `RAGSystem.add_course_folder("../docs")` → `DocumentProcessor` parses each file into `Course` + `CourseChunk` objects → `VectorStore` embeds and stores them in ChromaDB (`backend/chroma_db/`, two collections: `course_catalog` for metadata, `course_content` for searchable chunks).

**Data flow per query:** HTTP POST → `RAGSystem.query()` → `AIGenerator.generate_response()` → Claude API call #1 with the `search_course_content` tool available → if Claude invokes the tool, `CourseSearchTool.execute()` runs a semantic search via `VectorStore.search()` → results returned as a tool result → Claude API call #2 (no tools) synthesizes the final answer → sources extracted from `ToolManager.last_sources` → response returned to client.

**Key design decisions:**
- Conversation history is injected into the **system prompt** (not the messages array) as formatted text. `SessionManager` keeps the last `MAX_HISTORY` (2) exchanges in memory only — sessions are lost on restart.
- The tool is limited to **one search per query** by the system prompt instruction.
- Course name resolution uses a separate vector similarity search against `course_catalog` to handle fuzzy/partial names before filtering `course_content`.
- ChromaDB is persistent on disk; documents are skipped on reload if their course title already exists in the catalog.

**Document format** (`docs/*.txt`):
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<lesson content...>

Lesson 1: <title>
...
```

**Key configuration** (`backend/config.py`):
| Setting | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 800 chars | Max chars per content chunk |
| `CHUNK_OVERLAP` | 100 chars | Sentence overlap between chunks |
| `MAX_RESULTS` | 5 | Max chunks returned per search |
| `MAX_HISTORY` | 2 | Conversation exchanges retained |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | LLM used for generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformers model for embeddings |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence directory |

## Adding a new tool

1. Subclass `Tool` in `backend/search_tools.py`, implementing `get_tool_definition()` and `execute()`
2. Register it in `RAGSystem.__init__()` via `self.tool_manager.register_tool(your_tool)`

The tool definition must follow the Anthropic tool schema. If the tool tracks sources for the UI, add a `last_sources` list attribute — `ToolManager.get_last_sources()` checks for this automatically.
