# Work Log

## 2025-02-12 — Wire Chainlit to LangGraph

### Goal
Create a Chainlit chat UI that invokes the existing LangGraph `graph` on each user message.

### Changes

| Action | File | Description |
|--------|------|-------------|
| CREATE | `app.py` | Chainlit entry point with `on_chat_start`, `on_message`, and `_extract_ai_response` helper. Maintains per-session conversation history and delegates to the compiled LangGraph graph. |
| CREATE | `chainlit.md` | Welcome screen markdown displayed by Chainlit on chat start. |
| CREATE | `tests/agent/test_app.py` | 7 tests: 4 for `_extract_ai_response` (pure function), 1 for `on_chat_start` (session init), 2 for `on_message` (graph invocation + fallback). |
| EDIT | `.gitignore` | Added `.chainlit/` to prevent tracking auto-generated Chainlit config. |
| EDIT | `pyproject.toml` | Added `structlog>=25.1.0` as explicit dependency. |

### Architecture Decisions

- **`app.py` at project root** — Chainlit discovers it by default; it is an entry point, not part of the `agent` package.
- **`graph.ainvoke` (non-streaming)** — Simplest integration; switching to `graph.astream` later is a single-line change.
- **Session history via `cl.user_session`** — Conversation history is maintained per session and synced with the graph output after each invocation.
- **`_extract_ai_response` as a pure helper** — Extracts the last `AIMessage` from the graph result; easy to test without Chainlit context.
- **No `settings` import in `app.py`** — Respects the DI rules for entry points.

### Testing Notes

- **Chainlit mocking caveat**: `unittest.mock.patch("app.cl")` fails because Chainlit's `cl` module has a lazy-loading `__getattr__` that raises `KeyError` when `_is_async_obj` introspects for `__func__`. Solution: use `monkeypatch.setattr(app_module, "cl", SimpleNamespace(...))` instead.
- **Import ordering**: `app` is classified as first-party by ruff, so it must be grouped with local imports, not third-party.

### Verification

- `uv run python -m pytest tests/agent/test_app.py -v` — **7/7 passed**
- `uv run python -m ruff check app.py tests/agent/test_app.py` — **all checks passed**
- Manual smoke test: `chainlit run app.py` — UI loads, responds with placeholder ("No response generated." since `call_model` returns empty messages).

## 2025-02-13 — Architectural Analysis & RAG Foundation

### Goal
Assess the current LangGraph project structure and identify gaps for implementing the reasoning/RAG orchestration.

### Current Status
- **Folder Structure**: Highly adequate. Follows "Src Layout" with clear separation between `infrastructure`, `nodes`, `tools`, and `configuration`.
- **Infrastructure**: `GeminiClient`, `Neo4jClient`, and `PineconeClient` are implemented with dependency injection.
- **Tools**: Basic `vector_search` and `query_knowledge_graph` functions are ready in `src/agent/tools/`.
- **UI**: Chainlit is wired to the LangGraph `graph` via `app.py`.

### Identified Gaps
1. **Nodes**: `src/agent/nodes/retrieval.py` and `src/agent/nodes/generation.py` are skeletons (empty files).
2. **Graph Orchestration**: `src/agent/graph.py` only contains a placeholder `call_model` node. It needs to be refactored to include the RAG flow (`START -> retrieval -> generation -> END`).
3. **Reasoning Logic**: Need to implement query analysis/routing logic to decide when to use Vector vs. Graph search.

### Plan for Next Phase
1. Implement the `retrieval` node using Gemini for query embedding and Pinecone/Neo4j for searching.
2. Implement the `generation` node to synthesize answers from retrieved context.
3. Update `graph.py` to orchestrate these nodes.
4. Enhance `State` and `Configuration` to support advanced RAG parameters (e.g., top-k, hybrid search thresholds).
