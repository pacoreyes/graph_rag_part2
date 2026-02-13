# Work Log

## 2026-02-13 — Implementation of the "Surgical Pattern" Retrieval

### Goal
Bridge the gap between the Knowledge Graph (Neo4j) and the Vector Store (Pinecone) by injecting graph-derived metadata (`pagerank`, `entity_type`, `community_id`) into text chunks, allowing for high-precision filtering and authority-led re-ranking.

### Changes

| Action | File | Description |
|--------|------|-------------|
| EDIT | `src/agent/state.py` | Added `target_entity_types` to state to carry filtering instructions from analyzer to retriever. |
| EDIT | `src/agent/nodes/generation.py` | Updated `query_analyzer` to produce structured JSON identifying relevant `target_entity_types` based on the graph schema. |
| EDIT | `src/agent/nodes/retrieval.py` | Refactored `chunk_search` to implement the **Surgical Pattern**: filters by `entity_type`, expands discovery via `community_id`, and performs Authority-Led re-ranking using `pagerank`. |
| EDIT | `tests/agent/nodes/test_generation.py` | Updated tests to validate JSON parsing and type extraction in `query_analyzer`. |
| EDIT | `tests/agent/nodes/test_retrieval.py` | Updated tests to verify PageRank boosting logic and complex multi-factor filtering in `chunk_search`. |

### Architecture Decisions

- **Surgical Filtering** — Instead of blind semantic search, the agent now applies high-precision metadata filters:
    - **Semantic Scoping**: Uses `entity_type` (e.g., PERSON, GENRE) to prevent cross-type noise.
    - **Implicit Evidence Discovery**: Uses `community_id` to fetch context from the same thematic "scene" even if entities aren't explicitly linked in the graph.
- **Authority-Led Re-ranking (Anchor Retrieval)** — Implemented a re-ranking formula `score * log1p(pagerank)` to prioritize "canonical" facts from high-authority entities at the top of the LLM context window.
- **JSON-based Query Analysis** — Switched `query_analyzer` to structured JSON output to reliably extract multiple control signals (strategy + types).

### Verification Results
- **Unit Tests**: 29/29 node tests passed (including new re-ranking and filtering logic).
- **Schema Alignment**: Type extraction in `query_analyzer` is dynamically driven by `graph_schema.json`.

---

## 2025-02-13 — Multi-Strategy GraphRAG Refinement & Schema Integration

### Goal
Refine the GraphRAG agent based on the pipeline agent's architectural feedback, integrate the graph schema for dynamic context, and validate all strategies against live backends.

### Changes

| Action | File | Description |
|--------|------|-------------|
| EDIT | `src/agent/infrastructure/neo4j_client.py` | Implemented per-loop driver caching (`self._drivers: dict[int, AsyncDriver]`) to prevent `RuntimeError: Task got Future attached to a different loop` when running nodes in parallel. Added `max_connection_lifetime=300`. |
| EDIT | `src/agent/nodes/generation.py` | Refined `STRATEGY_PROMPT` with precise classification signals. Integrated `format_schema_for_prompt()` into both Query Analysis and Answer Synthesis. Updated `SYNTHESIS_PROMPT` to follow the Thematic-Factual-Evidence hierarchy. |
| EDIT | `src/agent/nodes/retrieval.py` | Implemented multi-hop expansion in `neighborhood_expand` (Drift strategy). Updated `chunk_search` to use strategy-dependent filtering (Local/Drift uses `article_id`, Global/Hybrid is open). Switched to explicit `pinecone_embed` for Pinecone queries due to missing integrated inference. Fixed Cypher syntax error in `SEMANTIC_RELATION_SEARCH`. |
| EDIT | `src/agent/tools/vector_store.py` | Added `pinecone_embed` function to generate 1024-dim Llama embeddings via Pinecone Inference API. |
| EDIT | `src/agent/tools/knowledge_graph.py` | Integrated `get_graph_schema` and `format_schema_for_prompt` utilities. Combined with original `query_knowledge_graph`. |
| EDIT | `src/agent/infrastructure/nomic_client.py` | Added `trust_remote_code=True` to both model and tokenizer loading to ensure non-interactive execution. |
| EDIT | `src/agent/configuration.py` | Updated default model to `gemini-2.0-flash`. Documented `-1` value for `community_level` (search all levels). |
| EDIT | `src/agent/tools/gemini.py` | Updated default model to `gemini-2.0-flash`. |
| CREATE | `tests/integration_tests/test_multi_strategy.py` | End-to-end validation for Local, Global, Drift, and Hybrid strategies against live Aura/Pinecone. |
| EDIT | `tests/agent/nodes/test_retrieval.py` | Fixed mock keys and updated assertions for 384-dim embeddings. |
| CREATE | `LANG_GRAPH_STATUS.md` | Comprehensive status report documenting current implementation, decisions, and validation results. |

### Architecture Decisions

- **Event Loop Stability** — The Neo4j driver is now thread/loop-safe within the LangGraph parallel execution environment by using `id(asyncio.get_event_loop())` as a cache key.
- **Drift Strategy (Option B)** — Adopting vector similarity (> 0.5) as the primary traversal filter instead of an LLM pre-planner. This maximizes performance while maintaining high precision via relationship embeddings.
- **Explicit Pinecone Embeddings** — Since the `chunks` and `community-summaries` indexes do not have integrated inference models, the agent now explicitly calls `pc.inference.embed` using `llama-text-embed-v2`.
- **Nomic Task Prefix** — Confirmed usage of `search_query: ` prefix for all Neo4j vector lookups.

### Verification Results

- **Integration Tests**: 4/4 strategies passed.
    - **Local**: Successful entity lookup and neighbor retrieval.
    - **Global**: Successful thematic framing using community reports.
    - **Drift**: Successful 1-hop semantic expansion (verified `new_entities=22` in logs).
    - **Hybrid**: Successful parallel execution of both paths.
- **Unit Tests**: 28/28 node tests passed.
- **Gemini 2.0 Flash**: Verified as the most robust model for query classification and synthesis.

### Data Ingestion Status
- **community-summaries**: 100% (18,912 vectors).
- **chunks**: In progress (~131 vectors and climbing). `chunk_search` is wired and will scale automatically.
