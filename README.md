# GraphRAG - Part 2: Multi-Strategy Retrieval Agent

*Last update: February 18, 2026*

A production-grade **GraphRAG** question-answering agent built with **LangGraph**, **Gemini**, **Neo4j**, and **Pinecone**. Given a natural-language question about electronic music, the agent autonomously selects a retrieval strategy, gathers evidence from a knowledge graph and vector store, and synthesizes a cited, source-attributed answer.

This is **Part 2** of the GraphRAG project. [Part 1](scripts/README.md) is the Dagster data pipeline that builds the knowledge graph and vector store from Wikidata, MusicBrainz, Last.fm, and Wikipedia.

<!-- TODO: Add screenshot of the Chainlit chat UI -->
<!-- ![Chat UI](docs/chainlit_ui.png) -->

## Domain: Electronic Music

The agent operates over a knowledge base of **4,600+ electronic music artists** spanning nearly a century of electronic music history — from early pioneers to contemporary producers across Techno, House, Ambient, IDM, Drum & Bass, and dozens of other sub-genres. The underlying data includes artist biographies, discographies (98,000+ releases), genre taxonomies, geographic origins, and community structures detected via the Leiden algorithm.

## How It Works

The agent implements an **Intelligent Router** that classifies each user query into one of five retrieval strategies, then orchestrates a multi-node retrieval pipeline that merges evidence from heterogeneous sources before generating a final answer with inline citations.

### Retrieval Strategies

| Strategy | When Used | Retrieval Path |
|---|---|---|
| **Local** | Specific entities (artists, albums, cities) | Entity search + NL-to-Cypher in parallel |
| **Global** | Broad thematic questions, genre movements | Community summaries (Leiden hierarchy) |
| **Drift** | Multi-hop reasoning, influence chains | Semantic relation traversal via DRIFT |
| **Structural** | Quantitative queries (counts, lists, filters) | LLM-generated Cypher with EXPLAIN guardrail |
| **Hybrid** | Complex queries needing both context and facts | Entity search + Community search combined |

A **fast-track** mode bypasses deep retrieval for simple single-entity factoid lookups.

### Agent Graph

```
START
  -> router              (Gemini classifies strategy + reasoning plan)
  -> embed_query          (Snowflake arctic-embed-s, 384-dim via HF API)
  -> [conditional parallel dispatch]
     |
     |-- Path A: Entity Discovery
     |   -> entity_search          (dual fulltext + vector search in Neo4j, PageRank-sorted)
     |   -> neighborhood_expand    (DRIFT: cosine similarity on relationship embeddings)
     |
     |-- Path B: Community Discovery
     |   -> community_search       (Pinecone: Leiden L1+L2 community summaries)
     |   -> community_members_search (Neo4j: member entities of discovered communities)
     |
     |-- Path C: Structural Discovery
     |   -> nl_to_cypher           (Gemini generates Cypher, EXPLAIN validation, 1 retry)
     |
  -> chunk_search         (Pinecone: graduated filter cascade, authority-led re-ranking)
  -> resolve_sources      (QID -> Wikipedia URL via Polars lazy parquet scan)
  -> synthesize_answer    (Gemini: USA Framework with inline [N] citations)
END
```

<!-- TODO: Add LangGraph Studio visualization -->
<!-- ![Agent Graph](docs/langgraph_studio.png) -->

### The USA Framework (Unified Source Attribution)

All retrieval results — graph entities, relationships, community summaries, text chunks, and Cypher query results — are homogenized into **Atomic Knowledge Units (AKUs)**. Each AKU receives an importance score based on relevance, mention frequency, and PageRank. The top AKUs (capped at 12) are numbered and passed to the synthesis LLM, which produces an answer with inline `[N]` citations.

A **resolve step** maps each citation to its origin (Graph DB or Vector DB), retrieval method, entity name, and a clickable Wikipedia URL, appended as a **Sources & Evidence Path** legend.

A heuristic **faithfulness check** detects low citation density and hallucinated references.

### Key Retrieval Techniques

- **Dual-path entity search**: Fulltext (Lucene) and vector search run in parallel against Neo4j, results merged and deduplicated with relative score pruning (80% threshold)
- **DRIFT expansion**: Semantic cosine similarity traversal over relationship embeddings in Neo4j for multi-hop reasoning
- **Graduated filter cascade**: Chunk search tries four progressively looser filter combinations before falling back to open search
- **Authority-led re-ranking**: `boosted_score = raw_score * log1p(pagerank)` prioritizes well-connected entities
- **EXPLAIN guardrail**: Generated Cypher is validated via `EXPLAIN` before execution, with error-feedback retry

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) (StateGraph, conditional edges, parallel dispatch) |
| **LLM** | [Google Gemini 2.0 Flash](https://ai.google.dev/) (routing, Cypher generation, synthesis) |
| **Knowledge Graph** | [Neo4j Aura](https://neo4j.com/cloud/aura/) (fulltext + vector indexes, Cypher) |
| **Vector Store** | [Pinecone](https://www.pinecone.io/) (two indexes: `chunks` + `community-summaries`) |
| **Embeddings** | [Snowflake arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) (384-dim, Neo4j) + [Llama Text Embed v2](https://docs.pinecone.io/guides/inference/understanding-inference#embedding-models) (1024-dim, Pinecone) |
| **Chat UI** | [Chainlit](https://chainlit.io/) (streaming step-by-step execution display) |
| **Config & Secrets** | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| **HTTP Client** | [curl-cffi](https://github.com/yifeikong/curl_cffi) |
| **Logging** | [Structlog](https://www.structlog.org/) |
| **Data Processing** | [Polars](https://pola.rs/) (lazy parquet scans for source resolution) |
| **Deployment** | [Docker](https://www.docker.com/) on [Google Cloud Run](https://cloud.google.com/run) |
| **Language** | [Python 3.13+](https://www.python.org/) |
| **Tooling** | [uv](https://docs.astral.sh/uv/), [Ruff](https://docs.astral.sh/ruff/), [pytest](https://docs.pytest.org/) |

## Architecture

The project follows a strict **separation of concerns** with dependency injection throughout:

```
src/agent/
├── graph.py              # LangGraph StateGraph definition (edges, routing)
├── state.py              # Graph state (additive reducers for parallel merging)
├── configuration.py      # Runtime config (model, retrieval_k, depth, thresholds)
├── settings.py           # Environment secrets via pydantic-settings
├── nodes/                # "The What" — graph nodes (business logic)
│   ├── generation.py     # Router + Synthesis (LLM-powered)
│   ├── retrieval.py      # Entity, community, chunk, neighborhood search
│   ├── structural.py     # NL-to-Cypher with EXPLAIN guardrail
│   └── models.py         # Pydantic response schemas for structured LLM output
├── tools/                # Pure functions — 100% reusable, no global config
│   ├── gemini.py         # Gemini API wrapper (structured + unstructured)
│   ├── embeddings.py     # Query embedding via HuggingFace Inference API
│   ├── knowledge_graph.py# Neo4j Cypher execution helpers
│   ├── vector_store.py   # Pinecone query + embed helpers
│   ├── source_resolver.py# QID -> Wikipedia URL resolution
│   └── synthesis.py      # AKU importance scoring, citation legend, faithfulness
└── infrastructure/       # Client singletons — the only place that reads secrets
    ├── clients.py        # Wiring: instantiates all clients from settings
    ├── gemini_client.py  # Lazy Gemini client with schema-injected system prompt
    ├── neo4j_client.py   # Per-event-loop AsyncDriver cache with weakref GC cleanup
    ├── pinecone_client.py# Lazy Pinecone client singleton
    └── hf_embedding_client.py  # curl-cffi HTTP client for HF Inference API
```

### Design Principles

- **Tools are pure**: All `tools/*.py` accept clients as arguments — zero imports from `settings` or `SecretStr`
- **Parallel-safe state**: Additive reducers (`operator.add`) on list fields allow LangGraph to safely merge results from concurrent nodes
- **Two embedding spaces**: 384-dim Snowflake for Neo4j entity/relation vectors; 1024-dim Llama for Pinecone document/community vectors — each optimized for its retrieval context
- **Schema-grounded LLM**: The Neo4j graph schema (`graph_schema.json`) is injected as a Gemini system instruction, grounding both the router and Cypher generator in the actual data model
- **Per-event-loop Neo4j caching**: The async Neo4j driver is cached per event loop with `weakref` finalizers, critical for LangGraph's async execution model

## Data Sources

The agent queries against data built by the [Part 1 pipeline](scripts/README.md):

| Source | Content | Size |
|---|---|---|
| **Neo4j Knowledge Graph** | Artists, releases, tracks, genres, countries + relationships | 98,677 nodes, 123,574 edges |
| **Pinecone `chunks` index** | Wikipedia article chunks with metadata | ~30,000 documents |
| **Pinecone `community-summaries` index** | Leiden community summaries (L0/L1/L2) | Hierarchical community reports |
| **Local Parquet files** | Source URL lookup tables, community metadata | `data_volume/assets/` |

## Getting Started

### Prerequisites

- **Python 3.13+**
- [**uv**](https://docs.astral.sh/uv/) (Astral's Python package manager)
- A **Neo4j Aura** instance (with the Part 1 knowledge graph loaded)
- A **Pinecone** account (with `chunks` and `community-summaries` indexes)
- A **Google Gemini** API key
- A **HuggingFace** access token

### Installation

```bash
git clone <repository-url>
cd graph_rag_part2
uv sync
```

### Environment Variables

Create a `.env` file in the project root:

```env
# Neo4j Aura
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>

# Google Gemini
GEMINI_API_KEY=<your-gemini-api-key>

# HuggingFace (for Snowflake embeddings)
HUGGING_FACE_HUB_TOKEN=<your-hf-token>

# Pinecone
PINECONE_API_KEY=<your-pinecone-api-key>
```

### Running the Agent

**Option 1: Chainlit Chat UI** (production)

```bash
uv run chainlit run app.py
```

Open [http://localhost:8080](http://localhost:8080) to interact with the agent. Each query displays a real-time step-by-step trace of the retrieval pipeline.

**Option 2: LangGraph Studio** (development & debugging)

```bash
langgraph dev
```

This launches the visual debugging IDE where you can inspect state at each node, edit past states, and replay from any point in the graph.

### Running with Docker

```bash
docker build -t graphrag-agent .
docker run -p 8080:8080 --env-file .env graphrag-agent
```

The Docker image uses a multi-stage build with `uv` for dependency resolution and deploys to **Google Cloud Run**.

### Running Tests

```bash
uv run pytest
```

Integration tests (requiring live backends) are marked separately:

```bash
uv run pytest -m integration      # Only integration tests
uv run pytest -m "not integration" # Only unit tests
```

## Example Queries

| Query | Strategy | What Happens |
|---|---|---|
| *"Tell me about Kraftwerk"* | Local (fast-track) | Direct entity lookup, minimal retrieval |
| *"What characterizes German techno artists?"* | Global | Community summaries for German electronic clusters |
| *"How did Detroit techno influence Berlin's scene?"* | Drift | Multi-hop DRIFT expansion across influence relationships |
| *"How many artists are from France?"* | Structural | NL-to-Cypher: `MATCH (a:Artist)-[:FROM_COUNTRY]->(c:Country {name: 'France'}) RETURN count(a)` |
| *"Compare the evolution of house and techno"* | Hybrid | Community summaries + entity details combined |

## Project Links

- **Part 1 — Data Pipeline**: [scripts/README.md](scripts/README.md) (Dagster pipeline that builds the knowledge graph)
- **Dataset**: [pacoreyes/electronic-music-wikipedia-rag](https://huggingface.co/datasets/pacoreyes/electronic-music-wikipedia-rag) on HuggingFace
- **GraphRAG Paper**: [From Local to Global (Microsoft, 2024)](https://arxiv.org/abs/2404.16130)

## License

MIT License. (C) 2025-2026 Juan-Francisco Reyes.
