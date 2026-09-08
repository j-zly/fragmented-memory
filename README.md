# Keepsake — Memory Plugin for Hermes Agent

The Keepsake system automatically retrieves relevant memories and injects them into the conversation context for each dialogue.

```text
User: "How did we set up that React project structure last time?"
                      ↓
         Keepsake System     ← Redis + RediSearch
                      ↓
    ┌─────────────────────────────────────┐
    │  [1] User prefers TypeScript + Vite    │
    │  [2] Previous projects used pinia state management │
    │  [3] Backend suggested using .NET 10 implementation       │
    └─────────────────────────────────────┘
                      ↓
         Model directly uses memories to answer
```

## Features

- **Full Entry Storage** — stores complete text as-is, no semantic splitting
- **BM25 Full-Text Search** — works out of the box with no external API
- **Optional Vector Search** — KNN via RediSearch (OpenAI / DashScope embedder)
- **Time Decay** — newer entries rank higher (60-day half-life configurable)
- **Sentiment Weighting** — emotional entries get priority
- **User Feedback** — mark entries useful/useless to improve ranking
- **Hot Topic Boost** — frequently discussed topics rank higher
- **Entity Extraction** — auto-tags entries with entities (people, places, crypto tickers, domain terms) at store time; searched alongside content text for higher recall
- **Entity Co-occurrence** — auto-track which entities appear together, expand search to co-occurring entities for associative recall ("Python" → also finds entries mentioning "Django")
- **Domain Dictionary** — jieba user dictionary auto-generated from corpus + synonym table, loaded on `/new` for better Chinese tokenization
- **Workflow Lock** — set `keepsake:workflow_lock` in Redis to globally disable memory retrieval (e.g. during automated workflows)
- **Skip Patterns** — define skip lists (via file) to avoid searching on trivial queries like "ok", "got it"
- **On-Demand Storage** — only `memory(action='add')` stores data; no automatic per-turn archiving
- **Search-Time Expiry** — `invalid_at` field in index: set a timestamp and the entry is filtered out at search time (no data loss, can be reverted)
- **Auto Maintenance** — consolidation (keyword clustering + LLM summarization) + selective forgetting (multi-dimension low-value detection) run every 2h to keep storage tidy
- **RRF Fusion Ranking (v1.3)** — Reciprocal Rank Fusion combines BM25 full-text and semantic KNN results into a single ranked list for better recall
- **Local Semantic Search** — optional ollama `nomic-embed-text` embedder (768-dim) runs fully on-premise, no external embedding API needed
- **Time-Aware Recall (v1.5)** — entity timelines (`keepsake:entity_timeline`) + versioned facts let searches leverage *when* things happened, not just what was said
- **Local Memory Distillation** — `scripts/memory_distill.py` uses a local model (qwen3:8b) to distill stale entries into compact summaries with watermark incremental updates (toggleable, ComfyUI off-peak aware)
- **Retrieval Quality Spot Checks** — `scripts/eval_spotcheck.py` runs 20 real-query regression tests to track search quality (v1.4: 60% → 67%)
- **Auto-Registered Cron Jobs** — when used as a Hermes plugin, three cron jobs (memory maintenance every 2h, deduplication every 1h, synonym discovery every 8h) are automatically registered on plugin initialization — zero manual setup
- **Hermes Plugin Wrapper** — ready-to-use `hermes-plugin/` directory with `plugin.yaml` and `__init__.py` for drop-in installation
- **Two-Phase Pipeline (v2, 2026-09)** — Mem0-style extract/update phases run async on a daemon thread, sealing old facts via `superseded_by` edges (not physical delete) so stale snapshots (e.g. completed tasks marked pending) never reappear. Cost-capped at `llm_pipeline.max_calls_per_window`; any LLM/JSON failure or budget breach falls back to v1 rule-gate storage — messages are never silently dropped.

## Design Philosophy: Clean Memory for LLMs

Keepsake stores **full, self-contained entries** — not split conversation snippets. The key insight is that LLMs need complete context to make use of stored information. An entry like "prefers TypeScript + Vite" without its surrounding context is useless; the full entry "User prefers TypeScript + Vite for frontend projects" is immediately actionable.

| Mechanism | Implementation |
|-----------|---------------|
| Complete Context | Stores full text entries, no splitting |
| Forgetting Curve | Time decay (60-day half-life) — old memories fade naturally |
| Emotion Deepens Memory | Emotional weight boost — intense moments stick |
| Repetition Reinforces | Attention tracking + hot topic scoring |
| Use It or Lose It | Feedback reinforcement (keepsake_feedback) |
| Association & Analogy | Synonym discovery (Jaccard co-occurrence statistics) — "deploy" ↔ "release" |
| Entity Association | Entity co-occurrence tracking — entries mentioning "BTC" also recall "halving" without being synonyms |
| Entity Tagging | Like the brain tagging memories with people/places/things — auto-extracted entities searched alongside content |
| On-Demand Storage | No automatic archiving; only saves when explicitly told to (memory tool) |
| Sleep Consolidation | Background maintenance every 2h: keyword-based clustering + LLM summarization |
| Context Isolation | agent_id tagging — different identities, separate memories |
| Fuzzy but Enough | BM25 full-text search — doesn't need an exact match to recall |

No vector database. No embedding API calls. No LLM inference for memory operations. Just **pure statistical methods** running on Redis + RediSearch — the same techniques the brain uses: frequency, recency, emotional salience, association, and feedback.

## Requirements

- **Python 3.10+**
- **Hermes Agent 0.12+** — provides `MemoryProvider` interface
- **Redis 7+** — with RediSearch module (v2.6+)
- **jieba** — Chinese tokenization (auto-installed)
- **Embedding API** (optional) — OpenAI / DashScope / any compatible `/v1/embeddings` service

## Installation

```bash
pip install keepsake
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/j-zly/keepsake.git
```

## Configuration

Configuration precedence (high to low): **Environment variables > JSON config file > config.yaml inline > defaults**

### 1. Configuration Methods

There are three ways to configure Keepsake, listed in order of priority:

1. **Environment Variables** (Highest precedence)  
   Set environment variables like `KEEPSAKE_REDIS_HOST`, `KEEPSAKE_REDIS_PASSWORD`, etc.

2. **JSON Config File** (~/.config/keepsake/config.json)  
   A complete JSON configuration file for all settings.

3. **Code Defaults** (Lowest precedence)  
   Default values defined in the code.

### 2. Complete Configuration Example

Here's a comprehensive example of the configuration file `~/.config/keepsake/config.json` with all available options:

```json
{
  // Redis connection
  "redis_host": "127.0.0.1",
  "redis_port": 6379,
  "redis_password": "",
  
  // Search settings
  "top_k": 5,
  "candidate_k": 10,
  "bm25_limit": 10,
  "tag_filter": "",

  // Skip patterns
  "skip_min_length": 2,
  "skip_patterns_file": "~/.config/keepsake/skip_patterns.txt",

  // Time decay
  "decay_half_days": 60,
  "hot_topic_decay_half_days": 30,
  
  // Ranking weights
  "sentiment_boost_positive": 1.5,
  "sentiment_boost_negative": 1.3,
  "feedback_positive_boost": 1.3,
  "feedback_negative_penalty": 0.5,
  "hot_topic_boost": 1.2,
  
  // Attention
  "attention_boost_max": 1.5,
  "attention_base_increment": 2.0,
  "attention_emotion_factor": 1.5,
  
  // Embedding (optional)
  "embedder": {
    "provider": "dashscope",
    "api_key": "sk-xxx",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "text-embedding-v2"
  },

  // LLM channel (consolidator + v2 pipeline) — empty = dashscope fallback
  // base_url: chat completions endpoint root (no trailing /v1 for zhipu etc.)
  // model: e.g. glm-4-flash, qwen-plus. api_key (testing) OR key_file (prod)
  "llm": {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4-flash", "key_file": "/path/to/key.pass"},

  // Auto maintenance
  "consolidate_min_group": 2,
  "consolidate_max_age_hours": 72,
  "forget_max_age_days": 30,
  "forget_dry_run": false,

  // Agent isolation
  "agent_id": "main-brain",
  "is_primary": true,

  // Synonym discovery
  "synonym_min_word_freq": 10,
  "synonym_jaccard_threshold": 0.5,
  "synonym_min_co_occurrence": 3,

  // Entity co-occurrence
  "entity_cooc_top_n": 3,
  "entity_cooc_min_count": 2,
  
  // Emotion intensity factor
  "emotion_intensity_factor": 0.4
}
```

> Note: Redis password compatibility: leave empty for no authentication, or provide password to automatically send AUTH command.

### 3. Environment Variables Reference

| Environment Variable | Corresponding Config Item | Description |
|----------------------|----------------------------|-------------|
| `KEEPSAKE_REDIS_HOST` | `redis_host` | Redis server host |
| `KEEPSAKE_REDIS_PORT` | `redis_port` | Redis server port |
| `KEEPSAKE_REDIS_PASSWORD` | `redis_password` | Redis password for authentication |
| `KEEPSAKE_TOP_K` | `top_k` | Number of final entries returned |
| `KEEPSAKE_CANDIDATE_K` | `candidate_k` | Candidate entries count (for KNN) |
| `KEEPSAKE_BM25_LIMIT` | `bm25_limit` | BM25 search candidate count |
| `KEEPSAKE_TAG_FILTER` | `tag_filter` | Tag filtering (comma-separated) |
| `KEEPSAKE_DECAY_HALF_DAYS` | `decay_half_days` | Time decay half-life (days) |
| `KEEPSAKE_HOT_TOPIC_DECAY_HALF_DAYS` | `hot_topic_decay_half_days` | Hot topic time decay half-life (days) |
| `KEEPSAKE_EMBED_CACHE_TTL` | `embed_cache_ttl` | Embedding cache TTL (seconds) |
| `KEEPSAKE_EMBEDDER` | `embedder.provider` | Embedding provider (`openai`, `dashscope`) |
| `KEEPSAKE_EMBEDDER_URL` | `embedder.base_url` | Embedding API endpoint |
| `KEEPSAKE_EMBEDDER_MODEL` | `embedder.model` | Embedding model name |
| `KEEPSAKE_CONSOLIDATE_MIN_GROUP` | `consolidate_min_group` | Minimum entries to trigger consolidation |
| `KEEPSAKE_CONSOLIDATE_MAX_AGE_HOURS` | `consolidate_max_age_hours` | Minimum age (hours) before entries can be consolidated |
| `KEEPSAKE_FORGET_MAX_AGE_DAYS` | `forget_max_age_days` | Number of days before entries might be forgotten |
| `KEEPSAKE_FORGET_DRY_RUN` | `forget_dry_run` | Safe mode: `true` = count only, `false` = actually delete |
| `KEEPSAKE_EMOTION_INTENSITY_FACTOR` | `emotion_intensity_factor` | Emotion intensity → weight coefficient (0=disabled, 1=max) |

> Note: Redis password is compatible with empty value (no auth) or password provided for AUTH command.  
> Note: Changes to config.json take effect immediately without restarting (just send `/new`).

### 4. Create Redis Index (first-time usage)

The code will auto-create (`ensure_index()`), or execute manually:

```bash
redis-cli FT.CREATE idx:memories ON HASH PREFIX 1 "memory:frag:" SCHEMA \
    content TEXT WEIGHT 1 \
    tags TAG SEPARATOR "," \
    category TAG SEPARATOR "," \
    source TEXT WEIGHT 1 \
    created TEXT WEIGHT 0 \
    entry_type TAG SEPARATOR "," \
    invalid_at TAG SEPARATOR "," \
    entities TAG SEPARATOR "," \
    embed_bin VECTOR FLAT 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE
```

> Dimension (DIM) is dynamically adjusted based on the embedding model used, default 1536.
> For Docker: `docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest`

### 5. Hermes Configuration

Enable in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: keepsake
```

If `embedder` is not configured, only BM25 full-text search mode will be used.

Also supports environment variable configuration (highest precedence):

```bash
export KEEPSAKE_REDIS_HOST=127.0.0.1
export KEEPSAKE_REDIS_PORT=6379
export KEEPSAKE_TOP_K=5
export KEEPSAKE_EMBEDDER=dashscope
export KEEPSAKE_EMBEDDER_MODEL=text-embedding-v2
export OPENAI_API_KEY=sk-xxx        # embedder API key
```

### 6. Workflow Lock

Temporarily disable memory retrieval during automated workflows (like batch processing):

```bash
# Lock (3600s TTL)
redis-cli SET keepsake:workflow_lock 1 EX 3600

# Unlock
redis-cli DEL keepsake:workflow_lock
```

### 7. Skip Patterns File

Create a file (one pattern per line, `#` for comments):

```text
# ~/.config/keepsake/skip_patterns.txt
好的
嗯
对
是
哦
可以
没错
ok
okay
yes
yeah
```

Then reference it in config.json:

```json
{
  "skip_min_length": 2,
  "skip_patterns_file": "~/.config/keepsake/skip_patterns.txt"
}
```

### 8. Restart Gateway

```bash
# For CLI mode, restart session is sufficient
# For Gateway mode, restart the process
```

## Configuration Reference

| Config Item | Environment Variable | Default Value | Description |
|-------------|---------------------|---------------|-------------|
| `redis_host` | `KEEPSAKE_REDIS_HOST` | `127.0.0.1` | Redis address |
| `redis_port` | `KEEPSAKE_REDIS_PORT` | `6379` | Redis port |
| `top_k` | `KEEPSAKE_TOP_K` | `5` | Number of final entries returned |
| `candidate_k` | `KEEPSAKE_CANDIDATE_K` | `10` | Candidate entries count (for KNN) |
| `tag_filter` | `KEEPSAKE_TAG_FILTER` | `""` | Tag filtering (comma-separated) |
| `bm25_limit` | `KEEPSAKE_BM25_LIMIT` | `10` | BM25 search candidate count |
| `decay_half_days` | `KEEPSAKE_DECAY_HALF_DAYS` | `60` | Time decay half-life (days) |
| `embed_cache_ttl` | `KEEPSAKE_EMBED_CACHE_TTL` | `3600` | Embedding cache TTL (seconds) |
| `sentiment_boost_positive` | — | `1.5` | Positive entry weight multiplier |
| `sentiment_boost_negative` | — | `1.3` | Negative entry weight multiplier |
| `feedback_positive_boost` | — | `1.3` | Positive feedback bonus weight |
| `feedback_negative_penalty` | — | `0.5` | Negative feedback penalty coefficient |
| `hot_topic_boost` | — | `1.2` | Hot topic weighting multiplier |
| `embedder.provider` | `KEEPSAKE_EMBEDDER` | `openai` | `openai` / `dashscope` |
| `embedder.api_key` | `OPENAI_API_KEY` | — | Embedding API key |
| `embedder.base_url` | `KEEPSAKE_EMBEDDER_URL` | `https://api.openai.com/v1` | API endpoint |
| `embedder.model` | `KEEPSAKE_EMBEDDER_MODEL` | `text-embedding-3-small` | Embedding model name |
| `consolidate_min_group` | — | `2` | Minimum entries to trigger consolidation |
| `consolidate_max_age_hours` | — | `72` | Minimum age (hours) before consolidation |
| `forget_max_age_days` | — | `30` | Max age (days) before deletion |
| `forget_dry_run` | — | `true` | Safe mode: `true` = count only, `false` = delete |
| `agent_id` | — | `""` | Agent identity tag for isolation (e.g. `"main-brain"`) |
| `is_primary` | — | `false` | `true` = sees all entries; `false` = only tagged ones |
| `hot_topic_decay_half_days` | — | `30` | Hot topic time decay half-life (days) |
| `emotion_intensity_factor` | — | `0.4` | Emotion intensity → weight coefficient |
| `skip_min_length` | — | `2` | Minimum query length to trigger search |
| `skip_patterns_file` | — | `""` | Path to skip patterns file |
| `attention_boost_max` | — | `1.5` | Max attention weighting value |
| `attention_base_increment` | — | `2.0` | Base attention increment per mention |
| `attention_emotion_factor` | — | `1.5` | Emotion amplification for attention |
| `synonym_min_word_freq` | — | `10` | Min frequency for synonym candidate |
| `synonym_jaccard_threshold` | — | `0.5` | Jaccard threshold for synonym detection |
| `synonym_min_co_occurrence` | — | `3` | Min co-occurrence for synonym detection |
| `entity_cooc_top_n` | — | `3` | Number of co-occurring entities to expand search |
| `entity_cooc_min_count` | — | `2` | Min co-occurrence for entity association |

> `sentiment_*`, `feedback_*`, `hot_topic_*` and other ranking weight parameters currently only support configuration through JSON config file, not environment variables. Set to `1.0` to disable the effect of that dimension.

### Embedding Models and Dimensions

| Model | Dimensions |
|-------|------------|
| OpenAI text-embedding-3-small | 1536 |
| OpenAI text-embedding-3-large | 3072 |
| OpenAI text-embedding-ada-002 | 1536 |
| DashScope text-embedding-v2 | 1536 |
| DashScope text-embedding-v3 | 1024 |

Dimensions are automatically detected, switching models doesn't require reconfiguration.

### Synonym Table

Stored in Redis Hash `keepsake:synonyms`, expanded at search time to improve recall:

```bash
redis-cli HSET keepsake:synonyms setup '["install","configure","deploy","setup"]'
redis-cli HSET keepsake:synonyms fix '["fix","modify","correct","repair","solve"]'
```

### Ingest Gate v1 (2026-09)

Write-side gate (`src/keepsake/ingest_gate.py`) intercepts every `sync_turn()` call before storage. Rules short-circuit in order: **R1** `[CONTEXT COMPACTION` prefix → reject; **R2** `len > max_len` (default 2000) → reject; **R3** blacklist / status questions / stripped length < 8 → reject; **R4** no letter/Chinese/digit at all → reject; **R5** `category="memory_tool"` → reject (MEMORY.md ↔ fragment library decoupled); **R6** same content hash already exists for `turn_memory`/`conversation` → `update_state` only (bump `updated_at`/`touch_count`, never overwrite `content`); **R7** else → store. Toggle in `config.json` under `"ingest_gate": {"enabled": true, "max_len": 2000}`. Defaults: enabled.

## Verification

Check logs after startup:

```
Memory provider 'keepsake' registered (0 tools)
keepsake: connected (session=xxx, top_k=5, tag_filter=(none))
keepsake: BM25-only mode (no embedder configured)
keepsake: auto-registered cron job 'memory-maintenance'
keepsake: auto-registered cron job 'synonym-discovery-daily'
keepsake: auto-registered cron job '记忆去重'
```

## Project Structure

```
keepsake/
├── src/keepsake/         # Python package — the memory provider
├── hermes-plugin/        # Hermes plugin wrapper (drop-in for ~/.hermes/plugins/)
│   ├── plugin.yaml
│   └── __init__.py
├── cron/                 # Cron wrapper scripts for scheduled tasks
│   ├── memory-maintenance.py   # Every 2h — consolidation + forgetting
│   ├── dedup-memory.sh         # Every 1h — deduplication
│   └── discover-synonyms.py    # Every 8h — synonym auto-discovery
├── scripts/              # Standalone utility scripts (dev/test)
├── README.md
└── pyproject.toml
```

The three cron jobs in `cron/` are **auto-registered** when the keepsake plugin initializes (on `/new` or gateway restart). No manual `hermes cron create` needed.

## v2 Two-Phase Pipeline (2026-09)

每轮对话走两相 LLM：**提取相**从窗口提炼长期事实（废话/纯确认自然无事实可提而消亡），**更新相**对每条 fact 拿 top-5 相似旧记忆做 ADD / UPDATE / DELETE / NOOP —— UPDATE/DELETE 给旧碎片只打 `superseded_by` 封边不物理删，检索侧统一排除；窗口内 LLM 调用硬顶 `llm_pipeline.max_calls_per_window`（默认 8），任一 LLM/JSON 失败或超预算 → 窗口整体回落 v1 规则闸门直存原文，**绝不静默丢消息**。配置键：`llm_pipeline.{enabled, model, window_pairs, window_seconds, max_calls_per_window, update_top_k, recent_context_size}`、`v2_min_score`。

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    User sends message                  │
└──────────────────┬─────────────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │   prefetch()       │  ← Automatically triggered on every user message
         │   ↓                │
         │  Workflow Lock?    │  ← Checks keepsake:workflow_lock
         │   ↓                │
         │  Skip patterns?    │  ← Length / exact match against skip list
         │   ↓                │
         │  BM25 Full-Text Search │  ← Default, zero cost, searches full entries
         │  (KNN Vector search) │  ← Optional (needs embedder)
         │  Entity co-occurrence │  ← Expand query with co-occurring entities
         │   ↓                │
         │  Six-dimensional Re-ranking │  ← Similarity × Time decay
         │                    │    × Emotion × Feedback × Hot Topic × Attention
         │   ↓                │
         │  Top N Injected into Context   │  ← Full entries returned as-is
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │   Model Response   │  ← Entries used directly (complete text)
         └───────────────────┘
                   │
         ┌─────────▼─────────┐
         │   on_memory_write()│  ← Only on memory(action='add')
         │   Stores Full Text │  ← Complete entry, no splitting
         │   Entity Extraction│  ← jieba + regex → entities TAG field
         │   Entity Co-occur. │  ← Track entity pairs in ZSET
         │   Attention Track  │  ← Extract keywords, increase attention score
         │   ↓                │
         │   Stored in Redis  │  ← Available for next retrieval as full text
         └───────────────────┘
                   │
         ┌─────────▼─────────┐
         │   [cron] Every 2h     │  ← Background maintenance
         │   ① Multi-level Consolidation  │  ← Same topic → keyword clustering → LLM → level+1
         │   ② Selective Forgetting  │  ← Age>30d + no feedback + low emotion + low attention → delete
         └───────────────────┘
```

## License

MIT
