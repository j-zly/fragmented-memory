"""
keepsake — Keepsake记忆系统 for Hermes Agent.

每次对话自动检索相关记忆注入上下文，支持：
  - 🔍 向量搜索 — RediSearch KNN 语义检索
  - ⏳ 时间衰减 — 新记忆权重高，旧记忆逐步降权
  - 📝 自动写入 — memory(action='add') 操作自动存档完整内容
  - 🏷️ 标签过滤 — 可选按标签范围搜索

安装: pip install keepsake
激活: config.yaml 中设置 memory.provider: keepsake

配置优先级: 环境变量 > 配置文件 > 默认值
配置文件: ~/.config/keepsake/config.json (或 KEEPSAKE_CONFIG 自定义路径)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
except ImportError:  # 独立测试环境回退；Hermes 实际运行时从 agent 包导入
    class MemoryProvider:
        def __init__(self, *args, **kwargs):
            pass

    def tool_error(msg):
        return f"[tool_error] {msg}"

from .embedder import create_embedder
from .storage import RedisStorage
from .consolidator import Consolidator
from .forgetter import Forgetter
# v2（2026-09）：写侧两相管线（Mem0 风格 + 封边取代）
from .pipeline import Pipeline, DEFAULT_PIPELINE_CONFIG as _LLM_PIPELINE_DEFAULTS

# ---------------------------------------------------------------------------
# 工具扇区（供 Hermes MemoryProvider 注册）
# ---------------------------------------------------------------------------

FEEDBACK_SCHEMA = {
    "name": "keepsake_feedback",
    "description": (
        "记录用户对一条记忆的反馈 — 标记有用/没用。"
        "正反馈让该记忆在未来搜索中排名更高，"
        "负反馈大幅降权（标记为没用的记忆几乎不会再出现）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fragment_key": {
                "type": "string",
                "description": "碎片的 Redis key（如 memory:frag:abc123），从相关碎片的 key 字段获得。",
            },
            "is_positive": {
                "type": "boolean",
                "description": "True = 这条记忆有用，False = 没用",
            },
        },
        "required": ["fragment_key", "is_positive"],
    },
}

HOT_TOPICS_SCHEMA = {
    "name": "keepsake_topics",
    "description": (
        "查询全局热门话题统计。返回跨会话出现最频繁的话题词。"
        "可选日榜/周榜/全局。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "返回条数（默认 10，最大 30）",
                "default": 10,
            },
            "period": {
                "type": "string",
                "enum": ["all", "daily", "weekly"],
                "description": "统计周期：all=全局, daily=日榜, weekly=周榜",
                "default": "all",
            },
        },
        "required": [],
    },
}


logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "~/.config/keepsake/config.json"


def _load_json_config() -> dict:
    """从 JSON 配置文件加载配置。

    路径来源（优先级高到低）:
      1. 环境变量 KEEPSAKE_CONFIG
      2. ~/.config/keepsake/config.json
    文件不存在时返回空 dict。
    """
    path_str = os.environ.get("KEEPSAKE_CONFIG") or _DEFAULT_CONFIG_PATH
    path = Path(path_str).expanduser()
    if not path.exists():
        logger.debug("keepsake: config file not found at %s", path)
        return {}
    try:
        with open(path) as f:
            cfg: dict = json.load(f)
        logger.info("keepsake: loaded config from %s", path)
        return cfg
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("keepsake: failed to load config from %s: %s", path, e)
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并两个 dict，override 覆盖 base。"""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class KeepsakeProvider(MemoryProvider):
    """
    Keepsake记忆提供者。

    和 Hermes builtin 内存共存，不冲突。每轮对话自动检索相关记忆
    注入上下文。仅 memory(action='add') 操作时存储完整内容。

    配置优先级（高→低）:
      1. 环境变量 (KEEPSAKE_REDIS_HOST, KEEPSAKE_EMBEDDER 等)
      2. JSON 配置文件 (~/.config/keepsake/config.json)
      3. config.yaml memory.keepsake 节（由 Hermes 传入）
      4. 硬编码默认值
    """

    _initialized: bool = False
    _storage: Optional[RedisStorage] = None
    _tag_filter: str = ""
    _consolidator: Optional[Consolidator] = None
    _forgetter: Optional[Forgetter] = None
    _last_maintenance: float = 0.0
    _maintenance_interval: float = 7200.0  # 每 2h 跑一次维护

    def __init__(self, **config):
        """
        参数（通过 config.yaml memory 节传入）:

            memory:
              provider: keepsake
                redis_host: 127.0.0.1
                redis_port: 6379
                top_k: 5
                candidate_k: 10
                tag_filter: ""
                embedder:
                  provider: openai
                  api_key: sk-xxx
                  base_url: https://api.openai.com/v1
                  model: text-embedding-3-small
        """
        super().__init__()
        self._config = config

    # ------------------------------------------------------------------
    # 配置合并
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_config(inline_cfg: dict) -> dict:
        """按优先级合并配置源，返回最终配置。

        合并顺序（后覆盖前）: 默认值 ← JSON 文件 ← 环境变量 ← inline
        inline = Hermes 的 config.yaml memory.keepsake 或 __init__ 传参
        """
        # 1. 硬编码默认值（不含 embedder — 由配置文件/环境变量按需开启）
        cfg: dict = {
            "redis_host": "127.0.0.1",
            "redis_port": 6379,
            "redis_password": "",
            "top_k": 5,
            "candidate_k": 10,
            "tag_filter": "",
            "synonym_min_word_freq": 10,
            "synonym_jaccard_threshold": 0.5,
            "synonym_min_co_occurrence": 3,
            "entity_cooc_top_n": 3,
            "entity_cooc_min_count": 2,
            # 写闸门 v1（2026-09）：enabled 默认开、max_len 默认 2000
            "ingest_gate": {"enabled": True, "max_len": 2000},
            # v2 写侧两相管线（2026-09）：默认开启，缺 LLM 时自动回落 v1
            "llm_pipeline": dict(_LLM_PIPELINE_DEFAULTS),
            # v2 检索侧相似度地板（按 _sim 归一化）
            "v2_min_score": 0.05,
        }

        # 2. JSON 配置文件覆盖
        json_cfg = _load_json_config()
        cfg = _deep_merge(cfg, json_cfg)

        # 3. 环境变量覆盖
        env_overrides = {
            "redis_host": os.environ.get("KEEPSAKE_REDIS_HOST"),
            "redis_port": os.environ.get("KEEPSAKE_REDIS_PORT"),
            "redis_password": os.environ.get("KEEPSAKE_REDIS_PASSWORD"),
            "top_k": os.environ.get("KEEPSAKE_TOP_K"),
            "candidate_k": os.environ.get("KEEPSAKE_CANDIDATE_K"),
            "tag_filter": os.environ.get("KEEPSAKE_TAG_FILTER"),
            "agent_id": os.environ.get("KEEPSAKE_AGENT_ID"),
            "is_primary": os.environ.get("KEEPSAKE_IS_PRIMARY"),
        }
        for key, val in env_overrides.items():
            if val is not None:
                cfg[key] = val

        # 4. inline（Hermes 传入的 config.yaml 配置）覆盖
        cfg = _deep_merge(cfg, inline_cfg)

        # 5. 验证 agent_id 必须配置
        agent_id = cfg.get("agent_id")
        if agent_id is None or agent_id == "":
            raise ValueError("agent_id must be configured in config file, environment variable, or inline config")

        # 6. 解析 is_primary，默认为 false
        is_primary = cfg.get("is_primary", False)
        if isinstance(is_primary, str):
            is_primary = is_primary.lower() in ("true", "1", "yes", "on")
        cfg["is_primary"] = bool(is_primary)

        # 7. 加载 skip patterns 配置
        # skip_min_length: int，默认 2，从 config.json 的 skip_min_length 读取
        skip_min_length = cfg.get("skip_min_length", 2)
        cfg["skip_min_length"] = skip_min_length

        # skip_patterns_file: str，默认空字符串，从 config.json 的 skip_patterns_file 读取
        skip_patterns_file = cfg.get("skip_patterns_file", "")
        if skip_patterns_file:
            skip_patterns_file = Path(skip_patterns_file).expanduser()
            if skip_patterns_file.exists():
                try:
                    with open(skip_patterns_file) as f:
                        patterns = set()
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                patterns.add(line.lower())
                    cfg["skip_patterns"] = patterns
                except Exception as e:
                    logger.warning("keepsake: failed to load skip patterns from %s: %s", skip_patterns_file, e)
            else:
                cfg["skip_patterns"] = set()
        else:
            cfg["skip_patterns"] = set()

        return cfg

    def _should_search(self, query: str) -> bool:
        """判断当前用户消息是否需要检索碎片。

        跳过条件：
          1. 长度 < skip_min_length（默认 2）
          2. query 精确匹配外部文件中的 skip pattern（忽略大小写）
        """
        q = query.strip()
        min_len = int(getattr(self, '_skip_min_length', 2))
        if len(q) < min_len:
            return False
        patterns = getattr(self, '_skip_patterns', [])
        if q.lower() in patterns:
            return False
        return True

    # ------------------------------------------------------------------
    # MemoryProvider 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "keepsake"

    def is_available(self) -> bool:
        try:
            import redis as _  # noqa: F401
        except ImportError:
            return False
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        """初始化 — 加载配置、连接 Redis、自动创建 index。"""
        cfg = self._resolve_config(self._config)

        # 加载/重载 jieba 自定义词典（发 /new 时生效）
        from .splitter import init_domain_dict
        init_domain_dict()

        redis_host = cfg.get("redis_host", "127.0.0.1")
        redis_port = int(cfg.get("redis_port", 6379))
        top_k = int(cfg.get("top_k", 5))
        candidate_k = int(cfg.get("candidate_k", 10))
        self._tag_filter = cfg.get("tag_filter", "")

        embed_cfg = cfg.get("embedder", {})
        embed_provider = embed_cfg.get("provider", "").strip().lower()
        # 只有显式配置了 embedder provider 才创建，否则走 BM25-only 模式
        if embed_provider and embed_provider not in ("", "default", "none"):
            embedder = create_embedder(
                provider=embed_cfg.get("provider", ""),
                api_key=embed_cfg.get("api_key", ""),
                base_url=embed_cfg.get("base_url", ""),
                model=embed_cfg.get("model", ""),
            )
            embed_dim = embedder.dimension
            logger.info(
                "keepsake: embedder enabled (%s, dim=%d)",
                embed_provider, embed_dim,
            )
        else:
            embedder = None
            embed_dim = 1536
            logger.info("keepsake: BM25-only mode (no embedder configured)")

        self._storage = RedisStorage(
            embedder=embedder,
            host=redis_host,
            port=redis_port,
            password=cfg.get("redis_password") or None,
            candidate_count=candidate_k,
            final_limit=top_k,
            embed_dim=embed_dim,
            bm25_limit=int(cfg.get("bm25_limit", 10)),
            decay_half_days=int(cfg.get("decay_half_days", 60)),
            embed_cache_ttl=int(cfg.get("embed_cache_ttl", 3600)),
            sentiment_boost_positive=float(cfg.get("sentiment_boost_positive", 1.5)),
            sentiment_boost_negative=float(cfg.get("sentiment_boost_negative", 1.3)),
            feedback_positive_boost=float(cfg.get("feedback_positive_boost", 1.3)),
            feedback_negative_penalty=float(cfg.get("feedback_negative_penalty", 0.5)),
            hot_topic_boost=float(cfg.get("hot_topic_boost", 1.2)),
            hot_topic_decay_half_days=int(cfg.get("hot_topic_decay_half_days", 30)),
            emotion_intensity_factor=float(cfg.get("emotion_intensity_factor", 0.4)),
            attention_boost_max=float(cfg.get("attention_boost_max", 1.5)),
            attention_base_increment=float(cfg.get("attention_base_increment", 2.0)),
            attention_emotion_factor=float(cfg.get("attention_emotion_factor", 1.5)),
            agent_id=cfg.get("agent_id", ""),
            is_primary=cfg.get("is_primary", False),
            synonym_min_word_freq=int(cfg.get("synonym_min_word_freq", 10)),
            synonym_jaccard_threshold=float(cfg.get("synonym_jaccard_threshold", 0.5)),
            synonym_min_co_occurrence=int(cfg.get("synonym_min_co_occurrence", 3)),
            entity_cooc_top_n=int(cfg.get("entity_cooc_top_n", 3)),
            entity_cooc_min_count=int(cfg.get("entity_cooc_min_count", 2)),
            # v2（2026-09）：检索侧相似度地板，按 _sim 归一化值过滤
            v2_min_score=float(cfg.get("v2_min_score", 0.05)),
        )

        # 自动创建/验证 index
        if not self._storage.ensure_index():
            logger.warning(
                "keepsake: Redis / RediSearch not ready at %s:%s",
                redis_host, redis_port,
            )
            return

        self._initialized = True
        logger.info(
            "keepsake: connected (session=%s, top_k=%d, tag_filter=%s)",
            session_id, top_k, self._tag_filter or "(none)",
        )

        # 初始化 skip patterns 配置
        self._skip_min_length = cfg.get("skip_min_length", 2)
        self._skip_patterns = cfg.get("skip_patterns", set())

        # 写闸门配置（ingest_gate v1，2026-09）
        self._gate_cfg = cfg.get("ingest_gate", {"enabled": True, "max_len": 2000})

        # 初始化 Consolidator 和 Forgetter（守护模式）
        self._consolidator = Consolidator(
            storage=self._storage,
            min_group_size=int(cfg.get("consolidate_min_group", 2)),
            max_age_hours=int(cfg.get("consolidate_max_age_hours", 72)),
        )
        self._forgetter = Forgetter(
            storage=self._storage,
            max_age_days=int(cfg.get("forget_max_age_days", 30)),
            dry_run=bool(cfg.get("forget_dry_run", True)),
        )
        logger.info("keepsake: maintenance engines initialized")

        # v2（2026-09）：写侧两相管线 — 仅在 enabled 且有 LLM 时启动
        self._init_pipeline(cfg.get("llm_pipeline", {}) or {})

        # 自动注册定时任务（仅在首次启动时创建）
        self._ensure_cron_jobs()

    # ------------------------------------------------------------------
    # 定时任务自动注册
    # ------------------------------------------------------------------

    CRON_JOBS = [
        {
            "name": "memory-maintenance",
            "script": "memory-maintenance.py",
            "schedule": {"kind": "interval", "minutes": 120, "display": "every 120m"},
        },
        {
            "name": "synonym-discovery-daily",
            "script": "discover_synonyms.py",
            "schedule": {"kind": "cron", "expr": "0 */8 * * *", "display": "0 */8 * * *"},
        },
        {
            "name": "记忆去重",
            "script": "dedup-memory.sh",
            "schedule": {"kind": "cron", "expr": "0 * * * *", "display": "0 * * * *"},
        },
    ]

    @staticmethod
    def _ensure_cron_jobs() -> None:
        """启动时自动注册三条定时任务（不存在才创建），并确保脚本文件到位。"""
        cron_dir = Path("~/.hermes/cron").expanduser()
        scripts_dir = Path("~/.hermes/scripts").expanduser()
        jobs_file = cron_dir / "jobs.json"

        if not jobs_file.exists():
            logger.info("keepsake: cron jobs.json not found, skipping auto-register")
            return

        # 读取现有任务
        try:
            data = json.loads(jobs_file.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("keepsake: failed to read jobs.json")
            return

        jobs = data.get("jobs", [])
        existing_names = {j.get("name") for j in jobs}

        # 获取插件包的 cron/ 目录（作为脚本源）
        pkg_cron = Path(__file__).resolve().parent.parent.parent / "cron"

        now = datetime.now(timezone.utc).isoformat()

        for jdef in KeepsakeProvider.CRON_JOBS:
            name = jdef["name"]
            script = jdef["script"]
            if name in existing_names:
                continue

            # 确保脚本文件存在
            target = scripts_dir / script
            if not target.exists() and pkg_cron.exists():
                src = pkg_cron / script
                if src.exists():
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(src.read_text())
                        target.chmod(0o755)
                        logger.info("keepsake: installed cron script %s", script)
                    except OSError as e:
                        logger.warning("keepsake: failed to install %s: %s", script, e)

            new_job = {
                "id": uuid.uuid4().hex[:12],
                "name": name,
                "prompt": "",
                "skills": [],
                "skill": None,
                "model": None,
                "provider": None,
                "base_url": None,
                "script": script,
                "no_agent": True,
                "context_from": None,
                "schedule": jdef["schedule"].copy(),
                "schedule_display": jdef["schedule"]["display"],
                "repeat": {"times": None, "completed": 0},
                "enabled": True,
                "state": "scheduled",
                "paused_at": None,
                "paused_reason": None,
                "created_at": now,
                "next_run_at": None,
                "last_run_at": None,
                "last_status": None,
                "last_error": None,
                "last_delivery_error": None,
                "deliver": "local",
                "origin": None,
                "enabled_toolsets": None,
                "workdir": None,
                "profile": None,
                "fire_claim": None,
            }
            jobs.append(new_job)
            logger.info("keepsake: auto-registered cron job '%s'", name)

        if any(jdef["name"] not in existing_names for jdef in KeepsakeProvider.CRON_JOBS):
            data["jobs"] = jobs
            data["updated_at"] = now
            try:
                jobs_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
                logger.info("keepsake: cron jobs.json updated")
            except OSError as e:
                logger.warning("keepsake: failed to write jobs.json: %s", e)

    def system_prompt_block(self) -> str:
        parts = [
            "你有Keepsake记忆系统（keepsake），连接在 Redis + RediSearch 上。",
            "当执行 memory(action='add') 操作时，系统会自动存储完整内容并支持后续检索。",
            "相关的记忆条目就在下面「相关记忆」段落里，直接使用即可。",
            "记忆综合排序 = BM25相似度 × 时间衰减 × 情感权重 × 反馈权重 × 热门话题权重。",
            "正反馈用 keepsake_feedback(key, positive=True) 标记有用，",
            "负反馈用 keepsake_feedback(key, positive=False) 标记没用。",
            "热门话题用 keepsake_topics() 查询。",
        ]
        return "\n".join(parts)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """根据用户消息检索相关碎片，注入到上下文。"""
        if not self._should_search(query):
            return ""

        if not query or not self._storage:
            return ""

        import time as _time

        # 检查工作流锁（复用 Redis client）
        lock_client = None
        try:
            lock_client = self._storage._get_client()
            if lock_client and lock_client.exists("keepsake:workflow_lock"):
                logger.debug("keepsake: workflow lock active, skipping search")
                return ""
        except Exception:
            pass

        start = _time.time()
        fragments = self._storage.search(
            query.strip(),
            tag_filter=self._tag_filter,
        )
        elapsed = _time.time() - start

        if not fragments:
            return ""

        lines = ["<keepsake>"]
        lines.append(f"# 相关记忆 (检索耗时 {elapsed:.1f}s)")
        lines.append("")
        for i, frag in enumerate(fragments, 1):
            lines.append(f"[{i}] {frag.get('content', '')}")
            tags = frag.get("tags", "")
            combined = frag.get("_combined_score", 0)
            weights = frag.get("_weights", {})
            info_parts = []
            if tags:
                info_parts.append(f"标签: {tags}")
            info_parts.append(f"综合: {combined:.2f}")
            if weights:
                info_parts.append(f"w: sim={weights.get('sim',0):.2f} decay={weights.get('decay',0):.2f} "
                                  f"emotion={weights.get('emotion',1):.1f} fb={weights.get('feedback',1):.1f} "
                                  f"hot={weights.get('hot_topic',1):.1f}")
            # 情感标签可视化
            sent_label = frag.get("sentiment_label", "")
            if sent_label and sent_label != "neutral":
                sent_score = frag.get("sentiment_score", "0")
                icon = "😊" if sent_label == "positive" else "😠"
                info_parts.append(f"{icon} {sent_label}({sent_score})")
            lines.append(f"    ({', '.join(info_parts)})")
            lines.append("")

        lines.append("</keepsake>")
        return "\n".join(lines)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """v2（2026-09）：格式闸门 → v2 pipeline 入队 → 兜底 v1 decide() 直存。"""
        if not self._storage or not user_content or not user_content.strip():
            return
        text = user_content.strip()
        decision, existing_meta = self._gate_decide(text, "turn_memory")
        if decision.action == "reject":
            return
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None and getattr(pipeline, "_llm_fn", None) is not None:
            try:
                pipeline.enqueue(text, assistant_content or "")
                return
            except Exception as e:
                logger.warning("keepsake: pipeline.enqueue failed (%s); falling back to v1", e)
        self._v1_store_after_decide(text, decision, existing_meta, "turn_memory", "hermes_agent", "")

    def _v1_fallback_store(self, user_text: str, category: str = "turn_memory") -> None:
        """v2 pipeline 失败时的窗口级兜底（与 sync_turn 的 v1 路径同源）。"""
        if not self._storage or not user_text or not user_text.strip():
            return
        decision, existing_meta = self._gate_decide(user_text.strip(), category)
        if decision.action == "reject":
            return
        self._v1_store_after_decide(user_text.strip(), decision, existing_meta, category, "pipeline_v2_fallback", "fallback:v1")

    def _gate_decide(self, text: str, category: str):
        """R1-R7 闸门裁决 + 探测同 hash 既有 meta（v1 与 v2 共享）。"""
        from .ingest_gate import decide
        frag_key = f"memory:frag:{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        existing_meta = None
        try:
            client = self._storage._get_client()
            if client and client.exists(frag_key):
                existing_meta = {"key": frag_key}
        except Exception:
            pass
        return decide(text, category, existing_meta, getattr(self, "_gate_cfg", None)), existing_meta

    def _v1_store_after_decide(self, text, decision, existing_meta, category, source, extra_tags) -> None:
        """decide() 之后直存（含 update_state 闭环；v1 与 v2 兜底共享）。"""
        from .ingest_gate import update_state_only
        if decision.action == "update_state":
            update_state_only(self._storage, existing_meta)
            return
        try:
            tags = "conversation" + ("," + extra_tags if extra_tags else "")
            self._storage.store(text=text, tags=tags, category=category, source=source,
                                fragment_type="memory")
        except Exception as e:
            logger.warning("keepsake: _v1_store_after_decide failed: %s", e)

    def _init_pipeline(self, llm_pipe_cfg: dict) -> None:
        """v2（2026-09）：构建 Pipeline 实例并启动 daemon（LLM 不可用则跳过）。"""
        self._pipeline = None
        if not llm_pipe_cfg.get("enabled", True):
            return
        from .consolidator import _call_llm, _get_api_key
        if not _get_api_key():
            logger.warning("keepsake: v2 pipeline disabled (no LLM API key); falls back to v1")
            return
        self._pipeline = Pipeline(
            storage=self._storage, llm_fn=_call_llm,
            model=llm_pipe_cfg.get("model") or "qwen-plus",
            window_pairs=int(llm_pipe_cfg.get("window_pairs", 4)),
            window_seconds=float(llm_pipe_cfg.get("window_seconds", 30.0)),
            max_calls_per_window=int(llm_pipe_cfg.get("max_calls_per_window", 8)),
            update_top_k=int(llm_pipe_cfg.get("update_top_k", 5)),
            recent_context_size=int(llm_pipe_cfg.get("recent_context_size", 8)),
            gate_fallback=self._v1_fallback_store,
        )
        self._pipeline.start()
        logger.info("keepsake: v2 pipeline started")

    def _maybe_maintain(self) -> None:
        """检查是否该执行维护，执行 Consolidation + Forget。"""
        import time as _time
        now = _time.time()
        if now - self._last_maintenance < self._maintenance_interval:
            return
        self._last_maintenance = now
        self.maintenance()

    def maintenance(self) -> Dict[str, Any]:
        """执行一轮完整维护：Consolidation → Forget。

        返回:
            维护统计
        """
        stats: Dict[str, Any] = {
            "consolidator": {"status": "skipped"},
            "forgetter": {"status": "skipped"},
        }

        # Step 1: Consolidation
        if self._consolidator:
            try:
                result = self._consolidator.consolidate()
                stats["consolidator"] = result
                logger.info("keepsake: consolidation done — %s", result)
            except Exception as e:
                logger.warning("keepsake: consolidation error: %s", e)
                stats["consolidator"] = {"status": "error", "reason": str(e)}

        # Step 2: Selective Forgetting
        if self._forgetter:
            try:
                result = self._forgetter.forget()
                stats["forgetter"] = result
                logger.info("keepsake: forgetting done — %s", result)
            except Exception as e:
                logger.warning("keepsake: forgetting error: %s", e)
                stats["forgetter"] = {"status": "error", "reason": str(e)}


        return stats

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FEEDBACK_SCHEMA, HOT_TOPICS_SCHEMA]

    def handle_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Route tool calls to the appropriate handler."""
        import json as _json

        if tool_name == "keepsake_feedback":
            return self._handle_feedback(args, _json)
        elif tool_name == "keepsake_topics":
            return self._handle_hot_topics(args, _json)
        return tool_error(f"Unknown keepsake memory tool: '{tool_name}'")

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_feedback(self, args: Dict[str, Any], _json) -> str:
        key = args.get("fragment_key", "")
        is_pos = bool(args.get("is_positive", True))
        if not key:
            return tool_error("fragment_key is required")
        if not self._storage:
            return tool_error("Memory storage not initialized")
        ok = self._storage.record_feedback(key, is_pos)
        if ok:
            action = "有用 👍" if is_pos else "没用 👎"
            return _json.dumps({"success": True, "action": action, "key": key})
        return tool_error("Failed to record feedback")

    def _handle_hot_topics(self, args: Dict[str, Any], _json) -> str:
        limit = min(int(args.get("limit", 10)), 30)
        period = args.get("period", "all")
        if not self._storage:
            return tool_error("Memory storage not initialized")
        topics = self._storage.get_hot_topics(limit=limit, period=period)
        return _json.dumps({"topics": topics, "count": len(topics)}, ensure_ascii=False)

    def shutdown(self) -> None:
        # v2（2026-09）：先排空管线，再关 storage；保证 shutdown 前队列内消息不丢
        pipeline = getattr(self, "_pipeline", None)
        if pipeline is not None:
            try:
                pipeline.stop(drain=True, timeout=5.0)
            except Exception as e:
                logger.warning("keepsake: pipeline stop failed: %s", e)
        if self._storage:
            self._storage.close()
        logger.info("keepsake memory provider shutdown")

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """builtin memory 写入时同步存到碎片库（完整内容，不做切分）。

        2026-09 ingest_gate v1：MEMORY.md 与碎片库分离，关闭同步双份入库。
        保留方法签名（Hermes MemoryProvider ABC 要求）。
        """
        # → ingest_gate R5（builtin_dup）：两套体系分离，不再复制 MEMORY.md 进碎片库
        return
