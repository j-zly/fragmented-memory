"""
Redis + RediSearch 存储层 — 碎片的读写与检索。

支持两种检索模式（可共存）：
  - BM25 全文搜索（默认，零成本）— 同义词扩展 + 标签过滤
  - KNN 向量搜索（可选）— 需要 embedder 配置
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis
from redis.commands.search.query import Query

from .embedder import Embedder
from .splitter import extract_keywords, extract_entities, segment_query
from .emotion import analyze_emotion
from .attention import record_attention, match_attention_boost

logger = logging.getLogger(__name__)

# RediSearch index 名称
RS_INDEX = "idx:memories"

# BM25 检索参数
DEFAULT_BM25_LIMIT = 10        # BM25 搜多少条候选
DEFAULT_FINAL_LIMIT = 5        # 最终返回条数

# KNN 参数（embedding 模式用）
DEFAULT_CANDIDATE_COUNT = 10   # KNN 候选数

# 时间衰减半衰期（天）
DECAY_HALF_DAYS = 60

# embedding 缓存 TTL（秒）
EMBED_CACHE_TTL = 3600

# 情感权重乘数（搜索排序用）
SENTIMENT_BOOST_POSITIVE = 1.5   # 正面碎片 ×1.5
SENTIMENT_BOOST_NEGATIVE = 1.3   # 负面碎片 ×1.3（用户明确表达不喜欢的也重要）
SENTIMENT_BOOST_NEUTRAL = 1.0    # 中性不变

# 反馈权重
FEEDBACK_POSITIVE_BOOST = 1.3    # 正反馈 ×1.3
FEEDBACK_NEGATIVE_PENALTY = 0.5  # 负反馈 ×0.5（标记没用的大幅降权）

# 热门话题加权
HOT_TOPIC_SET = "keepsake:hot_topics"
HOT_TOPIC_BOOST = 1.2           # 命中热门话题的碎片 ×1.2
HOT_TOPIC_DAILY = "keepsake:hot_topics:daily"  # 日榜
HOT_TOPIC_WEEKLY = "keepsake:hot_topics:weekly"  # 周榜
HOT_TOPIC_LAST_SEEN = "keepsake:hot_topics:last_seen"  # 最后提及时间
HOT_TOPIC_DECAY_HALF_DAYS = 30  # 热门话题时间衰减半衰期（天）

SYNONYM_HASH_KEY = "keepsake:synonyms"

# 实体共现关联
ENTITY_COOC_KEY = "keepsake:entity_cooc"
ENTITY_COOC_TTL = 2592000  # 30 天 TTL

# RediSearch 查询语法特殊字符（需要转义）
_QUERY_SPECIAL_CHARS = frozenset('@|()!*%~"\\/')


def _escape_query_term(term: str) -> str:
    """转义 RediSearch 查询语法中的特殊字符。"""
    for ch in _QUERY_SPECIAL_CHARS:
        term = term.replace(ch, f"\\{ch}")
    return term


def _escape_glob(term: str) -> str:
    """转义 Redis ZSCAN/ZSCAN MATCH 模式中的 glob 特殊字符。"""
    for ch in ('*', '?', '[', ']', '\\'):
        term = term.replace(ch, f"\\{ch}")
    return term


def _expand_terms(terms: List[str], synonym_map: Dict[str, set]) -> List[str]:
    """用同义词表展开搜索词列表。"""
    if not synonym_map:
        return terms
    expanded = list(terms)
    for t in terms:
        tl = t.lower()
        if tl in synonym_map:
            for syn in synonym_map[tl]:
                if syn not in expanded:
                    expanded.append(syn)
    return expanded


def _build_create_index_cmd(dim: int) -> str:
    """根据实际向量维度构建 FT.CREATE 命令。

    注意：返回命令字符串用于 split() 后 execute_command，不要加引号。
    """
    if dim < 1:
        logger.warning("_build_create_index_cmd: invalid dim=%d, falling back to 1536", dim)
        dim = 1536
    return (
        f"FT.CREATE {RS_INDEX} ON HASH PREFIX 1 memory:frag: LANGUAGE chinese SCHEMA "
        f"content TEXT WEIGHT 1 "
        f"tags TAG SEPARATOR , "
        f"category TAG SEPARATOR , "
        f"source TEXT WEIGHT 1 "
        f"created TEXT WEIGHT 0 "
        f"fragment_type TAG SEPARATOR , "
        f"invalid_at TAG SEPARATOR , "
        f"embed_bin VECTOR FLAT 6 TYPE FLOAT32 DIM {dim} DISTANCE_METRIC COSINE "
        f"entities TAG SEPARATOR ,"
    )


class RedisStorage:
    """碎片存储与检索。

    基于 Redis + RediSearch，同时支持 BM25 全文搜索（默认）和 KNN 向量搜索。"""
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        host: str = "127.0.0.1",
        port: int = 6379,
        password: Optional[str] = None,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
        final_limit: int = DEFAULT_FINAL_LIMIT,
        embed_dim: int = 1536,
        bm25_limit: int = DEFAULT_BM25_LIMIT,
        decay_half_days: int = DECAY_HALF_DAYS,
        embed_cache_ttl: int = EMBED_CACHE_TTL,
        sentiment_boost_positive: float = SENTIMENT_BOOST_POSITIVE,
        sentiment_boost_negative: float = SENTIMENT_BOOST_NEGATIVE,
        sentiment_boost_neutral: float = SENTIMENT_BOOST_NEUTRAL,
        feedback_positive_boost: float = FEEDBACK_POSITIVE_BOOST,
        feedback_negative_penalty: float = FEEDBACK_NEGATIVE_PENALTY,
        hot_topic_boost: float = HOT_TOPIC_BOOST,
        hot_topic_decay_half_days: int = HOT_TOPIC_DECAY_HALF_DAYS,
        emotion_intensity_factor: float = 0.4,
        attention_boost_max: float = 1.5,
        attention_base_increment: float = 2.0,
        attention_emotion_factor: float = 1.5,
        agent_id: str = "",
        is_primary: bool = False,
        synonym_min_word_freq: int = 10,
        synonym_jaccard_threshold: float = 0.5,
        synonym_min_co_occurrence: int = 3,
        entity_cooc_top_n: int = 3,
        entity_cooc_min_count: int = 2,
    ):
        self._embedder = embedder
        # 加固: embedder 存在时优先用它的真实维度，防止调用方漏传 embed_dim 建错索引
        if embedder is not None:
            embed_dim = embedder.dimension
        self._host = host
        self._port = port
        self._password = password
        self._candidate_count = candidate_count
        self._final_limit = final_limit
        self._embed_dim = embed_dim
        self._bm25_limit = bm25_limit
        self._decay_half_days = decay_half_days
        self._embed_cache_ttl = embed_cache_ttl
        self._sentiment_boost_positive = sentiment_boost_positive
        self._sentiment_boost_negative = sentiment_boost_negative
        self._sentiment_boost_neutral = sentiment_boost_neutral
        self._feedback_positive_boost = feedback_positive_boost
        self._feedback_negative_penalty = feedback_negative_penalty
        self._hot_topic_boost = hot_topic_boost
        self._hot_topic_decay_half_days = hot_topic_decay_half_days
        self._emotion_intensity_factor = emotion_intensity_factor
        self._attention_boost_max = attention_boost_max
        self._attention_base_increment = attention_base_increment
        self._attention_emotion_factor = attention_emotion_factor
        self._agent_id = agent_id
        self._is_primary = is_primary
        # 同义词发现参数
        self._synonym_min_word_freq = synonym_min_word_freq
        self._synonym_jaccard_threshold = synonym_jaccard_threshold
        self._synonym_min_co_occurrence = synonym_min_co_occurrence
        # 实体共现参数
        self._entity_cooc_top_n = entity_cooc_top_n
        self._entity_cooc_min_count = entity_cooc_min_count
        # 使用连接池（所有实例共享）
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._synonym_cache: Optional[Dict[str, set]] = None

    # ------------------------------------------------------------------
    # 连接管理（连接池）
    # ------------------------------------------------------------------

    def _get_client(self) -> Optional[redis.Redis]:
        if self._client is not None:
            try:
                self._client.ping()
                return self._client
            except redis.ConnectionError:
                self._client = None
                self._pool = None
        try:
            if self._pool is None:
                self._pool = redis.ConnectionPool(
                    host=self._host,
                    port=self._port,
                    password=self._password,
                    socket_connect_timeout=3,
                    socket_timeout=5,
                    decode_responses=False,
                    protocol=2,
                    max_connections=10,
                )
            self._client = redis.Redis(connection_pool=self._pool)
            self._client.ping()
            return self._client
        except redis.ConnectionError as e:
            logger.warning("storage: Redis not reachable (%s)", e)
            return None

    def _has_embedder(self) -> bool:
        """检查 embedder 是否可用。"""
        return self._embedder is not None and hasattr(self._embedder, "get_embedding")

    def ensure_index(self) -> bool:
        """初始化时自动创建/验证 RediSearch index。

        如果 index 已存在但向量维度与当前配置不匹配，打印警告
        但不自动重建（避免丢失已有数据）。
        """
        client = self._get_client()
        if not client:
            return False

        # 尝试检查 index 是否已存在
        try:
            client.execute_command("FT.INFO", RS_INDEX)
            idx_exists = True
        except redis.ResponseError:
            idx_exists = False
        except Exception as e:
            logger.debug("storage: FT.INFO check failed (will attempt recreate): %s", e)
            idx_exists = False

        # 如果 index 已存在，检查维度是否匹配
        if idx_exists:
            try:
                info = client.execute_command("FT.INFO", RS_INDEX)
                existing_dim = None
                # FT.INFO 返回扁平列表 [field, val, field, val, ...]
                for i in range(0, len(info) - 1, 2):
                    if isinstance(info[i], bytes) and info[i].decode() == "attributes":
                        attrs = info[i + 1]
                        if attrs and isinstance(attrs, list):
                            for attr in attrs:
                                for j in range(0, len(attr) - 1, 2):
                                    if isinstance(attr[j], bytes) and attr[j].decode() == "DIM":
                                        existing_dim = int(attr[j + 1])
                                        break
                if existing_dim is not None and existing_dim != self._embed_dim:
                    logger.warning(
                        "storage: index '%s' has dim=%d but configured dim=%d. "
                        "Index NOT rebuilt to preserve data. "
                        "Vector search may produce incorrect results.",
                        RS_INDEX, existing_dim, self._embed_dim,
                    )
            except Exception as e:
                logger.debug("storage: FT.INFO check failed: %s", e)

            # 尝试添加 invalid_at 字段（已存在则忽略）
            try:
                client.execute_command(
                    "FT.ALTER", RS_INDEX, "SCHEMA", "ADD", "invalid_at", "TAG", "SEPARATOR", ","
                )
                logger.info("storage: added invalid_at field to index '%s'", RS_INDEX)
            except redis.ResponseError:
                pass  # 字段已存在，忽略
            except Exception as e:
                logger.debug("storage: FT.ALTER failed (non-fatal): %s", e)

            # 尝试添加 entities 字段（已存在则忽略）
            try:
                client.execute_command(
                    "FT.ALTER", RS_INDEX, "SCHEMA", "ADD", "entities", "TAG", "SEPARATOR", ","
                )
                logger.info("storage: added entities field to index '%s'", RS_INDEX)
            except redis.ResponseError:
                pass  # 字段已存在，忽略
            except Exception as e:
                logger.debug("storage: FT.ALTER failed (non-fatal): %s", e)

            return True

        # 创建 index（如果不存在）
        if not idx_exists:
            try:
                cmd = _build_create_index_cmd(self._embed_dim)
                parts = cmd.split()
                client.execute_command(*parts)
                logger.info(
                    "storage: created RediSearch index '%s' (dim=%d)",
                    RS_INDEX, self._embed_dim,
                )
            except redis.ResponseError as e:
                if "already exists" in str(e).lower():
                    logger.info("storage: index '%s' already exists", RS_INDEX)
                else:
                    logger.warning("storage: failed to create index: %s", e)
                    return False
            except Exception as e:
                logger.warning("storage: failed to create index: %s", e)
                return False

        # 注册同义词组（幂等）
        self._ensure_synonyms(client)
        return True

    def _ensure_synonyms(self, client: redis.Redis) -> None:
        """已废弃 — 在 _load_synonym_map 中动态加载。"""
        pass

    def _load_synonym_map(self) -> Dict[str, set]:
        """从 Redis Hash keepsake:synonyms 加载同义词表（带实例级缓存）。"""
        if self._synonym_cache is not None:
            return self._synonym_cache
        client = self._get_client()
        if not client:
            return {}
        try:
            raw = client.hgetall(SYNONYM_HASH_KEY)
            if not raw:
                self._synonym_cache = {}
                return {}
            synonym_map: Dict[str, set] = {}
            for term_b, val_b in raw.items():
                term = term_b.decode("utf-8").lower().strip()
                if not term:
                    continue
                try:
                    syns = _json.loads(val_b.decode("utf-8"))
                except (_json.JSONDecodeError, UnicodeDecodeError):
                    continue
                terms_set = set()
                for s in syns:
                    sl = s.lower().strip()
                    if sl and sl != term:
                        terms_set.add(sl)
                if terms_set:
                    synonym_map[term] = terms_set
                    for s in terms_set:
                        if s not in synonym_map:
                            synonym_map[s] = set()
                        synonym_map[s].add(term)
            self._synonym_cache = synonym_map
            return synonym_map
        except Exception as e:
            logger.debug("storage: load synonyms error: %s", e)
            self._synonym_cache = {}
            return {}

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.disconnect()
            except Exception:
                pass
        self._client = None
        self._pool = None

    # ------------------------------------------------------------------
    # 向量化（带 MD5 缓存，仅 embedding 模式使用）
    # ------------------------------------------------------------------

    def _text_to_blob(self, text: str) -> Optional[bytes]:
        """文本 → float32 二进制 blob。"""
        md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        cache_key = f"embed_cache:{md5}"

        client = self._get_client()
        if client:
            try:
                cached = client.get(cache_key)
                if cached is not None and isinstance(cached, bytes):
                    return cached
            except Exception:
                pass

        if not self._has_embedder():
            return None
        vec = self._embedder.get_embedding(text)
        if not vec:
            return None
        blob = struct.pack(f"{len(vec)}f", *vec)

        if client:
            try:
                client.setex(cache_key, self._embed_cache_ttl, blob)
            except Exception:
                pass

        return blob

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def store(
        self,
        text: str,
        tags: str = "",
        category: str = "",
        source: str = "",
        fragment_type: str = "",
        sentiment_score: Optional[float] = None,
        sentiment_label: Optional[str] = None,
    ) -> bool:
        """将一段文本写入碎片库。

        embed_bin 是可选的（仅 embedding 模式需要）。
        BM25 全文搜索只需要 content + tags 字段。

        自动计算情感权重（除非显式传入 sentiment_*）。
        自动提取关键词并计入日榜/周榜。
        支持去重：内容相同则更新已有碎片（覆盖 feedback 外字段）。
        """
        client = self._get_client()
        if not client:
            return False

        # 情绪分析（除非明确传入）
        if sentiment_score is None or sentiment_label is None:
            intensity, label = analyze_emotion(text)
        else:
            intensity, label = sentiment_score, sentiment_label

        # 注意力追踪
        if client:
            try:
                keywords = extract_keywords(text, max_keywords=5)
                record_attention(
                    client, text, intensity, keywords,
                    base_increment=getattr(self, '_attention_base_increment', 2.0),
                    emotion_factor=getattr(self, '_attention_emotion_factor', 1.5),
                )
            except Exception:
                pass

        # 基于内容 hash 去重
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        key = f"memory:frag:{content_hash}"

        try:
            # 先去重检查：如果已存在，保留 feedback_score
            existing_feedback = "0"
            try:
                old = client.hget(key, "feedback_score")
                if old is not None:
                    existing_feedback = old if isinstance(old, str) else old.decode("utf-8")
            except Exception:
                pass

            # 处理 tags：自动添加 agent tag
            final_tags = tags
            if self._agent_id:
                # 将 agent tag 添加到 tags 列表中
                if not final_tags:
                    final_tags = f"agent:{self._agent_id}"
                else:
                    # 检查是否已经有 agent: 开头的 tag
                    tag_list = [t.strip() for t in final_tags.split(",") if t.strip()]
                    agent_tag = f"agent:{self._agent_id}"

                    # 移除已有的 agent tag
                    filtered_tags = [t for t in tag_list if not t.startswith("agent:")]
                    filtered_tags.append(agent_tag)
                    final_tags = ",".join(filtered_tags)

            mapping: Dict[str, Any] = {
                "content": text,
                "tags": final_tags,
                "category": category,
                "source": source,
                "created": datetime.now(timezone.utc).isoformat(),
                "sentiment_score": str(intensity),  # RediSearch Hash 存字符串
                "sentiment_label": label,
                "feedback_score": existing_feedback,  # 保留已有反馈，不重置
            }

            # 实体提取
            entities = extract_entities(text)
            if entities:
                mapping["entities"] = ",".join(entities)
                # 记录实体共现
                self._record_entity_cooccurrence(client, entities)
            if fragment_type:
                mapping["fragment_type"] = fragment_type

            # embed_bin 可选：有 embedder 时计算并存
            if self._has_embedder():
                blob = self._text_to_blob(text)
                if blob:
                    mapping["embed_bin"] = blob

            # HSET（去重：同 hash 会覆盖已有字段）
            client.hset(key, mapping=mapping)

            # 提取关键词并更新热门话题
            self._record_topics(client, text, label)

            return True
        except Exception as e:
            logger.warning("storage: store error: %s", e)
            return False

    # ------------------------------------------------------------------
    # 纠正标记 — 用户否定时降权前几轮碎片
    # ------------------------------------------------------------------

    def correct_fragments(self, keys: List[str]) -> int:
        """标记一批碎片为已纠正，降权使其几乎不出现在搜索结果中。

        做两件事：
          1. tags 中添加 'corrected' 标签
          2. feedback_score 设为 -1（已纠正标记，排序阶段直接压到最低）
        """
        client = self._get_client()
        if not client:
            return 0
        count = 0
        now = datetime.now(timezone.utc).isoformat()
        for key in keys:
            try:
                existing_tags = client.hget(key, "tags")
                if existing_tags is None:
                    continue
                if isinstance(existing_tags, bytes):
                    existing_tags = existing_tags.decode("utf-8")
                tag_list = [t.strip() for t in existing_tags.split(",") if t.strip()]
                if "corrected" not in tag_list:
                    tag_list.append("corrected")
                    client.hset(key, "tags", ",".join(tag_list))
                # 设 feedback_score 为负数，排序时大幅降权
                client.hset(key, "feedback_score", "-1")
                client.hset(key, "corrected_at", now)
                count += 1
            except Exception:
                continue
        if count:
            logger.info("storage: corrected %d fragments", count)
        return count

    # ------------------------------------------------------------------
    # 热门话题统计
    # ------------------------------------------------------------------

    _TOPIC_EXPIRE_SECONDS = {
        HOT_TOPIC_SET: 86400 * 7,       # 全局：7天过期
        HOT_TOPIC_DAILY: 86400 * 2,      # 日榜：2天过期（给次日看）
        HOT_TOPIC_WEEKLY: 86400 * 14,    # 周榜：14天过期
    }

    def _record_topics(
        self,
        client: redis.Redis,
        text: str,
        sentiment_label: str,
    ) -> None:
        """从文本提取关键词，计入热门话题 Sorted Set。"""
        try:
            keywords = extract_keywords(text, max_keywords=5)
            if not keywords:
                return

            # 情感权重：情感强烈的碎片关键词权重更高
            sentiment_weight = 1.0
            if sentiment_label == "positive":
                sentiment_weight = 1.5
            elif sentiment_label == "negative":
                sentiment_weight = 1.3

            for kw in keywords:
                for topic_set in (HOT_TOPIC_SET, HOT_TOPIC_DAILY, HOT_TOPIC_WEEKLY):
                    client.zincrby(topic_set, sentiment_weight, kw)
                    ttl = self._TOPIC_EXPIRE_SECONDS.get(topic_set, 86400)
                    client.expire(topic_set, ttl)

            # 记录热门话题最后提及时间（用于时间衰减）
            now_ts = datetime.now(timezone.utc).timestamp()
            for kw in keywords:
                client.hset(HOT_TOPIC_LAST_SEEN, kw, str(now_ts))
            client.expire(HOT_TOPIC_LAST_SEEN, 86400 * 30)  # 30天过期

        except Exception as e:
            logger.debug("storage: _record_topics error: %s", e)

    def _record_entity_cooccurrence(self, client: redis.Redis, entities: List[str]) -> None:
        """记录实体共现对。"""
        if len(entities) < 2:
            return
        try:
            sorted_ents = sorted(e.lower().strip() for e in entities if e.strip())
            for i in range(len(sorted_ents)):
                for j in range(i + 1, len(sorted_ents)):
                    pair = f"{sorted_ents[i]}||{sorted_ents[j]}"
                    client.zincrby(ENTITY_COOC_KEY, 1.0, pair)
                    client.expire(ENTITY_COOC_KEY, ENTITY_COOC_TTL)
        except Exception as e:
            logger.debug("storage: _record_entity_cooccurrence error: %s", e)

    def match_attention(self, content: str, top_n: int = 10) -> float:
        """检查碎片内容命中多少高注意力话题，返回加权值（1.0~max_boost）。"""
        client = self._get_client()
        if not client or not content:
            return 1.0
        try:
            boost_max = getattr(self, '_attention_boost_max', 1.5)
            return match_attention_boost(client, content, top_n=top_n, boost_max=boost_max)
        except Exception:
            return 1.0

    def match_hot_topics(self, text: str, limit: int = 10) -> float:
        """检查文本中包含多少个热门话题关键词（带时间衰减）。

        太久前的话题权重自动降低，最后提及时间越近权重越高。
        返回衰减后的有效命中数（非整数，有小数部分）。
        """
        if not text:
            return 0.0
        client = self._get_client()
        if not client:
            return 0.0
        try:
            raw = client.zrevrange(HOT_TOPIC_SET, 0, limit - 1, withscores=True)
            if not raw:
                return 0.0

            # 读取 last_seen 时间戳
            last_seen_raw = client.hgetall(HOT_TOPIC_LAST_SEEN) or {}
            last_seen = {}
            for k_b, v_b in last_seen_raw.items():
                k = k_b.decode("utf-8") if isinstance(k_b, bytes) else k_b
                v = v_b.decode("utf-8") if isinstance(v_b, bytes) else v_b
                try:
                    last_seen[k] = float(v)
                except (ValueError, TypeError):
                    pass

            now = datetime.now(timezone.utc).timestamp()
            decay_half = float(getattr(self, '_hot_topic_decay_half_days', HOT_TOPIC_DECAY_HALF_DAYS))

            text_lower = text.lower()
            weighted_hits = 0.0
            for topic_b, score_raw in raw:
                topic = topic_b.decode("utf-8") if isinstance(topic_b, bytes) else topic_b
                if isinstance(score_raw, bytes):
                    score_raw = score_raw.decode("utf-8")
                if len(topic) >= 2 and topic in text_lower:
                    # 时间衰减：最近提及的权重高，久远的低
                    seen_ts = last_seen.get(topic)
                    if seen_ts and seen_ts > 0:
                        days_ago = max(0, (now - seen_ts) / 86400.0)
                        decay = 2.0 ** (-days_ago / decay_half)
                    else:
                        decay = 0.5  # 无时间戳的折半
                    weighted_hits += decay

            return weighted_hits
        except Exception as e:
            logger.debug("storage: match_hot_topics error: %s", e)
            return 0.0

    def get_hot_topics(
        self,
        limit: int = 10,
        period: str = "all",
    ) -> List[Dict[str, Any]]:
        """获取热门话题。

        参数:
            limit: 返回条数
            period: "all" / "daily" / "weekly"

        返回:
            [{"topic": str, "count": float}, ...]
        """
        key = {
            "all": HOT_TOPIC_SET,
            "daily": HOT_TOPIC_DAILY,
            "weekly": HOT_TOPIC_WEEKLY,
        }.get(period, HOT_TOPIC_SET)

        client = self._get_client()
        if not client:
            return []

        try:
            raw = client.zrevrange(key, 0, limit - 1, withscores=True)
            return [{"topic": t.decode("utf-8") if isinstance(t, bytes) else t,
                     "count": round(s, 1)}
                    for t, s in raw]
        except Exception as e:
            logger.debug("storage: get_hot_topics error: %s", e)
            return []

    def is_hot_topic_keyword(self, keyword: str) -> bool:
        """检查一个词是否在热门话题库中（热度 > 1）。"""
        client = self._get_client()
        if not client:
            return False
        try:
            score = client.zscore(HOT_TOPIC_SET, keyword.lower())
            return score is not None and score > 1.0
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 人工反馈
    # ------------------------------------------------------------------

    def record_feedback(self, fragment_key: str, is_positive: bool) -> bool:
        """记录用户对碎片的反馈。

        参数:
            fragment_key: Redis key (如 "memory:frag:abc123")
            is_positive: True = 有用, False = 没用

        逻辑:
            - 有用：feedback_score += 1
            - 没用：feedback_score -= 2（负面反馈权重大）
        """
        client = self._get_client()
        if not client:
            return False
        try:
            delta = 1 if is_positive else -2
            client.hincrby(fragment_key, "feedback_score", delta)
            return True
        except Exception as e:
            logger.warning("storage: record_feedback error: %s", e)
            return False

    # ------------------------------------------------------------------
    # BM25 全文检索（默认，零成本）
    # ------------------------------------------------------------------

    def search_bm25(
        self,
        query: str,
        tag_filter: str = "",
        agent_id: str = "",
        is_primary: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """BM25 全文搜索，经时间衰减重排序后返回。

        流程:
          1. 构建 RediSearch 文本查询（自动扩展同义词）
          2. 可选标签过滤
          3. 时间衰减重排序
          4. 取 top final_limit
        """
        client = self._get_client()
        if not client or not query.strip():
            return []

        try:
            synonym_map = self._load_synonym_map()
            raw_terms = segment_query(query)
            expanded = _expand_terms(raw_terms, synonym_map)
            if not expanded:
                return []

            # 用 | 连接所有词（OR 语义），每个词单独转义
            safe_terms = "|".join(_escape_query_term(t) for t in expanded)

            # 实体共现扩展 — 从查询中提取实体，找关联实体扩充 entities 召回
            query_entities = extract_entities(query)
            if query_entities and self._entity_cooc_top_n > 0:
                try:
                    cooc_entities: set = set()
                    for ent in query_entities:
                        el = ent.lower().strip()
                        if not el:
                            continue
                        cursor = 0
                        while True:
                            cursor, members = client.zscan(
                                ENTITY_COOC_KEY, cursor=cursor, match=f"*{_escape_glob(el)}*", count=50
                            )
                            if not members:
                                break
                            for member_b, score_b in members:
                                pair = member_b.decode() if isinstance(member_b, bytes) else member_b
                                score = float(score_b)
                                if score < self._entity_cooc_min_count:
                                    continue
                                parts = pair.split("||")
                                for p in parts:
                                    if p != el:
                                        cooc_entities.add(p)
                            if cursor == 0:
                                break
                    # 按共现次数取 top-N
                    if cooc_entities:
                        scored = []
                        for ce in cooc_entities:
                            try:
                                pair = "||".join(sorted([ce, query_entities[0].lower().strip()]))
                                s = float(client.zscore(ENTITY_COOC_KEY, pair) or 0)
                                scored.append((s, ce))
                            except Exception:
                                scored.append((0, ce))
                        scored.sort(key=lambda x: -x[0])
                        top_cooc = [ce for _, ce in scored[:self._entity_cooc_top_n]]
                        raw_terms = list(raw_terms) + top_cooc
                        logger.debug("storage: entity cooc expanded %s -> %s", query_entities, top_cooc)
                except Exception as e:
                    logger.debug("storage: entity cooc expansion error: %s", e)

            # 构建基础查询表达式 — 同时搜 content 和 entities（OR 语义）
            # content 用括号包裹 OR 术语，避免与外部 OR 歧义
            if tag_filter:
                safe_tags = ",".join(
                    _escape_query_term(t.strip())
                    for t in tag_filter.split(",") if t.strip()
                )
                content_q = f"@tags:{{{safe_tags}}} @content:({safe_terms})"
            else:
                content_q = f"@content:({safe_terms})"

            # entities 字段只搜原始搜索词（不同义词扩展，避免TAG查询长度超限）
            raw_safe = "|".join(_escape_query_term(t) for t in raw_terms)
            entities_q = f"@entities:{{{raw_safe}}}"
            query_expr = f"({content_q} | {entities_q})"

            # 如果不是主脑且指定了 agent_id，则添加 agent 过滤条件
            effective_agent_id = agent_id if agent_id else self._agent_id
            effective_is_primary = is_primary if is_primary is not None else self._is_primary

            if not effective_is_primary and effective_agent_id:
                # 只能搜索 agent 指定的碎片或者 shared 标签的碎片
                agent_filter = f"@tags:{{agent:{effective_agent_id}}}"
                shared_filter = f"@tags:{{shared}}"
                # 两者之一即可
                query_expr = f"({agent_filter} || {shared_filter}) AND {query_expr}"
            elif not effective_is_primary and not effective_agent_id:
                # 如果没有 agent_id，只搜索 shared 标签
                query_expr = f"@tags:{{shared}} AND {query_expr}"

            q = (
                Query(query_expr)
                .paging(0, self._bm25_limit)
                .dialect(2)
                .return_fields("content", "tags", "category", "source", "created",
                               "sentiment_score", "sentiment_label", "feedback_score",
                               "invalid_at", "entities")
            )

            result = client.ft(RS_INDEX).search(q)

            fragments: List[Dict[str, Any]] = []
            for doc in result.docs:
                frag: Dict[str, Any] = {}
                for field in ("content", "tags", "category", "source", "created",
                             "sentiment_score", "sentiment_label", "feedback_score",
                             "invalid_at", "entities"):
                    val = getattr(doc, field, None)
                    if val is not None and val != "":
                        if isinstance(val, bytes):
                            val = val.decode("utf-8")
                        frag[field] = val
                # 跳过已过期的碎片
                invalid_at = getattr(doc, "invalid_at", None)
                if invalid_at:
                    continue
                if frag.get("content"):
                    # BM25 score 越大越相关
                    frag["_bm25_score"] = float(getattr(doc, "score", 0.0))
                    fragments.append(frag)

            fragments = self._rerank_with_decay(fragments, score_key="_bm25_score", storage=self)
            return fragments[: self._final_limit]

        except Exception as e:
            logger.debug("storage: BM25 search error: %s", e)
            return []

    # ------------------------------------------------------------------
    # KNN 向量检索（可选，需 embedder）
    # ------------------------------------------------------------------

    def search_knn(
        self,
        query: str,
        tag_filter: str = "",
        agent_id: str = "",
        is_primary: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """KNN 向量搜索，经时间衰减重排序后返回。"""
        if not self._has_embedder():
            return []

        blob = self._text_to_blob(query)
        if not blob:
            return []

        client = self._get_client()
        if not client:
            return []

        try:
            # 构建基础查询表达式
            if tag_filter:
                safe_tags = ",".join(
                    _escape_query_term(t.strip())
                    for t in tag_filter.split(",") if t.strip()
                )
                query_expr = f"@tags:{{{safe_tags}}}=>[KNN $K @embed_bin $vec AS score]"
            else:
                query_expr = "*=>[KNN $K @embed_bin $vec AS score]"

            # 如果不是主脑且指定了 agent_id，则添加 agent 过滤条件
            effective_agent_id = agent_id if agent_id else self._agent_id
            effective_is_primary = is_primary if is_primary is not None else self._is_primary

            if not effective_is_primary and effective_agent_id:
                # 只能搜索 agent 指定的碎片或者 shared 标签的碎片
                agent_filter = f"@tags:{{agent:{effective_agent_id}}}"
                shared_filter = f"@tags:{{shared}}"
                # 两者之一即可
                query_expr = f"({agent_filter} || {shared_filter}) AND {query_expr}"
            elif not effective_is_primary and not effective_agent_id:
                # 如果没有 agent_id，只搜索 shared 标签
                query_expr = f"@tags:{{shared}} AND {query_expr}"

            q = (
                Query(query_expr)
                .sort_by("score")
                .return_fields("content", "tags", "category", "source", "created",
                               "sentiment_score", "sentiment_label", "feedback_score",
                               "invalid_at", "entities")
                .dialect(2)
                .paging(0, self._candidate_count)
            )
            result = client.ft(RS_INDEX).search(
                q, query_params={"vec": blob, "K": self._candidate_count}
            )

            fragments: List[Dict[str, Any]] = []
            for doc in result.docs:
                frag: Dict[str, Any] = {}
                for field in ("content", "tags", "category", "source", "created",
                             "sentiment_score", "sentiment_label", "feedback_score",
                             "invalid_at", "entities"):
                    val = getattr(doc, field, None)
                    if val is not None and val != "":
                        if isinstance(val, bytes):
                            val = val.decode("utf-8")
                        frag[field] = val
                # 跳过已过期的碎片
                invalid_at = getattr(doc, "invalid_at", None)
                if invalid_at:
                    continue
                if frag.get("content"):
                    frag["_knn_score"] = float(getattr(doc, "score", 1.0))
                    fragments.append(frag)

            fragments = self._rerank_with_decay(fragments, score_key="_knn_score", is_knn=True, storage=self)
            return fragments[: self._final_limit]

        except Exception as e:
            logger.debug("storage: KNN search error: %s", e)
            return []

    # ------------------------------------------------------------------
    # 统一检索入口
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        tag_filter: str = "",
        agent_id: str = "",
        is_primary: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """统一检索入口。

        策略:
          1. 先走 BM25 全文搜索（零成本，有同义词扩展）
          2. 如果 BM25 无结果 且 embedder 可用，走 KNN 向量搜索
          3. BM25 有结果则直接返回，不做合并

        如果是主脑（is_primary=True），不限制搜索范围，否则只搜索指定 agent 或 shared 标签的碎片。
        """
        # 如果传入了参数，则优先使用参数；否则使用类实例中的配置
        effective_agent_id = agent_id if agent_id else self._agent_id
        effective_is_primary = is_primary if is_primary is not None else self._is_primary

        # RRF 融合检索（v1.3）: BM25 + KNN 两路排名融合（Reciprocal Rank Fusion）
        # 不比较原始分数（量纲不同），只按排名位置加权：score = Σ 1/(k + rank)
        # 两路都排名高的记忆排最前（交叉验证）；单路时退化为该路自身顺序
        bm25_results = self.search_bm25(query, tag_filter, effective_agent_id, effective_is_primary)

        if self._has_embedder():
            knn_results = self.search_knn(query, tag_filter, effective_agent_id, effective_is_primary)
            if knn_results:
                return self._rrf_fuse(bm25_results, knn_results)

        return bm25_results

    def _rrf_fuse(
        self,
        bm25_results: List[Dict[str, Any]],
        knn_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion：两路检索按排名位置融合，返回 final_limit 条。

        原理：score(content) = 1/(k+BM25排名) + 1/(k+KNN排名)，k 取 60（业界常用）。
        - 两路都命中 → 分数叠加 → 排最前（交叉验证）
        - 单路命中 → 得该路排名分
        - 按 content 去重合并（同一记忆条目只保留一份，字段取先出现者）
        """
        scores: dict = {}
        items: dict = {}
        for rank, frag in enumerate(bm25_results, 1):
            c = frag.get("content")
            if not c:
                continue
            scores[c] = scores.get(c, 0.0) + 1.0 / (k + rank)
            items[c] = frag
        for rank, frag in enumerate(knn_results, 1):
            c = frag.get("content")
            if not c:
                continue
            scores[c] = scores.get(c, 0.0) + 1.0 / (k + rank)
            if c not in items:
                items[c] = frag

        fused = sorted(items.values(), key=lambda f: -scores[f.get("content")])
        for frag in fused:
            frag["_combined_score"] = scores.get(frag.get("content"), 0.0)
        return fused[: self._final_limit]

    # ------------------------------------------------------------------
    # 综合得分重排序
    # ------------------------------------------------------------------

    def _rerank_with_decay(
        self,
        fragments: List[Dict[str, Any]],
        score_key: str = "_bm25_score",
        is_knn: bool = False,
        storage: Optional["RedisStorage"] = None,
    ) -> List[Dict[str, Any]]:
        """综合得分重排序。

        BM25 模式: combined = BM25归一化得分 × 时间衰减 × 情绪权重 × 反馈权重 × 热门权重 × 注意力权重
        KNN 模式:   combined = (1 - 余弦距离/2) × 时间衰减 × 情绪权重 × 反馈权重 × 热门权重 × 注意力权重

        权重参数见模块顶部常量。
        """
        if not fragments:
            return fragments

        now = datetime.now(timezone.utc)

        # ---- Step 1: 计算语义相似度 sim ----
        for frag in fragments:
            raw = float(frag.get(score_key, 0.0))
            if is_knn:
                # KNN: score 是余弦距离（0~2），越小越近
                frag["_sim"] = 1.0 - max(0.0, min(1.0, raw / 2.0))
            else:
                frag["_sim"] = raw  # 暂存原始 BM25 分数，后面归一化

        # ---- Step 2: BM25 模式用 min-max 动态归一化 ----
        if not is_knn and fragments:
            scores_raw = [float(f.get(score_key, 0.0)) for f in fragments]
            max_raw = max(scores_raw) if scores_raw else 1.0
            if max_raw < 0.001:
                max_raw = 1.0
            for frag in fragments:
                raw_val = float(frag.get(score_key, 0.0))
                frag["_sim"] = raw_val / max_raw

        # ---- Step 3: 六维权重综合 ----
        for frag in fragments:
            sim = float(frag.get("_sim", 0.0))

            # 0: 已纠正碎片直接压到最低，永远不出现在 Top-K
            tags = frag.get("tags", "")
            if "corrected" in (tags if isinstance(tags, str) else ""):
                frag["_combined_score"] = -1.0
                continue

            # 3a: 时间衰减
            created_str = frag.get("created", "")
            if not created_str:
                decay = 0.01
            else:
                try:
                    created = datetime.fromisoformat(created_str)
                    age_days = (now - created).total_seconds() / 86400.0
                    if age_days < 0:
                        age_days = 0
                except (ValueError, TypeError):
                    decay = 0.01
                    age_days = 0
                else:
                    decay = 2.0 ** (-age_days / self._decay_half_days)

            # 3b: 情绪权重（基于烈度，不再分正负）
            try:
                intensity = float(frag.get("sentiment_score", 0))
            except (ValueError, TypeError):
                intensity = 0.0
            # intensity 0.0~2.0 → 权重 1.0~1.0+2.0*factor
            emotion_factor = getattr(self, '_emotion_intensity_factor', 0.4)
            emotion_w = 1.0 + min(intensity, 2.0) * emotion_factor

            # 3c: 反馈权重
            try:
                fb = float(frag.get("feedback_score", 0))
            except (ValueError, TypeError):
                fb = 0.0
            if fb > 0:
                feedback_w = 1.0 + (self._feedback_positive_boost - 1.0) * min(fb / 3.0, 1.0)
            elif fb < 0:
                feedback_w = 1.0 - (1.0 - self._feedback_negative_penalty) * min(abs(fb) / 3.0, 1.0)
            else:
                feedback_w = 1.0

            # 3d: 热门话题加权
            hot_w = 1.0
            content = frag.get("content", "")
            if content and storage is not None and hasattr(storage, 'match_hot_topics'):
                try:
                    hits = storage.match_hot_topics(content, limit=10)
                    if hits >= 3:
                        hot_w = HOT_TOPIC_BOOST
                    elif hits >= 1:
                        hot_w = 1.0 + (HOT_TOPIC_BOOST - 1.0) * (hits / 3.0)
                except Exception:
                    pass

            # 3e: 注意力加权
            attn_w = 1.0
            if content and storage is not None and hasattr(storage, 'match_attention'):
                try:
                    attn_w = storage.match_attention(content)
                except Exception:
                    pass

            frag["_combined_score"] = sim * decay * emotion_w * feedback_w * hot_w * attn_w
            frag["_weights"] = {
                "sim": round(sim, 4),
                "decay": round(decay, 4),
                "emotion": round(emotion_w, 4),
                "feedback": round(feedback_w, 4),
                "hot_topic": round(hot_w, 4),
                "attention": round(attn_w, 4),
            }

        fragments.sort(key=lambda x: x.get("_combined_score", 0), reverse=True)
        return fragments

    def generate_jieba_dict(self, output_path: str = None) -> Dict[str, Any]:
        """从碎片库 + 同义词表生成 jieba 自定义词典。

        扫描全库碎片统计词频，合并同义词表中的术语，
        输出为 jieba.load_userdict() 可加载的词典文件。

        Args:
            output_path: 输出路径，默认 ~/.config/keepsake/jieba_dict.txt

        Returns:
            统计信息
        """
        from .splitter import _STOP_WORDS
        import jieba

        if output_path is None:
            output_path = str(Path.home() / '.config' / 'keepsake' / 'jieba_dict.txt')

        client = self._get_client()
        if not client:
            return {"written_terms": 0, "error": "no redis client"}

        # 1. 扫碎片库统计词频
        word_freq: Dict[str, int] = {}
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match="memory:frag:*", count=1000)
            for key in keys:
                try:
                    content = client.hget(key, "content")
                    if content is None:
                        continue
                    if isinstance(content, bytes):
                        content = content.decode("utf-8")
                    words = jieba.lcut(content)
                    filtered = [w for w in words
                                if len(w) >= 2
                                and w not in _STOP_WORDS
                                and not w.isdigit()]
                    for w in set(filtered):
                        word_freq[w] = word_freq.get(w, 0) + 1
                except Exception:
                    continue

            if cursor == 0:
                break

        # 2. 读同义词表补全术语
        try:
            synonyms = client.hgetall(SYNONYM_HASH_KEY)
            for term_b in synonyms.keys():
                term = term_b.decode("utf-8").lower().strip()
                if term and len(term) >= 2:
                    word_freq[term] = max(word_freq.get(term, 0), 3)
        except Exception:
            pass

        # 3. 过滤：至少出现 2 次
        selected = [(w, f) for w, f in word_freq.items() if f >= 2]

        # 4. 按词频降序写文件
        selected.sort(key=lambda x: -x[1])

        lines = [f"{word} {freq} nz\n" for word, freq in selected]

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(lines), encoding="utf-8")

        return {
            "written_terms": len(lines),
            "total_candidates": len(word_freq),
        }

    def discover_synonyms(self) -> Dict[str, Any]:
        """自动发现同义词组。

        扫描全库碎片，统计词频和共现关系，生成同义词组并写入 Redis Hash。

        Returns:
            统计信息字典
        """
        from .splitter import _STOP_WORDS
        import jieba

        client = self._get_client()
        if not client:
            return {"discovered_groups": 0, "total_terms": 0, "scanned_fragments": 0}

        # 统计词频和共现
        word_freq: Dict[str, int] = {}
        co_occur: Dict[Tuple[str, str], int] = {}
        scanned_fragments = 0

        # 遍历所有碎片
        cursor = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match="memory:frag:*", count=1000)
            for key in keys:
                try:
                    # 获取 content 字段
                    content = client.hget(key, "content")
                    if content is None:
                        continue
                    if isinstance(content, bytes):
                        content = content.decode("utf-8")

                    # 分词（使用 jieba）
                    words = jieba.lcut(content)

                    # 过滤停用词（复用 splitter.py 的 _STOP_WORDS）
                    stop_words = _STOP_WORDS

                    # 过滤长度≥2 且不在停用词中的词
                    filtered_words = [w for w in words
                                      if len(w) >= 2
                                      and w not in stop_words
                                      and not w.isdigit()]

                    if not filtered_words:
                        continue

                    scanned_fragments += 1

                    # 统计词频
                    unique_words = set(filtered_words)
                    for word in unique_words:
                        word_freq[word] = word_freq.get(word, 0) + 1

                    # 统计共现（用 set 去重，避免重复词导致计数偏差）
                    unique_list = sorted(unique_words)
                    for i in range(len(unique_list)):
                        for j in range(i + 1, len(unique_list)):
                            key_pair = (unique_list[i], unique_list[j])
                            co_occur[key_pair] = co_occur.get(key_pair, 0) + 1

                except Exception as e:
                    # 某条碎片解析失败跳过
                    logger.debug("storage: skip fragment %s due to parsing error: %s", key, e)
                    continue

            # SCAN 游标归零表示遍历完成
            if cursor == 0:
                break

        # 过滤候选词
        candidates = {word for word, freq in word_freq.items()
                      if freq >= self._synonym_min_word_freq}

        # 找出同义词组
        discovered_groups = 0
        new_synonym_map = {}

        # 对候选集中每一对词
        for word_a in candidates:
            for word_b in candidates:
                if word_a >= word_b:
                    continue

                # 获取共现次数
                c = co_occur.get((word_a, word_b), 0)

                # 计算 Jaccard 系数
                if word_freq[word_a] + word_freq[word_b] - c > 0:
                    jaccard = c / (word_freq[word_a] + word_freq[word_b] - c)
                else:
                    jaccard = 0.0

                # 满足任一阈值条件则认为是同义词
                if jaccard >= self._synonym_jaccard_threshold or c >= self._synonym_min_co_occurrence:
                    # 添加到结果中（双向）
                    if word_a not in new_synonym_map:
                        new_synonym_map[word_a] = set()
                    if word_b not in new_synonym_map:
                        new_synonym_map[word_b] = set()

                    new_synonym_map[word_a].add(word_b)
                    new_synonym_map[word_b].add(word_a)
                    discovered_groups += 1

        # 合并新发现的同义词到现有映射
        existing = client.hgetall(SYNONYM_HASH_KEY)
        merged_synonym_map = {}

        # 加载现有的同义词映射
        for term_b, val_b in existing.items():
            term = term_b.decode("utf-8").lower().strip()
            if not term:
                continue
            try:
                syns = _json.loads(val_b.decode("utf-8"))
            except (_json.JSONDecodeError, UnicodeDecodeError):
                continue
            terms_set = set()
            for s in syns:
                sl = s.lower().strip()
                if sl and sl != term:
                    terms_set.add(sl)
            if terms_set:
                merged_synonym_map[term] = terms_set
                for s in terms_set:
                    if s not in merged_synonym_map:
                        merged_synonym_map[s] = set()
                    merged_synonym_map[s].add(term)

        # 将新发现的同义词加入合并结果（避免覆盖手动添加的）
        for word, synonyms in new_synonym_map.items():
            if word in merged_synonym_map:
                # 手动已存在的，跳过（手动添加的优先级高）
                continue
            merged_synonym_map[word] = synonyms

            # 更新反向映射
            for syn in synonyms:
                if syn not in merged_synonym_map:
                    merged_synonym_map[syn] = set()
                merged_synonym_map[syn].add(word)

        # 写入 Redis Hash
        if merged_synonym_map:
            pipe = client.pipeline()
            for word, synonyms in merged_synonym_map.items():
                pipe.hset(SYNONYM_HASH_KEY, word, _json.dumps(list(synonyms)))
            pipe.execute()

        # 清除同义词缓存
        self._synonym_cache = None

        return {
            "discovered_groups": discovered_groups,
            "total_terms": len(merged_synonym_map),
            "scanned_fragments": scanned_fragments
        }
