#!/usr/bin/env python3
"""
碎片去重脚本 — 多信号融合判定两条碎片是否描述同一件事。

独立脚本，不修改插件代码。cron 定时跑，扫描新增碎片，对每条新碎片搜索
相似碎片，融合评分后标记旧碎片为已过期（invalid_at）。

运行方式:
  python3 scripts/dedup_fragments.py [--dry-run] [--threshold 0.65] [--batch-size 50]

阈值说明:
  >0.75  高度确信是同一件事 → 直接标记旧碎片过期
  0.65-0.75  可能是同一件事 → 取时间较早的标记过期
  <0.65  不是同一件事 → 跳过
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import jieba
import jieba.posseg as pseg
import redis
from redis.commands.search.query import Query

# fmt: off
# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

REDIS_HOST = os.environ.get("KEEPSAKE_REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("KEEPSAKE_REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("KEEPSAKE_REDIS_PASSWORD")
if not REDIS_PASSWORD:
    REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD")

# 回退：从 Keepsake 配置读取 Redis 连接（cron 环境通常没有环境变量）
if REDIS_HOST == "127.0.0.1":
    try:
        _cfg_path = os.path.expanduser("~/.config/keepsake/config.json")
        if os.path.exists(_cfg_path):
            import json as _json
            _cfg = _json.load(open(_cfg_path))
            REDIS_HOST = _cfg.get("redis_host", REDIS_HOST)
            REDIS_PORT = int(_cfg.get("redis_port", REDIS_PORT))
            REDIS_PASSWORD = _cfg.get("redis_password") or REDIS_PASSWORD
    except Exception:
        pass

# 同义词表 key
SYNONYM_KEY = "keepsake:synonyms"

# 情感极性门 — 同一情感类碎片极性差超过此值，直接不算同一事
POLARITY_GATE = 0.8

# 信号权重 — 情感类
WEIGHTS_EMOTION = {"jaccard": 0.40, "polarity": 0.30, "bm25": 0.20, "ngram": 0.10}

# 信号权重 — 事实类
WEIGHTS_FACT = {"entity": 0.40, "bm25": 0.30, "jaccard": 0.20, "ngram": 0.10}

# 默认融合阈值
DEFAULT_THRESHOLD = 0.65

# 实体词性（jieba 标注）— 认为有"标识意义"的词性
ENTITY_POS = {"nr", "ns", "nt", "nz", "n", "vn"}  # 人名/地名/机构/专名/名词/动名

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("dedup")
# fmt: on


# ══════════════════════════════════════════════
# Redis 连接
# ══════════════════════════════════════════════

def get_redis() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=False,
        protocol=2,  # 强制 RESP2 协议，兼容 redis-py 搜索解析
    )


def load_synonyms(r: redis.Redis) -> Dict[str, Set[str]]:
    """加载同义词表。key=小写原词, value=同义词集合(含原词本身)。"""
    raw = r.hgetall(SYNONYM_KEY)
    syn_map: Dict[str, Set[str]] = {}
    for word_b, syns_b in raw.items():
        word = word_b.decode("utf-8").lower()
        syns = json.loads(syns_b.decode("utf-8"))
        syn_map[word] = set(s.lower() for s in syns) | {word}
    return syn_map


# RediSearch 查询语法特殊字符
_QUERY_SPECIAL_CHARS = frozenset('@|()!*~%"\\\\/')


def _escape_query_term(term: str) -> str:
    for ch in _QUERY_SPECIAL_CHARS:
        term = term.replace(ch, f"\\\\{ch}")
    return term


# ══════════════════════════════════════════════
# 分词 & 特征提取
# ══════════════════════════════════════════════

_JIEBA_LOADED = False


def _ensure_jieba():
    global _JIEBA_LOADED
    if not _JIEBA_LOADED:
        jieba.initialize()
        _JIEBA_LOADED = True


def tokenize(text: str) -> List[str]:
    """jieba 分词，过滤停用词和单字。"""
    _ensure_jieba()
    words = jieba.lcut(text)
    # 过滤单字、空白、标点
    return [
        w
        for w in words
        if len(w) >= 2 and not re.match(r"^[，。！？、；：""''（）【】《》\\s\\W]+$", w)
    ]


def extract_entities(text: str) -> List[str]:
    """提取实体词：人名、地名、机构、专名、名词性核心词。"""
    _ensure_jieba()
    words = pseg.lcut(text)
    entities = []
    for w, flag in words:
        if flag in ENTITY_POS and len(w) >= 2:
            entities.append(w)
    return entities


def ngram_similarity(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram Jaccard 相似度。"""
    if not a or not b:
        return 0.0
    a_grams = {a[i : i + n] for i in range(len(a) - n + 1)}
    b_grams = {b[i : i + n] for i in range(len(b) - n + 1)}
    if not a_grams or not b_grams:
        return 0.0
    inter = a_grams & b_grams
    union = a_grams | b_grams
    return len(inter) / len(union)


def set_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ══════════════════════════════════════════════
# 多信号融合评分
# ══════════════════════════════════════════════

def calc_fusion_score(
    frag_a: Dict[str, Any],
    frag_b: Dict[str, Any],
    synonym_map: Dict[str, Set[str]],
    bm25_raw_score: float = 0.0,
) -> float:
    """
    计算两条碎片的多信号融合相似度。

    返回 0.0 ~ 1.0 的分数。
    """
    content_a = frag_a.get("content", "")
    content_b = frag_b.get("content", "")

    if not content_a or not content_b:
        return 0.0

    # --- 信号1: 分词 Jaccard + 同义词扩展 ---
    tokens_a = set(tokenize(content_a))
    tokens_b = set(tokenize(content_b))

    # 同义词扩展
    expanded_a: Set[str] = set()
    expanded_b: Set[str] = set()
    for t in tokens_a:
        expanded_a.add(t)
        tl = t.lower()
        if tl in synonym_map:
            expanded_a.update(synonym_map[tl])
    for t in tokens_b:
        expanded_b.add(t)
        tl = t.lower()
        if tl in synonym_map:
            expanded_b.update(synonym_map[tl])

    jaccard_score = set_jaccard(expanded_a, expanded_b)

    # --- 信号2: 实体词匹配 ---
    entities_a = set(extract_entities(content_a))
    entities_b = set(extract_entities(content_b))
    entity_score = set_jaccard(entities_a, entities_b)

    # --- 信号3: BM25 原始分（RediSearch BM25 一般在 0-30 范围）---
    bm25_score = min(bm25_raw_score / 20.0, 1.0) if bm25_raw_score > 0 else 0.0

    # --- 信号4: n-gram 字符重叠 ---
    ngram_score = ngram_similarity(content_a, content_b)

    # --- 信号5: 情感极性门（对情感类内容）---
    try:
        sent_a = float(frag_a.get("sentiment_score", 0))
    except (ValueError, TypeError):
        sent_a = 0.0
    try:
        sent_b = float(frag_b.get("sentiment_score", 0))
    except (ValueError, TypeError):
        sent_b = 0.0

    polar_diff = abs(sent_a - sent_b)

    # 判断是否情感类（任一条的情感强度 > 0.3）
    is_emotion_type = abs(sent_a) > 0.3 or abs(sent_b) > 0.3

    if is_emotion_type:
        # 情感极性门 — 极性差太大直接否决
        if polar_diff > POLARITY_GATE:
            return 0.0

        # 情感类融合
        score = (
            WEIGHTS_EMOTION["jaccard"] * jaccard_score
            + WEIGHTS_EMOTION["polarity"] * (1.0 - polar_diff)
            + WEIGHTS_EMOTION["bm25"] * bm25_score
            + WEIGHTS_EMOTION["ngram"] * ngram_score
        )
    else:
        # 事实类融合
        score = (
            WEIGHTS_FACT["entity"] * entity_score
            + WEIGHTS_FACT["bm25"] * bm25_score
            + WEIGHTS_FACT["jaccard"] * jaccard_score
            + WEIGHTS_FACT["ngram"] * ngram_score
        )

    # 兜底：实体词完全不匹配但其他信号都高 → 也减分
    if entities_a and entities_b and entity_score == 0.0:
        score *= 0.5  # 实体词一对都没对上，降权

    return max(0.0, min(1.0, score))


# ══════════════════════════════════════════════
# 碎片扫描 & 去重执行
# ══════════════════════════════════════════════

# 已检查的碎片 set key
CHECKED_SET_KEY = "keepsake:dedup:checked"


def scan_fragments(
    r: redis.Redis,
    batch_size: int = 200,
) -> List[Dict[str, Any]]:
    """
    扫描未检查、未过期的 memory:frag:*。

    扫描全部 key，过滤掉 checked set 中已有的和已标记 invalid_at 的。
    """
    # 加载已检查的 set
    checked = set()
    try:
        for k in r.sscan_iter(CHECKED_SET_KEY):
            checked.add(k.decode("utf-8") if isinstance(k, bytes) else k)
    except Exception:
        pass

    # 扫描所有 memory:frag:* key
    all_keys = []
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match="memory:frag:*", count=500)
        for k in keys:
            all_keys.append(k.decode("utf-8") if isinstance(k, bytes) else k)
        if cursor == 0:
            break

    logger.debug("Total memory:frag:* keys: %d, checked: %d", len(all_keys), len(checked))

    fragments = []
    for key in all_keys:
        if key in checked:
            continue

        data = r.hgetall(key)
        if not data:
            continue

        # 跳过已过期的
        if data.get(b"invalid_at"):
            continue
        if data.get(b"feedback_score") == b"-1":
            continue  # 跳过被纠正的

        frag = {
            "key": key,
            "content": (data.get(b"content") or b"").decode("utf-8", errors="replace"),
            "created": (data.get(b"created") or b"").decode("utf-8", errors="replace"),
            "sentiment_score": (data.get(b"sentiment_score") or b"0").decode("utf-8"),
            "sentiment_label": (data.get(b"sentiment_label") or b"").decode("utf-8"),
            "tags": (data.get(b"tags") or b"").decode("utf-8", errors="replace"),
            "invalid_at": (data.get(b"invalid_at") or b"").decode("utf-8", errors="replace"),
            "category": (data.get(b"category") or b"").decode("utf-8", errors="replace"),
        }
        fragments.append(frag)

        if len(fragments) >= batch_size:
            break

    return fragments, checked


def get_fragment_by_key(r: redis.Redis, key: str) -> Optional[Dict[str, Any]]:
    """按 Redis key 获取碎片数据。"""
    data = r.hgetall(key)
    if not data:
        return None
    return {
        "key": key,
        "content": (data.get(b"content") or b"").decode("utf-8", errors="replace"),
        "created": (data.get(b"created") or b"").decode("utf-8", errors="replace"),
        "sentiment_score": (data.get(b"sentiment_score") or b"0").decode("utf-8"),
        "sentiment_label": (data.get(b"sentiment_label") or b"").decode("utf-8"),
        "tags": (data.get(b"tags") or b"").decode("utf-8", errors="replace"),
        "invalid_at": (data.get(b"invalid_at") or b"").decode("utf-8", errors="replace"),
    }


def mark_invalid(r: redis.Redis, key: str):
    """给碎片标记 invalid_at = 当前时间。"""
    now = datetime.now(timezone.utc).isoformat()
    r.hset(key, "invalid_at", now)
    logger.info("  → marked invalid: %s", key)


def run_dedup(
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 50,
    dry_run: bool = False,
):
    """
    主流程：扫描一批碎片，对每条搜相似碎片，融合评分后标记过期。
    """
    r = get_redis()
    syn_map = load_synonyms(r)

    logger.info(
        "Dedup run: threshold=%.2f, batch=%d, dry_run=%s",
        threshold,
        batch_size,
        dry_run,
    )

    # 扫描
    fragments, checked_set = scan_fragments(r, batch_size=batch_size)
    logger.info("Scanned %d fragments to check", len(fragments))

    stats = {"checked": 0, "matched": 0, "marked": 0, "skipped": 0}

    for frag in fragments:
        content = frag["content"]
        if not content or len(content) < 4:
            stats["skipped"] += 1
            continue

        stats["checked"] += 1

        # 用 BM25 搜相似碎片
        keywords = tokenize(content)[:8]
        if not keywords:
            stats["skipped"] += 1
            continue

        # 用 | 连接所有词（OR 语义）
        safe_terms = "|".join(_escape_query_term(t) for t in keywords)
        query_expr = f"@content:{safe_terms}"

        try:
            q = (
                Query(query_expr)
                .paging(0, 20)
                .dialect(2)
                .return_fields("content", "tags", "category", "created",
                               "sentiment_score", "sentiment_label")
            )
            search_result = r.ft("idx:memories").search(q)
        except Exception as e:
            logger.debug("Search error for %s: %s", frag["key"], e)
            continue

        similar_fragments = []
        for doc in search_result.docs:
            doc_key = doc.id
            if doc_key == frag["key"]:
                continue  # 跳过自身

            hdata = r.hgetall(doc_key)
            if not hdata or hdata.get(b"invalid_at"):
                continue

            # 提取每条 doc 独立的 BM25 分数
            doc_bm25 = float(getattr(doc, "score", 0.0))

            similar = {
                "key": doc_key,
                "content": (hdata.get(b"content") or b"").decode("utf-8", errors="replace"),
                "created": (hdata.get(b"created") or b"").decode("utf-8", errors="replace"),
                "sentiment_score": (hdata.get(b"sentiment_score") or b"0").decode("utf-8"),
                "sentiment_label": (hdata.get(b"sentiment_label") or b"").decode("utf-8"),
                "tags": (hdata.get(b"tags") or b"").decode("utf-8", errors="replace"),
                "invalid_at": (hdata.get(b"invalid_at") or b"").decode("utf-8"),
                "_bm25": doc_bm25,
            }
            similar_fragments.append(similar)

        if not similar_fragments:
            continue

        # 逐个融合评分
        for similar in similar_fragments:
            score = calc_fusion_score(
                frag, similar, syn_map,
                bm25_raw_score=similar.get("_bm25", 0.0),
            )

            if score >= threshold:
                stats["matched"] += 1
                # 取时间较早的标记为过期
                try:
                    t_a = datetime.fromisoformat(frag["created"])
                    t_b = datetime.fromisoformat(similar["created"])
                except (ValueError, TypeError):
                    continue

                older_key = frag["key"] if t_a < t_b else similar["key"]
                newer_key = similar["key"] if older_key == frag["key"] else frag["key"]

                logger.info(
                    "  MATCH [%.2f]: %s...  ↔  %s...",
                    score,
                    frag["content"][:40],
                    similar["content"][:40],
                )
                logger.info("    older=%s  newer=%s", older_key, newer_key)

                if not dry_run:
                    mark_invalid(r, older_key)
                    stats["marked"] += 1
                else:
                    stats["marked"] += 1

    # 保存已检查的 key
    if not dry_run and stats["checked"] > 0:
        pipe = r.pipeline()
        for frag in fragments:
            pipe.sadd(CHECKED_SET_KEY, frag["key"])
        pipe.execute()
        logger.info("Saved %d checked keys", len(fragments))

    logger.info(
        "Done: checked=%d, matched=%d, marked=%d, skipped=%d",
        stats["checked"],
        stats["matched"],
        stats["marked"],
        stats["skipped"],
    )

    return stats


# ══════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    threshold = DEFAULT_THRESHOLD
    batch_size = 50

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--threshold"):
            if "=" in arg:
                threshold = float(arg.split("=")[1])
            elif i + 1 < len(sys.argv):
                threshold = float(sys.argv[i + 1])
                i += 1
        elif arg.startswith("--batch-size"):
            if "=" in arg:
                batch_size = int(arg.split("=")[1])
            elif i + 1 < len(sys.argv):
                batch_size = int(sys.argv[i + 1])
                i += 1
        i += 1

    run_dedup(threshold=threshold, batch_size=batch_size, dry_run=dry_run)
