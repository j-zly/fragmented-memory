"""Consolidation 引擎 — 将同主题碎片分层提炼为更高层记忆。

工作流程:
  1. 扫描所有未合并的碎片（fragment_type != "consolidated"）
  2. 用 jieba 关键词提取做主题分组
  3. 每组超过 min_group_size 条时，调 LLM 合并为一条高层次摘要
  4. 存为新碎片（fragment_type="consolidated", level=N+1）
  5. 删除原始碎片（或标记已合并）

配置参数:
  - min_group_size: 最少多少条碎片才触发合并（默认 3）
  - max_age_hours: 只合并超过此年龄的碎片（给新碎片时间积累，默认 72h）
  - llm_model: DashScope 模型名（默认 qwen-turbo，便宜够用）
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# DashScope API 端点
DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 默认参数
DEFAULT_MIN_GROUP_SIZE = 2  # 有重复内容就合
DEFAULT_MAX_AGE_HOURS = 72
DEFAULT_LLM_MODEL = "qwen-plus"
DEFAULT_BATCH_SIZE = 200  # 每次 consolidate 扫描的碎片数

# LLM 超时
LLM_TIMEOUT = 30

# 合并提示词
CONSOLIDATE_PROMPT = """你是一位知识提炼专家。以下是一组关于同一话题的对话片段。

请将它们合并成一条简洁、信息完整的高层记忆条目，要求：
1. 保留所有关键事实和结论，不丢信息
2. 去掉重复内容
3. 用陈述句表达，像一条知识条目
4. 如果片段之间存在矛盾，指出矛盾但不选边
5. 控制在 200 字以内

对话片段：
{segments}

合并后的知识条目："""


def _get_api_key() -> str:
    """获取 DashScope API key。"""
    key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")
    # fallback: 从 config.yaml 用 yaml 解析，优先取 providers.dashscope.api_key
    if not key:
        try:
            import yaml
            config_path = os.path.expanduser("~/.hermes/config.yaml")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg:
                    key = (
                        cfg.get("providers", {}).get("dashscope", {}).get("api_key")
                        or cfg.get("model", {}).get("api_key")
                        or ""
                    )
                    key = key.strip().strip("'\"")
        except Exception:
            pass
    return key


def resolve_llm_channel(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """解析 LLM 通道配置 —— base_url / model / api_key。

    优先级（高→低）:
      1. `cfg["llm"]` 节存在 → 取 base_url / model / api_key（直填优先）或 key_file（读文件）
      2. 无 `llm` 节或字段缺失 → 缺省回落现有 dashscope 路径（DASHSCOPE_BASE + DEFAULT_LLM_MODEL + _get_api_key）
      3. key_file 读失败 → 视同无 key，回落 _get_api_key（不抛）

    返回 dict 字段: base_url, model, api_key, source, key_file。
    日志安全：logger 只允许出现 key_file 路径 / 端点 host，绝不打印 key 内容。
    """
    llm_cfg: Dict[str, Any] = {}
    if cfg and isinstance(cfg, dict):
        llm_cfg = cfg.get("llm") or {}
        if not isinstance(llm_cfg, dict):
            llm_cfg = {}

    # 0. 缺省回落：dashscope 现状 —— 零配置=原样，行为向后兼容
    if not llm_cfg:
        return {
            "base_url": DASHSCOPE_BASE,
            "model": DEFAULT_LLM_MODEL,
            "api_key": _get_api_key(),
            "source": "dashscope_legacy",
            "key_file": "",
        }

    # 1. base_url —— rstrip("/") 防用户手抖
    base_url = (llm_cfg.get("base_url") or DASHSCOPE_BASE).rstrip("/")

    # 2. model —— 缺省用模块常量，缺省语义保持 qwen-plus
    model = (llm_cfg.get("model") or DEFAULT_LLM_MODEL).strip()

    # 3. api_key：api_key 直填 > key_file 读取 > _get_api_key() 兜底链
    api_key = (llm_cfg.get("api_key") or "").strip()
    key_file_path = llm_cfg.get("key_file") or ""
    if not api_key and key_file_path:
        try:
            with open(key_file_path) as f:
                api_key = f.read().strip()
            # 仅打印路径（host 等价信息），绝不打印 key 内容
            logger.debug(
                "resolve_llm_channel: loaded key_file=%s (len=%d)",
                key_file_path, len(api_key),
            )
        except (OSError, IOError) as e:
            # 读文件失败 → 视同无 key，走原回落链，不抛
            logger.debug(
                "resolve_llm_channel: key_file=%s unreadable: %s — fallback",
                key_file_path, e,
            )
            api_key = ""
    if not api_key:
        # 最后兜底：原 hermes/env 链（dashscope 等）
        api_key = _get_api_key()

    # 4. source 仅用于日志/监控归类（不影响行为）
    source = "configured"
    if "bigmodel.cn" in base_url:
        source = "bigmodel"
    elif "dashscope" in base_url:
        source = "dashscope"
    elif "openai.com" in base_url:
        source = "openai"

    return {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "source": source,
        "key_file": key_file_path,
    }


def _call_llm(messages: List[Dict[str, str]], model: str = DEFAULT_LLM_MODEL,
              *, channel: Optional[Dict[str, Any]] = None,
              max_retries: int = 2) -> Optional[str]:
    """调用 chat API 获取 LLM 回复。带重试。

    channel: 由 resolve_llm_channel 解析出的通道字典（含 base_url/model/api_key）；
             None → 走 dashscope 兜底链（向后兼容）。
    channel['model'] 优先于入参 model；二者都不存在时用 DEFAULT_LLM_MODEL。
    """
    if channel is None:
        channel = resolve_llm_channel(None)

    api_key = channel.get("api_key", "")
    if not api_key:
        logger.warning(
            "consolidator: no API key for LLM calls (source=%s)",
            channel.get("source", "?"),
        )
        return None

    base_url = channel["base_url"]
    actual_model = channel.get("model") or model or DEFAULT_LLM_MODEL
    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": actual_model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3,
    }).encode("utf-8")

    for attempt in range(1 + max_retries):
        if attempt > 0:
            wait = 2.0 * (2 ** (attempt - 1))  # 2s, 4s
            logger.debug("consolidator: retry %d/%d after %.0fs", attempt, max_retries, wait)
            time.sleep(wait)

        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                data = json.loads(resp.read())
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
            # Got response but no choices — don't retry
            logger.warning("consolidator: LLM returned no choices (attempt %d)", attempt + 1)
            return None
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            if attempt < max_retries:
                logger.debug("consolidator: LLM attempt %d failed: %s", attempt + 1, e)
            else:
                logger.warning("consolidator: LLM call failed after %d attempts: %s",
                               attempt + 1, e)
    return None


class Consolidator:
    """碎片分层提炼引擎。"""

    def __init__(
        self,
        storage: Any,  # RedisStorage instance (avoid circular import)
        min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
        max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
        llm_model: str = DEFAULT_LLM_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        channel: Optional[Dict[str, Any]] = None,
    ):
        self._storage = storage
        self._min_group_size = min_group_size
        self._max_age_hours = max_age_hours
        self._llm_model = llm_model
        self._batch_size = batch_size
        # channel：可选 resolve_llm_channel 输出；None → _call_llm 内部兜底
        self._channel = channel

    def consolidate(self) -> Dict[str, Any]:
        """执行一轮碎片合并。返回操作统计。"""
        client = self._storage._get_client()
        if not client:
            return {"status": "error", "reason": "Redis not available"}

        stats = {"scanned": 0, "groups_found": 0, "merged": 0, "skipped": 0, "errors": 0}

        # 1. 扫描未合并的碎片
        fragments = self._scan_unconsolidated(client)
        stats["scanned"] = len(fragments)
        if not fragments:
            return stats

        # 2. 按主题聚类
        groups = self._cluster_by_topic(fragments)
        stats["groups_found"] = len(groups)

        # 3. 对每个符合条件的组执行合并
        for group in groups:
            if len(group) < self._min_group_size:
                stats["skipped"] += len(group)
                continue

            result = self._merge_group(client, group)
            if result:
                stats["merged"] += len(group)
            else:
                stats["errors"] += len(group)

        return stats

    def _scan_unconsolidated(self, client) -> List[Dict[str, Any]]:
        """扫描符合合并条件的碎片。

        条件:
          - fragment_type != "consumed"（未被更高层合并吞掉的）
          - 创建时间 > max_age_hours（给新碎片时间积累）
          - 已合并的（consolidated）也参与扫描，实现多级提炼
        """
        try:
            cutoff = (datetime.now(timezone.utc).timestamp() - self._max_age_hours * 3600)
            cursor = 0
            fragments = []

            while True:
                cursor, keys = client.scan(
                    cursor=cursor,
                    match="memory:frag:*",
                    count=self._batch_size,
                )

                if not keys:
                    if cursor == 0:
                        break
                    continue

                # 用 pipeline 批量 HMGETALL，减少网络往返
                pipe = client.pipeline()
                for key_b in keys:
                    pipe.hgetall(key_b)
                pipe_results = pipe.execute()

                for key_b, data in zip(keys, pipe_results):
                    key = key_b.decode("utf-8") if isinstance(key_b, bytes) else key_b
                    if not data:
                        continue

                    # 解码
                    doc = {}
                    for k_b, v_b in data.items():
                        k = k_b.decode("utf-8") if isinstance(k_b, bytes) else k_b
                        v = v_b.decode("utf-8") if isinstance(v_b, bytes) else v_b
                        doc[k] = v

                    # 跳过已被更高层合并吞掉的
                    if doc.get("fragment_type", "") == "consumed":
                        continue

                    # 检查年龄
                    created_str = doc.get("created", "")
                    if created_str:
                        try:
                            created_ts = datetime.fromisoformat(created_str).timestamp()
                            if created_ts > cutoff:
                                continue  # 太新，等下次
                        except (ValueError, TypeError):
                            pass

                    doc["_key"] = key
                    fragments.append(doc)

                if cursor == 0:
                    break

            return fragments

        except Exception as e:
            logger.warning("consolidator: scan error: %s", e)
            return []

    def _cluster_by_topic(self, fragments: List[Dict]) -> List[List[Dict]]:
        """按关键词重叠做简单聚类。

        策略:
          - 对每个碎片提取关键词（用 jieba）
          - 关键词重叠 >= 2 的归为一组
          - 贪心算法，不追求最优聚类
        """
        from .splitter import extract_keywords

        # 提取每个碎片的关键词
        frag_data = []
        for f in fragments:
            content = f.get("content", "")
            if not content:
                continue
            kws = set(extract_keywords(content, max_keywords=5))
            frag_data.append({"frag": f, "keywords": kws})

        if not frag_data:
            return []

        # 贪心聚类
        groups: List[List[Dict]] = []
        assigned = set()

        for i, data in enumerate(frag_data):
            if i in assigned:
                continue
            group = [data["frag"]]
            assigned.add(i)

            for j, other in enumerate(frag_data):
                if j in assigned:
                    continue
                # 重叠 >= 2 个关键词
                overlap = len(data["keywords"] & other["keywords"])
                if overlap >= 2:
                    group.append(other["frag"])
                    assigned.add(j)

            groups.append(group)

        return groups

    def _merge_group(self, client, group: List[Dict]) -> bool:
        """用 LLM 合并一组碎片。"""
        # 计算新层级：取组内最高 level + 1
        max_level = 1
        for f in group:
            try:
                lv = int(f.get("level", "1"))
                if lv > max_level:
                    max_level = lv
            except (ValueError, TypeError):
                pass
        new_level = max_level + 1

        # 判断是否已有合并过的碎片
        has_consolidated = any(
            f.get("fragment_type") == "consolidated" or int(f.get("level", "1")) > 1
            for f in group
        )

        # 准备片段文本
        segments = []
        tags_set = set()
        for f in group:
            content = f.get("content", "")
            if content:
                segments.append(f"• {content[:300]}")
            tags = f.get("tags", "")
            if tags:
                for t in tags.split(","):
                    t = t.strip()
                    if t and not t.startswith("session:"):
                        tags_set.add(t)

        if len(segments) < self._min_group_size:
            return False

        # 根据是否已有提炼过的内容选择不同 prompt
        if has_consolidated:
            prompt = (
                "以下是一组已经提炼过的记忆条目和相关的原始对话片段。"
                "请将它们进一步提炼合并成一条更精炼的高层知识条目。\n\n"
                + "\n".join(segments)
                + "\n\n提炼后的高层知识条目："
            )
        else:
            prompt = CONSOLIDATE_PROMPT.format(segments="\n".join(segments))
        result = _call_llm([
            {"role": "system", "content": "你是一位知识提炼专家，擅长从对话中提取核心信息。"},
            {"role": "user", "content": prompt},
        ], model=self._llm_model, channel=self._channel)

        if not result:
            return False

        # 分析情绪
        from .splitter import analyze_sentiment
        sent_score, sent_label = analyze_sentiment(result)
        now_str = datetime.now(timezone.utc).isoformat()

        mapping = {
            "content": result,
            "tags": ",".join(sorted(tags_set)) if tags_set else "",
            "category": "consolidated",
            "source": "consolidator",
            "created": now_str,
            "fragment_type": "consolidated",
            "level": str(new_level),  # 多级：原始=1，首次合并=2，二次合并=3...
            "sentiment_score": str(sent_score),
            "sentiment_label": sent_label,
            "feedback_score": "0",
        }

        # 存 consolidated 碎片
        import hashlib
        content_hash = hashlib.sha256(result.encode()).hexdigest()[:12]
        consolidated_key = f"memory:frag:{content_hash}"

        # 如果没有相同 key（去重检查），就存
        existing = client.exists(consolidated_key)
        if existing:
            logger.debug("consolidator: duplicate consolidated result, skipping")
        else:
            client.hset(consolidated_key, mapping=mapping)

        # 软删除原始碎片（标记为已消费，不硬删）
        from datetime import datetime as _dt
        now_iso = _dt.now(timezone.utc).isoformat()
        consumed_count = 0
        for f in group:
            key = f.get("_key")
            if key:
                try:
                    client.hset(key, "consumed_by", consolidated_key)
                    client.hset(key, "consumed_at", now_iso)
                    client.hset(key, "fragment_type", "consumed")
                    consumed_count += 1
                except Exception:
                    pass

        logger.info(
            "consolidator: merged %d fragments → '%s...' (marked %d as consumed)",
            len(group), result[:60], consumed_count,
        )
        return True
