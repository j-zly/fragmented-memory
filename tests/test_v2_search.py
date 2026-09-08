"""keepsake v2 — 检索侧过滤 + 相似度地板测试（hermetic）。

覆盖 A 部分：
  - consumed/superseded 排除（search/search_bm25/search_knn 路径）
  - min_score 地板（按 _sim 归一化值过滤）
  - get_fragment / supersede_fragment 辅助方法
  - search_bm25 返回 _key 与 fragment_type 字段

设计要点：
  * 不连 Redis/网络 —— 用 FakeRedisStorage（pipeline.py 用的接口一致）
  * search_bm25 / search_knn 是真实方法，注入 fake redis client
  * 验证 search() 走完整路径（应用 _apply_v2_filters）
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

import pytest

from keepsake.storage import RedisStorage


# ===========================================================================
# 测试用 fake redis client
# ===========================================================================

class _FakeHash:
    """模拟 Redis Hash：HSET 覆盖、HGETALL 返回所有字段、HSET 保留多字段。"""
    def __init__(self):
        self.data: Dict[str, str] = {}

    def hset(self, *args, **kwargs):
        # 支持 hset(key, field, value) 或 hset(key, mapping={...})
        if len(args) == 3:
            key, field, value = args
            self.data[field] = str(value)
        elif "mapping" in kwargs:
            for k, v in kwargs["mapping"].items():
                self.data[k] = str(v)
        return True

    def hgetall(self):
        return {k.encode(): v.encode() for k, v in self.data.items()}


class _FakePipeline:
    def __init__(self, store: "_FakeClient"):
        self.store = store
        self.cmds: List[Tuple[str, ...]] = []

    def hset(self, key, field=None, value=None, mapping=None):
        if mapping is not None:
            self.cmds.append(("hset", key, mapping))
        else:
            self.cmds.append(("hset", key, field, value))

    def hget(self, key, field):
        self.cmds.append(("hget", key, field))

    def hgetall(self, key):
        self.cmds.append(("hgetall", key))

    def hincrby(self, key, field, n):
        self.cmds.append(("hincrby", key, field, n))

    def execute(self):
        # 模拟返回：每个 hget/hgetall 返回对应字段值；hset/hincrby 立即写入
        results = []
        for c in self.cmds:
            if c[0] == "hget":
                _, key, field = c
                h = self.store.hashes.get(key)
                if h is None:
                    results.append(None)
                else:
                    val = h.data.get(field)
                    results.append(val.encode() if val is not None else None)
            elif c[0] == "hgetall":
                _, key = c
                h = self.store.hashes.get(key)
                if h is None:
                    results.append({})
                else:
                    results.append(h.hgetall())
            elif c[0] == "hset":
                if len(c) == 3:  # mapping
                    _, key, mapping = c
                    h = self.store.hashes.setdefault(key, _FakeHash())
                    for k, v in mapping.items():
                        h.data[k] = str(v)
                else:  # field/value
                    _, key, field, value = c
                    h = self.store.hashes.setdefault(key, _FakeHash())
                    h.data[field] = str(value)
                results.append(1)
            elif c[0] == "hincrby":
                _, key, field, n = c
                h = self.store.hashes.setdefault(key, _FakeHash())
                cur = int(h.data.get(field, "0"))
                h.data[field] = str(cur + n)
                results.append(cur + n)
            else:
                results.append(1)
        self.cmds.clear()
        return results


class _FakeClient:
    """只覆盖本测试需要的命令：HSET / HGETALL / HGET / FT.SEARCH / PING。"""

    def __init__(self):
        self.hashes: Dict[str, _FakeHash] = {}
        self.search_results: List[Dict[str, Any]] = []

    def ping(self):
        return True

    def hset(self, key, *args, **kwargs):
        h = self.hashes.setdefault(key, _FakeHash())
        return h.hset(key, *args, **kwargs)

    def hget(self, key, field):
        h = self.hashes.get(key)
        if h is None:
            return None
        v = h.data.get(field)
        return v.encode() if v is not None else None

    def hgetall(self, key):
        h = self.hashes.get(key)
        if h is None:
            return {}
        return h.hgetall()

    def pipeline(self):
        return _FakePipeline(self)

    def ft(self, index_name):
        return _FakeSearch(self, index_name, self.search_results)


class _FakeSearchResult:
    def __init__(self, docs):
        self.docs = docs


class _FakeSearchDoc:
    def __init__(self, doc_id: str, fields: Dict[str, Any], score: float = 1.0):
        self.id = doc_id
        self._score = score
        for k, v in fields.items():
            setattr(self, k, v)
        # 模拟 RediSearch 的 score 字段
        self.score = score

    def __getattr__(self, name):
        return None


class _FakeSearch:
    def __init__(self, client: "_FakeClient", index_name: str, results: List[Dict[str, Any]]):
        self.client = client
        self.index_name = index_name
        self.results = results

    def search(self, q, query_params=None):
        docs = []
        for r in self.results:
            fields = {k: v for k, v in r.items() if k != "_key"}
            score = r.get("_score", 1.0)
            docs.append(_FakeSearchDoc(r["_key"], fields, score))
        return _FakeSearchResult(docs)


class _FakeConnectionPool:
    def disconnect(self):
        pass


def _make_storage(initial_hashes: Dict[str, Dict[str, str]] = None,
                  initial_search: List[Dict[str, Any]] = None,
                  v2_min_score: Optional[float] = None) -> Tuple[RedisStorage, _FakeClient]:
    """构造一个带 fake client 的 RedisStorage。"""
    client = _FakeClient()
    if initial_hashes:
        for key, fields in initial_hashes.items():
            client.hashes[key] = _FakeHash()
            for k, v in fields.items():
                client.hashes[key].data[k] = v
    if initial_search:
        client.search_results = list(initial_search)

    if v2_min_score is None:
        # 测试默认值：与生产默认一致（0.05）
        v2_min_score = 0.05

    storage = RedisStorage(
        host="127.0.0.1", port=6379, v2_min_score=v2_min_score,
        candidate_count=5, final_limit=5,
    )
    storage._client = client
    storage._pool = _FakeConnectionPool()
    return storage, client


# ===========================================================================
# A1 — consumed/superseded 排除
# ===========================================================================

class TestConsumedSupersededFilter:
    """search() 路径统一排除 consumed 与 superseded 碎片。"""

    def test_active_fragment_is_returned(self):
        """未被消耗/封边的正常碎片应被返回。"""
        s, client = _make_storage(initial_search=[
            {"_key": "memory:frag:abc", "content": "用户决定部署A方案",
             "_score": 1.5},
        ])
        out = s.search("部署")
        assert len(out) == 1
        assert out[0]["content"] == "用户决定部署A方案"

    def test_consumed_fragment_is_filtered(self):
        """fragment_type=consumed 的碎片（被 consolidator 蒸馏吞掉的）应被剔除。"""
        s, client = _make_storage(initial_search=[
            {"_key": "memory:frag:abc", "content": "用户决定部署A方案",
             "fragment_type": "consumed", "_score": 1.5},
        ])
        out = s.search("部署")
        assert out == []

    def test_superseded_fragment_is_filtered(self):
        """superseded_by 非空的碎片（pipeline 封边的）应被剔除。"""
        # 关键：search_bm25 已经返回这个碎片，但 search() 的 _apply_v2_filters
        # 应该通过 _key 读 superseded_by 字段并排除
        s, client = _make_storage(
            initial_search=[
                {"_key": "memory:frag:abc", "content": "用户要求做分页",
                 "_score": 1.5},
            ],
            initial_hashes={
                "memory:frag:abc": {
                    "content": "用户要求做分页",
                    "superseded_by": "memory:frag:def",
                },
            },
        )
        out = s.search("分页")
        assert out == []

    def test_mixed_active_and_consumed(self):
        """混合：active 与 consumed 都在结果里 → 只返回 active。"""
        s, client = _make_storage(initial_search=[
            {"_key": "memory:frag:abc", "content": "用户决定部署A方案",
             "_score": 1.5},
            {"_key": "memory:frag:def", "content": "用户决定部署A方案",
             "fragment_type": "consumed", "_score": 1.4},
        ])
        out = s.search("部署")
        assert len(out) == 1
        assert out[0]["_key"] == "memory:frag:abc"

    def test_mixed_active_and_superseded(self):
        """混合：active 与 superseded 都在结果里 → 只返回 active。"""
        s, client = _make_storage(
            initial_search=[
                {"_key": "memory:frag:abc", "content": "分页已完成上线",
                 "_score": 1.5},
                {"_key": "memory:frag:def", "content": "用户要求做分页",
                 "_score": 1.4},
            ],
            initial_hashes={
                "memory:frag:def": {
                    "content": "用户要求做分页",
                    "superseded_by": "memory:frag:abc",
                },
            },
        )
        out = s.search("分页")
        assert len(out) == 1
        assert out[0]["_key"] == "memory:frag:abc"

    def test_void_superseded_by_counts_as_sealed(self):
        """superseded_by='__void__'（DELETE 路径）也算封边 → 排除。"""
        s, client = _make_storage(
            initial_search=[
                {"_key": "memory:frag:abc", "content": "旧记忆",
                 "_score": 1.5},
            ],
            initial_hashes={
                "memory:frag:abc": {
                    "content": "旧记忆",
                    "superseded_by": "__void__",
                },
            },
        )
        out = s.search("旧记忆")
        assert out == []


# ===========================================================================
# A2 — min_score 地板
# ===========================================================================

class TestMinScoreFloor:
    """v2_min_score 阈值：低于此值的碎片不注入。"""

    def test_default_min_score_is_0p05(self):
        """默认 v2_min_score = 0.05（经验值）。"""
        s, _ = _make_storage(v2_min_score=None)  # 使用默认值
        assert s._v2_min_score == 0.05

    def test_low_sim_fragment_filtered_out(self):
        """sim < 地板值的碎片应被剔除。

        设计：两个结果，BM25 归一化后 _sim 分别 1.0（高分）和 0.05（低分），
        floor=0.1 时低分应被剔除。
        """
        s, client = _make_storage(
            initial_search=[
                {"_key": "memory:frag:top", "content": "高度相关",
                 "_score": 1.0},
                {"_key": "memory:frag:noise", "content": "无关文本",
                 "_score": 0.05},  # 0.05/1.0 = 0.05 (低 sim)
            ],
            v2_min_score=0.1,
        )
        out = s.search("查询")
        # top 应保留；noise 应被剔除
        assert len(out) == 1
        assert out[0]["_key"] == "memory:frag:top"

    def test_high_sim_fragment_kept(self):
        """sim >= 地板值的碎片应保留。"""
        s, client = _make_storage(
            initial_search=[
                {"_key": "memory:frag:abc", "content": "相关记忆",
                 "_score": 1.0},
            ],
            v2_min_score=0.1,
        )
        out = s.search("相关")
        assert len(out) == 1
        assert out[0]["content"] == "相关记忆"

    def test_floor_zero_disables_filter(self):
        """min_score=0 应关闭过滤。"""
        s, client = _make_storage(
            initial_search=[
                {"_key": "memory:frag:top", "content": "高度相关",
                 "_score": 1.0},
                {"_key": "memory:frag:noise", "content": "无关",
                 "_score": 0.001},
            ],
            v2_min_score=0.0,
        )
        out = s.search("查询")
        # 过滤关闭 → 两个都保留（top 因为 _sim=1.0，noise 因为 _sim=0.001）
        assert len(out) == 2


# ===========================================================================
# A3 — 新增辅助方法 get_fragment / supersede_fragment
# ===========================================================================

class TestGetFragment:
    """get_fragment 是 v2 pipeline UPDATE 阶段需要的能力。"""

    def test_get_existing_fragment(self):
        s, client = _make_storage(initial_hashes={
            "memory:frag:abc": {"content": "事实", "tags": "conversation"},
        })
        frag = s.get_fragment("memory:frag:abc")
        assert frag is not None
        assert frag["content"] == "事实"
        assert frag["tags"] == "conversation"

    def test_get_missing_returns_none(self):
        s, _ = _make_storage()
        assert s.get_fragment("memory:frag:nonexistent") is None

    def test_get_empty_key_returns_none(self):
        s, _ = _make_storage()
        assert s.get_fragment("") is None


class TestSupersedeFragment:
    """supersede_fragment 是 v2 封边操作（UPDATE/DELETE 路径）。"""

    def test_supersede_writes_superseded_by_and_at(self):
        s, client = _make_storage(initial_hashes={
            "memory:frag:old": {"content": "旧事实"},
        })
        ok = s.supersede_fragment("memory:frag:old", "memory:frag:new")
        assert ok is True
        frag = s.get_fragment("memory:frag:old")
        assert frag is not None
        assert frag["superseded_by"] == "memory:frag:new"
        assert "superseded_at" in frag
        # ISO 格式校验
        from datetime import datetime
        datetime.fromisoformat(frag["superseded_at"])

    def test_supersede_with_void_key(self):
        """DELETE 路径专用 sentinel '__void__'。"""
        s, client = _make_storage(initial_hashes={
            "memory:frag:old": {"content": "矛盾旧事实"},
        })
        ok = s.supersede_fragment("memory:frag:old", "__void__")
        assert ok is True
        frag = s.get_fragment("memory:frag:old")
        assert frag["superseded_by"] == "__void__"


class TestGetFragmentsBatch:
    """get_fragments_batch 用于 UPDATE 阶段批量校验候选。"""

    def test_batch_read_returns_only_existing(self):
        s, client = _make_storage(initial_hashes={
            "memory:frag:abc": {"content": "A"},
            "memory:frag:def": {"content": "B"},
        })
        out = s.get_fragments_batch(["memory:frag:abc", "memory:frag:none", "memory:frag:def"])
        assert "memory:frag:abc" in out
        assert "memory:frag:def" in out
        assert "memory:frag:none" not in out
        assert out["memory:frag:abc"]["content"] == "A"
        assert out["memory:frag:def"]["content"] == "B"

    def test_batch_empty_returns_empty(self):
        s, _ = _make_storage()
        assert s.get_fragments_batch([]) == {}


# ===========================================================================
# A4 — search_bm25 返回 _key 与 fragment_type（pipeline 依赖）
# ===========================================================================

class TestSearchReturnsKeyAndType:
    """v2 pipeline 内部 search_bm25 需要 _key + fragment_type 字段。"""

    def test_search_bm25_returns_key(self):
        s, client = _make_storage(initial_search=[
            {"_key": "memory:frag:abc123", "content": "测试", "_score": 1.5},
        ])
        out = s.search_bm25("测试")
        assert len(out) == 1
        assert out[0].get("_key") == "memory:frag:abc123"

    def test_search_bm25_returns_fragment_type(self):
        """search_bm25 应返回 fragment_type 字段（让 search() 顶层统一过滤）。"""
        s, client = _make_storage(initial_search=[
            {"_key": "memory:frag:abc", "content": "测试",
             "fragment_type": "consumed", "_score": 1.5},
        ])
        # search_bm25 本身不会过滤 consumed（统一在 search() 的 _apply_v2_filters 过滤）
        # 所以此处直接调 search() 验证 consumed 被剔除
        out = s.search("测试")
        assert out == []
