#!/usr/bin/env python3
"""Keepsake 记忆检索抽查集 — 用真实查询验证检索质量（防回归）

用法: python3 eval_spotcheck.py
每行: (查询, [期望命中的内容关键词, ...]) — top5 结果里任一包含关键词即算命中
"""
import json
import os
import sys

sys.path.insert(0, "/opt/fragmented-memory/src")

CFG = json.load(open(os.path.expanduser("~/.config/keepsake/config.json")))

# 抽查集: (查询, 期望关键词列表) — 基于记忆库真实内容
SPOTCHECK = [
    ("服务器密码", ["密码", "Redis"]),
    ("gost 代理怎么配", ["gost", "8443", "202"]),
    ("部署流程是什么", ["部署", "rsync", "88"]),
    ("FLUX 模型在哪下载", ["FLUX", "Kijai", "fp8"]),
    ("缠论怎么分析", ["缠论"]),
    ("数据库密码", ["密码", "MySQL", "MongoDB"]),
    ("202 服务器上有什么", ["202", "生产", "quartz"]),
    ("记忆插件状态", ["记忆", "keepsake", "Keepsake"]),
    ("定时任务有哪些", ["定时", "cron", "提炼"]),
    ("语义检索怎么实现的", ["语义", "embedding", "检索"]),
    ("怎么给 88 派活", ["88", "agent-worker", "任务"]),
    ("生图怎么弄", ["ComfyUI", "生图", "pipeline"]),
    ("怎么备份数据", ["备份", "数据库"]),
    ("量化回测", ["回测", "量化", "缠论"]),
    ("ClickHouse 数据", ["ClickHouse", "clickhouse"]),
    ("内存 swap 优化", ["swappiness", "swap"]),
    ("前端部署到哪", ["前端", "/opt/web", "web"]),
    ("项目放在哪个目录", ["claude_user", "/home/claude_user"]),
    ("记忆怎么自动提炼", ["提炼", "memory_distill", "qwen"]),
    ("PVE 上有什么服务", ["PVE", "容器", "ComfyUI"]),
]


def main():
    from keepsake.storage import RedisStorage
    from keepsake.embedder import create_embedder

    ec = CFG.get("embedder", {}) or {}
    embedder = None
    if ec.get("provider"):
        embedder = create_embedder(
            provider=ec.get("provider", ""),
            api_key=ec.get("api_key", ""),
            base_url=ec.get("base_url", ""),
            model=ec.get("model", ""),
        )
    stor = RedisStorage(
        host=CFG.get("redis_host", "127.0.0.1"),
        port=CFG.get("redis_port", 6379),
        password=CFG.get("redis_password") or None,
        embedder=embedder,
        is_primary=True,
    )

    passed = 0
    print(f"{'查询':<22} {'命中':<4} 期望关键词")
    print("-" * 60)
    for query, expect in SPOTCHECK:
        results = stor.search(query)
        top = [r.get("content", "") for r in results[:5]]
        hit_kw = [k for k in expect if any(k.lower() in c.lower() for c in top)]
        ok = len(hit_kw) > 0
        passed += ok
        mark = "✅" if ok else "❌"
        print(f"{query:<22} {mark:<4} {hit_kw if ok else '未命中: ' + str(expect)}")

    print("-" * 60)
    print(f"通过率: {passed}/{len(SPOTCHECK)} ({passed/len(SPOTCHECK)*100:.0f}%)")
    return 0 if passed / len(SPOTCHECK) >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())
