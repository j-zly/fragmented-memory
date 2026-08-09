#!/usr/bin/env python3
"""本地模型自动记忆提炼 — 扫最近会话 → qwen3:8b 提炼 → 写入 Keepsake

用法: python3 memory_distill.py [--hours 2] [--max-chars 4000] [--dry-run]
"""
import argparse
import json
import sqlite3
import sys
import time
import urllib.request
import subprocess
import os
import yaml

OLLAMA = "http://127.0.0.1:11434"
MODEL = "qwen3:8b"
DB = "/root/.hermes/state.db"
WATERMARK_FILE = "/tmp/memory_distill_watermark"
CONF_FILE = os.path.expanduser("~/scripts/memory_distill.conf")
PVE_SSH = ["ssh", "-p", "2224", "-o", "StrictHostKeyChecking=no",
           "-o", "ConnectTimeout=8", "root@127.0.0.1"]


def load_conf():
    """读开关配置: {"enabled": true, "hours": 2, "max_chars": 4000}"""
    try:
        with open(CONF_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

DISTILL_PROMPT = """你是记忆提炼助手。从下面的对话中提炼「值得长期记住」的信息，输出 JSON 数组。
只提炼：用户偏好/习惯、项目事实、环境配置、技术决策、踩坑教训、用户身份信息。
忽略：寒暄、临时任务进度、纯工具输出、重复内容。
每条: {"content": "一句话记忆内容(中文)", "category": "preference|fact|lesson|project|identity", "tags": "逗号分隔关键词"}
要求:
- content 具体明确，不写模糊的废话
- 最多 8 条，宁缺毋滥
- 只输出 JSON 数组，不要任何解释

对话:
{conversation}"""


def get_recent_messages(hours, last_id=0):
    """从 state.db 读增量消息（watermark 之后）"""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    ts = time.time() - hours * 3600
    rows = conn.execute(
        """SELECT id, role, content, timestamp FROM messages
           WHERE role IN ('user','assistant') AND content IS NOT NULL
           AND id > ? AND timestamp > ?
           ORDER BY id DESC LIMIT 300""", (last_id, ts)).fetchall()
    conn.close()
    max_id = rows[0][0] if rows else last_id
    # 按时间正序
    rows.reverse()
    # 截断超长内容
    out = []
    for mid, role, content, ts in rows:
        c = content.strip()
        if not c or len(c) < 20:
            continue
        if len(c) > 3000:
            c = c[:3000]
        out.append(f"[{role}] {c}")
    return "\n".join(out), max_id


def distill(conversation, max_chars):
    """调 qwen3:8b 提炼记忆"""
    if len(conversation) > max_chars:
        conversation = conversation[-max_chars:]
    prompt = DISTILL_PROMPT.replace("{conversation}", conversation)
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2, "num_predict": 1024}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=240).read())
        text = resp.get("response", "")
    except Exception as e:
        print(f"ollama 调用失败: {e}")
        return []
    # 提取 JSON 数组（容错：可能有多余文字）
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start:end + 1])
        # 过滤: 太短/无实质内容
        out = []
        for i in items:
            if not isinstance(i, dict):
                continue
            c = (i.get("content") or "").strip()
            if len(c) < 10:
                continue
            i["content"] = c
            out.append(i)
        return out
    except Exception:
        return []


def read_watermark():
    try:
        with open(WATERMARK_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def write_watermark(mid):
    try:
        with open(WATERMARK_FILE, "w") as f:
            f.write(str(mid))
    except Exception:
        pass


def get_redis_password():
    """从 Keepsake 配置 (~/.config/keepsake/config.json) 读 Redis 连接（不硬编码）"""
    try:
        with open(os.path.expanduser("~/.config/keepsake/config.json")) as f:
            cfg = json.load(f)
        return cfg.get("redis_password", ""), cfg.get("redis_host", "127.0.0.1"), cfg.get("redis_port", 6379)
    except Exception:
        return "", "127.0.0.1", 6379


def store_to_keepsake(items, dry_run):
    """写入 Keepsake (RedisStorage 直连)"""
    sys.path.insert(0, "/opt/fragmented-memory/src")
    from keepsake.storage import RedisStorage
    pwd, r_host, r_port = get_redis_password()
    stor = RedisStorage(host=r_host, port=r_port,
                        password=pwd or None)
    saved = 0
    for it in items:
        cat = it.get("category", "fact")
        tags = f"auto-distill,{cat}"
        if it.get("tags"):
            tags += "," + it["tags"]
        if dry_run:
            print(f"[DRY] [{cat}] {it['content']}")
            continue
        ok = stor.store(text=it["content"], tags=tags, category=cat,
                        source="auto-distill", fragment_type=cat)
        if ok:
            saved += 1
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 开关配置: enabled=false 时直接退出（crontab 保持挂着，改配置即可开关）
    conf = load_conf()
    if conf.get("enabled") is False:
        print("memory_distill 已禁用 (memory_distill.conf enabled=false)")
        return
    hours = args.hours or conf.get("hours", 2)
    max_chars = args.max_chars or conf.get("max_chars", 4000)

    last_id = read_watermark()
    conv, max_id = get_recent_messages(hours, last_id)
    if not conv:
        print(f"无新对话可提炼 (watermark={last_id})")
        return
    print(f"对话长度: {len(conv)} 字符 (watermark {last_id} → {max_id})")

    # 冲突检查: ComfyUI 队列在跑图时跳过 (qwen3:8b 自己常驻5.5GB, 不能用显存阈值判断)
    try:
        q = subprocess.run(PVE_SSH + ["curl", "-s", "-m", "8", "http://127.0.0.1:8188/queue"],
                           capture_output=True, text=True, timeout=15)
        import json as _json
        qd = _json.loads(q.stdout or "{}")
        if qd.get("queue_running") or qd.get("queue_pending"):
            print("ComfyUI 队列忙, 跳过(可能跑图中)")
            return
    except Exception:
        pass

    items = distill(conv, max_chars)
    print(f"提炼出 {len(items)} 条记忆")
    if not items:
        return
    saved = store_to_keepsake(items, args.dry_run)
    print(f"已写入 {saved} 条 (dry_run={args.dry_run})")
    if not args.dry_run and max_id > last_id:
        write_watermark(max_id)
        print(f"watermark 更新为 {max_id}")


if __name__ == "__main__":
    main()
