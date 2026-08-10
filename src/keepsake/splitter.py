"""文本切分工具 + 情感分析 — 按语义完整性切分文本，检测情感倾向。"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import jieba
import jieba.posseg  # 显式导入（该版本 jieba 无 lazy 加载，否则 posseg.lcut 抛 AttributeError 被吞）

# 自定义领域词典路径（由 discover_synonyms 自动生成）
_DOMAIN_DICT = Path.home() / '.config' / 'keepsake' / 'jieba_dict.txt'


def init_domain_dict() -> None:
    """加载/重载自定义领域词典。

    首次 import 时自动调用一次。插件 initialize() 时也调用一次，
    确保发 /new 后词典被重新加载（此时词典文件可能已被 discover_synonyms 更新）。
    重复调用安全（jieba.load_userdict 是累加的）。
    """
    if _DOMAIN_DICT.exists():
        jieba.load_userdict(str(_DOMAIN_DICT))


# 首次 import 时自动加载
init_domain_dict()

NEWLINE = "\n"

# ---------------------------------------------------------------------------
# 常见英文缩写（不在此边界断句）
# ---------------------------------------------------------------------------

_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "dept",
    "inc", "ltd", "co", "corp", "capt", "gen", "sgt", "lt", "maj", "col",
    "gov", "rep", "sen", "pres", "vice", "pres", "hon", "esq", "phd", "md",
    "ave", "blvd", "rd", "ct", "dr", "est", "inst", "univ",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "al", "fig", "vol", "no", "pp", "ex",
})

_STOP_WORDS = {
    "这个", "那个", "什么", "怎么", "为什么", "可以", "没有",
    "但是", "如果", "因为", "所以", "而且", "然后", "还是",
    "就是", "不是", "一个", "我们", "你们", "他们", "已经",
    "可以", "可能", "应该", "需要", "这样", "那样", "这里",
    "那里", "这个", "这些", "那些", "之后", "之前", "时候",
    "the", "this", "that", "what", "why", "how", "and",
    "but", "for", "with", "not", "are", "was", "had",
    "its", "has", "all", "can", "use", "get", "set",
    "的", "了", "在", "是", "我", "有", "和", "就", "不",
    "人", "都", "一", "一个", "上", "也", "很", "到", "说",
    "要", "去", "你", "会", "着", "没有", "看", "好", "自己",
    "这",
}


# ---------------------------------------------------------------------------
# 情感关键词表（支持中英文）
# ---------------------------------------------------------------------------

_POSITIVE_KEYWORDS: Dict[str, float] = {
    # 强烈正面 (×1.5)
    "太棒了": 1.5, "太好了": 1.5, "非常满意": 1.5, "爱了": 1.5,
    "绝了": 1.5, "牛逼": 1.5, "太强了": 1.5,
    # 中等正面 (×1.0)
    "喜欢": 1.0, "不错": 1.0, "好用": 1.0, "满意": 1.0,
    "推荐": 1.0, "值得": 1.0, "好使": 1.0, "方便": 0.8,
    "清晰": 0.8, "直观": 0.8, "nice": 0.8, "good": 0.8,
    "great": 1.0, "awesome": 1.2, "excellent": 1.3,
}

_NEGATIVE_KEYWORDS: Dict[str, float] = {
    # 强烈负面 (×1.5)
    "太差了": 1.5, "垃圾": 1.5, "恶心": 1.5, "废物": 1.5,
    "烂": 1.3, "垃圾东西": 1.5,
    # 中等负面 (×1.0)
    "讨厌": 1.0, "不好用": 1.0, "不满意": 1.0, "糟糕": 1.0,
    "烦": 0.8, "失望": 1.0, "差评": 1.0, "不行": 0.8,
    "bad": 1.0, "terrible": 1.3, "hate": 1.2, "useless": 1.0,
}

# 否定词 — 出现在情感词前 3 个字符内会反转情感
_NEGATORS = {"不", "没", "别", "勿", "无", "not", "no", "n't", "never"}


def analyze_sentiment(text: str) -> Tuple[float, str]:
    """分析文本情感倾向。

    返回 (sentiment_score, sentiment_label):
        sentiment_score: -1.0 ~ 1.0, 负值=负面, 正值=正面
        sentiment_label: "positive" / "negative" / "neutral"

    算法:
      1. 扫描正负面关键词，匹配时检测前文否定词
      2. 累计正负得分，归一化到 -1~1
      3. |score| < 0.15 视为 neutral
    """
    if not text or not text.strip():
        return 0.0, "neutral"

    text_lower = text.lower().strip()
    pos_score = 0.0
    neg_score = 0.0

    # 扫描正面关键词
    for kw, intensity in _POSITIVE_KEYWORDS.items():
        idx = text_lower.find(kw.lower())
        if idx == -1:
            continue
        # 检查前文是否有否定词
        start = max(0, idx - 4)
        prefix = text_lower[start:idx].strip()
        negated = any(n in prefix for n in _NEGATORS)
        if negated:
            neg_score += intensity * 0.5
        else:
            pos_score += intensity

    # 扫描负面关键词
    for kw, intensity in _NEGATIVE_KEYWORDS.items():
        idx = text_lower.find(kw.lower())
        if idx == -1:
            continue
        start = max(0, idx - 4)
        prefix = text_lower[start:idx].strip()
        negated = any(n in prefix for n in _NEGATORS)
        if negated:
            pos_score += intensity * 0.5
        else:
            neg_score += intensity

    # 归一化到 -1~1
    total = pos_score + neg_score
    if total == 0:
        return 0.0, "neutral"

    score = (pos_score - neg_score) / max(total, 0.1)
    score = max(-1.0, min(1.0, score))

    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"

    return round(score, 4), label


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """从文本中提取关键词（基于 jieba 分词）。

    使用 jieba 进行中文分词 + 词频统计，过滤停用词后返回
    高频词作为关键词。英文词单独提取（3 字母以上）。
    """
    if not text or not text.strip():
        return []

    text_lower = text.lower().strip()
    candidates: List[str] = []

    # 1. 用 jieba 做中文分词
    words = jieba.lcut(text)
    # 过滤停用词 + 长度 >= 2（单字词一般是语气词/助词）
    chinese_words = [w for w in words
                     if len(w) >= 2
                     and w not in _STOP_WORDS
                     and not w.isdigit()
                     and len(set(w)) > 1]  # 过滤 "哈哈" "AA" 类重复词
    candidates.extend(chinese_words)

    # 2. 提取英文词（3 字母以上）
    eng_words = re.findall(r"\b[a-zA-Z]{3,}\b", text_lower)
    candidates.extend([w for w in eng_words if w not in _STOP_WORDS])

    # 3. 按频次降序
    freq = Counter(candidates)
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:max_keywords]]


# ---------------------------------------------------------------------------
# 句子切分
# ---------------------------------------------------------------------------

# 预编译边界正则 — 同时匹配中文和英文句末标点
# 中文：。！？ 后直接切（不跟在实际可能出现的引号后）
# 英文：.！？后跟空白、大写字母、标点或行尾才切
# 保护场景在后处理 _split_sentences 中处理
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[。！？])'                                            # 中文句末标点
    r'|'
    r'(?<=[！？])'                                              # 中英文通用叹号/问号
    r'|'
    # 英文 .!? 后跟空白字符或文本结束才切
    r'(?<=[.!?])(?=\s|$)'                                      # 仅空白或行尾
)


def _split_sentences(text: str) -> List[str]:
    """将段落切分为句级片段，优先保留语义完整。

    保护规则:
      - 数字间句点（3.14、v1.0）
      - 缩写句点（Mr.、Dr.、U.S.A.）
      - 省略号（... → …）
      - 中文 。 后跟引号不单独切
    """
    # 1. 归一化连点 → …（保留至少 3 个点时才归一化）
    text = re.sub(r'\.{3,}', '…', text)
    text = re.sub(r'…{2,}', '…', text)

    # 2. 用正则切分
    raw_parts = _SENTENCE_SPLIT_RE.split(text)

    # 3. 后处理：合并因缩写/数字/单字母误切的部分
    merged: List[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        if merged:
            last = merged[-1]
            # 如果上一片段的末尾看起像是一个缩写或数字（如 Mr、3.14、U.S.）
            # 或者当前片段很短（<=3 字符）且不以大写字母开头 → 应合并
            prev_tail = last.rstrip()
            is_abbrev = (
                _looks_like_abbreviation(prev_tail)
                or _ends_with_digit_dot(prev_tail)
            )
            is_fragment = len(part) <= 3 and not part[0].isupper()
            if is_abbrev or is_fragment:
                merged[-1] = last + part
                continue

        merged.append(part)

    return merged


def _looks_like_abbreviation(text: str) -> bool:
    """检查文本末尾是否像缩写（Mr.、Dr.、U.S.A.、etc.）。"""
    # 去掉末尾空白
    text = text.rstrip()
    m = re.search(r'\b([A-Za-z]{1,5})\.$', text)
    if not m:
        return False
    word = m.group(1).lower()
    # 在缩写列表中，或是 1-2 个全大写字母（如 U.S. -> U 和 S 各一段）
    return word in _ABBREVIATIONS or (word.isupper() and len(word) <= 2)


def _ends_with_digit_dot(text: str) -> bool:
    """检查文本是否以数字+句点结尾（如 '3.14' 中的 '3.'）。"""
    return bool(re.search(r'\d\.$', text.rstrip()))


# ---------------------------------------------------------------------------
# 主切分入口
# ---------------------------------------------------------------------------


def split_text(text: str, max_chars: int = 500) -> List[str]:
    """按语义完整性切分文本。

    策略：
      1. 先按段落（连续空行）切分
      2. 超长段落按智能句子边界切分（保护数字、缩写、引号）
      3. 过短碎片（<10 字）与相邻片段合并

    参数:
        text: 要切分的文本
        max_chars: 单个碎片最大字符数

    返回:
        切分后的文本列表
    """
    if not text or not text.strip():
        return []

    raw_paras = re.split(r"\n\s*\n", text.strip())
    segments: List[str] = []

    for para in raw_paras:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chars:
            segments.append(para)
            continue

        # 超长段落：按智能句子边界切
        sentences = _split_sentences(para)
        chunk = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(chunk) + len(s) > max_chars and chunk:
                segments.append(chunk.strip())
                chunk = s
            else:
                chunk += s + NEWLINE
        if chunk.strip():
            segments.append(chunk.strip())

    # 合并过短碎片（<10 字符）到前一个
    final: List[str] = []
    for seg in segments:
        if len(seg) < 10 and final:
            final[-1] += NEWLINE + seg
        else:
            final.append(seg)

    return [s for s in final if len(s) >= 10]


# ---------------------------------------------------------------------------
# 实体提取 — 用于实体关系图
# ---------------------------------------------------------------------------

# 大写缩写/英文实体: BTC, ETH, ZG, MACD 等
_ENTITY_ENGLISH_RE = re.compile(r"[A-Z][A-Z0-9]{1,}(?:/[A-Z0-9]+)*")  # BTC, ETH, ZG, ZD

# v1.5: 英文碎词噪音（jieba 把长英文拆成的片段/常见虚词）
_ENTITY_ENG_STOP = frozenset({
    "in", "an", "ce", "be", "dd", "em", "g", "of", "to", "on", "at", "is",
    "am", "or", "nd", "st", "th", "nt", "ab", "cd", "ef", "gh",
})

# 中文平台/项目名常见词缀
_ENTITY_CHN_SUFFIXES = {"公司", "平台", "集团", "科技", "网络", "学院", "大学", "社区", "基金", "项目", "团队", "部门"}

# 已知高频实体白名单（本领域常见名词）
_ENTITY_KNOWN = frozenset({
    "缠论", "中枢", "三买", "三卖", "顶背驰", "底背驰", "金叉", "死叉",
    "以太坊", "比特币", "知乎", "B站", "抖音", "微博", "公众号", "视频号",
    "微信", "QQ", "Telegram", "币安", "Bitget", "Binance",
    "小红书", "TradeApi", "Hermes",
})

# 价格数字: 5-6位数（常见crypto价格）
_ENTITY_PRICE_RE = re.compile(r"(?<!\d)([6-9]\d{4,5})(?!\d)")

def segment_query(query: str) -> list[str]:
    """对搜索查询进行分词。

    对包含中文的查询使用 jieba 分词，英文按空格切分。
    返回分词后的词条列表，用于 RediSearch 全文搜索。
    """
    if not query or not query.strip():
        return []

    query = query.strip()
    # 检测是否包含中文
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))

    if not has_chinese:
        # 纯英文/数字：按空格切分
        return [t for t in query.split() if t.strip()]

    # 中文：jieba 分词
    words = jieba.lcut(query)
    words = [w for w in words if w.strip()]

    # v1.4: 补充完整英文/数字词 — jieba 可能把英文切碎（embedding → em/be/dd/in/g），
    # 整词追加保证 TAG/entities 精确匹配能命中
    eng_words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", query)
    lower_set = {w.lower() for w in words}
    for w in eng_words:
        if w.lower() not in lower_set:
            words.append(w)
            lower_set.add(w.lower())

    return words




def extract_entities(text: str) -> list[str]:
    """从文本中提取候选实体（零LLM，纯jieba + regex + 白名单）。

    返回去重后的实体名列表，按出现顺序排列。
    """
    if not text or not text.strip():
        return []

    entities: list[str] = []
    seen: set[str] = set()

    # 0. 已知白名单实体
    text_lower = text.lower()
    for known in _ENTITY_KNOWN:
        if known.lower() in text_lower:
            key = known.lower()
            if key not in seen:
                entities.append(known)
                seen.add(key)

    # 1. jieba posseg 提取
    try:
        words = jieba.posseg.lcut(text)
        for w, flag in words:
            w_stripped = w.strip()
            if len(w_stripped) < 2:
                continue
            # 人名/机构/地名
            if flag in ("nr", "nr1", "nr2", "nrj", "nrf", "nt", "ns", "nsf"):
                key = w_stripped.lower()
                if key not in seen:
                    entities.append(w_stripped)
                    seen.add(key)
            # 其他专名（nz）— 只收含中文或首字母大写的英文词
            # （HMM/领域词典会把英文碎片标 nz，如 Binance→in/an；也会把 Redis 标 nz）
            elif flag == "nz":
                is_chinese = bool(re.search(r"[\u4e00-\u9fff]", w_stripped))
                is_title_eng = (
                    w_stripped[0].isupper()
                    and w_stripped.isalnum()
                    and len(w_stripped) >= 3
                )
                if is_chinese or is_title_eng:
                    key = w_stripped.lower()
                    if key not in seen:
                        entities.append(w_stripped)
                        seen.add(key)
            # 英文词（BTC, ETH 等）
            elif flag == "eng":
                key = w_stripped.lower()
                # v1.5: 过滤英文碎词噪音（jieba 把长英文拆成 in/an/ce 等）+ 常见虚词
                if len(key) >= 3 and key not in _ENTITY_ENG_STOP and key not in seen:
                    entities.append(w_stripped.upper())
                    seen.add(key)
    except Exception:
        pass

    # 2. regex 补充 — 大写缩写
    for m in _ENTITY_ENGLISH_RE.finditer(text):
        token = m.group()
        key = token.lower()
        if len(token) >= 2 and key not in seen:
            entities.append(token)
            seen.add(key)

    # 3. regex 补充 — 技术术语
    for token in ("ZG", "ZD", "MACD", "DIF", "DEA", "HIST", "RSI", "OBV", "EMA", "SMA", "BOLL",
                  "buy1", "sell1", "buy2", "sell2"):
        if token.lower() in text_lower:
            key = token.lower()
            if key not in seen:
                entities.append(token.upper())
                seen.add(key)

    # 4. 中文名+词缀组合: 如 "小米公司" 里的 "小米"
    for suffix in _ENTITY_CHN_SUFFIXES:
        idx = text.find(suffix)
        if idx >= 1:
            # 取 suffix 前一个词（1-6字）
            start = max(0, idx - 12)
            prefix = text[start:idx]
            # 尝试用 jieba 分词找最后一个有意义的词
            try:
                words = jieba.lcut(prefix)
                for w in reversed(words):
                    w = w.strip()
                    if len(w) >= 2 and w not in _STOP_WORDS:
                        key = w.lower()
                        if key not in seen:
                            entities.append(w)
                            seen.add(key)
                        break
            except Exception:
                pass

    # 5. 价格数字（5-6位数，如 63000）
    for m in _ENTITY_PRICE_RE.finditer(text):
        token = m.group()
        key = token
        if key not in seen:
            entities.append(token)
            seen.add(key)

    return entities

