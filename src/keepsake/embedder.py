"""Embedding 客户端 — 支持 OpenAI / DashScope / 自定义兼容端点。"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型 → 维度映射
# ---------------------------------------------------------------------------

_MODEL_DIMENSIONS: dict[str, int] = {
    # OpenAI text-embedding-3 系列
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # DashScope
    "text-embedding-v2": 1536,
    "text-embedding-v3": 1024,
    # 本地 ollama
    "nomic-embed-text": 768,
}

_DEFAULT_DIM = 1536


def resolve_dimension(model: str) -> int:
    """根据模型名返回向量维度，未知模型返回默认值 1536。"""
    model_key = model.strip().lower()
    return _MODEL_DIMENSIONS.get(model_key, _DEFAULT_DIM)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class Embedder(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> Optional[list[float]]:
        """输入文本，返回 float 向量。"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回当前模型输出的向量维度。"""
        ...


# ---------------------------------------------------------------------------
# OpenAI 兼容
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_URL = "https://api.openai.com/v1/embeddings"
_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class OpenAIEmbedder(Embedder):
    """兼容 OpenAI Embedding API 的客户端。

    也兼容 DashScope 等提供 /v1/embeddings 端点的服务。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = _DEFAULT_OPENAI_URL,
        model: str = _DEFAULT_OPENAI_MODEL,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = resolve_dimension(model)

    @property
    def dimension(self) -> int:
        return self._dim

    def get_embedding(self, text: str) -> Optional[list[float]]:
        if not self._api_key:
            logger.warning("embedder: no API key configured")
            return None

        payload = json.dumps({
            "model": self._model,
            "input": text,
        }).encode("utf-8")

        req = Request(
            f"{self._base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                emb = data["data"][0]["embedding"]
                # 如果服务返回的维度与预期不符，更新 self._dim
                if len(emb) != self._dim:
                    logger.info(
                        "embedder: model %s returned %d dims (expected %d), updating",
                        self._model, len(emb), self._dim,
                    )
                    self._dim = len(emb)
                return emb
        except Exception as e:
            logger.debug("embedder: request failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

_EMBEDDER_PROVIDERS: dict[str, type[Embedder]] = {
    "openai": OpenAIEmbedder,
    "dashscope": OpenAIEmbedder,  # DashScope 也走 /v1/embeddings
}


def create_embedder(
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Embedder:
    """根据配置创建 Embedder 实例。

    参数:
        provider: "openai" | "dashscope" | 自定义
        api_key: API 密钥
        base_url: API 端点
        model: 模型名
    """
    provider = provider or os.environ.get("KEEPSAKE_EMBEDDER", "openai").lower()
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("KEEPSAKE_EMBEDDER_URL", _DEFAULT_OPENAI_URL)
    model = model or os.environ.get("KEEPSAKE_EMBEDDER_MODEL", _DEFAULT_OPENAI_MODEL)

    if provider == "dashscope":
        base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = model or "text-embedding-v2"

    cls = _EMBEDDER_PROVIDERS.get(provider, OpenAIEmbedder)
    return cls(api_key=api_key, base_url=base_url, model=model)
