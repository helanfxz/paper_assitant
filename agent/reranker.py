"""专用 rerank 客户端。

rerank 模型不是 chat 模型，不能通过 ChatOpenAI 调用。
这里按 DashScope text-rerank 接口协议直接发送 HTTP 请求。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


@dataclass(frozen=True)
class RerankHit:
    """一条重排命中结果。"""

    index: int  # 命中的原始 documents 下标。
    score: float  # rerank 模型返回的相关性分数。
    text: str  # 命中的文档文本。


class DashScopeReranker:
    """DashScope text-rerank 客户端。"""

    model: str
    api_key: str
    url: str
    timeout: int

    def __init__(
        self,
        model: str,
        api_key: str,
        url: str = DEFAULT_DASHSCOPE_RERANK_URL,
        timeout: int = 10,
    ):
        self.model = model
        self.api_key = api_key
        self.url = url or DEFAULT_DASHSCOPE_RERANK_URL
        self.timeout = timeout

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[RerankHit]:
        """根据 query 对候选 documents 重排，返回按相关性排序后的命中结果。
        失败时抛异常，由调用方决定是否回退。"""
        cleaned_query = " ".join(str(query).strip().split())
        cleaned_documents = [str(document).strip() for document in documents if str(document).strip()]
        if not cleaned_query or not cleaned_documents:
            return []

        payload = {
            "model": self.model,
            "input": {
                "query": cleaned_query,
                "documents": cleaned_documents,
            },
            "parameters": {
                "return_documents": True,
                "top_n": max(1, min(top_n, len(cleaned_documents))),
            },
        }

        response_data = self._post_json(payload)
        return self._parse_response(response_data, cleaned_documents)

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """发送 JSON 请求，失败时直接抛异常。"""
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"rerank HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"rerank 网络错误: {exc}") from exc

    def _parse_response(self, response_data: dict[str, Any], documents: list[str]) -> list[RerankHit]:
        """解析 DashScope rerank 返回结构。"""
        output = response_data.get("output", {})
        results = output.get("results", []) if isinstance(output, dict) else []
        if not isinstance(results, list):
            return []

        hits: list[RerankHit] = []
        for result in results:
            if not isinstance(result, dict):
                continue

            index = self._parse_index(result, documents)
            if index < 0 or index >= len(documents):
                continue

            score = result.get("relevance_score", result.get("score", 0.0))
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = 0.0

            hits.append(RerankHit(index=index, score=score_value, text=documents[index]))

        return hits

    def _parse_index(self, result: dict[str, Any], documents: list[str]) -> int:
        """优先读取返回的 index；缺失时用返回文档文本反查原始下标。"""
        raw_index = result.get("index")
        if isinstance(raw_index, int):
            return raw_index
        if isinstance(raw_index, str) and raw_index.isdigit():
            return int(raw_index)

        returned_document = result.get("document")
        if isinstance(returned_document, dict):
            returned_text = str(
                returned_document.get("text")
                or returned_document.get("content")
                or returned_document.get("document")
                or ""
            ).strip()
        else:
            returned_text = str(returned_document or "").strip()

        if returned_text:
            for index, document in enumerate(documents):
                if document == returned_text:
                    return index
        return -1
