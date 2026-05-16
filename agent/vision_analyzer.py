"""视觉分析器：PP-Structure 版面分析 + GLM-4V-Flash 图表描述。

PP-Structure 用于检测页面中的 text / table / figure 区域并提取表格结构；
GLM-4V-Flash 对 figure 区域生成自然语言描述。

当任一组件不可用时自动降级：
- PP-Structure 不可用 → 返回空版面分析结果，调用方回退到纯 OCR 文本
- GLM-4V 不可用 → figure 区域使用占位描述文本
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image as PILImage

from agent.config import GLM4V_MAX_TOKENS, GLM4V_TIMEOUT_SECONDS


# ── 数据类 ──

@dataclass
class LayoutRegion:
    """版面分析识别出的一个区域。"""
    region_type: str  # "text" | "table" | "figure"
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) 像素坐标
    confidence: float = 0.0
    content: str = ""  # table → Markdown 表格；figure → GLM-4V 描述；text → 空（由 OCR 处理）


@dataclass
class PageLayoutResult:
    """单页版面分析结果。"""
    page_number: int = 0
    regions: list[LayoutRegion] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def table_regions(self) -> list[LayoutRegion]:
        return [r for r in self.regions if r.region_type == "table"]

    @property
    def figure_regions(self) -> list[LayoutRegion]:
        return [r for r in self.regions if r.region_type == "figure"]


# ── 主类 ──

class VisionAnalyzer:
    """视觉分析融合器：PP-Structure 版面分析 + GLM-4V 图表描述。

    使用方式:
        analyzer = VisionAnalyzer(vision_llm=get_vision_llm())
        result = analyzer.analyze_page(page_image)

    PP-Structure 免费本地运行；GLM-4V 仅对 figure 区域按需调用。
    """

    def __init__(self, vision_llm=None):
        self._vision_llm = vision_llm       # 注入来自 config.get_vision_llm()
        self._layout_engine = None           # PP-Structure 延迟加载
        self._layout_available = True

    def cache_profile(self) -> str:
        """返回影响视觉分析结果的配置指纹，供 PDF 页级缓存使用。"""
        model_name = "none"
        if self._vision_llm is not None:
            model_name = str(
                getattr(self._vision_llm, "model_name", "")
                or getattr(self._vision_llm, "model", "")
                or type(self._vision_llm).__qualname__
            )
        return (
            "VisionAnalyzer;"
            "layout=PPStructure(table=True,ocr=False);"
            f"vision_model={model_name};"
            f"GLM4V_MAX_TOKENS={GLM4V_MAX_TOKENS};"
            f"GLM4V_TIMEOUT_SECONDS={GLM4V_TIMEOUT_SECONDS}"
        )

    # ── 版面分析 ──

    def _init_layout_engine(self) -> None:
        """延迟初始化 PP-Structure，避免 import 耗时影响启动。"""
        if self._layout_engine is not None:
            return
        try:
            from paddleocr import PPStructure
            self._layout_engine = PPStructure(table=True, ocr=False, show_log=False)
        except ImportError:
            self._layout_available = False
            print("[Vision] PP-Structure 未安装，版面分析不可用")
        except Exception as exc:
            self._layout_available = False
            print(f"[Vision] PP-Structure 初始化失败: {exc}")

    def _predict_layout(self, image: Any) -> list[dict[str, Any]]:
        """调用 PP-Structure 预测版面区域。

        Returns:
            list[dict]: 每个元素含 type / bbox / confidence / res 字段。
            返回空列表表示版面分析不可用或未识别出区域。
        """
        if not self._layout_available:
            return []
        self._init_layout_engine()
        if self._layout_engine is None:
            return []

        try:
            result = self._layout_engine(image)
            if isinstance(result, list):
                return result
            return []
        except Exception as exc:
            print(f"[Vision] 版面分析失败: {exc}")
            return []

    # ── 表格提取 ──

    @staticmethod
    def _extract_table(layout_item: dict[str, Any]) -> str:
        """从 PP-Structure 的 table 结果中提取 Markdown 表格。

        PP-Structure 对 table 区域的 res 字段包含直接可用的 HTML/Markdown。
        根据实际返回格式尝试解析。
        """
        res = layout_item.get("res", {})
        if isinstance(res, str):
            return res.strip()
        if isinstance(res, dict):
            # 可能需要从 HTML 转换，优先返回 cell_text 拼接
            cells = res.get("cell_text", []) or res.get("cells", [])
            if cells:
                return _cells_to_markdown(cells)
        return ""

    # ── 图表描述 ──

    def _describe_figure(self, pil_image: PILImage.Image) -> str:
        """用 GLM-4V-Flash 生成图表结构化描述。

        当 vision_llm 不可用时返回空字符串（降级为无描述）。
        """
        if self._vision_llm is None:
            return ""

        # 限制图片尺寸，避免 base64 过大
        w, h = pil_image.size
        max_dim = 1024
        if w > max_dim or h > max_dim:
            ratio = max_dim / max(w, h)
            pil_image = pil_image.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS)

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        prompt = (
            "请详细描述这张学术论文图表的内容：\n"
            "1. 图表类型（折线图、柱状图、流程图、结构图等）\n"
            "2. 横纵坐标含义及数据趋势（如适用）\n"
            "3. 主要发现和数据对比结论\n"
            "4. 图表标题（如有）\n"
            "用中文回答，不超过 200 字。"
        )

        try:
            from langchain_core.messages import HumanMessage
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_base64}",
                    "detail": "high",
                }},
            ])
            response = self._vision_llm.invoke([message])
            return response.content.strip()
        except Exception as exc:
            print(f"[Vision] 图表描述失败 (GLM-4V): {exc}")
            return ""

    # ── 主入口 ──

    def analyze_page(
        self,
        page_image: Any,
        page_number: int = 1,
    ) -> PageLayoutResult:
        """对单页做完整视觉分析。

        流程：版面分析 → table 提取 Markdown → figure 调用 GLM-4V 描述。

        Args:
            page_image: numpy 数组 (H, W, C)、PIL Image 或图片文件路径。
            page_number: 页码（用于结果标注）。

        Returns:
            PageLayoutResult: 包含所有识别区域的完整结果。
        """
        start = time.time()

        # Step 1: 版面分析
        layout_results = self._predict_layout(page_image)
        if not layout_results:
            elapsed = time.time() - start
            return PageLayoutResult(page_number=page_number, elapsed_seconds=elapsed)

        regions: list[LayoutRegion] = []
        # 暂存 figure 的裁剪图片，稍后批量描述
        pending_figures: list[tuple[int, PILImage.Image]] = []

        for item in layout_results:
            item_type = item.get("type", "text")
            bbox = _normalize_bbox(item.get("bbox", []))
            confidence = float(item.get("confidence", 0.0))

            region = LayoutRegion(
                region_type=item_type,
                bbox=bbox,
                confidence=confidence,
            )

            if item_type == "table":
                region.content = self._extract_table(item)

            elif item_type == "figure":
                cropped = _crop_region(page_image, bbox)
                if cropped is not None:
                    pending_figures.append((len(regions), cropped))

            # text 区域不填充 content，由 OCR 引擎负责
            regions.append(region)

        # Step 2: 批量描述 figure 区域
        for region_idx, cropped_img in pending_figures:
            description = self._describe_figure(cropped_img)
            regions[region_idx].content = description

        elapsed = time.time() - start
        return PageLayoutResult(
            page_number=page_number,
            regions=regions,
            elapsed_seconds=elapsed,
        )


# ── 辅助函数 ──

def _normalize_bbox(raw_bbox: list[float]) -> tuple[float, float, float, float]:
    """标准化 bbox 为 4 元组。"""
    if len(raw_bbox) >= 4:
        return (float(raw_bbox[0]), float(raw_bbox[1]),
                float(raw_bbox[2]), float(raw_bbox[3]))
    return (0.0, 0.0, 0.0, 0.0)


def _crop_region(image: Any, bbox: tuple[float, float, float, float]) -> PILImage.Image | None:
    """从图片中裁剪指定区域。"""
    try:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        if isinstance(image, PILImage.Image):
            pil = image
        else:
            import numpy as np
            pil = PILImage.fromarray(np.asarray(image))
        return pil.crop((int(x1), int(y1), int(x2), int(y2)))
    except Exception as exc:
        print(f"[Vision] 裁剪区域失败: {exc}")
        return None


def _cells_to_markdown(cells: list[list[str]]) -> str:
    """将二维单元格列表转为 Markdown 表格。"""
    if not cells:
        return ""
    lines = []
    # 表头
    header = cells[0]
    lines.append("| " + " | ".join(str(c) for c in header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    # 数据行
    for row in cells[1:]:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)
