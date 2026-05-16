"""OCR 引擎：PaddleOCR 主引擎 + Tesseract 降级备选。"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.config import OCR_LANG, OCR_USE_GPU, OCR_FALLBACK_LANG


@dataclass
class OcrResult:
    """单页 OCR 结果。"""
    page_number: int = 0
    text: str = ""
    text_lines: list[dict[str, Any]] = field(default_factory=list)
    engine_used: str = "unknown"
    confidence: float = 0.0
    elapsed_seconds: float = 0.0


class OcrEngine:
    """OCR 引擎封装：PaddleOCR 优先，不可用时降级 Tesseract。

    使用方式:
        engine = OcrEngine()
        result = engine.extract_text(image)  # numpy 数组或文件路径
    """

    def __init__(
        self,
        lang: str | None = None,
        use_gpu: bool | None = None,
        fallback_lang: str | None = None,
    ):
        self._paddle_ocr = None
        self._paddle_available = True  # 设为 False 后永久降级
        self._lang = lang or OCR_LANG
        self._use_gpu = use_gpu if use_gpu is not None else OCR_USE_GPU
        self._fallback_lang = fallback_lang or OCR_FALLBACK_LANG

    def _init_paddle(self) -> None:
        """延迟初始化 PaddleOCR，避免 import 耗时影响启动。"""
        if self._paddle_ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
            self._paddle_ocr = PaddleOCR(
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
        except ImportError:
            self._paddle_available = False
            print("[OCR] PaddleOCR 未安装，将使用 Tesseract 降级")
        except Exception as exc:
            self._paddle_available = False
            print(f"[OCR] PaddleOCR 初始化失败: {exc}，降级 Tesseract")

    def _ocr_with_paddle(self, image: Any) -> list[dict[str, Any]]:
        """使用 PaddleOCR 识别。"""
        ocr_result = self._paddle_ocr.ocr(image, cls=False)
        if not ocr_result or not ocr_result[0]:
            return []
        lines = []
        for line_info in ocr_result[0]:
            _bbox, (text, confidence) = line_info
            lines.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": _bbox,
            })
        return lines

    def _ocr_with_tesseract(self, image: Any) -> list[dict[str, Any]]:
        """使用 Tesseract 降级识别。"""
        try:
            import numpy as np
            import pytesseract
            from PIL import Image as PILImage

            if isinstance(image, (str, Path)):
                pil_image = PILImage.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(image)
            else:
                pil_image = image

            text = pytesseract.image_to_string(pil_image, lang=self._fallback_lang)
            return [{"text": text, "confidence": 0.5, "bbox": None}]
        except ImportError:
            raise RuntimeError(
                "OCR 不可用：PaddleOCR 和 Tesseract 均未安装。"
                "请执行: pip install paddleocr 或 apt install tesseract-ocr"
            )

    def extract_text(self, image: Any) -> OcrResult:
        """从图片中提取文本。

        Args:
            image: numpy 数组 (H, W, C)、PIL Image 或图片文件路径。

        Returns:
            OcrResult: 包含提取文本和元信息。
        """
        start = time.time()

        try:
            if self._paddle_available:
                self._init_paddle()

            if self._paddle_ocr is not None:
                try:
                    lines = self._ocr_with_paddle(image)
                    engine = "paddleocr"
                except Exception as exc:
                    print(f"[OCR] PaddleOCR 调用失败，降级 Tesseract: {exc}")
                    self._paddle_available = False
                    lines = self._ocr_with_tesseract(image)
                    engine = "tesseract"
            else:
                lines = self._ocr_with_tesseract(image)
                engine = "tesseract"
        except Exception as exc:
            print(f"[OCR] 文本提取失败: {exc}")
            elapsed = time.time() - start
            return OcrResult(
                text=f"(OCR 失败: {exc})",
                engine_used="failed",
                elapsed_seconds=elapsed,
            )

        text = "\n".join(line["text"] for line in lines)
        avg_conf = (
            sum(line["confidence"] for line in lines) / len(lines)
            if lines else 0.0
        )
        elapsed = time.time() - start

        return OcrResult(
            text=text,
            text_lines=lines,
            engine_used=engine,
            confidence=avg_conf,
            elapsed_seconds=elapsed,
        )

    def extract_text_batch(
        self, images: list[Any], page_offset: int = 1
    ) -> list[OcrResult]:
        """批量 OCR 多张图片（顺序执行，单页错误不中断）。

        Args:
            images: 图片列表，每张对应一页。
            page_offset: 页码起始偏移（默认为 1）。

        Returns:
            list[OcrResult]: 按页码排序的结果列表。
        """
        results = []
        for idx, image in enumerate(images):
            page_num = idx + page_offset
            try:
                result = self.extract_text(image)
                result.page_number = page_num
            except Exception as exc:
                result = OcrResult(
                    page_number=page_num,
                    text=f"(第 {page_num} 页 OCR 失败: {exc})",
                    engine_used="failed",
                )
            results.append(result)
        return results
