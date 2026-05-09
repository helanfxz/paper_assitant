"""Skill 加载层。

Skill 是一组按需加载的任务说明，不是直接执行的工具函数。
系统每次初始化时只扫描摘要，完整内容通过 load_skill 工具进入当前上下文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"  # 项目默认 skill 目录。
SKILL_FILENAME = "SKILL.md"  # 每个 skill 目录中必须存在的说明文件名。
FRONTMATTER_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?", re.DOTALL)


@dataclass(frozen=True)
class SkillInfo:
    """一条 skill 摘要，供 system prompt 低成本注入。"""

    name: str  # skill 名称，默认来自 frontmatter.name，缺省时使用目录名。
    description: str  # skill 简介，用来帮助模型判断是否需要加载该 skill。
    path: Path  # SKILL.md 的真实路径，便于调试和后续扩展。


class SkillLoader:
    """扫描 skill 目录并按需加载完整 skill 正文。"""

    skills_dir: Path  # skill 根目录，默认是项目根目录下的 skills。
    skills: dict[str, SkillInfo]  # skill 名称到摘要信息的缓存。

    def __init__(self, skills_dir: Path | str | None = None):
        self.skills_dir = Path(skills_dir) if skills_dir else DEFAULT_SKILLS_DIR
        self.skills = {}
        self.refresh()

    def refresh(self) -> None:
        """重新扫描 skill 目录，更新内存中的 skill 摘要列表。"""
        self.skills = {}
        if not self.skills_dir.exists():
            return

        # 每个 SKILL.md 代表一个可加载 skill，递归扫描便于后续按领域分目录组织。
        for skill_path in sorted(self.skills_dir.rglob(SKILL_FILENAME)):
            if not skill_path.is_file():
                continue
            metadata, _ = self._read_skill_parts(skill_path)
            name = str(metadata.get("name") or skill_path.parent.name).strip()
            description = str(metadata.get("description") or name).strip()
            if not name:
                continue
            self.skills[name] = SkillInfo(name=name, description=description, path=skill_path)

    def list_skills(self) -> list[SkillInfo]:
        """返回当前已发现的 skill 摘要列表。"""
        return list(self.skills.values())

    def build_prompt_block(self) -> str:
        """把 skill 摘要列表转成 system prompt 中的可用技能区块。"""
        skill_infos = self.list_skills()
        if not skill_infos:
            return ""

        lines = [
            "【可用 Skill】",
            "这些 skill 是按需加载的专项说明。当前只展示摘要；如果任务需要某个 skill，请调用 load_skill(name) 读取完整内容后再执行。",
            "",
        ]
        for skill_info in skill_infos:
            lines.append(f"- {skill_info.name}：{skill_info.description}")
        return "\n".join(lines)

    def load_skill(self, name: str) -> str:
        """按名称读取完整 skill 正文，返回内容不包含顶部 frontmatter。"""
        cleaned_name = str(name).strip()
        if not cleaned_name:
            return "加载失败：skill name 不能为空。"

        skill_info = self.skills.get(cleaned_name)
        if skill_info is None:
            available_names = ", ".join(sorted(self.skills)) or "无"
            return f"加载失败：未知 skill={cleaned_name}。当前可用 skill：{available_names}"

        _, body = self._read_skill_parts(skill_info.path)
        cleaned_body = body.strip()
        if not cleaned_body:
            return f"skill={cleaned_name} 没有正文内容。"
        return f"【已加载 Skill：{cleaned_name}】\n{cleaned_body}"

    def _read_skill_parts(self, skill_path: Path) -> tuple[dict[str, str], str]:
        """读取 SKILL.md，并拆分顶部 frontmatter 与正文。"""
        text = skill_path.read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(text)
        if not match:
            return {}, text

        metadata: dict[str, str] = {}
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")

        return metadata, text[match.end():]
