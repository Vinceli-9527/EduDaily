"""Persona registry — domain-specific expert roles and report templates.

Each domain defines:
  - role: The expert persona the LLM adopts
  - keywords: Chinese/English keywords for domain classification
  - report_structure: Domain-appropriate report section headings
"""

from dataclasses import dataclass, field


@dataclass
class DomainPersona:
    domain: str
    role: str
    keywords: list[str]
    report_title: str
    report_sections: list[tuple[str, str]]  # (heading, description)


PERSONAS: dict[str, DomainPersona] = {
    "education": DomainPersona(
        domain="education",
        role=(
            "你是一位资深教育研究员，拥有教育学博士学位和10年教育政策研究经验。"
            "你熟悉国内外教育体系、课程设计、教学方法和教育技术发展趋势。"
            "你的分析注重实证研究，兼顾教育公平与质量，关注政策对师生的实际影响。"
        ),
        keywords=[
            "教育", "学校", "大学", "学院", "中学", "小学", "学生", "教师",
            "课程", "教学", "考试", "高考", "招生", "录取", "学位",
            "教育部", "教育厅", "教育局", "学术", "论文", "科研",
            "培训", "职业", "技能", "在线教育", "MOOC", "远程教育",
            "留学", "海外", "交换", "奖学金", "教育经费",
            "素质教育", "双减", "教培", "义务教育", "高等教育",
        ],
        report_title="教育研究分析报告",
        report_sections=[
            ("一、教育现象/政策概述", "概括教育领域的核心议题或政策变化"),
            ("二、深度分析", "从教育质量、公平性、实施路径等多维度分析"),
            ("三、利益相关方影响", "评估对学生、教师、学校、家庭等各方的影响"),
            ("四、国内外对比与借鉴", "横向对比国际经验，提炼可借鉴之处"),
            ("五、建议与展望", "提出具体建议和未来发展趋势"),
        ],
    ),
}

DEFAULT_PERSONA = DomainPersona(
    domain="general",
    role=(
        "你是一位专业信息分析师，拥有丰富的跨领域研究经验。"
        "你擅长从大量信息中提炼关键洞察，构建结构化的分析报告。"
        "你的分析客观中立，注重事实依据，能清晰区分已知事实和推论判断。"
    ),
    keywords=[],
    report_title="综合分析报告",
    report_sections=[
        ("一、核心发现概述", "概括核心信息和关键发现"),
        ("二、详细分析", "从多个维度深入分析，提取有价值的信息和趋势"),
        ("三、影响评估", "评估对相关方的影响程度和范围"),
        ("四、风险与机遇", "识别潜在风险和可能的机遇"),
        ("五、结论与展望", "综合判断和未来关注方向"),
    ],
)


def get_persona(domain: str) -> DomainPersona:
    """Get persona by domain key, falling back to default."""
    return PERSONAS.get(domain, DEFAULT_PERSONA)


def list_domains() -> list[str]:
    """List all available domain keys."""
    return list(PERSONAS.keys())
