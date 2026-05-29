"""Generation prompt templates for DeepSeek API — education domain.

Supports dynamic persona selection based on content domain.
"""

from prompts.personas import DomainPersona, DEFAULT_PERSONA


def build_system_prompt(persona: DomainPersona) -> str:
    """Build the system prompt from a domain persona."""
    rules = [
        "1. 基于提供的上下文信息，不编造数据",
        "2. 使用中文撰写，专业但不晦涩",
        "3. 包含明确的数据引用（在括号中注明来源）",
        "4. 在分析中区分确定事实和推断结论",
        "5. 使用 Markdown 格式组织内容",
    ]
    return persona.role + "\n\n你的报告必须：\n" + "\n".join(rules)


def build_report_template(persona: DomainPersona) -> str:
    """Build the report structure template from a domain persona's sections."""
    lines = []
    for heading, description in persona.report_sections:
        lines.append(f"### {heading}")
        lines.append(f"（{description}）")
        lines.append("")
    return "\n".join(lines)


GENERATION_USER_TEMPLATE = """请根据以下信息回答用户的问题。

## 用户问题
{user_query}

## 检索到的相关文本片段
{retrieved_contexts}

## 结构化数据摘要
{structured_summary}

请按以下结构生成{report_title}：

{report_template}
请开始生成报告："""


def build_generation_messages(
    user_query: str,
    retrieved_contexts: str,
    structured_summary: str,
    persona: DomainPersona | None = None,
) -> list[dict]:
    """Build the message list for a report generation API call.

    Args:
        user_query: The user's question
        retrieved_contexts: Formatted retrieved chunk texts
        structured_summary: Extracted entity summary
        persona: Domain persona to use. If None, uses DEFAULT_PERSONA.
    """
    if persona is None:
        persona = DEFAULT_PERSONA

    system_prompt = build_system_prompt(persona)
    report_template = build_report_template(persona)

    user_content = GENERATION_USER_TEMPLATE.format(
        user_query=user_query,
        retrieved_contexts=retrieved_contexts,
        structured_summary=structured_summary,
        report_title=persona.report_title,
        report_template=report_template,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
