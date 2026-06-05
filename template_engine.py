#!/usr/bin/env python3
"""Content template engine — rewrite summaries for different content platforms.

Approach: AI-native rewriting (Approach A).
Each platform has a .j2 template in templates/ that serves as the system
prompt. The template contains {{ title }} and {{ summary }} placeholders
that get injected before sending to the AI. The AI returns the complete
platform-styled text.

Usage:
    from template_engine import rewrite_for_platform, generate_all_platforms

    text = rewrite_for_platform(client, summary, "xhs", "教育部新政策")
    generate_all_platforms(client, summary, "教育部新政策", "output/")
"""

import logging
from datetime import datetime
from pathlib import Path

import openai
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

import config

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"

# ── Platform registry ────────────────────────────────────────────────────

PLATFORMS = {
    "wechat": {
        "name": "微信公众号",
        "template": "wechat.j2",
        "extension": ".md",
        "description": "深度分析长文，专业结构化，适合教育从业者和家长",
    },
    "xhs": {
        "name": "小红书",
        "template": "xiaohongshu.j2",
        "extension": ".txt",
        "description": "活泼短笔记，emoji + 短句 + 话题标签，适合年轻用户",
    },
    "douyin": {
        "name": "抖音脚本",
        "template": "douyin.j2",
        "extension": ".txt",
        "description": "60秒口播脚本，黄金3秒开头 + 画面提示 + 互动引导",
    },
    "podcast": {
        "name": "播客口播稿",
        "template": "podcast.j2",
        "extension": ".md",
        "description": "3-5分钟口播稿，口语化 + 转场过渡 + 演播提示",
    },
}


def list_platforms() -> list[dict]:
    """Return metadata for all available platforms."""
    return [
        {
            "key": key,
            "name": cfg["name"],
            "description": cfg["description"],
            "extension": cfg["extension"],
        }
        for key, cfg in PLATFORMS.items()
    ]


def _get_jinja_env() -> Environment:
    """Create a Jinja2 environment pointing to templates/."""
    if not TEMPLATES_DIR.is_dir():
        raise FileNotFoundError(f"Templates directory not found: {TEMPLATES_DIR}")
    return Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))


def load_prompt(platform: str, summary: str, title: str = "", date_str: str = "") -> str:
    """Load and render the prompt template for a given platform.

    Args:
        platform: Platform key ("wechat", "xhs", "douyin", "podcast")
        summary: The article summary text to inject
        title: Article title
        date_str: Date string (YYYY-MM-DD)

    Returns:
        Rendered prompt string ready to send to the AI
    """
    if platform not in PLATFORMS:
        raise ValueError(
            f"Unknown platform '{platform}'. Available: {', '.join(PLATFORMS)}"
        )

    cfg = PLATFORMS[platform]
    env = _get_jinja_env()

    try:
        template = env.get_template(cfg["template"])
    except TemplateNotFound:
        raise FileNotFoundError(
            f"Template file '{cfg['template']}' not found in {TEMPLATES_DIR}"
        )

    return template.render(
        summary=summary,
        title=title,
        date=date_str or datetime.now().strftime("%Y-%m-%d"),
        source=f"EduDaily 教育日报 {date_str or datetime.now().strftime('%Y-%m-%d')}",
    )


def rewrite_for_platform(
    client: openai.OpenAI,
    summary: str,
    platform: str,
    title: str = "",
    date_str: str = "",
) -> str:
    """Use AI to rewrite a summary in the style of a target platform.

    Args:
        client: OpenAI-compatible client
        summary: Original summary text
        platform: Target platform key
        title: Article title for context
        date_str: Date string

    Returns:
        Platform-styled rewritten text
    """
    if platform not in PLATFORMS:
        raise ValueError(
            f"Unknown platform '{platform}'. Available: {', '.join(PLATFORMS)}"
        )

    cfg = PLATFORMS[platform]
    prompt = load_prompt(platform, summary, title, date_str)

    logger.info(
        "Rewriting for %s (%s) — prompt length: %d chars",
        platform, cfg["name"], len(prompt),
    )

    resp = client.chat.completions.create(
        model=config.DEEPSEEK_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"你是{cfg['name']}内容创作助手。"
                    "严格遵循用户提供的内容格式和风格要求，"
                    "直接输出格式化内容，不要额外解释。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,  # higher temperature for creative rewriting
        max_tokens=4096,
        timeout=config.API_TIMEOUT_SECONDS,
        **config.deepseek_chat_options(),
    )

    result = resp.choices[0].message.content or ""
    logger.info(
        "Rewrite complete for %s — output: %d chars",
        platform, len(result),
    )
    return result.strip()


def generate_platform_content(
    client: openai.OpenAI,
    summary: str,
    platform: str,
    title: str = "",
    output_dir: str | None = None,
    date_str: str = "",
) -> dict:
    """Rewrite summary for a platform and save to file.

    Args:
        client: OpenAI-compatible client
        summary: Original summary text
        platform: Target platform key
        title: Article title
        output_dir: Directory to save output
        date_str: Date string (YYYY-MM-DD)

    Returns:
        dict with platform, title, content, file_path
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")

    cfg = PLATFORMS[platform]
    content = rewrite_for_platform(client, summary, platform, title, date_str)

    file_path = None
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize title for filename
        safe_title = title[:40].replace("/", "_").replace("\\", "_").replace(":", "_")
        safe_title = "".join(c for c in safe_title if c not in r'*?"<>|')
        safe_title = safe_title.strip() or "article"

        filename = f"daily_{date_str}_{platform}{cfg['extension']}"
        file_path = out_dir / filename

        # Avoid overwrite: append counter
        counter = 1
        while file_path.exists():
            filename = f"daily_{date_str}_{platform}_{counter}{cfg['extension']}"
            file_path = out_dir / filename
            counter += 1

        file_path.write_text(content, encoding="utf-8")
        logger.info("Saved %s content: %s", platform, file_path)

    return {
        "platform": platform,
        "platform_name": cfg["name"],
        "title": title,
        "content": content,
        "file_path": str(file_path) if file_path else None,
    }


def generate_all_platforms(
    client: openai.OpenAI,
    summary: str,
    title: str = "",
    output_dir: str | None = None,
    date_str: str = "",
    platforms: list[str] | None = None,
) -> list[dict]:
    """Rewrite summary for all (or specified) platforms.

    Args:
        client: OpenAI-compatible client
        summary: Original summary text
        title: Article title
        output_dir: Directory to save outputs
        date_str: Date string
        platforms: Specific platforms to generate (default: all)

    Returns:
        List of dicts, one per platform
    """
    if platforms is None:
        platforms = list(PLATFORMS.keys())

    results = []
    for platform in platforms:
        if platform not in PLATFORMS:
            logger.warning("Skipping unknown platform: %s", platform)
            continue
        try:
            result = generate_platform_content(
                client, summary, platform, title, output_dir, date_str
            )
            results.append(result)
        except Exception as e:
            logger.error("Failed to generate %s content: %s", platform, e)
            results.append({
                "platform": platform,
                "platform_name": PLATFORMS.get(platform, {}).get("name", platform),
                "title": title,
                "content": "",
                "file_path": None,
                "error": str(e),
            })

    return results
