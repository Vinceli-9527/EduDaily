#!/usr/bin/env python3
"""Clipboard utilities — copy report content to system clipboard.

Cross-platform support via pyperclip with fallback to native commands.

Usage:
    python clipboard.py                        # Copy today's default report
    python clipboard.py --platform xhs         # Copy today's Xiaohongshu version
    python clipboard.py --file output/daily_summary_2026-06-02.md  # Copy specific file

Dependencies:
    pip install pyperclip
    Linux users may also need: sudo apt install xclip  (or xsel)

API:
    from clipboard import copy_to_clipboard, copy_latest_report, copy_report_file
    copy_to_clipboard("some text")
    copy_latest_report("output/", platform="wechat")
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config
from template_engine import PLATFORMS

logger = logging.getLogger(__name__)


def _pyperclip_copy(text: str) -> bool:
    """Copy using pyperclip (cross-platform). Returns True on success."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning("pyperclip failed: %s", e)
        return False


def _native_copy(text: str) -> bool:
    """Copy using OS-native commands as fallback. Returns True on success."""
    if sys.platform == "darwin":
        # macOS
        try:
            proc = subprocess.run(
                ["pbcopy"], input=text, text=True, timeout=5,
                check=True, capture_output=True,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    elif sys.platform == "win32":
        # Windows — use PowerShell clip
        try:
            proc = subprocess.run(
                ["powershell", "-Command", "Set-Clipboard", "-Value", f"$input"],
                input=text, text=True, timeout=5,
                check=True, capture_output=True,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        # Fallback: built-in clip.exe
        try:
            proc = subprocess.run(
                ["clip"], input=text, text=True, timeout=5,
                check=True, capture_output=True,
            )
            return proc.returncode == 0
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    else:
        # Linux — try xclip then xsel
        for cmd in ["xclip", "xsel"]:
            try:
                args = (
                    ["xclip", "-selection", "clipboard"] if cmd == "xclip"
                    else ["xsel", "--clipboard", "--input"]
                )
                proc = subprocess.run(
                    args, input=text, text=True, timeout=5,
                    check=True, capture_output=True,
                )
                return proc.returncode == 0
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        return False


def copy_to_clipboard(content: str) -> bool:
    """Copy text to the system clipboard.

    Tries pyperclip first, then native OS commands as fallback.

    Args:
        content: The text to copy

    Returns:
        True if copy succeeded, False otherwise
    """
    if not content or not content.strip():
        logger.warning("Attempted to copy empty content.")
        print("[!] 警告: 内容为空，未执行复制。")
        return False

    # Method 1: pyperclip
    if _pyperclip_copy(content):
        logger.info("Copied %d chars via pyperclip", len(content))
        return True

    # Method 2: Native command
    if _native_copy(content):
        logger.info("Copied %d chars via native command", len(content))
        return True

    logger.error("All clipboard methods failed.")
    print("[!] 复制失败: 未找到可用的剪贴板工具。")
    print("    请安装: pip install pyperclip")
    if sys.platform.startswith("linux"):
        print("    Linux 用户还需: sudo apt install xclip")
    return False


def find_latest_report(
    output_dir: str | None = None,
    platform: str | None = None,
    date_str: str | None = None,
) -> Path | None:
    """Find the most recent report file in the output directory.

    Args:
        output_dir: Directory to search (default: config.OUTPUT_DIR)
        platform: Filter by platform key (e.g. "wechat", "xhs"). None = default report.
        date_str: Date filter (YYYY-MM-DD). None = today.

    Returns:
        Path to the report file, or None if not found
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    out_path = Path(output_dir)
    if not out_path.is_dir():
        logger.warning("Output directory not found: %s", output_dir)
        return None

    if platform and platform in PLATFORMS:
        extension = PLATFORMS[platform]["extension"]
        pattern = f"daily_{date_str}_{platform}{extension}"
    else:
        pattern = f"daily_summary_{date_str}.md"

    candidate = out_path / pattern
    if candidate.exists():
        return candidate

    # Try finding with counter suffix (e.g. daily_2026-06-02_xhs_1.txt)
    if platform:
        base_pattern = f"daily_{date_str}_{platform}"
        matches = sorted(
            out_path.glob(f"{base_pattern}*{PLATFORMS[platform]['extension']}")
        )
        if matches:
            return matches[-1]  # highest counter = most recent

    return None


def copy_report_file(filepath: str | Path) -> tuple[bool, int]:
    """Read a report file and copy its content to clipboard.

    Args:
        filepath: Path to the report file

    Returns:
        Tuple of (success: bool, char_count: int)
    """
    path = Path(filepath)
    if not path.exists():
        print(f"[!] 文件不存在: {filepath}")
        return False, 0

    content = path.read_text(encoding="utf-8")
    ok = copy_to_clipboard(content)
    if ok:
        print(f"✓ 已复制 {len(content)} 字符到剪贴板: {path.name}")
    return ok, len(content)


def copy_latest_report(
    output_dir: str | None = None,
    platform: str | None = None,
    date_str: str | None = None,
) -> bool:
    """Find the latest daily report and copy it to clipboard.

    Args:
        output_dir: Output directory (default: config.OUTPUT_DIR)
        platform: Platform key to select platform-specific report
        date_str: Date to find report for

    Returns:
        True if a report was found and copied successfully
    """
    report_path = find_latest_report(output_dir, platform, date_str)

    if report_path is None:
        when = date_str or datetime.now().strftime("%Y-%m-%d")
        plat_name = PLATFORMS.get(platform, {}).get("name", "默认")
        if platform:
            print(f"[!] 未找到 {date_str or when} 的 {plat_name} 日报。")
            print(f"    请先运行: python batch_processor.py --platform {platform}")
        else:
            print(f"[!] 未找到 {when} 的默认日报。")
            print("    请先运行: python batch_processor.py")
        return False

    ok, chars = copy_report_file(report_path)
    if ok and platform:
        plat_name = PLATFORMS.get(platform, {}).get("name", platform)
    return ok


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="EduDaily Clipboard — 一键复制日报到剪贴板"
    )
    parser.add_argument(
        "--platform", "-p", type=str, default=None,
        help="复制指定平台版本 (wechat / xhs / douyin / podcast)",
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="复制指定文件路径（优先级高于 --platform）",
    )
    parser.add_argument(
        "--date", "-d", type=str, default=None,
        help="指定日期 YYYY-MM-DD（默认: 今天）",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录路径（默认: output）",
    )
    args = parser.parse_args()

    if args.file:
        copy_report_file(args.file)
    else:
        copy_latest_report(
            output_dir=args.output_dir,
            platform=args.platform,
            date_str=args.date,
        )


if __name__ == "__main__":
    main()
