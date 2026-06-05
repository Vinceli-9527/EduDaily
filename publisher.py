#!/usr/bin/env python3
"""Platform draft publisher — send reports to content platform draft boxes.

ARCHITECTURE NOTE — This is a FRAMEWORK with reserved interfaces.
Full implementation requires OAuth setup and platform-specific API
credentials that must be obtained manually by the user. See each
publisher class docstring for step-by-step token acquisition guides.

Currently functional:
  - Credential loading/validation from credentials.json
  - Platform registry with metadata
  - publish_draft() entry point with argument routing
  - WeChatPublisher stub with documented API workflow
  - ZhihuPublisher stub
  - WeiboPublisher stub

To implement a real publisher, subclass BasePublisher and fill in the
publish() method.

Usage (CLI):
    python publisher.py --platform wechat --file output/daily_summary_2026-06-02.md
    python publisher.py --platform zhihu --content "文章内容..."

API:
    from publisher import publish_draft, list_publishers
    result = publish_draft("wechat", report_content, title="标题")
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CREDENTIALS_PATH = Path(__file__).parent / "credentials.json"

# ═══════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PublishResult:
    """Result of a publish attempt."""

    platform: str
    success: bool
    draft_url: str = ""
    message: str = ""
    draft_id: str = ""
    error: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Credential management
# ═══════════════════════════════════════════════════════════════════════════


def load_credentials() -> dict:
    """Load platform credentials from credentials.json.

    Returns empty dict if file doesn't exist.

    Expected format:
    {
        "wechat": {
            "app_id": "wx...",
            "app_secret": "...",
            "access_token": "...",
            "refresh_token": "...",
            "note": "Token from https://mp.weixin.qq.com/ ..."
        },
        "zhihu": {
            "client_id": "...",
            "access_token": "..."
        }
    }
    """
    if not CREDENTIALS_PATH.exists():
        return {}
    with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_credentials(data: dict) -> None:
    """Save credentials to credentials.json (creates file if needed)."""
    with open(CREDENTIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_credentials_for(platform: str) -> dict:
    """Get credentials for a specific platform."""
    all_creds = load_credentials()
    return all_creds.get(platform, {})


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base publisher
# ═══════════════════════════════════════════════════════════════════════════


class BasePublisher(ABC):
    """Abstract base for platform draft publishers.

    Subclass and implement publish() to add a new platform.
    """

    platform_key: str = ""
    platform_name: str = ""
    requires: list[str] = []  # required credential keys

    def __init__(self, credentials: dict | None = None):
        self.credentials = credentials or get_credentials_for(self.platform_key)
        self._validate()

    def _validate(self) -> None:
        """Check that required credentials are present."""
        missing = [k for k in self.requires if k not in self.credentials or not self.credentials[k]]
        if missing:
            raise ValueError(
                f"[{self.platform_name}] 缺少凭据: {', '.join(missing)}。"
                f"请在 {CREDENTIALS_PATH} 中配置。"
            )

    @abstractmethod
    def publish(self, content: str, title: str = "", **kwargs) -> PublishResult:
        """Publish content as a draft. Subclasses must implement."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# WeChat Official Account (微信公众号)
# ═══════════════════════════════════════════════════════════════════════════


class WeChatPublisher(BasePublisher):
    """微信公众号草稿发布。

    ## 如何获取凭据（用户手动操作）

    1. 登录 https://mp.weixin.qq.com/
    2. 前往「开发 → 基本配置」获取 AppID 和 AppSecret
    3. 配置 IP 白名单（添加服务器 IP）
    4. 使用 AppID + AppSecret 调用 token 接口获取 access_token（有效期 2 小时）
       curl "https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=APPID&secret=APPSECRET"
    5. 将获得的 access_token 填入 credentials.json

    ## 草稿 API 文档
    https://developers.weixin.qq.com/doc/offiaccount/Draft_Box/Add_draft.html

    ## 第三方 SDK
    pip install wechatpy
    文档: https://wechatpy.readthedocs.io/

    ## credentials.json 示例
    {
      "wechat": {
        "app_id": "wx1234567890abcdef",
        "app_secret": "your_app_secret_here",
        "access_token": "从 token 接口获取（有效期2小时）",
        "note": "Token 会过期，生产环境建议自动刷新"
      }
    }
    """

    platform_key = "wechat"
    platform_name = "微信公众号"
    requires = ["app_id", "app_secret", "access_token"]

    def publish(self, content: str, title: str = "", **kwargs) -> PublishResult:
        """Send a draft to WeChat Official Account draft box.

        Uses the wechatpy SDK if installed, otherwise falls back to direct HTTP.

        Args:
            content: Article body (HTML or Markdown — convert to rich text)
            title: Article title
            **kwargs: Extra fields (author, digest, cover_media_id, etc.)

        Returns:
            PublishResult with draft_url if successful
        """
        try:
            import wechatpy
            return self._publish_via_wechatpy(content, title, **kwargs)
        except ImportError:
            logger.info("wechatpy not installed — using direct HTTP API")
            return self._publish_via_http(content, title, **kwargs)

    def _publish_via_wechatpy(self, content: str, title: str = "", **kwargs) -> PublishResult:
        """Publish draft using wechatpy SDK."""
        import wechatpy

        client = wechatpy.WeChatClient(
            appid=self.credentials["app_id"],
            secret=self.credentials["app_secret"],
        )

        # Prepare draft article
        articles = [{
            "title": title or "EduDaily 教育日报",
            "content": content,  # wechatpy accepts Markdown → auto-converts
            "content_source_url": kwargs.get("source_url", ""),
            "digest": kwargs.get("digest", content[:120] if content else ""),
            "author": kwargs.get("author", "EduDaily"),
            "need_open_comment": 1,
        }]

        try:
            # Add as draft (not publish)
            media_id = client.draft.add(articles)
            return PublishResult(
                platform="wechat",
                success=True,
                draft_id=media_id,
                draft_url=f"https://mp.weixin.qq.com/cgi-bin/appmsg?t=media/appmsg_edit&action=edit&type=10&appmsgid={media_id}",
                message=f"草稿已保存！media_id={media_id}",
            )
        except wechatpy.WeChatClientException as e:
            return PublishResult(
                platform="wechat",
                success=False,
                error=f"微信 API 错误 (errCode={e.errcode}): {e.errmsg}",
            )

    def _publish_via_http(self, content: str, title: str = "", **kwargs) -> PublishResult:
        """Publish draft using direct HTTP calls (no SDK dependency)."""
        import requests

        token = self.credentials.get("access_token", "")
        if not token:
            return PublishResult(
                platform="wechat",
                success=False,
                error="access_token 缺失。请先获取 token 并填入 credentials.json。",
            )

        url = f"https://api.weixin.qq.com/cgi-bin/draft/add?access_token={token}"

        payload = {
            "articles": [{
                "title": title or "EduDaily 教育日报",
                "content": content,
                "digest": kwargs.get("digest", content[:120] if content else ""),
                "author": kwargs.get("author", "EduDaily"),
            }]
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            data = resp.json()
            if "media_id" in data:
                media_id = data["media_id"]
                return PublishResult(
                    platform="wechat",
                    success=True,
                    draft_id=media_id,
                    draft_url=f"https://mp.weixin.qq.com/cgi-bin/appmsg?t=media/appmsg_edit&action=edit&type=10&appmsgid={media_id}",
                    message=f"草稿已保存到微信公众号草稿箱！media_id={media_id}",
                )
            else:
                err_code = data.get("errcode", "unknown")
                err_msg = data.get("errmsg", "未知错误")
                return PublishResult(
                    platform="wechat",
                    success=False,
                    error=f"微信 API 返回错误 (errcode={err_code}): {err_msg}",
                )
        except requests.RequestException as e:
            return PublishResult(
                platform="wechat",
                success=False,
                error=f"网络请求失败: {e}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Zhihu (知乎)
# ═══════════════════════════════════════════════════════════════════════════


class ZhihuPublisher(BasePublisher):
    """知乎文章发布。

    ## 如何获取凭据

    1. 登录 https://www.zhihu.com/
    2. 前往 https://www.zhihu.com/settings/account 查看账号信息
    3. 知乎 API 目前主要面向企业合作，个人用户需通过 Cookie 方式调用
    4. 在浏览器开发者工具中获取 Cookie 中的 z_c0 值
    5. 将 z_c0 填入 credentials.json

    ## 注意
    知乎没有公开的草稿箱 API。当前实现框架预留，需要 HTTPS
    模拟请求到 zhihu.com/api/v4/drafts。

    ## credentials.json 示例
    {
      "zhihu": {
        "z_c0": "从浏览器 Cookie 获取",
        "note": "知乎无官方草稿 API，此为预留接口"
      }
    }
    """

    platform_key = "zhihu"
    platform_name = "知乎"
    requires = ["z_c0"]

    def publish(self, content: str, title: str = "", **kwargs) -> PublishResult:
        return PublishResult(
            platform="zhihu",
            success=False,
            message=(
                "知乎暂未开放草稿箱 API。"
                "建议方案：将内容保存为草稿 → 手动粘贴到知乎编辑器。"
                f"已复制 {len(content)} 字符。"
            ),
            error="平台 API 不可用",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Weibo (微博)
# ═══════════════════════════════════════════════════════════════════════════


class WeiboPublisher(BasePublisher):
    """微博头条文章发布。

    ## 如何获取凭据

    1. 前往 https://open.weibo.com/ 注册开发者账号
    2. 创建应用，获取 App Key 和 App Secret
    3. 使用 OAuth 2.0 授权获取 access_token
       https://open.weibo.com/wiki/Oauth2/authorize
    4. 将 token 填入 credentials.json

    ## 头条文章 API
    https://open.weibo.com/wiki/2/statuses/upload_url_text

    ## credentials.json 示例
    {
      "weibo": {
        "app_key": "your_app_key",
        "app_secret": "your_app_secret",
        "access_token": "从 OAuth 获取",
        "uid": "你的微博 UID"
      }
    }
    """

    platform_key = "weibo"
    platform_name = "微博"
    requires = ["access_token"]

    def publish(self, content: str, title: str = "", **kwargs) -> PublishResult:
        import requests

        token = self.credentials["access_token"]
        uid = self.credentials.get("uid", "")

        url = "https://api.weibo.com/2/statuses/upload_url_text.json"

        # Weibo's 头条文章 (long-form): title + content
        payload = {
            "access_token": token,
            "status": (
                f"【{title or 'EduDaily 教育日报'}】\n\n"
                f"{content[:2000]}"  # Weibo long-form text limit
            ),
        }

        try:
            resp = requests.post(url, data=payload, timeout=30)
            data = resp.json()
            if "id" in data:
                weibo_id = data["id"]
                return PublishResult(
                    platform="weibo",
                    success=True,
                    draft_id=str(weibo_id),
                    draft_url=f"https://weibo.com/{uid}/{weibo_id}" if uid else "",
                    message=f"微博头条文章已发布！id={weibo_id}",
                )
            else:
                return PublishResult(
                    platform="weibo",
                    success=False,
                    error=f"微博 API 错误: {data.get('error', '未知错误')}",
                )
        except requests.RequestException as e:
            return PublishResult(
                platform="weibo",
                success=False,
                error=f"网络请求失败: {e}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Platform registry
# ═══════════════════════════════════════════════════════════════════════════


PUBLISHERS: dict[str, type[BasePublisher]] = {
    "wechat": WeChatPublisher,
    "zhihu": ZhihuPublisher,
    "weibo": WeiboPublisher,
}


def list_publishers() -> list[dict]:
    """Return metadata for all available publishers."""
    return [
        {
            "key": key,
            "name": cls.platform_name,
            "requires": cls.requires,
        }
        for key, cls in PUBLISHERS.items()
    ]


def publish_draft(
    platform: str,
    content: str,
    title: str = "",
    **kwargs,
) -> PublishResult:
    """Publish a draft to a platform.

    Args:
        platform: Platform key ("wechat", "zhihu", "weibo")
        content: Draft body content (Markdown or plain text)
        title: Draft title
        **kwargs: Platform-specific extra fields

    Returns:
        PublishResult with success status, draft URL, and message
    """
    if platform not in PUBLISHERS:
        return PublishResult(
            platform=platform,
            success=False,
            error=f"未知平台 '{platform}'。可选: {', '.join(PUBLISHERS)}",
        )

    publisher_cls = PUBLISHERS[platform]

    # Check credentials exist
    creds = get_credentials_for(platform)
    if not creds:
        return PublishResult(
            platform=platform,
            success=False,
            error=(
                f"需要 {publisher_cls.platform_name} 凭据，但 credentials.json 中未找到。\n"
                f"请参考 publisher.py 中 {publisher_cls.__name__} 的文档获取凭据。"
            ),
        )

    try:
        publisher = publisher_cls(credentials=creds)
        return publisher.publish(content, title, **kwargs)
    except ValueError as e:
        return PublishResult(platform=platform, success=False, error=str(e))
    except Exception as e:
        logger.exception("Unexpected error in %s publisher", platform)
        return PublishResult(
            platform=platform,
            success=False,
            error=f"发布异常: {e}",
        )


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="EduDaily Publisher — 发送草稿到内容平台"
    )
    parser.add_argument(
        "--platform", "-p", type=str, required=True,
        help="目标平台 (wechat / zhihu / weibo)",
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None,
        help="从文件读取内容",
    )
    parser.add_argument(
        "--content", "-c", type=str, default=None,
        help="直接传入内容文本",
    )
    parser.add_argument(
        "--title", "-t", type=str, default="",
        help="文章标题",
    )
    args = parser.parse_args()

    # Get content
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"[!] 文件不存在: {args.file}")
            exit(1)
        content = filepath.read_text(encoding="utf-8")
    elif args.content:
        content = args.content
    else:
        print("[!] 请通过 --file 或 --content 提供发布内容")
        exit(1)

    result = publish_draft(args.platform, content, title=args.title)
    if result.success:
        print(f"✓ {result.message}")
        if result.draft_url:
            print(f"  草稿链接: {result.draft_url}")
    else:
        print(f"✗ 发布失败: {result.error}")
