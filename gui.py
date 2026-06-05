#!/usr/bin/env python3
"""EduDaily Desktop GUI — 图形化操作界面

为教师和自媒体初学者提供零门槛操作体验。
所有 CLI 功能均可通过点击按钮完成。

启动方式:
    python gui.py

依赖: 内置 tkinter（Python 自带，无需额外安装）
"""

import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Toplevel, messagebox, filedialog
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import config


# ═══════════════════════════════════════════════════════════════════════════
# 线程安全输出重定向
# ═══════════════════════════════════════════════════════════════════════════


class QueueStream:
    """将 print 输出重定向到队列，GUI 定时拉取显示。"""

    def __init__(self, q: queue.Queue):
        self._queue = q

    def write(self, text: str):
        if text and text.strip():
            self._queue.put(text)

    def flush(self):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 主 GUI 类
# ═══════════════════════════════════════════════════════════════════════════


class EduDailyGUI(Tk):
    """EduDaily 桌面主窗口"""

    def __init__(self):
        super().__init__()

        self.title("EduDaily — 每日教育资讯工具")
        self.geometry("1020x720")
        self.minsize(860, 600)

        # ── 应用状态 ──
        self._conn = None
        self._client = None
        self._embedding_model = None
        self._collection = None
        self._docs = []
        self._ready = False
        self._scheduler_running = False
        self._scheduler_stop_event = threading.Event()

        # ── 输出队列 ──
        self._output_queue = queue.Queue()
        self._orig_stdout = sys.stdout

        # ── UI ──
        self._build_ui()
        self._poll_output_queue()

        # ── 启动后自动初始化管道 ──
        self.after(300, self._init_pipeline)

    # ── UI 构建 ───────────────────────────────────────────────────────

    def _build_ui(self):
        # 顶部状态栏
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(10, 0))

        ttk.Label(top, text="EduDaily", font=("", 16, "bold")).pack(side="left")
        self._status_label = ttk.Label(top, text="⏳ 初始化中…", foreground="gray")
        self._status_label.pack(side="right")
        self._api_label = ttk.Label(top, text="", foreground="gray")
        self._api_label.pack(side="right", padx=(0, 15))

        # 标签页
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self._tabs = {}
        for name, label in [
            ("dashboard", "🏠 首页"),
            ("sources", "📡 信息源"),
            ("fetch", "🔄 抓取 & 分析"),
            ("platform", "📱 平台输出"),
            ("schedule", "⏰ 定时任务"),
            ("settings", "⚙ 设置"),
        ]:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=label)
            self._tabs[name] = frame

        self._create_dashboard_tab()
        self._create_sources_tab()
        self._create_fetch_tab()
        self._create_platform_tab()
        self._create_schedule_tab()
        self._create_settings_tab()

        # 底部操作日志
        bottom = ttk.LabelFrame(self, text="操作日志", padding=5)
        bottom.pack(fill="both", padx=10, pady=(0, 10))
        self._log_widget = ScrolledText(bottom, height=6, state="disabled", wrap="word")
        self._log_widget.pack(fill="both", expand=True)

    # ── 首页 ──────────────────────────────────────────────────────────

    def _create_dashboard_tab(self):
        tab = self._tabs["dashboard"]
        tab.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(tab, text="欢迎使用 EduDaily", font=("", 18, "bold")).grid(
            row=row, column=0, pady=(30, 5)
        )
        row += 1
        ttk.Label(
            tab,
            text="教育资讯智能抓取 · AI 自动分析 · 多平台一键适配",
            font=("", 10),
        ).grid(row=row, column=0)

        row += 1
        ttk.Separator(tab, orient="horizontal").grid(
            row=row, column=0, sticky="ew", pady=20, padx=40
        )

        # 快速操作按钮
        row += 1
        btn_frame = ttk.Frame(tab)
        btn_frame.grid(row=row, column=0, pady=10)

        actions = [
            ("📰 获取今日资讯", self._on_fetch),
            ("📊 批量分析文章", self._on_batch_analyze),
            ("📋 复制最新日报", self._on_copy_latest),
            ("📱 生成多平台版本", lambda: self._select_tab("platform")),
        ]
        for i, (text, cmd) in enumerate(actions):
            btn = ttk.Button(btn_frame, text=text, command=cmd, width=20)
            btn.grid(row=0, column=i, padx=5)

        # 统计信息
        row += 1
        self._stats_frame = ttk.LabelFrame(tab, text="知识库概况", padding=15)
        self._stats_frame.grid(row=row, column=0, pady=25, padx=60, sticky="ew")
        for i in range(5):
            self._stats_frame.columnconfigure(i, weight=1)

        self._stats_labels = {}
        for i, (key, label) in enumerate([
            ("docs", "文档数"), ("chunks", "分块数"),
            ("sources", "信息源"), ("articles", "文章数"),
            ("vectors", "向量数"),
        ]):
            ttk.Label(self._stats_frame, text="--", font=("", 16, "bold")).grid(
                row=0, column=i, padx=10
            )
            self._stats_labels[key] = ttk.Label(
                self._stats_frame, text=label, font=("", 9), foreground="gray"
            )
            self._stats_labels[key].grid(row=1, column=i, padx=10)

    def _refresh_stats(self):
        if not self._conn:
            return
        try:
            cur = self._conn.execute
            vals = {
                "docs": cur("SELECT COUNT(*) FROM documents").fetchone()[0],
                "chunks": cur("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "sources": cur(
                    "SELECT COUNT(*) FROM news_sources WHERE enabled=1"
                ).fetchone()[0],
                "articles": cur("SELECT COUNT(*) FROM edu_articles").fetchone()[0],
                "vectors": self._collection.count() if self._collection else 0,
            }
        except Exception:
            vals = {"docs": "--", "chunks": "--", "sources": "--", "articles": "--", "vectors": "--"}

        for key, label in self._stats_labels.items():
            label_text = label.cget("text")
        # Update stats widget children
        children = self._stats_frame.winfo_children()
        idx = 0
        for key in ["docs", "chunks", "sources", "articles", "vectors"]:
            if idx < len(children):
                children[idx].configure(text=str(vals.get(key, "--")))
                idx += 2  # skip the description label

    # ── 信息源管理 ────────────────────────────────────────────────────

    def _create_sources_tab(self):
        tab = self._tabs["sources"]
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        # 添加表单
        form = ttk.Frame(tab)
        form.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ttk.Label(form, text="名称:").pack(side="left")
        self._src_name = ttk.Entry(form, width=18)
        self._src_name.pack(side="left", padx=(5, 10))

        ttk.Label(form, text="网址:").pack(side="left")
        self._src_url = ttk.Entry(form, width=40)
        self._src_url.pack(side="left", padx=(5, 10))

        ttk.Button(form, text="添加信息源", command=self._on_add_source).pack(side="left")

        # 列表
        cols = ("名称", "网址", "类型", "分类", "状态", "上次抓取")
        self._src_tree = ttk.Treeview(tab, columns=cols, show="headings", height=8)
        for c in cols:
            self._src_tree.heading(c, text=c)
        self._src_tree.column("名称", width=140)
        self._src_tree.column("网址", width=280)
        self._src_tree.column("类型", width=60)
        self._src_tree.column("分类", width=80)
        self._src_tree.column("状态", width=60)
        self._src_tree.column("上次抓取", width=120)
        self._src_tree.grid(row=2, column=0, sticky="nsew", padx=10)

        scroll = ttk.Scrollbar(tab, orient="vertical", command=self._src_tree.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self._src_tree.configure(yscrollcommand=scroll.set)

        btn_row = ttk.Frame(tab)
        btn_row.grid(row=3, column=0, sticky="w", padx=10, pady=10)
        ttk.Button(btn_row, text="删除选中", command=self._on_delete_source).pack(
            side="left", padx=(0, 5)
        )
        ttk.Button(btn_row, text="刷新列表", command=self._refresh_sources).pack(side="left")

    def _refresh_sources(self):
        if not self._conn:
            return
        for item in self._src_tree.get_children():
            self._src_tree.delete(item)
        try:
            from db.repository import list_news_sources
            for s in list_news_sources(self._conn):
                self._src_tree.insert("", "end", values=(
                    s.get("name", ""),
                    s.get("url", ""),
                    s.get("source_type", ""),
                    s.get("category", ""),
                    "启用" if s.get("enabled") else "停用",
                    s.get("last_fetched_at") or "从未",
                ), iid=str(s["id"]))
        except Exception as e:
            self._log(f"加载信息源失败: {e}")

    def _on_add_source(self):
        name = self._src_name.get().strip()
        url = self._src_url.get().strip()
        if not name or not url:
            messagebox.showwarning("提示", "请填写名称和网址。")
            return
        if not url.startswith(("http://", "https://")):
            messagebox.showwarning("提示", "网址需要以 http:// 或 https:// 开头。")
            return
        try:
            from db.repository import insert_news_source
            insert_news_source(self._conn, name, url)
            self._src_name.delete(0, "end")
            self._src_url.delete(0, "end")
            self._refresh_sources()
            self._log(f"已添加信息源: {name}")
        except Exception as e:
            self._log(f"添加失败: {e}")

    def _on_delete_source(self):
        sel = self._src_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择要删除的信息源。")
            return
        if not messagebox.askyesno("确认", "确定删除选中的信息源？关联的文章也会被删除。"):
            return
        try:
            from db.repository import delete_news_source
            for iid in sel:
                delete_news_source(self._conn, int(iid))
            self._refresh_sources()
            self._log(f"已删除 {len(sel)} 个信息源")
        except Exception as e:
            self._log(f"删除失败: {e}")

    # ── 抓取 & 分析 ─────────────────────────────────────────────────

    def _create_fetch_tab(self):
        tab = self._tabs["fetch"]
        tab.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(tab, text="每日新闻抓取与分析", font=("", 13, "bold")).grid(
            row=row, column=0, pady=(15, 5)
        )

        row += 1
        desc = ttk.Label(
            tab,
            text="从所有已启用的信息源抓取今日文章，自动归档索引。然后批量生成 AI 摘要与日报。",
            wraplength=700,
        )
        desc.grid(row=row, column=0, pady=(0, 15))

        row += 1
        btn_row = ttk.Frame(tab)
        btn_row.grid(row=row, column=0, pady=5)

        ttk.Button(
            btn_row, text="📰 抓取今日新闻", command=self._threaded(self._do_fetch), width=18
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_row, text="📊 批量分析已有文章",
            command=self._threaded(self._do_batch_analyze), width=18,
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_row, text="🔄 完整流程（抓取 + 分析）",
            command=self._threaded(self._do_full_pipeline), width=22,
        ).pack(side="left", padx=5)

        row += 1
        self._fetch_progress = ttk.Progressbar(tab, mode="indeterminate", length=500)
        self._fetch_progress.grid(row=row, column=0, pady=15)

        row += 1
        self._fetch_status = ttk.Label(tab, text="就绪", foreground="gray")
        self._fetch_status.grid(row=row, column=0)

    def _do_fetch(self):
        self._fetch_status.configure(text="正在抓取…", foreground="blue")
        self._fetch_progress.start()

        from modules.daily_fetcher import run_daily_fetch
        result = run_daily_fetch(
            self._client, self._conn,
            self._embedding_model, self._collection, self._docs,
        )

        self._fetch_progress.stop()
        self._fetch_status.configure(
            text=f"抓取完成: {result['total_articles']} 篇文章 | "
                 f"{result['sources_checked']} 个信息源",
            foreground="green",
        )
        for art in result.get("articles", [])[:5]:
            self._log(f"  📄 {art.get('title', '--')} [{art.get('publish_date', '--')}]")
        if result.get("errors"):
            for err in result["errors"]:
                self._log(f"  ⚠ {err}")
        self._refresh_stats()

    def _do_batch_analyze(self):
        self._fetch_status.configure(text="正在批量分析…", foreground="blue")
        self._fetch_progress.start()

        from batch_processor import process_batch
        result = process_batch(
            self._client, self._conn,
            self._embedding_model, self._collection,
        )

        self._fetch_progress.stop()
        status = f"分析完成: {result['processed']}/{result['total']} 篇"
        if result.get("report_path"):
            status += f" | 日报: {Path(result['report_path']).name}"
        self._fetch_status.configure(text=status, foreground="green")
        self._refresh_stats()

    def _do_full_pipeline(self):
        self._do_fetch()
        self._do_batch_analyze()

    # ── 平台输出 ──────────────────────────────────────────────────────

    def _create_platform_tab(self):
        tab = self._tabs["platform"]
        tab.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(tab, text="多平台内容适配", font=("", 13, "bold")).grid(
            row=row, column=0, pady=(15, 5)
        )

        row += 1
        ttk.Label(
            tab, text="选择平台，将最新日报转换为对应风格。输出文件保存在 output/ 目录。",
            wraplength=700,
        ).grid(row=row, column=0, pady=(0, 15))

        from template_engine import PLATFORMS, list_platforms

        row += 1
        pf_frame = ttk.LabelFrame(tab, text="选择平台", padding=15)
        pf_frame.grid(row=row, column=0, pady=5, padx=40, sticky="ew")

        self._platform_vars = {}
        self._platform_labels = {}
        for i, pinfo in enumerate(list_platforms()):
            var = tk.BooleanVar(value=False)
            self._platform_vars[pinfo["key"]] = var
            cb = ttk.Checkbutton(
                pf_frame,
                text=f"{pinfo['name']} — {pinfo['description']}",
                variable=var,
            )
            cb.grid(row=i, column=0, sticky="w", pady=2)
            self._platform_labels[pinfo["key"]] = cb

        # 全选 / 取消全选
        row += 1
        sel_frame = ttk.Frame(tab)
        sel_frame.grid(row=row, column=0, pady=5)
        ttk.Button(
            sel_frame, text="全选", command=lambda: self._toggle_platforms(True)
        ).pack(side="left", padx=5)
        ttk.Button(
            sel_frame, text="取消全选", command=lambda: self._toggle_platforms(False)
        ).pack(side="left", padx=5)

        row += 1
        btn_row = ttk.Frame(tab)
        btn_row.grid(row=row, column=0, pady=10)
        ttk.Button(
            btn_row, text="🚀 为最新日报生成平台版本",
            command=self._threaded(self._do_generate_platforms), width=28,
        ).pack(side="left", padx=5)
        ttk.Button(
            btn_row, text="📋 复制选中平台内容到剪贴板",
            command=self._on_copy_platform, width=28,
        ).pack(side="left", padx=5)

        row += 1
        self._plat_progress = ttk.Progressbar(tab, mode="indeterminate", length=500)
        self._plat_progress.grid(row=row, column=0, pady=10)

        row += 1
        self._plat_status = ttk.Label(tab, text="就绪", foreground="gray")
        self._plat_status.grid(row=row, column=0)

    def _toggle_platforms(self, state: bool):
        for var in self._platform_vars.values():
            var.set(state)

    def _get_selected_platforms(self):
        return [k for k, v in self._platform_vars.items() if v.get()]

    def _do_generate_platforms(self):
        platforms = self._get_selected_platforms()
        if not platforms:
            messagebox.showwarning("提示", "请至少选择一个平台。")
            return

        # Find latest summary content
        from clipboard import find_latest_report

        report = find_latest_report()
        if not report:
            self._plat_status.configure(text="未找到日报，请先执行分析。", foreground="red")
            return

        self._plat_status.configure(
            text=f"正在生成 {len(platforms)} 个平台版本…", foreground="blue"
        )
        self._plat_progress.start()

        content = report.read_text(encoding="utf-8")
        from template_engine import generate_all_platforms

        results = generate_all_platforms(
            self._client, content, "日报",
            output_dir=config.OUTPUT_DIR, platforms=platforms,
        )

        self._plat_progress.stop()
        for r in results:
            if r.get("file_path"):
                self._log(f"  ✓ {r['platform_name']}: {Path(r['file_path']).name}")
            elif r.get("error"):
                self._log(f"  ✗ {r['platform_name']}: {r['error']}")

        self._plat_status.configure(
            text=f"已生成 {sum(1 for r in results if r.get('file_path'))} 个平台版本",
            foreground="green",
        )

    def _on_copy_platform(self):
        platforms = self._get_selected_platforms()
        if not platforms:
            messagebox.showwarning("提示", "请至少选择一个平台。")
            return

        from clipboard import copy_latest_report

        ok_any = False
        for p in platforms:
            if copy_latest_report(platform=p):
                ok_any = True

        if ok_any:
            self._plat_status.configure(text="已复制到剪贴板", foreground="green")
        else:
            self._plat_status.configure(
                text="复制失败，请确认日报已生成且 pyperclip 已安装", foreground="red"
            )

    # ── 定时任务 ──────────────────────────────────────────────────────

    def _create_schedule_tab(self):
        tab = self._tabs["schedule"]
        tab.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(tab, text="定时自动运行", font=("", 13, "bold")).grid(
            row=row, column=0, pady=(15, 5)
        )

        row += 1
        ttk.Label(
            tab,
            text="设置在每天的固定时间自动执行：抓取新闻 → 批量分析 → 生成日报。\n启动后窗口保持打开，到时间自动运行。",
            wraplength=700,
        ).grid(row=row, column=0, pady=(0, 15))

        row += 1
        cfg_frame = ttk.LabelFrame(tab, text="时间设置", padding=15)
        cfg_frame.grid(row=row, column=0, pady=10)

        ttk.Label(cfg_frame, text="每天运行时间:").grid(row=0, column=0, padx=(0, 10))
        self._sched_hour = ttk.Combobox(
            cfg_frame, values=[f"{h:02d}" for h in range(24)],
            width=4, state="readonly",
        )
        self._sched_hour.grid(row=0, column=1)
        self._sched_hour.set(
            getattr(config, "SCHEDULE_TIME", "07:00").split(":")[0]
        )
        ttk.Label(cfg_frame, text=":").grid(row=0, column=2)
        self._sched_minute = ttk.Combobox(
            cfg_frame, values=[f"{m:02d}" for m in range(0, 60, 5)],
            width=4, state="readonly",
        )
        self._sched_minute.grid(row=0, column=3)
        self._sched_minute.set(
            getattr(config, "SCHEDULE_TIME", "07:00").split(":")[1]
        )

        ttk.Label(
            cfg_frame,
            text="（修改后需重启定时任务生效）",
            foreground="gray",
        ).grid(row=0, column=4, padx=(15, 0))

        row += 1
        btn_row = ttk.Frame(tab)
        btn_row.grid(row=row, column=0, pady=15)
        self._sched_start_btn = ttk.Button(
            btn_row, text="▶ 启动定时任务", command=self._on_start_scheduler, width=18,
        )
        self._sched_start_btn.pack(side="left", padx=5)
        self._sched_stop_btn = ttk.Button(
            btn_row, text="⏹ 停止", command=self._on_stop_scheduler,
            width=12, state="disabled",
        )
        self._sched_stop_btn.pack(side="left", padx=5)

        row += 1
        self._sched_status = ttk.Label(
            tab, text="定时任务未启动", foreground="gray", font=("", 10)
        )
        self._sched_status.grid(row=row, column=0, pady=10)

        row += 1
        ttk.Label(
            tab,
            text="💡 提示：保持本窗口打开即可。如需开机自启，可将 python gui.py 加入系统启动项。",
            foreground="gray",
            wraplength=700,
        ).grid(row=row, column=0, pady=(0, 20))

    def _on_start_scheduler(self):
        if self._scheduler_running:
            return

        from scheduler import run_full_pipeline

        hour = self._sched_hour.get()
        minute = self._sched_minute.get()
        schedule_time = f"{hour}:{minute}"

        self._scheduler_stop_event.clear()
        self._scheduler_running = True
        self._sched_start_btn.configure(state="disabled")
        self._sched_stop_btn.configure(state="normal")
        self._sched_status.configure(
            text=f"⏰ 定时任务运行中 — 每天 {schedule_time} 自动执行",
            foreground="blue",
        )
        self._log(f"定时任务已启动: 每天 {schedule_time}")

        def _schedule_loop():
            last_run_date = None
            while not self._scheduler_stop_event.is_set():
                now = datetime.now()
                today = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%H:%M")

                if current_time == schedule_time and last_run_date != today:
                    last_run_date = today
                    self._output_queue.put(f"[{now.strftime('%H:%M:%S')}] ⏰ 定时触发，开始执行...")
                    try:
                        run_full_pipeline(
                            self._client, self._conn,
                            self._embedding_model, self._collection, self._docs,
                        )
                    except Exception as e:
                        self._output_queue.put(f"定时任务出错: {e}")

                time.sleep(30)  # Check every 30 seconds

            self._scheduler_running = False
            self.after(0, lambda: self._sched_status.configure(
                text="定时任务已停止", foreground="gray",
            ))
            self.after(0, lambda: self._sched_start_btn.configure(state="normal"))
            self.after(0, lambda: self._sched_stop_btn.configure(state="disabled"))
            self._output_queue.put("定时任务已停止。")

        thread = threading.Thread(target=_schedule_loop, daemon=True)
        thread.start()

    def _on_stop_scheduler(self):
        self._scheduler_stop_event.set()
        self._log("正在停止定时任务…")

    # ── 设置 ──────────────────────────────────────────────────────────

    def _create_settings_tab(self):
        tab = self._tabs["settings"]
        tab.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(tab, text="系统设置", font=("", 13, "bold")).grid(
            row=row, column=0, pady=(15, 5), sticky="w", padx=20
        )

        # API Key
        row += 1
        api_frame = ttk.LabelFrame(tab, text="DeepSeek API 配置", padding=15)
        api_frame.grid(row=row, column=0, padx=20, pady=10, sticky="ew")

        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky="w")
        self._api_key_var = tk.StringVar(value=self._mask_key(config.DEEPSEEK_API_KEY))
        self._api_key_entry = ttk.Entry(api_frame, textvariable=self._api_key_var, width=55, show="*")
        self._api_key_entry.grid(row=0, column=1, padx=(10, 0))

        show_btn = ttk.Button(
            api_frame, text="👁",
            command=lambda: self._toggle_key_visibility(self._api_key_entry),
            width=3,
        )
        show_btn.grid(row=0, column=2, padx=(5, 0))

        ttk.Label(
            api_frame,
            text="API Key 存储在 .env 文件中。修改后需重启应用生效。",
            foreground="gray",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # 路径
        row += 1
        path_frame = ttk.LabelFrame(tab, text="数据路径", padding=15)
        path_frame.grid(row=row, column=0, padx=20, pady=10, sticky="ew")

        paths = [
            ("文档目录:", config.SAMPLE_DOCS_DIR),
            ("输出目录:", config.OUTPUT_DIR),
            ("数据库:", config.SQLITE_DB_PATH),
        ]
        for i, (label, val) in enumerate(paths):
            ttk.Label(path_frame, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Label(path_frame, text=val, foreground="gray").grid(
                row=i, column=1, sticky="w", padx=(10, 0), pady=2
            )

        # 依赖检查
        row += 1
        dep_frame = ttk.LabelFrame(tab, text="依赖检查", padding=15)
        dep_frame.grid(row=row, column=0, padx=20, pady=10, sticky="ew")

        deps = [
            ("pyperclip", "剪贴板复制"),
            ("schedule", "定时任务（增强模式）"),
            ("jinja2", "多平台模板引擎"),
        ]
        for i, (mod, desc) in enumerate(deps):
            try:
                __import__(mod)
                status = "✓ 已安装"
                color = "green"
            except ImportError:
                status = "✗ 未安装"
                color = "red"
            ttk.Label(dep_frame, text=f"{desc}:").grid(row=i, column=0, sticky="w", pady=2)
            ttk.Label(dep_frame, text=status, foreground=color).grid(
                row=i, column=1, sticky="w", padx=(10, 0), pady=2
            )

        # 操作按钮
        row += 1
        btn_row = ttk.Frame(tab)
        btn_row.grid(row=row, column=0, pady=15, padx=20, sticky="w")

        ttk.Button(
            btn_row, text="🔄 重新初始化管道", command=self._init_pipeline, width=20,
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            btn_row, text="📂 打开输出文件夹",
            command=lambda: __import__("os").startfile(config.OUTPUT_DIR)
            if sys.platform == "win32"
            else __import__("subprocess").run(
                ["open" if sys.platform == "darwin" else "xdg-open", config.OUTPUT_DIR]
            ),
            width=20,
        ).pack(side="left")

    @staticmethod
    def _mask_key(key: str) -> str:
        if not key or key == "sk-your-key-here":
            return ""
        if len(key) > 12:
            return key[:8] + "****" + key[-4:]
        return key

    def _toggle_key_visibility(self, entry):
        current = entry.cget("show")
        entry.configure(show="" if current == "*" else "*")

    # ── Tab 切换 ──────────────────────────────────────────────────────

    def _select_tab(self, name: str):
        notebook = self._tabs[name].master
        for idx, tab_id in enumerate(notebook.tabs()):
            if notebook.tab(tab_id, "text").endswith(
                {"dashboard": "首页", "sources": "信息源", "fetch": "分析",
                 "platform": "平台", "schedule": "定时", "settings": "设置"}[name]
            ):
                notebook.select(idx)
                return

    # ── 管道初始化 ────────────────────────────────────────────────────

    def _init_pipeline(self):
        self._log("正在初始化系统…")
        self._status_label.configure(text="⏳ 初始化中…", foreground="gray")

        import sqlite3
        import chromadb
        import openai
        from sentence_transformers import SentenceTransformer

        from db.schema import init_db
        from modules.embedder import build_chroma_collection
        from utils.helpers import setup_logging

        try:
            setup_logging(config.LOG_FILE)

            # SQLite
            db_dir = Path(config.SQLITE_DB_PATH).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(config.SQLITE_DB_PATH)
            self._conn.row_factory = sqlite3.Row
            init_db(self._conn)

            # DeepSeek
            api_key = config.DEEPSEEK_API_KEY
            has_key = bool(api_key) and api_key not in ("", "sk-your-key-here")
            if has_key:
                self._client = openai.OpenAI(
                    api_key=api_key, base_url=config.DEEPSEEK_BASE_URL
                )
            else:
                self._client = None

            # Embedding
            self._embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

            # ChromaDB
            chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            try:
                self._collection = chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
            except Exception:
                self._collection = build_chroma_collection(
                    config.CHROMA_PERSIST_DIR, config.CHROMA_COLLECTION_NAME
                )

            # Dirs
            Path(config.SAMPLE_DOCS_DIR).mkdir(parents=True, exist_ok=True)
            Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

            self._ready = True
            self._status_label.configure(text="✓ 就绪", foreground="green")

            if has_key:
                self._api_label.configure(text="🔑 API 已配置", foreground="green")
            else:
                self._api_label.configure(
                    text="⚠ API 未配置（请在设置中查看）", foreground="orange"
                )

            chunk_count = self._conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            vec_count = self._collection.count()

            self._log(
                f"初始化完成: {chunk_count} 个分块, {vec_count} 个向量索引"
            )
            self._refresh_sources()
            self._refresh_stats()

        except Exception as e:
            self._status_label.configure(text="✗ 初始化失败", foreground="red")
            self._log(f"初始化失败: {e}")
            raise

    # ── 日志输出 ──────────────────────────────────────────────────────

    def _log(self, text: str):
        self._log_widget.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._log_widget.insert("end", f"[{timestamp}] {text}\n")
        self._log_widget.see("end")
        self._log_widget.configure(state="disabled")

    def _poll_output_queue(self):
        """定期拉取队列中的输出并显示到日志控件。"""
        try:
            while True:
                text = self._output_queue.get_nowait()
                self._log(text)
        except queue.Empty:
            pass
        self.after(200, self._poll_output_queue)

    # ── 线程工具 ──────────────────────────────────────────────────────

    def _threaded(self, func):
        """装饰器：将函数放入后台线程执行。自动重定向 stdout 到日志队列。"""
        def wrapper():
            old_stdout = sys.stdout
            sys.stdout = QueueStream(self._output_queue)
            try:
                func()
            except Exception as e:
                self._output_queue.put(f"✗ 错误: {e}")
            finally:
                sys.stdout = old_stdout
                self._output_queue.put("完成。")

        return lambda: threading.Thread(target=wrapper, daemon=True).start()

    # ── 快速操作回调 ──────────────────────────────────────────────────

    def _on_fetch(self):
        self._select_tab("fetch")
        self._threaded(self._do_fetch)()

    def _on_batch_analyze(self):
        self._select_tab("fetch")
        self._threaded(self._do_batch_analyze)()

    def _on_copy_latest(self):
        from clipboard import copy_latest_report
        if copy_latest_report():
            self._log("已复制最新日报到剪贴板")
        else:
            self._log("复制失败 — 请确认日报已生成且 pyperclip 已安装")

    # ── 关闭 ──────────────────────────────────────────────────────────

    def destroy(self):
        self._scheduler_stop_event.set()
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        super().destroy()


# ═══════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = EduDailyGUI()
    app.protocol("WM_DELETE_WINDOW", app.destroy)
    app.mainloop()
