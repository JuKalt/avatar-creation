"""Full-screen Textual interface for registered method execution."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import traceback
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, RichLog, Static, TabbedContent, TabPane, Tabs

from .registry import MethodSpec
from .runner import PreparedMethod, prepare_method


_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _tab_pane_id(method: str) -> str:
    return _widget_id("method", method)


def _widget_id(prefix: str, method: str) -> str:
    return f"{prefix}-{_ID_RE.sub('-', method)}"


class MethodPane(Vertical):
    def __init__(self, method: str, *, single: bool = False) -> None:
        super().__init__(
            id=_widget_id("pane", method),
            classes="method-pane single" if single else "method-pane",
        )
        self.method = method

    def compose(self) -> ComposeResult:
        yield Static(
            f"[b]{self.method}[/b]  [dim]waiting[/dim]",
            id=_widget_id("status", self.method),
            classes="method-status",
            markup=True,
        )
        yield Static(
            "",
            id=_widget_id("progress", self.method),
            classes="method-progress",
        )
        yield RichLog(
            id=_widget_id("log", self.method),
            classes="method-log",
            wrap=True,
            highlight=False,
            markup=False,
            auto_scroll=True,
        )

    @property
    def output(self) -> RichLog:
        return self.query_one(f"#{_widget_id('log', self.method)}", RichLog)

    def set_progress(self, value: str) -> None:
        progress = self.query_one(
            f"#{_widget_id('progress', self.method)}", Static
        )
        progress.update(Text.from_ansi(value))

    def set_status(self, label: str, style: str) -> None:
        status = self.query_one(
            f"#{_widget_id('status', self.method)}", Static
        )
        status.update(f"[b]{self.method}[/b]  [{style}]{label}[/{style}]")


class MethodRunnerApp(App[None]):
    TITLE = "VCI Method Runner"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Stop"),
        ("tab", "focus_next", "Next log"),
        ("shift+tab", "focus_previous", "Previous log"),
    ]
    CSS = """
    Screen {
        background: #111418;
        color: #e6e8eb;
    }
    Header {
        background: #20262d;
    }
    #summary {
        height: 3;
        padding: 1 2;
        background: #171c21;
        color: #aeb7c2;
    }
    #method-tabs {
        height: 1fr;
        padding: 0 1 1 1;
    }
    #method-tabs TabPane {
        padding: 0;
    }
    #method-tabs Tabs {
        background: #171c21;
    }
    .method-pane {
        height: 1fr;
        min-height: 10;
        border: solid #46515d;
        background: #0c0f12;
    }
    .method-pane:focus-within {
        border: solid #63a9d8;
    }
    .method-pane.single {
        column-span: 2;
    }
    .method-status {
        height: 3;
        padding: 1 1;
        background: #20262d;
    }
    .method-progress {
        height: 1;
        padding: 0 1;
        background: #171c21;
        color: #8fc7e8;
        overflow: hidden;
    }
    .method-log {
        height: 1fr;
        padding: 0 1;
        scrollbar-color: #637080;
        scrollbar-background: #171c21;
    }
    Footer {
        background: #20262d;
    }
    """

    def __init__(
        self,
        specs: list[MethodSpec],
        *,
        dataset_kind: str,
        dataset_path: Path | None,
        output_dir: Path | None,
        required_keys_value: str | None,
        extra_args: list[str],
        subject_ids: tuple[str, ...],
        devices: tuple[str, ...],
        verbose_subject_output: bool,
        dry_run: bool,
        training_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.specs = specs
        self.dataset_kind = dataset_kind
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.required_keys_value = required_keys_value
        self.extra_args = extra_args
        self.training_config = training_config
        self.subject_ids = subject_ids
        self.devices = devices
        self.verbose_subject_output = verbose_subject_output
        self.dry_run = dry_run
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.method_return_code = 0
        self.finished_count = 0
        self.succeeded_count = 0
        self.failed_count = 0
        self.method_results: dict[str, tuple[str, Path | None]] = {}
        self.method_logs: dict[str, Path] = {}
        self.current_method: str | None = None
        self._stopping = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            f"Dataset: [b]{self.dataset_kind}[/b]   Finished: [b]0/{len(self.specs)}[/b]   Succeeded: 0   Failed: 0",
            id="summary",
            markup=True,
        )
        with TabbedContent(id="method-tabs"):
            for spec in self.specs:
                with TabPane(f"[ ] {spec.name}", id=_tab_pane_id(spec.name)):
                    yield MethodPane(spec.name, single=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(MethodPane).output.focus()
        self.run_worker(self._run_all(), name="method-runner", exclusive=True)

    def _pane(self, method: str) -> MethodPane:
        return self.query_one(f"#{_widget_id('pane', method)}", MethodPane)

    def _update_summary(self, *, final: bool = False) -> None:
        summary = self.query_one("#summary", Static)
        current = (
            f"   Active: [cyan]{self.current_method}[/cyan]"
            if self.current_method is not None
            else ""
        )
        suffix = "   All methods complete." if final else ""
        summary.update(
            f"Dataset: [b]{self.dataset_kind}[/b]   "
            f"Finished: [b]{self.finished_count}/{len(self.specs)}[/b]   "
            f"Succeeded: [green]{self.succeeded_count}[/green]   "
            f"Failed: [red]{self.failed_count}[/red]"
            f"{current}{suffix}"
        )

    def _set_method_status(
        self, method: str, label: str, style: str, marker: str
    ) -> None:
        self._pane(method).set_status(label, style)
        tabs = self.query_one("#method-tabs Tabs", Tabs)
        tab = tabs.get_tab(f"--content-tab-{_tab_pane_id(method)}")
        if tab is not None:
            tab.label = Text(f"{marker} {method}", style=style)

    async def _prepare(self, spec: MethodSpec) -> PreparedMethod:
        return await asyncio.to_thread(
            prepare_method,
            spec,
            dataset_kind=self.dataset_kind,
            dataset_path=self.dataset_path,
            output_dir=self.output_dir,
            required_keys_value=self.required_keys_value,
            extra_args=self.extra_args,
            training_config=self.training_config,
            subject_ids=self.subject_ids,
            devices=self.devices,
            verbose_subject_output=self.verbose_subject_output,
            warn_missing=False,
            persist_paths=not self.dry_run,
        )

    async def _run_all(self) -> None:
        self._update_summary()
        for spec in self.specs:
            if self._stopping:
                break
            self.current_method = spec.name
            self.query_one("#method-tabs", TabbedContent).active = _tab_pane_id(spec.name)
            self._update_summary()
            pane = self._pane(spec.name)
            self._set_method_status(spec.name, "preparing", "yellow", "[>]")
            pane.output.write("Preparing dataset manifest and command...")
            prepared: PreparedMethod | None = None
            try:
                prepared = await self._prepare(spec)
                pane.output.write(Text(prepared.summary))
                self.method_results[spec.name] = ("running", prepared.output_dir)
                pane.output.write(Text("Command: " + " ".join(prepared.command), style="cyan"))
                pane.output.write(f"Output: {prepared.output_dir}")
                if self.dry_run:
                    self._set_method_status(spec.name, "dry run complete", "green", "[ok]")
                    self.succeeded_count += 1
                    self.method_results[spec.name] = (
                        "dry run complete", prepared.output_dir
                    )
                else:
                    return_code = await self._run_process(prepared, pane)
                    if return_code == 0:
                        self._set_method_status(spec.name, "succeeded", "green", "[ok]")
                        self.succeeded_count += 1
                        self.method_results[spec.name] = (
                            "succeeded", prepared.output_dir
                        )
                    else:
                        self.method_return_code = 1
                        self.failed_count += 1
                        self._set_method_status(spec.name, f"failed ({return_code})", "red", "[x]")
                        self.method_results[spec.name] = (
                            f"failed ({return_code})", prepared.output_dir
                        )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                self.method_return_code = 1
                self.failed_count += 1
                self._set_method_status(spec.name, "failed", "red", "[x]")
                pane.output.write(Text(f"{type(exc).__name__}: {exc}", style="bold red"))
                output_path = prepared.output_dir if prepared is not None else None
                self.method_results[spec.name] = ("failed", output_path)
                log_path = self.method_logs.get(spec.name)
                if log_path is None:
                    log_path = (
                        prepared.output_dir / "runner.log"
                        if prepared is not None
                        else Path.cwd() / "method_logs" / spec.name / "runner.log"
                    )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                    if log_file.tell():
                        log_file.write("\n")
                    log_file.write(traceback.format_exc())
                self.method_logs[spec.name] = log_path
            finally:
                self.finished_count += 1
                self.current_method = None
                self._update_summary()

        if self._stopping:
            self.query_one("#summary", Static).update(
                f"[yellow]Stopping active methods...[/yellow]   "
                f"Finished: {self.finished_count}/{len(self.specs)}"
            )
        else:
            self._update_summary(final=True)
            self.exit()

    async def _run_process(
        self, prepared: PreparedMethod, pane: MethodPane
    ) -> int:
        prepared.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = prepared.output_dir / "runner.log"
        self.method_logs[prepared.spec.name] = log_path
        log_path.touch()
        pane.output.write(f"Full log: {log_path}")
        self._set_method_status(prepared.spec.name, "running", "bold cyan", "[>]")
        process = await asyncio.create_subprocess_exec(
            *prepared.command,
            cwd=prepared.cwd,
            env=prepared.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes[prepared.spec.name] = process
        try:
            assert process.stdout is not None
            pending = ""
            with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
                log_file.write(prepared.summary)
                log_file.flush()
                while chunk := await process.stdout.read(4096):
                    text = chunk.decode("utf-8", errors="replace")
                    log_file.write(text)
                    log_file.flush()
                    for character in text:
                        if character == "\r":
                            if pending:
                                pane.set_progress(pending)
                            pending = ""
                        elif character == "\n":
                            if pending:
                                pane.output.write(Text.from_ansi(pending))
                            pending = ""
                            pane.set_progress("")
                        else:
                            pending += character
                if pending:
                    pane.output.write(Text.from_ansi(pending))
                    pane.set_progress("")
            return await process.wait()
        finally:
            self.processes.pop(prepared.spec.name, None)

    async def _terminate_processes(self) -> None:
        self._stopping = True
        active = list(self.processes.values())
        for process in active:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if active:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(process.wait() for process in active)),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                for process in active:
                    if process.returncode is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    async def action_quit(self) -> None:
        await self._terminate_processes()
        self.exit()


def _print_run_overview(app: MethodRunnerApp) -> None:
    table = Table(title="Method run overview")
    table.add_column("Method", style="bold")
    table.add_column("Status")
    table.add_column("Output directory")
    failure_logs: list[tuple[str, Path]] = []

    for spec in app.specs:
        status, output_path = app.method_results.get(spec.name, ("not run", None))
        if status.startswith("succeeded") or status == "dry run complete":
            rendered_status = f"[green]{status}[/green]"
        elif status.startswith("failed"):
            rendered_status = f"[red]{status}[/red]"
        else:
            rendered_status = f"[yellow]{status}[/yellow]"
        table.add_row(spec.name, rendered_status, str(output_path or "-"))
        if status.startswith("failed"):
            log_path = app.method_logs.get(spec.name)
            if log_path is not None and log_path.is_file():
                failure_logs.append((spec.name, log_path))

    table.caption = (
        f"Finished: {app.finished_count}/{len(app.specs)}  "
        f"Succeeded: {app.succeeded_count}  Failed: {app.failed_count}"
    )
    console = Console()
    console.print("", table)
    if failure_logs:
        console.print("[bold red]Failure logs:[/bold red]")
        for method, log_path in failure_logs:
            console.print(f"  [red]{method}:[/red] {log_path}")


def run_textual_methods(
    specs: list[MethodSpec],
    *,
    dataset_kind: str,
    dataset_path: Path | None,
    output_dir: Path | None,
    required_keys_value: str | None,
    extra_args: list[str],
    subject_ids: tuple[str, ...],
    devices: tuple[str, ...],
    verbose_subject_output: bool,
    dry_run: bool,
    training_config: dict | None = None,
) -> int:
    app = MethodRunnerApp(
        specs,
        dataset_kind=dataset_kind,
        dataset_path=dataset_path,
        output_dir=output_dir,
        required_keys_value=required_keys_value,
        extra_args=extra_args,
        training_config=training_config,
        subject_ids=subject_ids,
        devices=devices,
        verbose_subject_output=verbose_subject_output,
        dry_run=dry_run,
    )
    app.run()
    _print_run_overview(app)
    return app.method_return_code
