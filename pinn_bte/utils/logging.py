"""Structured logging utilities with Rich formatting."""

import logging
import subprocess
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple, TextIO

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel
from rich.theme import Theme

# Custom theme for training logs
PINN_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "epoch": "bold magenta",
    "loss": "blue",
    "time": "dim cyan",
})


class TeeConsole(Console):
    """Console that writes to both stdout and a log file."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_file: Optional[TextIO] = None
        self._plain_console: Optional[Console] = None

    def set_log_file(self, log_file: TextIO) -> None:
        """Set the log file to also write to."""
        self._log_file = log_file
        # Create a plain console for file output (no colors)
        self._plain_console = Console(file=StringIO(), force_terminal=False, no_color=True)

    def close_log_file(self) -> None:
        """Close the log file."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._plain_console = None

    def print(self, *args, **kwargs) -> None:
        """Print to console and optionally to log file."""
        # Print to console with colors
        super().print(*args, **kwargs)

        # Also write to log file (plain text)
        if self._log_file and self._plain_console:
            # Capture plain text output
            self._plain_console.file = StringIO()
            self._plain_console.print(*args, **kwargs)
            plain_text = self._plain_console.file.getvalue()
            self._log_file.write(plain_text)
            self._log_file.flush()


# Global console instance (with tee capability)
console = TeeConsole(theme=PINN_THEME)

# Track current log file path
_current_log_file: Optional[Path] = None


def setup_file_logging(output_dir: Path, run_id: str) -> Path:
    """Setup file logging for a training run."""
    global _current_log_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / f"training_{run_id}.log"
    log_file = open(log_file_path, "w", encoding="utf-8")

    # Write header
    log_file.write(f"PINN-BTE Training Log\n")
    log_file.write(f"Run ID: {run_id}\n")
    log_file.write(f"Started: {datetime.now().isoformat()}\n")
    log_file.write("=" * 70 + "\n\n")
    log_file.flush()

    # Set tee console to also write to this file
    console.set_log_file(log_file)
    _current_log_file = log_file_path

    # Also add file handler to Python logger
    logger = logging.getLogger("pinn_bte")
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    return log_file_path


def close_file_logging() -> None:
    """Close the current log file."""
    console.close_log_file()


def get_current_log_file() -> Optional[Path]:
    """Get the path to the current log file."""
    return _current_log_file


def get_logger(name: str = "pinn_bte", level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger with Rich handler."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Rich handler for pretty console output
    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
    )
    handler.setLevel(level)

    # Format
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def log_training_start(
    mode: str,
    num_epochs: int,
    device: str,
    run_id: str
) -> None:
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="bold")
    info_table.add_column("Value")
    info_table.add_row("Mode", mode)
    info_table.add_row("Epochs", str(num_epochs))
    info_table.add_row("Device", device)
    info_table.add_row("Run ID", run_id)

    console.print(Panel(
        info_table,
        title="[bold green]Starting Training",
        border_style="green"
    ))


def log_training_complete(
    elapsed_time: float,
    output_path: str,
    model_files: Optional[list] = None
) -> None:
    """Log training completion with summary."""
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="bold")
    info_table.add_column("Value")
    info_table.add_row("Time", f"{elapsed_time:.2f}s")
    info_table.add_row("Output", output_path)

    if model_files:
        for f in model_files:
            info_table.add_row("Saved", f"[dim]{f}[/dim]")

    console.print(Panel(
        info_table,
        title="[bold green]Training Complete",
        border_style="green"
    ))


def log_epoch_progress(
    epoch: int,
    losses: dict,
    total_loss: float,
    elapsed: float
) -> None:
    """Log epoch progress with formatted loss values."""
    loss_parts = " | ".join(
        f"[blue]{name}[/blue]={val:.2e}"
        for name, val in losses.items()
    )

    console.print(
        f"[dim]{elapsed:7.1f}s[/dim] | "
        f"[bold magenta]Epoch {epoch:5d}[/bold magenta] | "
        f"{loss_parts} | "
        f"[bold]Total[/bold]={total_loss:.2e}"
    )


def log_file_saved(filepath: str, description: str = "File") -> None:
    console.print(f"[success]Saved[/success] {description}: [dim]{filepath}[/dim]")


# Default logger instance
logger = get_logger()


def get_git_info() -> Tuple[Optional[str], Optional[str], bool]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return None, None, False

        full_sha = result.stdout.strip()
        short_sha = full_sha[:7]

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_dirty = bool(result.stdout.strip())

        return short_sha, full_sha, is_dirty

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None, None, False


def save_git_info(output_dir: Path, run_id: str) -> Optional[str]:
    short_sha, full_sha, is_dirty = get_git_info()

    if full_sha is None:
        console.print("[warning]Not in a git repository, skipping git info[/warning]")
        return None

    sha_str = f"{short_sha}-dirty" if is_dirty else short_sha

    # Save to file
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_file = output_dir / f"git_info_{run_id}.txt"
    with open(git_file, "w") as f:
        f.write(f"Git Commit: {full_sha}\n")
        f.write(f"Short SHA: {short_sha}\n")
        f.write(f"Dirty: {is_dirty}\n")
        f.write(f"Display: {sha_str}\n")

    console.print(f"[info]Git SHA:[/info] {sha_str}")

    return sha_str


def log_experiment_info(
    experiment_name: str,
    mode: str,
    run_id: str,
    output_dir: str,
    extra_info: Optional[dict] = None
) -> None:
    short_sha, _, is_dirty = get_git_info()
    sha_str = None
    if short_sha:
        sha_str = f"{short_sha}-dirty" if is_dirty else short_sha

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="bold")
    info_table.add_column("Value")
    info_table.add_row("Experiment", experiment_name)
    info_table.add_row("Mode", mode)
    info_table.add_row("Run ID", run_id)
    if sha_str:
        info_table.add_row("Git SHA", sha_str)
    info_table.add_row("Output", output_dir)

    if extra_info:
        for key, value in extra_info.items():
            info_table.add_row(key, str(value))

    console.print(Panel(
        info_table,
        title="[bold cyan]Experiment Configuration",
        border_style="cyan"
    ))
