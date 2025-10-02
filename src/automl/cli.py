"""State-of-the-art command-line interface with Rich terminal UI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__

__all__ = ["app", "main"]

# Create Typer app with rich help
app = typer.Typer(
    name="automl",
    help="🚀 State-of-the-art AutoML platform with cutting-edge optimization",
    add_completion=True,
    rich_markup_mode="rich",
)

# Rich console for beautiful output
console = Console()


@app.command()
def version():
    """Display the installed AutoML version."""
    console.print(Panel(
        f"[bold cyan]AutoML[/bold cyan] version [bold green]{__version__}[/bold green]",
        title="Version Information",
        border_style="cyan",
    ))


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Option(
            "--config", "-c",
            help="Path to configuration file (YAML/TOML)",
            exists=True,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir", "-o",
            help="Output directory for results and artifacts",
        ),
    ] = None,
    n_trials: Annotated[
        int,
        typer.Option(
            "--n-trials", "-n",
            help="Number of optimization trials",
            min=1,
        ),
    ] = 100,
    n_jobs: Annotated[
        int,
        typer.Option(
            "--n-jobs", "-j",
            help="Number of parallel jobs (-1 for all CPUs)",
        ),
    ] = -1,
    gpu: Annotated[
        bool,
        typer.Option(
            "--gpu/--no-gpu",
            help="Enable GPU acceleration",
        ),
    ] = False,
    distributed: Annotated[
        bool,
        typer.Option(
            "--distributed",
            help="Enable distributed computing with Ray",
        ),
    ] = False,
    track: Annotated[
        Optional[str],
        typer.Option(
            "--track",
            help="Experiment tracking backend (mlflow, wandb, tensorboard)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help="Enable verbose logging",
        ),
    ] = False,
):
    """🎯 Run AutoML optimization experiment.
    
    Execute a complete AutoML workflow with hyperparameter optimization,
    model selection, and ensemble building.
    """
    console.print(Panel(
        "[bold cyan]AutoML Experiment Runner[/bold cyan]\n"
        f"Configuration: [yellow]{config}[/yellow]\n"
        f"Trials: [green]{n_trials}[/green] | "
        f"Jobs: [green]{n_jobs}[/green] | "
        f"GPU: [{'green' if gpu else 'red'}]{gpu}[/]\n"
        f"Distributed: [{'green' if distributed else 'red'}]{distributed}[/] | "
        f"Tracking: [yellow]{track or 'None'}[/yellow]",
        title="🚀 Starting Experiment",
        border_style="cyan",
    ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load configuration
            task = progress.add_task("[cyan]Loading configuration...", total=None)
            config_dict = _load_config(config)
            progress.update(task, completed=True)

            # Initialize engine
            task = progress.add_task("[cyan]Initializing AutoML engine...", total=None)
            from .core.engine import default_engine
            engine = default_engine()
            progress.update(task, completed=True)

            # Run optimization
            task = progress.add_task(
                f"[cyan]Running optimization ({n_trials} trials)...",
                total=None,
            )
            
            # Here we would actually run the optimization
            # For now, show the structure
            console.print("\n[yellow]⚠️  Full execution engine implementation in progress[/yellow]")
            
            progress.update(task, completed=True)

        # Display results
        _display_results({})

    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def info():
    """📊 Display system and environment information."""
    import platform
    import sklearn
    
    table = Table(title="System Information", border_style="cyan")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Version/Value", style="green")

    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())
    table.add_row("AutoML", __version__)
    table.add_row("scikit-learn", sklearn.__version__)
    
    try:
        import numpy
        table.add_row("NumPy", numpy.__version__)
    except ImportError:
        pass
    
    try:
        import pandas
        table.add_row("Pandas", pandas.__version__)
    except ImportError:
        pass
    
    try:
        import optuna
        table.add_row("Optuna", optuna.__version__)
    except ImportError:
        pass
    
    try:
        import torch
        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA Available", str(torch.cuda.is_available()))
    except ImportError:
        pass

    console.print(table)


@app.command()
def validate(
    config: Annotated[
        Path,
        typer.Argument(help="Path to configuration file to validate"),
    ],
):
    """✅ Validate configuration file without running experiment."""
    console.print(f"[cyan]Validating configuration:[/cyan] {config}")
    
    try:
        config_dict = _load_config(config)
        
        # Validate using Pydantic models
        from .core.config import AutoMLConfig
        validated_config = AutoMLConfig(**config_dict)
        
        console.print("[green]✓[/green] Configuration is valid!")
        console.print(Panel(
            f"[bold]Configuration Summary[/bold]\n\n"
            f"Dataset: [cyan]{validated_config.dataset.name}[/cyan]\n"
            f"Model: [cyan]{validated_config.pipeline.model.name}[/cyan]\n"
            f"Optimizer: [cyan]{validated_config.optimizer.name}[/cyan]\n"
            f"CV Folds: [cyan]{validated_config.optimizer.cv_folds}[/cyan]",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]❌ Validation failed:[/bold red] {e}")
        raise typer.Exit(code=1)


def _load_config(config_path: Path) -> dict:
    """Load configuration from YAML or TOML file."""
    import yaml
    
    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path) as f:
            return yaml.safe_load(f)
    elif config_path.suffix == ".toml":
        import tomllib
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def _display_results(results: dict):
    """Display experiment results in rich format."""
    if not results:
        return
    
    table = Table(title="Experiment Results", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in results.items():
        table.add_row(key, str(value))
    
    console.print(table)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the automl console script."""
    try:
        app()
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
