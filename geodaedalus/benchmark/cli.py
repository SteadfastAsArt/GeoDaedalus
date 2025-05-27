"""CLI for GeoDataBench benchmark suite."""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="geo-bench",
    help="GeoDataBench: Benchmark suite for evaluating geoscience data extraction systems",
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Dataset name to benchmark against"),
    model: str = typer.Option("geodaedalus", "--model", "-m", help="Model/system to evaluate"),
    output_dir: Path = typer.Option(Path("benchmark_results"), "--output", "-o", help="Output directory"),
    split: str = typer.Option("test", "--split", help="Dataset split to use"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Maximum number of samples to evaluate"),
) -> None:
    """Run benchmark evaluation on specified dataset."""
    
    console.print(f"[blue]Running GeoDataBench evaluation[/blue]")
    console.print(f"Dataset: {dataset}")
    console.print(f"Model: {model}")
    console.print(f"Split: {split}")
    
    # TODO: Implement benchmark runner
    console.print("[yellow]Benchmark implementation coming soon![/yellow]")


@app.command()
def list_datasets() -> None:
    """List available benchmark datasets."""
    
    # TODO: Implement dataset listing
    datasets = [
        "intent_understanding",
        "literature_retrieval", 
        "data_extraction",
        "data_fusion"
    ]
    
    table = Table(title="Available Benchmark Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Samples", style="yellow")
    
    for dataset in datasets:
        table.add_row(dataset, f"Description for {dataset}", "TBD")
    
    console.print(table)


@app.command()
def create_dataset(
    name: str = typer.Argument(..., help="Dataset name"),
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file"),
    output_dir: Path = typer.Option(Path("datasets"), "--output", "-o", help="Output directory"),
) -> None:
    """Create a new benchmark dataset."""
    
    console.print(f"[blue]Creating benchmark dataset: {name}[/blue]")
    
    # TODO: Implement dataset creation
    console.print("[yellow]Dataset creation coming soon![/yellow]")


benchmark_app = app 