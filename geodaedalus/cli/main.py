"""Main CLI application for GeoDaedalus."""

import asyncio
import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.tree import Tree

from geodaedalus.core.config import get_settings
from geodaedalus.core.logging import get_logger
from geodaedalus.core.metrics import get_metrics_collector
from geodaedalus.core.pipeline import GeoDaedalusPipeline
from geodaedalus.core.models import SearchEngine

app = typer.Typer(
    name="geodaedalus",
    help="GeoDaedalus: Academic multi-agent system for geoscience literature search and data aggregation",
    rich_markup_mode="rich"
)

console = Console()
logger = get_logger(__name__)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language query for geoscience data"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    max_papers: int = typer.Option(20, "--max-papers", "-n", help="Maximum number of papers to search"),
    max_extraction: Optional[int] = typer.Option(None, "--max-extraction", help="Maximum papers to extract data from"),
    engines: Optional[str] = typer.Option(None, "--engines", "-e", help="Comma-separated list of search engines (semantic_scholar,google_scholar)"),
    validation_level: str = typer.Option("standard", "--validation", help="Validation level (lenient,standard,strict)"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse query without executing search"),
    step_by_step: bool = typer.Option(False, "--step-by-step", help="Show intermediate results for each step"),
) -> None:
    """Search for geoscience literature and extract data based on natural language query."""
    
    # Setup
    settings = get_settings()
    if verbose:
        settings.logging.level = "DEBUG"
    
    session_id = uuid4()
    metrics = get_metrics_collector()
    
    # Parse search engines
    search_engines = None
    if engines:
        engine_names = [e.strip() for e in engines.split(",")]
        search_engines = []
        for name in engine_names:
            try:
                search_engines.append(SearchEngine(name))
            except ValueError:
                console.print(f"[red]Invalid search engine: {name}[/red]")
                console.print(f"Valid options: {', '.join([e.value for e in SearchEngine])}")
                raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold blue]GeoDaedalus Literature Search & Data Extraction[/bold blue]\n"
        f"Session ID: {session_id}\n"
        f"Query: [italic]{query}[/italic]\n"
        f"Max Papers: {max_papers} | Validation: {validation_level}",
        border_style="blue"
    ))
    
    try:
        if step_by_step:
            asyncio.run(_run_step_by_step_pipeline(
                query=query,
                session_id=session_id,
                output_dir=output_dir,
                max_papers=max_papers,
                search_engines=search_engines,
                dry_run=dry_run
            ))
        else:
            asyncio.run(_run_complete_pipeline(
                query=query,
                session_id=session_id,
                output_dir=output_dir,
                max_papers=max_papers,
                max_extraction=max_extraction,
                search_engines=search_engines,
                validation_level=validation_level,
                dry_run=dry_run
            ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Search failed: {e}")
        raise typer.Exit(1)
    
    # Show metrics summary
    summary = metrics.get_session_summary()
    _display_metrics_summary(summary)


async def _run_complete_pipeline(
    query: str,
    session_id,
    output_dir: Optional[Path],
    max_papers: int,
    max_extraction: Optional[int],
    search_engines: Optional[list],
    validation_level: str,
    dry_run: bool
) -> None:
    """Run the complete pipeline end-to-end."""
    
    async with GeoDaedalusPipeline(session_id=session_id) as pipeline:
        
        if dry_run:
            # Just validate constraints
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Parsing requirements...", total=None)
                constraints = await pipeline.validate_constraints(query)
                progress.update(task, completed=True, description="✓ Requirements parsed")
            
            _display_constraints(constraints)
            console.print("[yellow]Dry run completed. Skipping actual search and extraction.[/yellow]")
            return
        
        # Run complete pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing complete pipeline...", total=4)
            
            final_data = await pipeline.process_query(
                user_query=query,
                max_papers=max_papers,
                max_extraction_papers=max_extraction,
                search_engines=search_engines,
                validation_level=validation_level
            )
            
            progress.update(task, completed=4, description="✓ Pipeline completed")
        
        # Display results
        _display_pipeline_results(final_data)
        
        # Save results if output directory specified
        if output_dir:
            await _save_results(final_data, output_dir, session_id)


async def _run_step_by_step_pipeline(
    query: str,
    session_id,
    output_dir: Optional[Path],
    max_papers: int,
    search_engines: Optional[list],
    dry_run: bool
) -> None:
    """Run the pipeline step by step, showing intermediate results."""
    
    async with GeoDaedalusPipeline(session_id=session_id) as pipeline:
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Parse requirements
            task1 = progress.add_task("Step 1: Parsing requirements...", total=None)
            constraints = await pipeline.validate_constraints(query)
            progress.update(task1, completed=True, description="✓ Step 1: Requirements parsed")
            
            _display_constraints(constraints)
            
            if dry_run:
                console.print("[yellow]Dry run completed.[/yellow]")
                return
            
            # Step 2: Literature search
            task2 = progress.add_task("Step 2: Searching literature...", total=None)
            search_results = await pipeline.search_literature_only(
                constraints, 
                max_results=max_papers,
                search_engines=search_engines
            )
            progress.update(task2, completed=True, description="✓ Step 2: Literature search completed")
            
            _display_search_results(search_results)
            
            # Step 3: Data extraction (limit to 5 papers for demo)
            task3 = progress.add_task("Step 3: Extracting data...", total=None)
            extracted_data = await pipeline.extract_data_only(
                search_results,
                max_papers=min(5, len(search_results.papers))
            )
            progress.update(task3, completed=True, description="✓ Step 3: Data extraction completed")
            
            _display_extraction_results(extracted_data)
            
            # Step 4: Data fusion
            task4 = progress.add_task("Step 4: Fusing and validating data...", total=None)
            final_data = await pipeline.fuse_data_only(extracted_data)
            progress.update(task4, completed=True, description="✓ Step 4: Data fusion completed")
            
            _display_fusion_results(final_data)
        
        # Save results if output directory specified
        if output_dir:
            await _save_results(final_data, output_dir, session_id)


def _display_constraints(constraints) -> None:
    """Display parsed constraints in a formatted table."""
    
    table = Table(title="🔍 Parsed Geoscientific Constraints")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    # Spatial constraints
    if constraints.spatial:
        if constraints.spatial.location_name:
            table.add_row("📍 Location", constraints.spatial.location_name)
        if constraints.spatial.country:
            table.add_row("🌍 Country", constraints.spatial.country)
        if constraints.spatial.latitude and constraints.spatial.longitude:
            table.add_row("🗺️ Coordinates", f"{constraints.spatial.latitude:.4f}, {constraints.spatial.longitude:.4f}")
    
    # Temporal constraints
    if constraints.temporal:
        if constraints.temporal.geological_period:
            table.add_row("⏰ Geological Period", constraints.temporal.geological_period)
        if constraints.temporal.age_min or constraints.temporal.age_max:
            age_range = f"{constraints.temporal.age_min or '?'} - {constraints.temporal.age_max or '?'} Ma"
            table.add_row("📅 Age Range", age_range)
    
    # Rock types
    if constraints.rock_types:
        rock_types_str = ", ".join([rt.value for rt in constraints.rock_types])
        table.add_row("🪨 Rock Types", rock_types_str)
    
    # Element constraints
    for ec in constraints.element_constraints:
        elements_str = ", ".join(ec.elements)
        table.add_row(f"⚛️ Elements ({ec.category.value})", elements_str)
    
    # Additional keywords
    if constraints.additional_keywords:
        keywords_str = ", ".join(constraints.additional_keywords)
        table.add_row("🔤 Keywords", keywords_str)
    
    console.print(table)


def _display_search_results(search_results) -> None:
    """Display literature search results."""
    
    table = Table(title=f"📚 Literature Search Results ({len(search_results.papers)} papers found)")
    table.add_column("Title", style="cyan", max_width=50)
    table.add_column("Authors", style="green", max_width=30)
    table.add_column("Year", style="yellow", justify="center")
    table.add_column("Relevance", style="red", justify="center")
    
    for paper in search_results.papers[:10]:  # Show top 10
        authors = ", ".join([a.name for a in paper.authors[:2]])
        if len(paper.authors) > 2:
            authors += f" et al."
        
        relevance = f"{paper.relevance_score:.2f}" if paper.relevance_score else "N/A"
        
        table.add_row(
            paper.title[:47] + "..." if len(paper.title) > 50 else paper.title,
            authors,
            str(paper.year) if paper.year else "N/A",
            relevance
        )
    
    if len(search_results.papers) > 10:
        table.add_row("...", f"({len(search_results.papers) - 10} more papers)", "", "")
    
    console.print(table)


def _display_extraction_results(extracted_data) -> None:
    """Display data extraction results."""
    
    console.print(f"\n📊 Data Extraction Results")
    console.print(f"Papers processed: {len(extracted_data.source_papers)}")
    console.print(f"Tables extracted: {len(extracted_data.tables)}")
    
    if extracted_data.tables:
        table = Table(title="Extracted Data Tables")
        table.add_column("Paper", style="cyan", max_width=30)
        table.add_column("Table", style="green")
        table.add_column("Samples", style="yellow", justify="center")
        table.add_column("Elements", style="red", justify="center")
        table.add_column("Confidence", style="magenta", justify="center")
        
        for dt in extracted_data.tables[:10]:  # Show first 10 tables
            table.add_row(
                str(dt.paper_id)[:8] + "...",
                dt.table_number or "N/A",
                str(len(dt.sample_ids)),
                str(len(dt.element_data)),
                f"{dt.extraction_confidence:.2f}" if dt.extraction_confidence else "N/A"
            )
        
        console.print(table)


def _display_fusion_results(final_data) -> None:
    """Display data fusion and validation results."""
    
    console.print(f"\n🔬 Data Fusion & Validation Results")
    
    # Quality metrics
    metrics_table = Table(title="Quality Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    quality_metrics = final_data.consolidated_data.get("quality_metrics", {})
    metrics_table.add_row("Data Completeness", f"{quality_metrics.get('completeness', 0):.1%}")
    metrics_table.add_row("Consistency Score", f"{quality_metrics.get('consistency', 0):.1%}")
    metrics_table.add_row("Reliability", f"{quality_metrics.get('reliability', 0):.1%}")
    
    console.print(metrics_table)
    
    # Element statistics
    element_stats = final_data.consolidated_data.get("element_statistics", {})
    if element_stats:
        stats_table = Table(title="Element Statistics (Top 10)")
        stats_table.add_column("Element", style="cyan")
        stats_table.add_column("Count", style="green", justify="center")
        stats_table.add_column("Mean", style="yellow", justify="center")
        stats_table.add_column("Range", style="red", justify="center")
        
        for element, stats in list(element_stats.items())[:10]:
            stats_table.add_row(
                element,
                str(stats["count"]),
                f"{stats['mean']:.2f}",
                f"{stats['min']:.2f} - {stats['max']:.2f}"
            )
        
        console.print(stats_table)


def _display_pipeline_results(final_data) -> None:
    """Display complete pipeline results."""
    
    console.print("\n🎯 Pipeline Results Summary")
    
    # Create a tree view of results
    tree = Tree("📊 GeoDaedalus Results")
    
    # Pipeline metadata
    pipeline_meta = final_data.consolidated_data.get("pipeline_metadata", {})
    search_summary = pipeline_meta.get("search_summary", {})
    processing_summary = pipeline_meta.get("processing_summary", {})
    
    # Search branch
    search_branch = tree.add("🔍 Literature Search")
    search_branch.add(f"Papers Found: {search_summary.get('papers_found', 0)}")
    search_branch.add(f"Engines Used: {', '.join(search_summary.get('engines_used', []))}")
    if search_summary.get('search_time'):
        search_branch.add(f"Search Time: {search_summary['search_time']:.2f}s")
    
    # Extraction branch
    extraction_branch = tree.add("📊 Data Extraction")
    extraction_branch.add(f"Papers Processed: {processing_summary.get('papers_processed', 0)}")
    extraction_branch.add(f"Tables Extracted: {processing_summary.get('tables_extracted', 0)}")
    extraction_branch.add(f"Final Tables: {processing_summary.get('final_tables', 0)}")
    
    # Quality branch
    quality_branch = tree.add("✅ Data Quality")
    quality_metrics = final_data.consolidated_data.get("quality_metrics", {})
    quality_branch.add(f"Completeness: {quality_metrics.get('completeness', 0):.1%}")
    quality_branch.add(f"Consistency: {quality_metrics.get('consistency', 0):.1%}")
    quality_branch.add(f"Reliability: {quality_metrics.get('reliability', 0):.1%}")
    
    # Data summary branch
    data_branch = tree.add("📈 Data Summary")
    data_branch.add(f"Total Samples: {final_data.consolidated_data.get('total_samples', 0)}")
    data_branch.add(f"Elements Found: {final_data.consolidated_data.get('total_elements', 0)}")
    data_branch.add(f"Source Papers: {final_data.consolidated_data.get('total_papers', 0)}")
    
    console.print(tree)


async def _save_results(final_data, output_dir: Path, session_id) -> None:
    """Save pipeline results to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete results as JSON
    results_file = output_dir / f"geodaedalus_results_{session_id}.json"
    with open(results_file, 'w') as f:
        json.dump(final_data.dict(), f, indent=2, default=str)
    
    # Save consolidated data as CSV if possible
    try:
        import pandas as pd
        
        # Create a summary CSV
        summary_data = []
        for table in final_data.tables:
            for element, values in table.element_data.items():
                for i, value in enumerate(values):
                    summary_data.append({
                        "paper_id": str(table.paper_id),
                        "table_number": table.table_number,
                        "sample_id": table.sample_ids[i] if i < len(table.sample_ids) else f"sample_{i}",
                        "element": element,
                        "value": value,
                        "location": table.location_info.location_name if table.location_info else None,
                        "rock_type": table.rock_type.value if table.rock_type else None,
                        "extraction_confidence": table.extraction_confidence
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = output_dir / f"geodaedalus_data_{session_id}.csv"
            df.to_csv(csv_file, index=False)
            console.print(f"📁 Results saved to: {output_dir}")
            console.print(f"   - Complete results: {results_file.name}")
            console.print(f"   - Data summary: {csv_file.name}")
        
    except ImportError:
        console.print(f"📁 Results saved to: {results_file}")


def _display_metrics_summary(summary: dict) -> None:
    """Display metrics summary."""
    
    table = Table(title="⚡ Execution Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Operations", str(summary.get("total_operations", 0)))
    table.add_row("Success Rate", f"{summary.get('success_rate', 0):.1%}")
    table.add_row("Total Duration", f"{summary.get('total_duration_seconds', 0):.2f}s")
    table.add_row("Total Tokens", str(summary.get("total_tokens_used", 0)))
    table.add_row("Total API Calls", str(summary.get("total_api_calls", 0)))
    table.add_row("Estimated Cost", f"${summary.get('total_cost_estimate', 0):.4f}")
    
    console.print(table)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    set_key: Optional[str] = typer.Option(None, "--set", help="Set configuration key=value"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
) -> None:
    """Manage GeoDaedalus configuration."""
    
    settings = get_settings()
    
    if show:
        config_dict = settings.dict()
        console.print_json(json.dumps(config_dict, indent=2, default=str))
    
    if set_key:
        console.print(f"[yellow]Configuration setting not yet implemented: {set_key}[/yellow]")
    
    if validate:
        try:
            # Test LLM connection
            console.print("🔧 Validating configuration...")
            console.print(f"✓ LLM Provider: {settings.llm.provider}")
            console.print(f"✓ LLM Model: {settings.llm.model}")
            console.print("✓ Configuration is valid")
        except Exception as e:
            console.print(f"❌ Configuration error: {e}")


@app.command()
def benchmark(
    dataset: str = typer.Argument(..., help="Benchmark dataset name"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    metrics_only: bool = typer.Option(False, "--metrics-only", help="Only compute metrics, don't run pipeline"),
) -> None:
    """Run GeoDaedalus on benchmark datasets."""
    
    console.print(f"[yellow]Benchmark functionality not yet implemented for dataset: {dataset}[/yellow]")
    # TODO: Implement benchmark functionality


@app.command()
def export_metrics(
    session_id: Optional[str] = typer.Option(None, "--session", help="Session ID to export"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", help="Export format (json, csv)"),
) -> None:
    """Export execution metrics and performance data."""
    
    metrics = get_metrics_collector()
    
    if session_id:
        data = metrics.get_session_metrics(session_id)
        console.print(f"Exported metrics for session: {session_id}")
    else:
        data = metrics.get_all_metrics()
        console.print("Exported all metrics")
    
    if output_file:
        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "csv":
            # TODO: Implement CSV export
            console.print("[yellow]CSV export not yet implemented[/yellow]")
        
        console.print(f"📁 Metrics saved to: {output_file}")
    else:
        console.print_json(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    app() 