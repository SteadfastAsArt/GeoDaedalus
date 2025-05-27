"""
GeoDaedalus Demo Script
======================

This script demonstrates the complete GeoDaedalus geoscientific data extraction pipeline.
It includes:
1. Natural language requirement understanding
2. Academic literature search and retrieval
3. Document processing and data extraction
4. Data fusion and quality assessment
5. Benchmark evaluation system

Run with: python demo.py [--mode {quick|full|benchmark}]
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from geodaedalus.core.config import Settings
from geodaedalus.core.pipeline import GeoDaedalusPipeline
from geodaedalus.benchmark.datasets import GeoDataBenchDataset
from geodaedalus.benchmark.evaluator import BenchmarkEvaluator


class GeoDaedalusDemo:
    """Comprehensive demo of GeoDaedalus system."""
    
    def __init__(self):
        self.console = Console()
        self.settings = Settings()
        self.demo_queries = [
            "Find volcanic rocks from Hawaii with major element compositions",
            "Get Cretaceous basalts from the Deccan Traps with trace element data",
            "Search for Archean komatiites with rare earth element patterns",
            "Find sedimentary rocks from the Permian period with carbonate chemistry",
            "Get metamorphic rocks from the Alps with pressure-temperature conditions"
        ]
    
    def show_welcome(self):
        """Display welcome message."""
        welcome_text = """
[bold blue]Welcome to GeoDaedalus![/bold blue]

A comprehensive geoscientific data extraction pipeline that:
• 🧠 Understands natural language queries
• 📚 Searches academic literature
• 📄 Processes documents and extracts data
• 🔬 Fuses data from multiple sources
• 📊 Provides quality assessments

Ready to explore geoscientific data extraction!
        """
        
        self.console.print(Panel(welcome_text, title="GeoDaedalus Demo", border_style="blue"))
    
    def show_system_overview(self):
        """Display system architecture overview."""
        tree = Tree("🏗️ [bold]GeoDaedalus Architecture[/bold]")
        
        # Core components
        core_branch = tree.add("🔧 [cyan]Core Components[/cyan]")
        core_branch.add("⚙️ Configuration & Settings")
        core_branch.add("📝 Logging & Monitoring") 
        core_branch.add("🔄 Pipeline Orchestrator")
        core_branch.add("📊 Data Models & Schemas")
        
        # Agents
        agents_branch = tree.add("🤖 [green]Intelligent Agents[/green]")
        agents_branch.add("🧠 Requirement Understanding Agent")
        agents_branch.add("🔍 Literature Search Agent")
        agents_branch.add("📄 Data Extraction Agent")
        agents_branch.add("🔬 Data Fusion Agent")
        
        # Services  
        services_branch = tree.add("🛠️ [yellow]Services[/yellow]")
        services_branch.add("🤖 LLM Service (OpenAI)")
        services_branch.add("🔍 Academic Search Service")
        services_branch.add("📄 Document Processing Service")
        
        # Benchmarking
        bench_branch = tree.add("📏 [magenta]Evaluation & Benchmarking[/magenta]")
        bench_branch.add("📊 Benchmark Datasets")
        bench_branch.add("🎯 Task-Specific Evaluators")
        bench_branch.add("📈 Performance Metrics")
        bench_branch.add("📋 Comprehensive Reporting")
        
        self.console.print(tree)
    
    async def quick_demo(self):
        """Run a quick demonstration of the pipeline."""
        self.console.print("\n[bold yellow]🚀 Quick Demo Mode[/bold yellow]")
        
        # Initialize pipeline
        with Progress(SpinnerColumn(), TextColumn("Initializing pipeline..."), console=self.console) as progress:
            task = progress.add_task("setup", total=1)
            pipeline = GeoDaedalusPipeline(self.settings)
            progress.advance(task)
        
        # Demo query
        demo_query = self.demo_queries[0]
        self.console.print(f"\n[bold]Demo Query:[/bold] {demo_query}")
        
        try:
            # Step 1: Requirement Understanding
            self.console.print("\n[cyan]Step 1: Understanding Requirements[/cyan]")
            with Progress(SpinnerColumn(), TextColumn("Parsing natural language..."), console=self.console) as progress:
                task = progress.add_task("understanding", total=1)
                constraints = await pipeline.agent1.process(demo_query)
                progress.advance(task)
            
            self._display_constraints(constraints)
            
            # Step 2: Literature Search
            self.console.print("\n[cyan]Step 2: Searching Literature[/cyan]")
            with Progress(SpinnerColumn(), TextColumn("Searching academic papers..."), console=self.console) as progress:
                task = progress.add_task("searching", total=1)
                papers = await pipeline.agent2.process(constraints, max_papers=5)
                progress.advance(task)
            
            self._display_papers(papers)
            
            # Step 3: Document Processing (if we have papers)
            if papers:
                self.console.print("\n[cyan]Step 3: Processing Documents[/cyan]")
                with Progress(SpinnerColumn(), TextColumn("Extracting data..."), console=self.console) as progress:
                    task = progress.add_task("processing", total=1)
                    extracted_data = await pipeline.agent3.process(papers[:2])  # Process first 2 papers
                    progress.advance(task)
                
                self._display_extracted_data(extracted_data)
                
                # Step 4: Data Fusion
                if extracted_data:
                    self.console.print("\n[cyan]Step 4: Fusing Data[/cyan]")
                    with Progress(SpinnerColumn(), TextColumn("Fusing and validating..."), console=self.console) as progress:
                        task = progress.add_task("fusing", total=1)
                        fused_data = await pipeline.agent4.process(extracted_data)
                        progress.advance(task)
                    
                    self._display_fused_data(fused_data)
        
        except Exception as e:
            self.console.print(f"[red]Demo failed: {e}[/red]")
        
        finally:
            # Pipeline cleanup (no close method needed for now)
            pass
    
    async def full_demo(self):
        """Run a comprehensive demo with multiple queries."""
        self.console.print("\n[bold yellow]🎯 Full Demo Mode[/bold yellow]")
        
        pipeline = GeoDaedalusPipeline(self.settings)
        
        try:
            for i, query in enumerate(self.demo_queries[:3], 1):
                self.console.print(f"\n[bold blue]Demo {i}/3: {query}[/bold blue]")
                
                # Run complete pipeline
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console
                ) as progress:
                    task = progress.add_task(f"Processing query {i}...", total=1)
                    
                    result = await pipeline.process_query(
                        user_query=query,
                        max_papers=3
                    )
                    
                    progress.advance(task)
                
                # Display results summary
                self._display_pipeline_result(result, i)
                
                # Brief pause between demos
                await asyncio.sleep(1)
        
        except Exception as e:
            self.console.print(f"[red]Full demo failed: {e}[/red]")
        
        finally:
            await pipeline.close()
    
    async def benchmark_demo(self):
        """Run benchmark evaluation demo."""
        self.console.print("\n[bold yellow]📊 Benchmark Demo Mode[/bold yellow]")
        
        try:
            # Generate benchmark datasets
            self.console.print("\n[cyan]Generating Benchmark Datasets[/cyan]")
            benchmark_data = GeoDataBenchDataset()
            datasets_dir = Path("benchmark_datasets")
            
            with Progress(SpinnerColumn(), TextColumn("Creating datasets..."), console=self.console) as progress:
                task = progress.add_task("datasets", total=1)
                benchmark_data.generate_all_datasets(datasets_dir)
                progress.advance(task)
            
            # Show dataset statistics
            stats = benchmark_data.get_statistics()
            self._display_benchmark_stats(stats)
            
            # Run evaluation on a subset
            self.console.print("\n[cyan]Running Benchmark Evaluation[/cyan]")
            evaluator = BenchmarkEvaluator(self.settings)
            
            results_dir = Path("benchmark_results")
            
            # Quick evaluation with limited samples
            report = await evaluator.run_comprehensive_evaluation(
                dataset_dir=datasets_dir,
                output_dir=results_dir,
                max_samples_per_task=5  # Limit for demo
            )
            
            self.console.print(f"\n[green]✅ Benchmark evaluation completed![/green]")
            self.console.print(f"📁 Results saved to: {results_dir}")
            
            await evaluator.close()
        
        except Exception as e:
            self.console.print(f"[red]Benchmark demo failed: {e}[/red]")
    
    def _display_constraints(self, constraints):
        """Display extracted constraints."""
        table = Table(title="Extracted Constraints")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="green")
        
        # Spatial constraints
        if constraints.spatial:
            spatial_details = []
            if constraints.spatial.location_name:
                spatial_details.append(f"Location: {constraints.spatial.location_name}")
            if constraints.spatial.country:
                spatial_details.append(f"Country: {constraints.spatial.country}")
            table.add_row("Spatial", "\n".join(spatial_details) if spatial_details else "None")
        
        # Temporal constraints
        if constraints.temporal:
            temporal_details = []
            if constraints.temporal.geological_period:
                temporal_details.append(f"Period: {constraints.temporal.geological_period}")
            if constraints.temporal.age_min and constraints.temporal.age_max:
                temporal_details.append(f"Age: {constraints.temporal.age_min}-{constraints.temporal.age_max} Ma")
            table.add_row("Temporal", "\n".join(temporal_details) if temporal_details else "None")
        
        # Rock types
        if constraints.rock_types:
            rock_types = [rt.value for rt in constraints.rock_types]
            table.add_row("Rock Types", ", ".join(rock_types))
        
        # Element constraints
        if constraints.element_constraints:
            for ec in constraints.element_constraints:
                elements = ", ".join(ec.elements[:5])  # Show first 5 elements
                if len(ec.elements) > 5:
                    elements += f" (+ {len(ec.elements) - 5} more)"
                table.add_row(f"Elements ({ec.category.value})", elements)
        
        self.console.print(table)
    
    def _display_papers(self, papers):
        """Display found papers."""
        if not papers:
            self.console.print("[yellow]No papers found[/yellow]")
            return
        
        table = Table(title=f"Found Papers ({len(papers)})")
        table.add_column("Title", style="cyan", max_width=50)
        table.add_column("Authors", style="green", max_width=30)
        table.add_column("Year", style="yellow")
        table.add_column("Citations", style="magenta")
        
        for paper in papers[:5]:  # Show first 5 papers
            authors = ", ".join([a.name for a in paper.authors[:2]])
            if len(paper.authors) > 2:
                authors += f" (+ {len(paper.authors) - 2} more)"
            
            table.add_row(
                paper.title[:47] + "..." if len(paper.title) > 50 else paper.title,
                authors,
                str(paper.year) if paper.year else "N/A",
                str(paper.citation_count) if paper.citation_count else "N/A"
            )
        
        self.console.print(table)
    
    def _display_extracted_data(self, extracted_data):
        """Display extracted data."""
        if not extracted_data:
            self.console.print("[yellow]No data extracted[/yellow]")
            return
        
        table = Table(title="Extracted Data Summary")
        table.add_column("Paper", style="cyan")
        table.add_column("Tables Found", style="green")
        table.add_column("Data Points", style="yellow")
        table.add_column("Status", style="magenta")
        
        for data in extracted_data[:3]:  # Show first 3 extractions
            paper_title = data.paper_title[:30] + "..." if len(data.paper_title) > 33 else data.paper_title
            tables_count = len(data.extracted_tables)
            data_points = sum(len(table.data) for table in data.extracted_tables)
            
            table.add_row(
                paper_title,
                str(tables_count),
                str(data_points),
                "✅ Success" if data.extraction_status == "success" else "⚠️ Partial"
            )
        
        self.console.print(table)
    
    def _display_fused_data(self, fused_data):
        """Display fused data results."""
        if not fused_data or not fused_data.fused_tables:
            self.console.print("[yellow]No fused data available[/yellow]")
            return
        
        # Summary table
        table = Table(title="Data Fusion Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Fused Tables", str(len(fused_data.fused_tables)))
        table.add_row("Quality Score", f"{fused_data.quality_metrics.overall_quality:.2f}")
        table.add_row("Data Completeness", f"{fused_data.quality_metrics.data_completeness:.2f}")
        table.add_row("Average Confidence", f"{fused_data.quality_metrics.average_confidence:.2f}")
        
        if fused_data.quality_metrics.duplicate_papers_removed:
            table.add_row("Duplicates Removed", str(fused_data.quality_metrics.duplicate_papers_removed))
        if fused_data.quality_metrics.outliers_detected:
            table.add_row("Outliers Detected", str(fused_data.quality_metrics.outliers_detected))
        
        self.console.print(table)
    
    def _display_pipeline_result(self, result: Dict[str, Any], demo_number: int):
        """Display complete pipeline result."""
        panel_title = f"Pipeline Result {demo_number}"
        
        # Create summary text
        summary_lines = []
        summary_lines.append(f"📚 Papers Found: {len(result.get('papers', []))}")
        summary_lines.append(f"📄 Data Extracted: {len(result.get('extracted_data', []))}")
        summary_lines.append(f"🔬 Fused Tables: {len(result.get('fused_data', {}).get('fused_tables', []))}")
        
        quality_metrics = result.get('quality_metrics', {})
        if quality_metrics:
            summary_lines.append(f"📊 Quality Score: {quality_metrics.get('overall_quality', 0.0):.2f}")
            summary_lines.append(f"✅ Completeness: {quality_metrics.get('data_completeness', 0.0):.2f}")
        
        summary_text = "\n".join(summary_lines)
        
        self.console.print(Panel(summary_text, title=panel_title, border_style="green"))
    
    def _display_benchmark_stats(self, stats: Dict[str, Any]):
        """Display benchmark dataset statistics."""
        # Overall stats
        table = Table(title="Benchmark Dataset Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Samples", str(stats["total_samples"]))
        table.add_row("Tasks", str(len(stats["tasks"])))
        table.add_row("Difficulty Levels", str(len(stats["difficulty_distribution"])))
        
        self.console.print(table)
        
        # Task breakdown
        task_table = Table(title="Samples by Task")
        task_table.add_column("Task", style="cyan")
        task_table.add_column("Samples", style="green")
        task_table.add_column("Easy", style="yellow")
        task_table.add_column("Medium", style="orange3")
        task_table.add_column("Hard", style="red")
        
        for task, task_stats in stats["tasks"].items():
            breakdown = task_stats["difficulty_breakdown"]
            task_table.add_row(
                task.replace('_', ' ').title(),
                str(task_stats["sample_count"]),
                str(breakdown.get("easy", 0)),
                str(breakdown.get("medium", 0)),
                str(breakdown.get("hard", 0))
            )
        
        self.console.print(task_table)


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="GeoDaedalus Demo")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "benchmark"],
        default="quick",
        help="Demo mode to run"
    )
    
    args = parser.parse_args()
    
    demo = GeoDaedalusDemo()
    
    # Show welcome and overview
    demo.show_welcome()
    demo.show_system_overview()
    
    # Run selected demo mode
    try:
        if args.mode == "quick":
            await demo.quick_demo()
        elif args.mode == "full":
            await demo.full_demo()
        elif args.mode == "benchmark":
            await demo.benchmark_demo()
        
        demo.console.print("\n[bold green]🎉 Demo completed successfully![/bold green]")
        demo.console.print("\n[italic]Thank you for exploring GeoDaedalus![/italic]")
        
    except KeyboardInterrupt:
        demo.console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        demo.console.print(f"\n[red]Demo failed with error: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main()) 