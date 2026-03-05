"""
main.py
────────
CLI entry point for the Financial Analyst Agent.

Usage:
    # Single query
    python main.py --query "What are the key credit risk factors for large US banks?"

    # Single query with verbose graph tracing
    python main.py --query "..." --verbose

    # Interactive mode (run multiple queries without restarting)
    python main.py --interactive

    # Print the agent graph as Mermaid diagram
    python main.py --show-graph
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.graph import get_graph
from src.agent.state import initial_state

console = Console()

EXAMPLE_QUERIES = [
    "What are the key credit risk factors for large US banks and how does Basel III address them?",
    "How does JPMorgan's capital position compare to Basel III minimum requirements?",
    "What systemic risks does the Federal Reserve identify, and what macro context does the ECB provide?",
]


def run_query(query: str, verbose: bool = False) -> str:
    """Run a single query through the agent and return the final report."""
    graph = get_graph()

    if verbose:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")

    state = initial_state(query)

    # Stream the graph execution
    final_state = None

    if verbose:
        # Stream mode: print each node's output as it runs
        for step in graph.stream(state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]
            console.print(f"[dim]── Node: {node_name}[/dim]")

            if node_name == "decompose_query" and "sub_questions" in node_output:
                sqs = node_output["sub_questions"]
                if sqs:
                    console.print(f"  Decomposed into {len(sqs)} sub-questions:")
                    for sq in sqs:
                        scope = f" [{sq['document_scope']}]" if sq.get("document_scope") else ""
                        console.print(f"    {sq['id']}: {sq['question'][:80]}{scope}")

            elif node_name == "retrieve_context":
                idx = node_output.get("current_sq_index", 0)
                chunks = sum(
                    len(sq.get("retrieved_chunks", []))
                    for sq in node_output.get("sub_questions", [])
                )
                console.print(f"  Retrieved for sub-question {idx}, total chunks so far: {chunks}")

            elif node_name == "evaluate_sufficiency":
                score = node_output.get("overall_sufficiency", 0)
                console.print(f"  Sufficiency score: {score:.2f}")
                reasons = node_output.get("insufficiency_reasons", [])
                if reasons:
                    console.print(f"  Issues: {reasons[0][:80]}")

            elif node_name == "refine_retrieval":
                console.print("  Reformulating sub-questions and retrying...")

            final_state = {**state, **node_output}
    else:
        # Silent mode: just invoke and collect result
        with console.status("[bold green]Thinking...[/bold green]"):
            final_state = graph.invoke(state)

    return final_state.get("final_report", "Error: No report generated.")


def print_report(report: str) -> None:
    """Render the markdown report to the terminal."""
    console.print()
    console.print(Rule("[bold]Financial Analysis Report[/bold]", style="green"))
    console.print()
    try:
        console.print(Markdown(report))
    except Exception:
        console.print(report)
    console.print()


def show_graph() -> None:
    """Print the Mermaid graph diagram."""
    graph = get_graph()
    try:
        mermaid = graph.get_graph().draw_mermaid()
        console.print("\n[bold]Agent Graph (Mermaid):[/bold]\n")
        console.print(mermaid)
        console.print("\nPaste the above at https://mermaid.live/ to visualize.\n")
    except Exception as e:
        console.print(f"[yellow]Could not render graph: {e}[/yellow]")
        console.print("Ensure langgraph is up to date: pip install -U langgraph")


def interactive_mode() -> None:
    """Interactive REPL for running multiple queries."""
    console.print(Panel(
        "[bold green]Financial Analyst Agent[/bold green]\n"
        "Type your query and press Enter. Type 'quit' to exit.\n"
        "Type 'examples' to see sample queries.",
        title="Interactive Mode",
    ))

    while True:
        try:
            query = console.input("\n[bold cyan]Query>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break
        if query.lower() == "examples":
            for i, q in enumerate(EXAMPLE_QUERIES, 1):
                console.print(f"  {i}. {q}")
            continue

        try:
            report = run_query(query, verbose=False)
            print_report(report)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial Analyst Agent — Multi-step RAG over financial documents"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", "-q", type=str, help="Query to analyze")
    group.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    group.add_argument("--show-graph", action="store_true", help="Print agent graph as Mermaid")

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show step-by-step agent execution",
    )

    args = parser.parse_args()

    if args.show_graph:
        show_graph()
        return

    if args.interactive:
        interactive_mode()
        return

    if args.query:
        report = run_query(args.query, verbose=args.verbose)
        print_report(report)
        return

    # No arguments: show help and example
    parser.print_help()
    console.print("\n[bold]Example:[/bold]")
    console.print(
        '  python main.py --query "What are the key credit risk factors for large US banks?" --verbose'
    )
    console.print("\n[bold]Sample queries:[/bold]")
    for q in EXAMPLE_QUERIES:
        console.print(f"  • {q}")


if __name__ == "__main__":
    main()
