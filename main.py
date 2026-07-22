"""
main.py
─────────
Interactive terminal chat loop for the Email RAG Agent.

Run in a separate terminal from ingest.py:

    python main.py

Commands inside the chat:
  • Type any natural-language question, e.g.
        "Did Stripe email me today?"
        "Summarize the conversation with hr@acme.com last week"
        "Anything from amazon.com on 2025-05-30?"
  • /reset   — clear the conversation history (start fresh)
  • /help    — show this help
  • exit     — quit
"""

from __future__ import annotations

import sys
import traceback

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent import ask
from common import log

console = Console()


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                  📧  EMAIL RAG AGENT  —  v1.0                    ║
║   LangGraph · Qdrant · Gemini · Hugging Face · IMAPClient IDLE   ║
╚══════════════════════════════════════════════════════════════════╝

Type your question and hit Enter.
Examples:
  • Did Stripe email me today?
  • Summarize emails from hr@acme.com last week
  • What was the latest thing GitHub sent me?

Commands:
  /reset   clear conversation history
  /help    show usage
  exit     quit
"""


def print_help() -> None:
    console.print(Panel.fit(BANNER.strip(), border_style="cyan"))


def main() -> None:
    print_help()

    history: list = []

    while True:
        try:
            user_input = console.input("\n[bold green]you ›[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[magenta]👋 Bye![/magenta]")
            return

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in {"exit", "quit", ":q"}:
            console.print("[magenta]👋 Bye![/magenta]")
            return
        if cmd in {"/help", "help", "?"}:
            print_help()
            continue
        if cmd in {"/reset", "/clear"}:
            history = []
            console.print("[yellow]🔄 Conversation history cleared.[/yellow]")
            continue

        # Real query → run through the LangGraph agent.
        try:
            answer, history = ask(user_input, history)
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            continue
        except Exception as e:
            log("💥 [Error]", f"{e}", "bold red")
            traceback.print_exc()
            continue

        console.print()
        console.print(
            Panel(
                Markdown(answer) if answer else "[dim](no answer generated)[/dim]",
                title="🤖 agent",
                border_style="cyan",
                padding=(1, 2),
            )
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
