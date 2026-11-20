"""Command Line Interface for the Recipe Step Autocompleter.

Provides:
- 'hello': Greeting and system diagnostics (required by project test/spec)
- 'autocomplete' / 'suggest': Autocomplete incomplete recipe instructions
- 'interactive': Interactive REPL session for home cooks
- 'stats': Display corpus and n-gram statistics + perplexity evaluation
- 'add': Add new recipe instructions to the corpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from app.core import NGramModel, detokenize, tokenize
from app.llm import generate_recipe_step_llm, get_llm_config, is_llm_available, llm_hello
from app.storage import (
    DEFAULT_RECIPES_FILE,
    append_recipe_step,
    load_recipe_corpus,
    write_builtin_corpus_file,
)


def get_trained_model(
    corpus_path: Optional[str] = None, n: int = 3, method: str = "laplace"
) -> NGramModel:
    """Load corpus and return a trained NGramModel."""
    corpus = load_recipe_corpus(corpus_path)
    model = NGramModel(n=n)
    model.train(corpus)
    return model


def cmd_hello(args: argparse.Namespace) -> int:
    """Handle 'hello' command: print greeting, system diagnostics, and quick start."""
    info = llm_hello()
    corpus = load_recipe_corpus()

    print("=" * 65)
    print(" 🍳 Recipe Step Autocompleter CLI - Welcome Home Chef! 🍳")
    print("=" * 65)
    print(f"Status       : Ready to help home cooks autocomplete recipe steps.")
    print(f"Corpus Steps : {len(corpus)} authentic recipe instructions loaded.")
    print(f"Default Model: {args.n if hasattr(args, 'n') else 3}-Gram Statistical Language Model")
    print(f"Smoothing    : Laplace (Add-1) + Lidstone + Backoff supported")
    print(f"LLM Provider : {info['provider']} (Active: {info['llm_ready']})")
    print("-" * 65)
    print("Quick Start Examples:")
    print('  python -m app.cli autocomplete "Heat olive oil in a"')
    print('  python -m app.cli autocomplete "Preheat oven to" --length 12 --num-suggestions 3')
    print('  python -m app.cli autocomplete "Whisk together flour" --method lidstone')
    print('  python -m app.cli interactive')
    print("=" * 65)
    return 0


def cmd_autocomplete(args: argparse.Namespace) -> int:
    """Handle 'autocomplete' / 'suggest' command."""
    prompt = args.prompt
    if not prompt:
        try:
            prompt = input("Enter incomplete recipe step: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 1

    if not prompt:
        print("Error: Empty prompt provided.")
        return 1

    model = get_trained_model(corpus_path=args.corpus, n=args.n)

    print("\n" + "=" * 65)
    print(f"Input Partial Step: \"{prompt}\"")
    print(f"Model: {args.n}-gram | Smoothing: {args.method} | Length: {args.length} words")
    print("=" * 65)

    if args.num_suggestions > 1:
        suggestions = model.generate_suggestions(
            prompt=prompt,
            num_suggestions=args.num_suggestions,
            length=args.length,
            method=args.method,
            temperature=args.temperature,
            order=args.n,
        )
        print(f"\nTop {len(suggestions)} Suggestions:")
        for idx, item in enumerate(suggestions, 1):
            print(f"\n  [{idx}] Continuation: {item['continuation']}")
            print(f"      Full Step   : {item['full_step']}")
    else:
        result = model.generate_continuation(
            prompt=prompt,
            max_words=args.length,
            method=args.method,
            temperature=args.temperature,
            order=args.n,
        )
        print(f"\nGenerated Continuation:")
        print(f"  👉 {result['continuation']}")
        print(f"\nComplete Recipe Step:")
        print(f"  🍳 {result['full_step']}")

    # If requested or LLM active, show comparison
    if args.compare_llm:
        print("\n" + "-" * 65)
        print("🤖 LLM Completion Comparison:")
        llm_result = generate_recipe_step_llm(prompt)
        print(f"  {llm_result}")

    print("=" * 65 + "\n")
    return 0


def cmd_interactive(args: argparse.Namespace) -> int:
    """Interactive shell for home cooks to type partial recipe steps and get suggestions."""
    print("=" * 65)
    print(" 🍳 Interactive Recipe Step Autocompleter 🍳")
    print(" Type a partial recipe step to get autocompletions.")
    print(" Commands: 'exit' or 'quit' to leave, ':n <int>' to change n-gram order.")
    print("=" * 65 + "\n")

    current_n = args.n
    model = get_trained_model(corpus_path=args.corpus, n=current_n)

    while True:
        try:
            line = input("Chef > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye and happy cooking!")
            break

        if not line:
            continue
        if line.lower() in ("exit", "quit", "q"):
            print("Goodbye and happy cooking!")
            break
        if line.startswith(":n "):
            try:
                new_n = int(line[3:].strip())
                if new_n >= 1:
                    current_n = new_n
                    model = get_trained_model(corpus_path=args.corpus, n=current_n)
                    print(f"Switched to {current_n}-gram model.")
                else:
                    print("n must be at least 1.")
            except ValueError:
                print("Invalid number for :n.")
            continue

        suggestions = model.generate_suggestions(
            prompt=line,
            num_suggestions=args.num_suggestions,
            length=args.length,
            method=args.method,
            temperature=args.temperature,
            order=current_n,
        )

        print("-" * 50)
        for idx, item in enumerate(suggestions, 1):
            print(f"  [{idx}] ... {item['continuation']}")
            print(f"      Full: {item['full_step']}")
        print("-" * 50)

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Display corpus and n-gram statistics and evaluate model perplexity."""
    corpus = load_recipe_corpus(args.corpus)
    model = get_trained_model(corpus_path=args.corpus, n=args.n)

    print("=" * 65)
    print(" 📊 Recipe Corpus & N-Gram Model Statistics 📊")
    print("=" * 65)
    print(f"Recipe Steps in Corpus : {len(corpus)}")
    print(f"Total Tokens Trained   : {model.total_tokens}")
    print(f"Vocabulary Size (|V|)  : {model.vocab_size}")
    print(f"Configured N-Gram Order: {model.n}")

    for k in range(1, model.n + 1):
        num_contexts = len(model.context_counts[k])
        total_ngrams = sum(sum(w.values()) for w in model.ngram_counts[k].values())
        print(f"  Order {k}: {num_contexts} distinct contexts, {total_ngrams} n-gram occurrences")

    # Evaluate perplexity across smoothing methods on sample test steps
    test_samples = [
        "Heat olive oil in a skillet over medium heat.",
        "Add garlic and onions, cooking until soft and fragrant.",
        "Bake in preheated oven for 20 minutes until golden brown.",
    ]
    print("\nPerplexity Evaluation (lower is better):")
    for method in ("laplace", "lidstone", "backoff"):
        pp = model.calculate_perplexity(test_samples, method=method)
        print(f"  Smoothing: {method:<10} => Perplexity: {pp:.2f}")

    print("=" * 65)
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    """Add a new recipe instruction step to the corpus."""
    step = args.step.strip()
    if not step:
        print("Error: Recipe step cannot be empty.")
        return 1

    target = Path(args.corpus) if args.corpus else DEFAULT_RECIPES_FILE
    append_recipe_step(step, target)
    print(f"Successfully added recipe step to {target}:")
    print(f"  \"{step}\"")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="recipe-step-autocompleter",
        description="Command-line interface to help home cooks autocomplete recipe instructions.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 'hello' subcommand
    subparsers.add_parser("hello", help="Greet user and verify system readiness.")

    # 'autocomplete' / 'suggest' subcommand
    ac_parser = subparsers.add_parser("autocomplete", aliases=["suggest"], help="Autocomplete a partial recipe step.")
    ac_parser.add_argument("prompt", nargs="?", default="", help="Incomplete recipe step prompt.")
    ac_parser.add_argument("-n", type=int, default=3, help="Order of n-gram model (default: 3).")
    ac_parser.add_argument("-l", "--length", type=int, default=10, help="Continuation length in words (default: 10).")
    ac_parser.add_argument("-k", "--num-suggestions", type=int, default=1, help="Number of suggestions (default: 1).")
    ac_parser.add_argument(
        "-s", "--method", choices=["laplace", "lidstone", "backoff", "mle"], default="laplace",
        help="Smoothing technique (default: laplace)."
    )
    ac_parser.add_argument("-t", "--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0 greedy).")
    ac_parser.add_argument("--corpus", type=str, default=None, help="Path to custom recipe corpus file.")
    ac_parser.add_argument("--compare-llm", action="store_true", help="Compare with LLM generation.")

    # 'interactive' subcommand
    it_parser = subparsers.add_parser("interactive", help="Start interactive completion session.")
    it_parser.add_argument("-n", type=int, default=3, help="Order of n-gram model (default: 3).")
    it_parser.add_argument("-l", "--length", type=int, default=10, help="Continuation length in words (default: 10).")
    it_parser.add_argument("-k", "--num-suggestions", type=int, default=3, help="Number of suggestions (default: 3).")
    it_parser.add_argument("-s", "--method", choices=["laplace", "lidstone", "backoff", "mle"], default="laplace", help="Smoothing technique.")
    it_parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Sampling temperature.")
    it_parser.add_argument("--corpus", type=str, default=None, help="Path to custom recipe corpus file.")

    # 'stats' subcommand
    stats_parser = subparsers.add_parser("stats", aliases=["evaluate"], help="Display corpus statistics and evaluate model.")
    stats_parser.add_argument("-n", type=int, default=3, help="N-gram order.")
    stats_parser.add_argument("--corpus", type=str, default=None, help="Path to custom corpus.")

    # 'add' subcommand
    add_parser = subparsers.add_parser("add", help="Add a new recipe step to the corpus.")
    add_parser.add_argument("step", type=str, help="Recipe instruction text to add.")
    add_parser.add_argument("--corpus", type=str, default=None, help="Target corpus file path.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "hello":
        return cmd_hello(args)
    elif args.command in ("autocomplete", "suggest"):
        return cmd_autocomplete(args)
    elif args.command == "interactive":
        return cmd_interactive(args)
    elif args.command in ("stats", "evaluate"):
        return cmd_stats(args)
    elif args.command == "add":
        return cmd_add(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
