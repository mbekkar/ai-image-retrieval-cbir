"""
CBIR CLI — Command-Line Interface
==================================
Usage examples:
    python search.py index  --dir ./dataset
    python search.py search --query ./query.jpg --top 5
    python search.py eval   --k 5

Author : Mounir Bekkar
"""

import argparse
import sys
from pathlib import Path

from cbir_engine import (
    build_index, load_index,
    find_similar, evaluate_precision_at_k,
    INDEX_FILE
)


def cmd_index(args):
    """Build and save the feature index."""
    print(f"\n[CBIR] Building index from: {args.dir}")
    index = build_index(args.dir, save_path=args.save)
    print(f"[CBIR] Done — {len(index)} images indexed.\n")


def cmd_search(args):
    """Search for similar images."""
    print(f"\n[CBIR] Loading index from: {args.index}")
    index = load_index(args.index)

    if not Path(args.query).exists():
        print(f"[ERROR] Query image not found: {args.query}")
        sys.exit(1)

    print(f"[CBIR] Searching for top-{args.top} similar images to: {args.query}\n")
    results = find_similar(args.query, index, top_k=args.top)

    if not results:
        print("[CBIR] No results found.")
        return

    print(f"{'Rank':<6} {'Similarity':>10}  {'Path'}")
    print("─" * 60)
    for rank, (path, score) in enumerate(results, 1):
        bar   = "█" * int(score * 20)
        label = "Très similaire" if score > 0.85 else "Similaire" if score > 0.65 else "Modéré" if score > 0.45 else "Faible"
        print(f"#{rank:<5} {score:>10.4f}  {path}")
        print(f"       [{bar:<20}] {label}")
    print()


def cmd_eval(args):
    """Evaluate Precision@k."""
    print(f"\n[CBIR] Loading index from: {args.index}")
    index = load_index(args.index)

    print(f"[CBIR] Evaluating Precision@{args.k} ...")
    precision = evaluate_precision_at_k(index, k=args.k)
    print(f"\n[CBIR] Mean Precision@{args.k} = {precision:.4f} ({precision*100:.1f}%)\n")


def main():
    parser = argparse.ArgumentParser(
        description="CBIR — Content-Based Image Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1 — build index
  python search.py index --dir ./dataset

  # Step 2 — search
  python search.py search --query ./my_photo.jpg --top 5

  # Step 3 — evaluate (optional)
  python search.py eval --k 5
        """
    )
    sub = parser.add_subparsers(dest="command")

    # ── index ──
    p_index = sub.add_parser("index", help="Build feature index from image folder")
    p_index.add_argument("--dir",  required=True, help="Path to image dataset folder")
    p_index.add_argument("--save", default=INDEX_FILE, help=f"Output index file (default: {INDEX_FILE})")

    # ── search ──
    p_search = sub.add_parser("search", help="Find similar images to a query")
    p_search.add_argument("--query", required=True, help="Path to query image")
    p_search.add_argument("--top",   type=int, default=5, help="Number of results (default: 5)")
    p_search.add_argument("--index", default=INDEX_FILE, help=f"Index file to use (default: {INDEX_FILE})")

    # ── eval ──
    p_eval = sub.add_parser("eval", help="Evaluate Precision@k on the indexed dataset")
    p_eval.add_argument("--k",     type=int, default=5, help="k for Precision@k (default: 5)")
    p_eval.add_argument("--index", default=INDEX_FILE, help=f"Index file to use (default: {INDEX_FILE})")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
