#!/usr/bin/env python3
"""
List all experiments with beautiful formatting

Displays experiments in a clean table with metrics, timestamps, and summary statistics.
Supports filtering by model family and sorting options.
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.manager import list_experiments


def format_timestamp(ts: str) -> str:
    """Convert timestamp to human-readable format"""
    try:
        # Try parsing full datetime
        if len(ts) == 15:  # YYYYMMDD_HHMMSS
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M")
        else:
            return ts
    except:
        return ts


def truncate_name(name: str, max_len: int = 100) -> str:
    """Truncate long experiment names"""
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + "..."


def print_experiments_table(df, show_path: bool = False):
    """Print experiments in a beautiful formatted table"""
    
    if df.empty:
        print("\n❌ No experiments found")
        return
    
    # Sort by timestamp (newest first)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp', ascending=False)
    
    # Format columns
    df_display = df.copy()
    
    # Format score with 4 decimals
    if 'score' in df_display.columns:
        df_display['score'] = df_display['score'].apply(lambda x: f"{float(x):.4f}" if x else "N/A")
    
    # Format timestamp
    if 'timestamp' in df_display.columns:
        df_display['display_time'] = df_display['timestamp'].apply(format_timestamp)
    
    # Truncate experiment names
    if 'experiment_name' in df_display.columns:
        df_display['exp_name'] = df_display['experiment_name'].apply(lambda x: truncate_name(x, 100))
        
    # FORMAT has_thinkings as True/False string
    if 'has_thinkings' in df_display.columns:
        df_display['has_thinkings'] = df_display['has_thinkings'].apply(
            lambda x: 'True' if x else 'False'
        )
    
    # Select columns to display
    display_cols = []
    col_mapping = {
        'exp_name': 'Experiment',
        'model_name': 'Model',
        'score': 'Score',
        'num_questions': 'Questions',
        'num_topics': 'Topics',
        'has_thinkings': 'Thinking',
        'display_time': 'Timestamp'
    }
    
    for col in col_mapping.keys():
        if col in df_display.columns:
            display_cols.append(col)
    
    if show_path and 'relative_path' in df_display.columns:
        display_cols.append('relative_path')
        col_mapping['relative_path'] = 'Path'
    
    # Rename columns
    df_final = df_display[display_cols].rename(columns=col_mapping)
    
    # Print header
    print("\n" + "="*120)
    print("📊 EXPERIMENT RESULTS")
    print("="*120)
    
    # Print table
    print(df_final.to_string(index=False, max_colwidth=50))
    
    # Print summary statistics
    print("\n" + "-"*120)
    print(f"📈 SUMMARY")
    print("-"*120)
    
    total_experiments = len(df)
    unique_models = df['model_name'].nunique() if 'model_name' in df.columns else 0
    
    if 'score' in df.columns:
        # Convert back to float for stats
        scores = df['score'].dropna().astype(float)
        if not scores.empty:
            best_score = scores.max()
            avg_score = scores.mean()
            best_exp = df.loc[df['score'].astype(float).idxmax(), 'experiment_name']
        else:
            best_score = avg_score = 0.0
            best_exp = "N/A"
    else:
        best_score = avg_score = 0.0
        best_exp = "N/A"
    
    print(f"Total experiments:    {total_experiments}")
    print(f"Unique models:        {unique_models}")
    print(f"Best score:           {best_score:.4f} ({truncate_name(best_exp, 40)})")
    print(f"Average score:        {avg_score:.4f}")
    
    # Model breakdown
    if 'model_name' in df.columns and total_experiments > 1:
        print(f"\n📊 By Model:")
        model_counts = df['model_name'].value_counts()
        for model, count in model_counts.items():
            model_scores = df[df['model_name'] == model]['score'].dropna()
            if not model_scores.empty:
                avg = model_scores.astype(float).mean()
                print(f"   {model:30s} {count:3d} experiments  (avg: {avg:.4f})")
            else:
                print(f"   {model:30s} {count:3d} experiments")
    
    print("\n" + "="*120)
    print(f"💡 Tip: Use --filter <model> to show only specific models")
    print(f"💡 Tip: Use --path to show experiment paths")
    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="List all experiments with beautiful formatting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  python scripts/list_experiments.py
  
  # Filter by model
  python scripts/list_experiments.py --filter claude-sonnet-4.5
  
  # Show paths
  python scripts/list_experiments.py --path
  
  # Sort by score (best first)
  python scripts/list_experiments.py --sort score
        """
    )
    
    parser.add_argument("--filter", type=str, metavar="MODEL",
                        help="Filter by model name (e.g., 'claude-sonnet-4.5')")
    parser.add_argument("--path", action="store_true",
                        help="Show experiment paths")
    parser.add_argument("--sort", type=str, choices=["time", "score", "name"],
                        default="time",
                        help="Sort by: time (default), score, or name")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Show only N most recent experiments")
    
    args = parser.parse_args()
    
    # Load experiments
    print("🔍 Loading experiments...")
    df = list_experiments(model_filter=args.filter, as_dataframe=True)
    
    if df.empty:
        print("\n❌ No experiments found")
        if args.filter:
            print(f"   Filter: {args.filter}")
        print("\n💡 Run an experiment first:")
        print("   python scripts/run_experiment.py --model-family claude --version claude-sonnet-4.5 --limit 10")
        return
    
    # Apply sorting
    if args.sort == "score" and 'score' in df.columns:
        df = df.sort_values('score', ascending=False)
        print("   Sorted by: score (best first)")
    elif args.sort == "name" and 'experiment_name' in df.columns:
        df = df.sort_values('experiment_name')
        print("   Sorted by: name (alphabetical)")
    else:
        # Default: time (already sorted in list_experiments)
        print("   Sorted by: time (newest first)")
    
    # Apply limit
    if args.limit:
        df = df.head(args.limit)
        print(f"   Showing: top {args.limit} experiments")
    
    # Print table
    print_experiments_table(df, show_path=args.path)


if __name__ == "__main__":
    main()
