#!/usr/bin/env python3
"""
Generate HTML dashboard for experiment results
"""
import sys
from pathlib import Path
import argparse
import webbrowser
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.dashboard import create_dashboard
from src.experiments.manager import get_latest_experiment, list_experiments


def main():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard for experiment")
    parser.add_argument("--path", type=str, help="Experiment folder path")
    parser.add_argument("--latest", action="store_true", help="Use latest experiment")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    parser.add_argument("--output", type=str, help="Custom output file path")
    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\n📋 Available Experiments:")
        print("="*80)
        df = list_experiments()
        print(df.to_string(index=False))
        print("\n💡 Use --path <experiment_path> to generate dashboard")
        return

    # Determine experiment path
    if args.latest:
        exp_path = get_latest_experiment()
        if not exp_path:
            print("❌ No experiments found")
            return
    elif args.path:
        exp_path = args.path
    else:
        print("❌ Please specify --path or --latest")
        print("\n💡 Try: python scripts/create_dashboard.py --list")
        return

    # Generate dashboard
    print(f"\n🎨 Generating dashboard for: {exp_path}")
    print("="*80)

    output_file = create_dashboard(exp_path, output_file=args.output)

    # Open in browser
    if args.open:
        print("\n🌐 Opening in browser...")
        webbrowser.open(f"file://{Path(output_file).absolute()}")
    else:
        print("\n💡 To open in browser:")
        print(f"   file://{Path(output_file).absolute()}")


if __name__ == "__main__":
    main()
