#!/usr/bin/env python3
"""
Quick Launcher for MUSE Pantheon Universal Ingest Pipeline
Provides sensible defaults for common use cases
"""
import sys
import argparse
from pathlib import Path
import subprocess

def main():
    """Quick launcher with sensible defaults."""
    parser = argparse.ArgumentParser(description="Quick launcher for MUSE Pantheon Pipeline")
    parser.add_argument('--scan-root', default='.',
                       help='Root directory to scan (default: current directory)')
    parser.add_argument('--workdir', default='./_work',
                       help='Working directory for output (default: ./_work)')
    parser.add_argument('--user', default='user',
                       help='User identifier (default: user)')
    parser.add_argument('--apply', action='store_true',
                       help='Actually run the pipeline (default: dry run)')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (default: common types)')
    
    args = parser.parse_args()
    
    # Set default file types if not specified
    if not args.types:
        args.types = ['.py', '.md', '.txt', '.json', '.jpg', '.jpeg', '.png', '.pdf']
    
    # Get the script directory
    script_dir = Path(__file__).parent
    workspace_root = script_dir
    
    print("🚀 MUSE Pantheon Quick Launcher")
    print("=" * 40)
    print(f"Scan Root: {args.scan_root}")
    print(f"Work Directory: {args.workdir}")
    print(f"File Types: {', '.join(args.types)}")
    print(f"User: {args.user}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print()
    
    if not args.apply:
        print("This is a DRY RUN. Use --apply to actually run the pipeline.")
        print()
        print("Proposed command:")
        cmd = [
            sys.executable,
            str(script_dir / "warden" / "tools" / "run_full_pipeline.py"),
            "--roots", args.scan_root,
            "--workspace", str(workspace_root)
        ]
        if args.types:
            cmd.extend(["--types"] + args.types)
        
        print(" ".join(cmd))
        return 0
    
    # Run the actual pipeline
    cmd = [
        sys.executable,
        str(script_dir / "warden" / "tools" / "run_full_pipeline.py"),
        "--roots", args.scan_root,
        "--workspace", str(workspace_root)
    ]
    
    if args.types:
        cmd.extend(["--types"] + args.types)
    
    print("Running pipeline...")
    result = subprocess.run(cmd, cwd=workspace_root)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())