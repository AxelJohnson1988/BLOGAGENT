#!/usr/bin/env python3
"""
Quick launcher for MUSE Pantheon Universal Ingest Pipeline
"""
import sys
import pathlib
import subprocess

def main():
    # Default configuration for typical use cases
    workspace = pathlib.Path.cwd()
    roots = [str(workspace)]  # Process current directory by default
    
    print("🎯 MUSE Pantheon Universal Ingest Pipeline - Quick Start")
    print("=" * 60)
    print(f"Workspace: {workspace}")
    print(f"Processing: {roots}")
    print()
    
    # Run the full pipeline
    cmd = [
        sys.executable,
        str(workspace / "warden" / "tools" / "run_full_pipeline.py"),
        "--roots"
    ] + roots
    
    try:
        print("🚀 Starting pipeline...")
        result = subprocess.run(cmd, cwd=workspace)
        
        if result.returncode == 0:
            print("\n✅ Pipeline completed successfully!")
            print(f"📁 Check results in: {workspace / '_work' / 'pipeline_output'}")
        else:
            print(f"\n❌ Pipeline failed with exit code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()