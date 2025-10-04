#!/usr/bin/env python3
"""
MUSE Pantheon Pipeline Quick Launcher
Simplified interface for common pipeline operations
"""
import sys
import pathlib
import argparse
from datetime import datetime

def run_default_pipeline():
    """Run pipeline with sensible defaults for typical use cases"""
    
    # Default configuration
    workspace = pathlib.Path.cwd()
    common_roots = [
        ".",
        "warden",
        "common",
        "_work"
    ]
    
    # Filter to existing directories
    existing_roots = [str(root) for root in common_roots if pathlib.Path(root).exists()]
    
    if not existing_roots:
        print("❌ No valid root directories found in current workspace")
        print("Available directories:", list(pathlib.Path.cwd().iterdir()))
        return False
    
    # Common file types for development projects
    file_types = ['.py', '.md', '.json', '.txt', '.yaml', '.yml', '.html', '.js', '.css']
    
    print("🚀 MUSE Pantheon Quick Launch")
    print("=" * 40)
    print(f"📁 Workspace: {workspace}")
    print(f"🎯 Scanning: {', '.join(existing_roots)}")
    print(f"📄 File types: {', '.join(file_types)}")
    print("")
    
    # Import and run the main orchestrator
    try:
        from warden.tools.run_full_pipeline import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator(str(workspace))
        
        # Validate environment first
        if not orchestrator.validate_environment():
            print("❌ Environment validation failed")
            return False
        
        # Run the pipeline
        success = orchestrator.run_full_pipeline(existing_roots, file_types)
        
        if success:
            print("\n🎉 Quick launch completed successfully!")
            print(f"📁 Check results in: {workspace / '_work' / 'pipeline_output'}")
        else:
            print("\n❌ Quick launch failed")
        
        return success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory with the warden/ tools installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_custom_pipeline(roots, types=None, workspace="."):
    """Run pipeline with custom configuration"""
    
    print("🚀 MUSE Pantheon Custom Launch")
    print("=" * 40)
    print(f"📁 Workspace: {workspace}")
    print(f"🎯 Scanning: {', '.join(roots)}")
    if types:
        print(f"📄 File types: {', '.join(types)}")
    print("")
    
    try:
        from warden.tools.run_full_pipeline import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator(workspace)
        
        # Validate environment
        if not orchestrator.validate_environment():
            print("❌ Environment validation failed")
            return False
        
        # Run the pipeline
        success = orchestrator.run_full_pipeline(roots, types)
        
        if success:
            print("\n🎉 Custom launch completed successfully!")
        else:
            print("\n❌ Custom launch failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def show_status(workspace="."):
    """Show pipeline status and recent results"""
    
    workspace_path = pathlib.Path(workspace)
    output_dir = workspace_path / "_work" / "pipeline_output"
    
    print("📊 MUSE Pantheon Pipeline Status")
    print("=" * 40)
    print(f"📁 Workspace: {workspace_path.absolute()}")
    print(f"📂 Output directory: {output_dir}")
    print("")
    
    if not output_dir.exists():
        print("❌ No pipeline output found. Run the pipeline first.")
        return
    
    # Check each phase
    phases = [
        ("memory_blocks", "🧠 Memory Blocks"),
        ("clustering_results", "🎯 Clustering Results"), 
        ("project_assignments", "📋 Project Assignments")
    ]
    
    for dir_name, label in phases:
        phase_dir = output_dir / dir_name
        if phase_dir.exists():
            file_count = len(list(phase_dir.glob("*.json")))
            print(f"✅ {label}: {file_count} files")
        else:
            print(f"❌ {label}: Not found")
    
    # Check summary file
    summary_file = output_dir / "pipeline_summary.txt"
    if summary_file.exists():
        print(f"\n📄 Latest summary:")
        print("-" * 20)
        with open(summary_file, 'r') as f:
            print(f.read())
    else:
        print("\n❌ No summary file found")

def clean_output(workspace="."):
    """Clean pipeline output directory"""
    
    workspace_path = pathlib.Path(workspace)
    output_dir = workspace_path / "_work" / "pipeline_output"
    
    if not output_dir.exists():
        print("✅ Output directory doesn't exist - nothing to clean")
        return
    
    print(f"🧹 Cleaning pipeline output: {output_dir}")
    
    try:
        import shutil
        shutil.rmtree(output_dir)
        print("✅ Output directory cleaned successfully")
    except Exception as e:
        print(f"❌ Failed to clean output directory: {e}")

def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Pipeline Quick Launcher")
    parser.add_argument('action', choices=['run', 'status', 'clean', 'custom'],
                       help='Action to perform')
    parser.add_argument('--roots', nargs='+',
                       help='Root directories to scan (for custom action)')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (for custom action)')
    parser.add_argument('--workspace', default='.',
                       help='Workspace directory (default: current)')
    
    args = parser.parse_args()
    
    if args.action == 'run':
        success = run_default_pipeline()
        sys.exit(0 if success else 1)
    
    elif args.action == 'custom':
        if not args.roots:
            print("❌ Custom action requires --roots argument")
            sys.exit(1)
        success = run_custom_pipeline(args.roots, args.types, args.workspace)
        sys.exit(0 if success else 1)
    
    elif args.action == 'status':
        show_status(args.workspace)
    
    elif args.action == 'clean':
        clean_output(args.workspace)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()