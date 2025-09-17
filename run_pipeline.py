#!/usr/bin/env python3
"""
MUSE Pantheon Quick Launcher
Launches the universal ingest pipeline with sensible defaults.
"""
import sys
import argparse
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent))

from warden.tools.run_full_pipeline import PipelineOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="MUSE Pantheon Quick Launcher - Universal Ingest Pipeline with Sensible Defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Examples:
  # Process current directory
  ./run_pipeline.py

  # Process MUSE_UNIFIED (from problem statement)  
  ./run_pipeline.py --target /Users/jakobaxelpaper/MUSE_UNIFIED

  # Process with file type filter
  ./run_pipeline.py --target ./documents --filter-docs

  # Process code files only
  ./run_pipeline.py --target ./src --filter-code
        """
    )
    
    parser.add_argument('--target', default='.',
                       help='Target directory to process (default: current directory)')
    parser.add_argument('--filter-docs', action='store_true',
                       help='Process only document files (.txt, .md, .pdf)')
    parser.add_argument('--filter-code', action='store_true',
                       help='Process only code files (.py, .js, .ts, .java, etc.)')
    parser.add_argument('--filter-media', action='store_true',
                       help='Process only media files (.jpg, .png, .mp4, etc.)')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependency status and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator('.')

        # Check dependencies if requested
        if args.check_deps:
            orchestrator.show_dependency_status()
            return

        # Determine file types filter
        file_types = None
        if args.filter_docs:
            file_types = ['.txt', '.md', '.pdf', '.doc', '.docx', '.rtf']
        elif args.filter_code:
            file_types = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rs', '.php']
        elif args.filter_media:
            file_types = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.mp4', '.mov', '.avi', '.mp3', '.wav']

        print(f"🚀 MUSE Pantheon Quick Launch")
        print(f"📁 Target: {args.target}")
        print(f"🔍 Filter: {file_types if file_types else 'All file types'}")
        print(f"📊 Output: ./_work/pipeline_output/")
        print()

        # Expand target path
        target_path = Path(args.target).expanduser().resolve()
        if not target_path.exists():
            print(f"❌ Target path does not exist: {target_path}")
            sys.exit(1)

        # Run pipeline
        success = orchestrator.run_full_pipeline([str(target_path)], file_types)

        if success:
            print(f"\n🎉 Quick launch completed successfully!")
            print(f"📁 Check results in: ./_work/pipeline_output/")
            print(f"📄 Human summary: ./_work/pipeline_output/pipeline_summary.txt")
            print(f"📊 JSON report: ./_work/pipeline_output/pipeline_report.json")
        else:
            print(f"\n❌ Quick launch failed - check logs for details")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Quick launch error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()