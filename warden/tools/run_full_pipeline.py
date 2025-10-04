#!/usr/bin/env python3
"""
MUSE Pantheon Universal Ingest Pipeline Orchestrator
Complete pipeline: Ingest → Cluster → Assign Projects → Report
"""
import sys
import pathlib
import argparse
import subprocess
import json
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete MUSE Pantheon ingest pipeline"""

    def __init__(self, workspace_root: str):
        self.workspace_root = pathlib.Path(workspace_root)
        self.tools_dir = self.workspace_root / "warden" / "tools"
        self.output_base = self.workspace_root / "_work" / "pipeline_output"
        self.output_base.mkdir(parents=True, exist_ok=True)

    def run_ingestion(self, roots: List[str], file_types: List[str] = None) -> bool:
        """Run the universal ingestion phase"""
        print("🚀 PHASE 1: Universal File Ingestion")
        print("-" * 40)

        cmd = [
            sys.executable,
            str(self.tools_dir / "universal_ingest.py"),
            "--roots"
        ] + roots + [
            "--output", str(self.output_base / "memory_blocks")
        ]

        if file_types:
            cmd.extend(["--types"] + file_types)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0:
                print("✅ Ingestion completed successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"❌ Ingestion failed: {result.stderr}")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                return False
        except Exception as e:
            print(f"❌ Ingestion error: {e}")
            return False

    def run_clustering(self) -> bool:
        """Run the clustering and embedding phase"""
        print("\n🧠 PHASE 2: Clustering & Embeddings")
        print("-" * 40)

        blocks_dir = self.output_base / "memory_blocks"
        if not blocks_dir.exists():
            print("❌ Memory blocks directory not found. Run ingestion first.")
            return False

        cmd = [
            sys.executable,
            str(self.tools_dir / "cluster_embeddings.py"),
            "--blocks-dir", str(blocks_dir),
            "--output-dir", str(self.output_base / "clustering_results"),
            "--assign-projects"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0:
                print("✅ Clustering completed successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"❌ Clustering failed: {result.stderr}")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                return False
        except Exception as e:
            print(f"❌ Clustering error: {e}")
            return False

    def run_project_assignment(self) -> bool:
        """Run the project assignment phase"""
        print("\n🎯 PHASE 3: Project Assignment")
        print("-" * 40)

        blocks_dir = self.output_base / "memory_blocks"
        clusters_file = self.output_base / "clustering_results" / "cluster_assignments.json"

        if not blocks_dir.exists():
            print("❌ Memory blocks directory not found. Run ingestion first.")
            return False

        cmd = [
            sys.executable,
            str(self.tools_dir / "assign_projects.py"),
            "--blocks-dir", str(blocks_dir),
            "--output-dir", str(self.output_base / "project_assignments"),
            "--update-blocks"
        ]

        if clusters_file.exists():
            cmd.extend(["--clusters-file", str(clusters_file)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0:
                print("✅ Project assignment completed successfully")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"❌ Project assignment failed: {result.stderr}")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                return False
        except Exception as e:
            print(f"❌ Project assignment error: {e}")
            return False

    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        print("\n📊 PHASE 4: Final Report Generation")
        print("-" * 40)

        report = {
            'pipeline_execution': {
                'timestamp': datetime.utcnow().isoformat() + "Z",
                'workspace': str(self.workspace_root),
                'output_base': str(self.output_base)
            },
            'phases': {},
            'summary': {}
        }

        # Check each phase output
        phases = [
            ('ingestion', 'memory_blocks'),
            ('clustering', 'clustering_results'),
            ('project_assignment', 'project_assignments')
        ]

        total_blocks = 0
        total_projects = 0
        
        for phase_name, dir_name in phases:
            phase_dir = self.output_base / dir_name
            if phase_dir.exists():
                files = list(phase_dir.glob('*.json'))
                report['phases'][phase_name] = {
                    'status': 'completed',
                    'output_count': len(files),
                    'output_dir': str(phase_dir)
                }
                if phase_name == 'ingestion':
                    total_blocks = len(files)
                elif phase_name == 'project_assignment':
                    # Count unique projects
                    try:
                        metadata_file = phase_dir / 'project_metadata.json'
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                total_projects = len(metadata)
                    except Exception:
                        pass
            else:
                report['phases'][phase_name] = {'status': 'not_run'}

        report['summary'] = {
            'total_memory_blocks': total_blocks,
            'total_projects': total_projects,
            'pipeline_status': 'completed' if total_blocks > 0 else 'failed'
        }

        self._create_human_readable_report(report)
        return report

    def _create_human_readable_report(self, report: Dict[str, Any]):
        """Create a human-readable summary file"""
        summary_path = self.output_base / "pipeline_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("MUSE Pantheon Pipeline Execution Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Executed: {report['pipeline_execution']['timestamp']}\n")
            f.write(f"Workspace: {report['pipeline_execution']['workspace']}\n\n")
            
            f.write("Phase Results:\n")
            for phase, data in report['phases'].items():
                f.write(f"  {phase.title()}: {data['status']}")
                if 'output_count' in data:
                    f.write(f" ({data['output_count']} outputs)")
                f.write("\n")
            
            f.write(f"\nTotal Memory Blocks: {report['summary']['total_memory_blocks']}\n")
            f.write(f"Total Projects: {report['summary']['total_projects']}\n")
            f.write(f"Overall Status: {report['summary']['pipeline_status']}\n")
            
            # Add file locations
            f.write("\nOutput Locations:\n")
            for phase, data in report['phases'].items():
                if 'output_dir' in data:
                    f.write(f"  {phase.title()}: {data['output_dir']}\n")

    def run_full_pipeline(self, roots: List[str], file_types: List[str] = None) -> bool:
        """Run the complete pipeline"""
        print("🎯 Starting MUSE Pantheon Universal Ingest Pipeline")
        print("=" * 60)
        
        success = True
        
        # Phase 1: Ingestion
        if not self.run_ingestion(roots, file_types):
            success = False
        
        # Phase 2: Clustering (only if ingestion succeeded)
        if success and not self.run_clustering():
            success = False
            
        # Phase 3: Project Assignment (only if clustering succeeded)
        if success and not self.run_project_assignment():
            success = False
            
        # Phase 4: Generate final report
        report = self.generate_final_report()
        
        print(f"\n🎉 Pipeline {'completed successfully' if success else 'failed'}")
        print(f"📁 Output directory: {self.output_base}")
        print(f"📄 Summary: {self.output_base / 'pipeline_summary.txt'}")
        
        # Print key statistics
        if success and report['summary']:
            print(f"📊 {report['summary']['total_memory_blocks']} MemoryBlocks created")
            print(f"🎯 {report['summary']['total_projects']} projects identified")
        
        return success

    def validate_environment(self) -> bool:
        """Validate that required tools and directories exist"""
        print("🔍 Validating environment...")
        
        # Check tool scripts exist
        required_tools = [
            "universal_ingest.py",
            "cluster_embeddings.py", 
            "assign_projects.py"
        ]
        
        for tool in required_tools:
            tool_path = self.tools_dir / tool
            if not tool_path.exists():
                print(f"❌ Missing required tool: {tool_path}")
                return False
        
        # Check workspace structure
        self.workspace_root.mkdir(exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        print("✅ Environment validation passed")
        return True


def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Universal Ingest Pipeline Orchestrator")
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (e.g., .txt .pdf .jpg)')
    parser.add_argument('--workspace', default='.',
                       help='Workspace root directory (default: current directory)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate environment, do not run pipeline')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(args.workspace)

    # Validate environment
    if not orchestrator.validate_environment():
        print("❌ Environment validation failed")
        sys.exit(1)

    if args.validate_only:
        print("✅ Environment validation completed successfully")
        sys.exit(0)

    # Validate root directories exist
    for root in args.roots:
        root_path = pathlib.Path(root)
        if not root_path.exists():
            print(f"❌ Root directory does not exist: {root}")
            sys.exit(1)

    # Run full pipeline
    success = orchestrator.run_full_pipeline(args.roots, args.types)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()