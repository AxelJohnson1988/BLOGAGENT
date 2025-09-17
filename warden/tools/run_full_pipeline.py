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
            "--output", str(self.output_base)
        ]

        if file_types:
            cmd.extend(["--types"] + file_types)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            print(f"Ingestion stdout: {result.stdout}")
            if result.stderr:
                print(f"Ingestion stderr: {result.stderr}")
            
            if result.returncode == 0:
                print("✅ Ingestion completed successfully")
                return True
            else:
                print(f"❌ Ingestion failed with return code {result.returncode}")
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
            "--output-dir", str(self.output_base),
            "--assign-projects"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            print(f"Clustering stdout: {result.stdout}")
            if result.stderr:
                print(f"Clustering stderr: {result.stderr}")
            
            if result.returncode == 0:
                print("✅ Clustering completed successfully")
                return True
            else:
                print(f"❌ Clustering failed with return code {result.returncode}")
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
            "--output-dir", str(self.output_base)
        ]

        if clusters_file.exists():
            cmd.extend(["--clusters-file", str(clusters_file)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            print(f"Project assignment stdout: {result.stdout}")
            if result.stderr:
                print(f"Project assignment stderr: {result.stderr}")
            
            if result.returncode == 0:
                print("✅ Project assignment completed successfully")
                return True
            else:
                print(f"❌ Project assignment failed with return code {result.returncode}")
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
                'timestamp': datetime.now().isoformat() + "Z",
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
            else:
                report['phases'][phase_name] = {'status': 'not_run'}

        report['summary'] = {
            'total_memory_blocks': total_blocks,
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
            f.write(f"Overall Status: {report['summary']['pipeline_status']}\n")

        # Also save as JSON
        json_path = self.output_base / "pipeline_summary.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

    def run_full_pipeline(self, roots: List[str], file_types: List[str] = None) -> bool:
        """Run the complete pipeline"""
        print("🎯 Starting MUSE Pantheon Universal Ingest Pipeline")
        print("=" * 60)
        
        success = True
        
        if not self.run_ingestion(roots, file_types):
            success = False
        
        if success and not self.run_clustering():
            success = False
            
        if success and not self.run_project_assignment():
            success = False
            
        report = self.generate_final_report()
        
        print(f"\n🎉 Pipeline {'completed successfully' if success else 'failed'}")
        print(f"📁 Output directory: {self.output_base}")
        print(f"📄 Summary: {self.output_base / 'pipeline_summary.txt'}")
        
        return success

def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Universal Ingest Pipeline Orchestrator")
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (e.g., .txt .pdf .jpg)')
    parser.add_argument('--workspace', default='.',
                       help='Workspace root directory')

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(args.workspace)

    # Run full pipeline
    success = orchestrator.run_full_pipeline(args.roots, args.types)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()