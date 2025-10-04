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
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

# Setup logging
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
        logger.info("🚀 PHASE 1: Universal File Ingestion")
        logger.info("-" * 40)

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
                logger.info("✅ Ingestion completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Ingestion failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Ingestion error: {e}")
            return False

    def run_clustering(self) -> bool:
        """Run the clustering and embedding phase"""
        logger.info("\n🧠 PHASE 2: Clustering & Embeddings")
        logger.info("-" * 40)

        blocks_dir = self.output_base / "memory_blocks"
        if not blocks_dir.exists() or not any(blocks_dir.glob("*.json")):
            logger.error("❌ Memory blocks directory not found or empty. Run ingestion first.")
            return False

        cmd = [
            sys.executable,
            str(self.tools_dir / "cluster_embeddings.py"),
            "--blocks-dir", str(blocks_dir),
            "--output-dir", str(self.output_base / "clustering_results")
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0:
                logger.info("✅ Clustering completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Clustering failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Clustering error: {e}")
            return False

    def run_project_assignment(self) -> bool:
        """Run the project assignment phase"""
        logger.info("\n🎯 PHASE 3: Project Assignment")
        logger.info("-" * 40)

        blocks_dir = self.output_base / "memory_blocks"
        clusters_file = self.output_base / "clustering_results" / "clustering_results.json"

        if not blocks_dir.exists():
            logger.error("❌ Memory blocks directory not found. Run ingestion first.")
            return False

        cmd = [
            sys.executable,
            str(self.tools_dir / "assign_projects.py"),
            "--blocks-dir", str(blocks_dir),
            "--output-dir", str(self.output_base / "project_assignments")
        ]

        if clusters_file.exists():
            cmd.extend(["--clusters-file", str(clusters_file)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0:
                logger.info("✅ Project assignment completed successfully")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Project assignment failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Project assignment error: {e}")
            return False

    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        logger.info("\n📊 PHASE 4: Final Report Generation")
        logger.info("-" * 40)

        report = {
            'pipeline_execution': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
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
                    total_blocks = len([f for f in files if f.name.startswith('memory_block_')])
            else:
                report['phases'][phase_name] = {'status': 'not_run'}

        # Get clustering statistics
        clustering_results_file = self.output_base / "clustering_results" / "clustering_results.json"
        if clustering_results_file.exists():
            try:
                with open(clustering_results_file, 'r') as f:
                    clustering_data = json.load(f)
                    report['clustering_stats'] = {
                        'n_clusters': clustering_data.get('clustering_results', {}).get('n_clusters', 0),
                        'visualization_available': (self.output_base / "clustering_results" / "cluster_visualization.html").exists()
                    }
            except Exception as e:
                logger.warning(f"Failed to load clustering stats: {e}")

        # Get project assignment statistics
        project_report_file = self.output_base / "project_assignments" / "project_assignment_report.json"
        if project_report_file.exists():
            try:
                with open(project_report_file, 'r') as f:
                    project_data = json.load(f)
                    report['project_stats'] = project_data.get('project_assignment_summary', {})
            except Exception as e:
                logger.warning(f"Failed to load project stats: {e}")

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
            f.write(f"Workspace: {report['pipeline_execution']['workspace']}\n")
            f.write(f"Output Directory: {report['pipeline_execution']['output_base']}\n\n")
            
            f.write("Phase Results:\n")
            f.write("-" * 14 + "\n")
            for phase, data in report['phases'].items():
                f.write(f"  {phase.title()}: {data['status']}")
                if 'output_count' in data:
                    f.write(f" ({data['output_count']} outputs)")
                f.write("\n")
            
            f.write(f"\nTotal Memory Blocks: {report['summary']['total_memory_blocks']}\n")
            f.write(f"Overall Status: {report['summary']['pipeline_status']}\n")
            
            # Add clustering info if available
            if 'clustering_stats' in report:
                f.write(f"\nClustering Results:\n")
                f.write(f"  Number of Clusters: {report['clustering_stats']['n_clusters']}\n")
                f.write(f"  Visualization Available: {report['clustering_stats']['visualization_available']}\n")
            
            # Add project info if available
            if 'project_stats' in report:
                f.write(f"\nProject Assignment:\n")
                f.write(f"  Total Projects: {report['project_stats'].get('total_projects', 'N/A')}\n")
            
            f.write(f"\nOutput Files:\n")
            f.write(f"  Memory Blocks: {self.output_base / 'memory_blocks'}\n")
            f.write(f"  Clustering Results: {self.output_base / 'clustering_results'}\n")
            f.write(f"  Project Assignments: {self.output_base / 'project_assignments'}\n")
            f.write(f"  Cluster Visualization: {self.output_base / 'clustering_results' / 'cluster_visualization.html'}\n")

    def run_full_pipeline(self, roots: List[str], file_types: List[str] = None) -> bool:
        """Run the complete pipeline"""
        logger.info("🎯 Starting MUSE Pantheon Universal Ingest Pipeline")
        logger.info("=" * 60)
        
        success = True
        
        # Phase 1: Ingestion
        if not self.run_ingestion(roots, file_types):
            success = False
            return success  # Early exit on ingestion failure
        
        # Phase 2: Clustering
        if success and not self.run_clustering():
            success = False
            # Continue to other phases even if clustering fails
            
        # Phase 3: Project Assignment
        if not self.run_project_assignment():
            success = False
            # Continue to reporting even if project assignment fails
            
        # Phase 4: Final Report
        report = self.generate_final_report()
        
        logger.info(f"\n🎉 Pipeline {'completed successfully' if success else 'completed with errors'}")
        logger.info(f"📁 Output directory: {self.output_base}")
        logger.info(f"📄 Summary: {self.output_base / 'pipeline_summary.txt'}")
        
        if report['summary']['total_memory_blocks'] > 0:
            logger.info(f"✅ Created {report['summary']['total_memory_blocks']} memory blocks")
            
            # Show clustering info if available
            if 'clustering_stats' in report:
                logger.info(f"🧠 Organized into {report['clustering_stats']['n_clusters']} semantic clusters")
                
            # Show project info if available
            if 'project_stats' in report:
                logger.info(f"🎯 Assigned to {report['project_stats'].get('total_projects', 'N/A')} projects")
                
            logger.info(f"🌐 Interactive visualization: {self.output_base / 'clustering_results' / 'cluster_visualization.html'}")
        
        return success

    def validate_environment(self) -> bool:
        """Validate that the environment is set up correctly"""
        logger.info("🔍 Validating environment...")
        
        # Check if tools directory exists
        if not self.tools_dir.exists():
            logger.error(f"❌ Tools directory not found: {self.tools_dir}")
            return False
            
        # Check if individual tools exist
        required_tools = [
            "universal_ingest.py",
            "cluster_embeddings.py", 
            "assign_projects.py"
        ]
        
        for tool in required_tools:
            tool_path = self.tools_dir / tool
            if not tool_path.exists():
                logger.error(f"❌ Required tool not found: {tool_path}")
                return False
                
        # Check if common module exists
        common_dir = self.workspace_root / "common"
        if not common_dir.exists() or not (common_dir / "memory_block.py").exists():
            logger.error(f"❌ Common module not found: {common_dir / 'memory_block.py'}")
            return False
            
        logger.info("✅ Environment validation passed")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MUSE Pantheon Universal Ingest Pipeline Orchestrator")
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (e.g., .txt .pdf .jpg)')
    parser.add_argument('--workspace', default='.',
                       help='Workspace root directory')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate environment, do not run pipeline')

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(args.workspace)

    # Validate environment
    if not orchestrator.validate_environment():
        logger.error("❌ Environment validation failed")
        return 1

    # If validation-only mode, exit here
    if args.validate_only:
        logger.info("✅ Environment validation successful")
        return 0

    # Run full pipeline
    success = orchestrator.run_full_pipeline(args.roots, args.types)

    # Exit with appropriate code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())