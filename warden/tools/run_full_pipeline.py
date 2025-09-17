#!/usr/bin/env python3
"""
MUSE Pantheon Universal Ingest Pipeline Orchestrator
Complete pipeline: Ingest → Cluster → Assign Projects → Report
"""
import sys
import pathlib
import argparse
import subprocess
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
        
        # Ensure tools directory exists
        if not self.tools_dir.exists():
            logger.error(f"Tools directory not found: {self.tools_dir}")
            raise FileNotFoundError(f"Tools directory not found: {self.tools_dir}")

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
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            
            if result.returncode == 0:
                logger.info("✅ Ingestion completed successfully")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Ingestion failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                return False
        except Exception as e:
            logger.error(f"❌ Ingestion error: {e}")
            return False

    def run_clustering(self) -> bool:
        """Run the clustering and embedding phase"""
        logger.info("\n🧠 PHASE 2: Clustering & Embeddings")
        logger.info("-" * 40)

        blocks_dir = self.output_base / "memory_blocks"
        if not blocks_dir.exists():
            logger.error("❌ Memory blocks directory not found. Run ingestion first.")
            return False

        cmd = [
            sys.executable,
            str(self.tools_dir / "cluster_embeddings.py"),
            "--blocks-dir", str(blocks_dir),
            "--output-dir", str(self.output_base / "clustering_results"),
            "--visualize"
        ]

        try:
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            
            if result.returncode == 0:
                logger.info("✅ Clustering completed successfully")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Clustering failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                return False
        except Exception as e:
            logger.error(f"❌ Clustering error: {e}")
            return False

    def run_project_assignment(self) -> bool:
        """Run the project assignment phase"""
        logger.info("\n🎯 PHASE 3: Project Assignment")
        logger.info("-" * 40)

        blocks_dir = self.output_base / "memory_blocks"
        clusters_file = self.output_base / "clustering_results" / "clusters.json"

        if not blocks_dir.exists():
            logger.error("❌ Memory blocks directory not found. Run ingestion first.")
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
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            
            if result.returncode == 0:
                logger.info("✅ Project assignment completed successfully")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Project assignment failed with return code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error: {result.stderr}")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
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
            f.write(f"Workspace: {report['pipeline_execution']['workspace']}\n")
            f.write(f"Output Directory: {report['pipeline_execution']['output_base']}\n\n")
            
            f.write("Phase Results:\n")
            f.write("-" * 20 + "\n")
            for phase, data in report['phases'].items():
                f.write(f"  {phase.title()}: {data['status']}")
                if 'output_count' in data:
                    f.write(f" ({data['output_count']} outputs)")
                f.write("\n")
            
            f.write(f"\nSummary:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Memory Blocks: {report['summary']['total_memory_blocks']}\n")
            f.write(f"Overall Status: {report['summary']['pipeline_status'].upper()}\n")
            
            if report['summary']['pipeline_status'] == 'completed':
                f.write(f"\n✅ Pipeline executed successfully!\n")
                f.write(f"📁 All outputs available in: {report['pipeline_execution']['output_base']}\n")
            else:
                f.write(f"\n❌ Pipeline execution failed.\n")
                f.write(f"Check logs for details.\n")

    def run_full_pipeline(self, roots: List[str], file_types: List[str] = None) -> bool:
        """Run the complete pipeline"""
        logger.info("🎯 Starting MUSE Pantheon Universal Ingest Pipeline")
        logger.info("=" * 60)
        
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
            
        # Phase 4: Final Report
        report = self.generate_final_report()
        
        logger.info(f"\n🎉 Pipeline {'completed successfully' if success else 'failed'}")
        logger.info(f"📁 Output directory: {self.output_base}")
        logger.info(f"📄 Summary: {self.output_base / 'pipeline_summary.txt'}")
        
        if success:
            logger.info("\n📋 Next Steps:")
            logger.info("1. Review MemoryBlocks in the memory_blocks/ directory")
            logger.info("2. Explore clustering results and visualizations")
            logger.info("3. Check project assignments and hierarchy")
            logger.info("4. Integrate MemoryBlocks into your vector store for semantic search")
        
        return success

def main():
    parser = argparse.ArgumentParser(
        description="MUSE Pantheon Universal Ingest Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific directories with all file types
  python run_full_pipeline.py --roots /path/to/docs /path/to/code
  
  # Process only specific file types
  python run_full_pipeline.py --roots /path/to/files --types .py .md .json
  
  # Set custom workspace
  python run_full_pipeline.py --roots /path/to/files --workspace /custom/workspace
        """
    )
    
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (e.g., .txt .pdf .jpg)')
    parser.add_argument('--workspace', default='.',
                       help='Workspace root directory (default: current directory)')

    args = parser.parse_args()

    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(args.workspace)

        # Run full pipeline
        success = orchestrator.run_full_pipeline(args.roots, args.types)

        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()