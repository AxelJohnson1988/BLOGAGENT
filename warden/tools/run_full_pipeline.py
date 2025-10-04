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
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0:
                logger.info("✅ Ingestion completed successfully")
                if result.stdout:
                    logger.info(f"Ingestion output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Ingestion failed: {result.stderr}")
                if result.stdout:
                    logger.error(f"Ingestion stdout: {result.stdout}")
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
            "--assign-projects"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0:
                logger.info("✅ Clustering completed successfully")
                if result.stdout:
                    logger.info(f"Clustering output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Clustering failed: {result.stderr}")
                if result.stdout:
                    logger.error(f"Clustering stdout: {result.stdout}")
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
            "--output-dir", str(self.output_base / "project_assignments")
        ]

        if clusters_file.exists():
            cmd.extend(["--clusters-file", str(clusters_file)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_root)
            if result.returncode == 0:
                logger.info("✅ Project assignment completed successfully")
                if result.stdout:
                    logger.info(f"Project assignment output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Project assignment failed: {result.stderr}")
                if result.stdout:
                    logger.error(f"Project assignment stdout: {result.stdout}")
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

        # Load additional statistics if available
        try:
            # Load ingestion stats
            ingest_stats_file = self.output_base / "memory_blocks" / "ingest_stats.json"
            if ingest_stats_file.exists():
                with open(ingest_stats_file, 'r') as f:
                    ingest_stats = json.load(f)
                report['phases']['ingestion'].update(ingest_stats)
            
            # Load clustering metadata
            clusters_file = self.output_base / "clustering_results" / "clusters.json"
            if clusters_file.exists():
                with open(clusters_file, 'r') as f:
                    cluster_data = json.load(f)
                report['phases']['clustering']['metadata'] = cluster_data.get('metadata', {})
            
            # Load project hierarchy
            hierarchy_file = self.output_base / "project_assignments" / "project_hierarchy.json"
            if hierarchy_file.exists():
                with open(hierarchy_file, 'r') as f:
                    hierarchy_data = json.load(f)
                report['phases']['project_assignment']['hierarchy'] = hierarchy_data.get('summary', {})
        
        except Exception as e:
            logger.warning(f"Could not load detailed statistics: {e}")

        report['summary'] = {
            'total_memory_blocks': total_blocks,
            'pipeline_status': 'completed' if total_blocks > 0 else 'failed'
        }

        self._create_human_readable_report(report)
        self._save_machine_readable_report(report)

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
                status = data['status']
                f.write(f"  {phase.title()}: {status}")
                if 'output_count' in data:
                    f.write(f" ({data['output_count']} outputs)")
                if 'processed' in data:
                    f.write(f" (processed: {data['processed']}, failed: {data.get('failed', 0)})")
                f.write("\n")
                
                # Add detailed stats if available
                if 'file_types' in data:
                    f.write(f"    File types: {data['file_types']}\n")
                if 'metadata' in data and 'num_clusters' in data['metadata']:
                    f.write(f"    Clusters: {data['metadata']['num_clusters']}\n")
                if 'hierarchy' in data and 'total_projects' in data['hierarchy']:
                    f.write(f"    Projects: {data['hierarchy']['total_projects']}\n")
            
            f.write(f"\nOverall Results:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Memory Blocks: {report['summary']['total_memory_blocks']}\n")
            f.write(f"Pipeline Status: {report['summary']['pipeline_status']}\n")
            
            f.write(f"\nOutput Structure:\n")
            f.write("-" * 20 + "\n")
            f.write(f"📁 {self.output_base}/\n")
            f.write(f"├── memory_blocks/           # Raw MemoryBlock JSON files\n")
            f.write(f"├── clustering_results/      # Interactive visualizations & cluster data\n")
            f.write(f"├── project_assignments/     # Project hierarchies & updated blocks\n")
            f.write(f"├── pipeline_summary.txt     # This human-readable report\n")
            f.write(f"└── pipeline_report.json     # Machine-readable execution data\n")

    def _save_machine_readable_report(self, report: Dict[str, Any]):
        """Save machine-readable report"""
        report_path = self.output_base / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    def run_full_pipeline(self, roots: List[str], file_types: List[str] = None) -> bool:
        """Run the complete pipeline"""
        logger.info("🎯 Starting MUSE Pantheon Universal Ingest Pipeline")
        logger.info("=" * 60)
        logger.info(f"Workspace: {self.workspace_root}")
        logger.info(f"Target roots: {roots}")
        logger.info(f"File types filter: {file_types or 'All types'}")
        logger.info(f"Output directory: {self.output_base}")
        
        success = True
        phase_results = {}
        
        # Phase 1: Ingestion
        if not self.run_ingestion(roots, file_types):
            success = False
            phase_results['ingestion'] = 'failed'
        else:
            phase_results['ingestion'] = 'completed'
        
        # Phase 2: Clustering (only if ingestion succeeded)
        if success and not self.run_clustering():
            success = False
            phase_results['clustering'] = 'failed'
        elif success:
            phase_results['clustering'] = 'completed'
        else:
            phase_results['clustering'] = 'skipped'
            
        # Phase 3: Project Assignment (only if previous phases succeeded)
        if success and not self.run_project_assignment():
            success = False
            phase_results['project_assignment'] = 'failed'
        elif success:
            phase_results['project_assignment'] = 'completed'
        else:
            phase_results['project_assignment'] = 'skipped'
            
        # Phase 4: Generate report (always run)
        report = self.generate_final_report()
        
        # Final summary
        logger.info(f"\n{'='*60}")
        if success:
            logger.info("🎉 Pipeline completed successfully!")
            logger.info("✅ All phases executed without errors")
        else:
            logger.info("⚠️  Pipeline completed with errors")
            logger.info("❌ Some phases failed - check logs for details")
        
        logger.info(f"📁 Output directory: {self.output_base}")
        logger.info(f"📄 Summary report: {self.output_base / 'pipeline_summary.txt'}")
        logger.info(f"📊 Detailed report: {self.output_base / 'pipeline_report.json'}")
        
        # Show quick stats
        summary = report['summary']
        if summary['total_memory_blocks'] > 0:
            logger.info(f"📈 Generated {summary['total_memory_blocks']} MemoryBlocks")
            
            # Show cluster info if available
            if 'clustering' in report['phases'] and 'metadata' in report['phases']['clustering']:
                cluster_meta = report['phases']['clustering']['metadata']
                if 'num_clusters' in cluster_meta:
                    logger.info(f"🧠 Created {cluster_meta['num_clusters']} semantic clusters")
            
            # Show project info if available
            if 'project_assignment' in report['phases'] and 'hierarchy' in report['phases']['project_assignment']:
                proj_meta = report['phases']['project_assignment']['hierarchy']
                if 'total_projects' in proj_meta:
                    logger.info(f"🎯 Assigned {proj_meta['total_projects']} project categories")
        
        logger.info(f"{'='*60}")
        
        return success

    def check_dependencies(self) -> Dict[str, bool]:
        """Check for optional dependencies and tools"""
        deps = {}
        
        # Check Python packages
        try:
            import sklearn
            deps['sklearn'] = True
        except ImportError:
            deps['sklearn'] = False
        
        try:
            import plotly
            deps['plotly'] = True
        except ImportError:
            deps['plotly'] = False
        
        # Check external tools
        try:
            subprocess.run(['tesseract', '--version'], capture_output=True, check=True)
            deps['tesseract'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            deps['tesseract'] = False
        
        try:
            subprocess.run(['pdftotext', '-v'], capture_output=True)
            deps['pdftotext'] = True
        except FileNotFoundError:
            deps['pdftotext'] = False
        
        return deps

    def show_dependency_status(self):
        """Show status of optional dependencies"""
        deps = self.check_dependencies()
        
        logger.info("📋 Dependency Status:")
        logger.info("-" * 20)
        
        for dep, available in deps.items():
            status = "✅ Available" if available else "❌ Missing"
            logger.info(f"  {dep}: {status}")
        
        # Show impact of missing dependencies
        if not deps.get('sklearn', True):
            logger.warning("  ⚠️  Without sklearn: Using basic clustering instead of advanced K-means")
        if not deps.get('plotly', True):
            logger.warning("  ⚠️  Without plotly: No interactive visualizations will be generated")
        if not deps.get('tesseract', True):
            logger.warning("  ⚠️  Without tesseract: OCR text extraction from images disabled")
        if not deps.get('pdftotext', True):
            logger.warning("  ⚠️  Without pdftotext: PDF text extraction may be limited")


def main():
    parser = argparse.ArgumentParser(
        description="MUSE Pantheon Universal Ingest Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in current directory
  python run_full_pipeline.py --roots .

  # Process specific directories with file type filter
  python run_full_pipeline.py --roots /path/to/docs /path/to/images --types .txt .pdf .jpg

  # Process MUSE_UNIFIED directory (from problem statement)
  python run_full_pipeline.py --roots /Users/jakobaxelpaper/MUSE_UNIFIED

  # Process with custom workspace
  python run_full_pipeline.py --roots ./data --workspace /custom/workspace
        """
    )
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (e.g., .txt .pdf .jpg)')
    parser.add_argument('--workspace', default='.',
                       help='Workspace root directory (default: current directory)')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependency status and exit')

    args = parser.parse_args()

    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(args.workspace)

        # Check dependencies if requested
        if args.check_deps:
            orchestrator.show_dependency_status()
            return

        # Show dependency status before running
        orchestrator.show_dependency_status()

        # Run full pipeline
        success = orchestrator.run_full_pipeline(args.roots, args.types)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()