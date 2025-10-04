#!/usr/bin/env python3
"""
Project Assignment System for MUSE Pantheon
Auto-assigns project IDs based on content analysis and archetype mapping
"""
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re

# Add common directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "common"))
from memory_block import MemoryBlock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectClassifier:
    """Classifies MemoryBlocks into projects based on content and patterns."""
    
    def __init__(self):
        """Initialize the project classifier."""
        self.project_patterns = {
            'muse.pantheon': {
                'keywords': ['memory', 'block', 'muse', 'pantheon', 'archetype', 'guardian', 'oracle', 'sage'],
                'file_patterns': ['memory_block', 'warden', 'pantheon'],
                'archetypes': ['Guardian', 'Oracle', 'Sage', 'Shaman']
            },
            'blog.agent': {
                'keywords': ['blog', 'post', 'article', 'content', 'writing', 'publish'],
                'file_patterns': ['blog', 'post', 'article', 'content'],
                'archetypes': ['Scribe', 'Storyteller', 'Weaver']
            },
            'warden.system': {
                'keywords': ['warden', 'security', 'monitor', 'guardian', 'protection', 'ethics'],
                'file_patterns': ['warden', 'security', 'monitor'],
                'archetypes': ['Guardian', 'Warden', 'Protector']
            },
            'ai.assistant': {
                'keywords': ['assistant', 'ai', 'artificial', 'intelligence', 'agent', 'copilot'],
                'file_patterns': ['assistant', 'ai', 'agent', 'copilot'],
                'archetypes': ['Oracle', 'Sage', 'Navigator']
            },
            'legal.documentation': {
                'keywords': ['legal', 'contract', 'agreement', 'terms', 'policy', 'compliance'],
                'file_patterns': ['legal', 'contract', 'terms', 'policy'],
                'archetypes': ['Guardian', 'Scribe', 'Sage']
            },
            'data.analytics': {
                'keywords': ['data', 'analytics', 'analysis', 'statistics', 'metrics', 'report'],
                'file_patterns': ['data', 'analytics', 'report', 'stats'],
                'archetypes': ['Oracle', 'Alchemist', 'Sage']
            },
            'creative.content': {
                'keywords': ['creative', 'design', 'art', 'visual', 'image', 'media'],
                'file_patterns': ['creative', 'design', 'art', 'media'],
                'archetypes': ['Visionary', 'Architect', 'Storyteller']
            },
            'development.tools': {
                'keywords': ['development', 'tools', 'utility', 'script', 'automation'],
                'file_patterns': ['tools', 'utils', 'scripts', 'dev'],
                'archetypes': ['Alchemist', 'Architect', 'Navigator']
            }
        }
        
        self.archetype_project_mapping = {
            'Guardian': ['muse.pantheon', 'warden.system', 'legal.documentation'],
            'Oracle': ['ai.assistant', 'data.analytics', 'muse.pantheon'],
            'Sage': ['ai.assistant', 'legal.documentation', 'data.analytics'],
            'Scribe': ['blog.agent', 'legal.documentation', 'creative.content'],
            'Storyteller': ['blog.agent', 'creative.content', 'muse.pantheon'],
            'Visionary': ['creative.content', 'muse.pantheon', 'development.tools'],
            'Alchemist': ['development.tools', 'data.analytics', 'muse.pantheon'],
            'Architect': ['development.tools', 'creative.content', 'muse.pantheon'],
            'Weaver': ['blog.agent', 'creative.content', 'development.tools'],
            'Navigator': ['ai.assistant', 'development.tools', 'muse.pantheon'],
            'Shaman': ['muse.pantheon', 'warden.system', 'creative.content'],
            'Discoverer': ['muse.pantheon', 'data.analytics', 'development.tools']
        }
    
    def classify_memory_block(self, memory_block: MemoryBlock) -> str:
        """Classify a memory block into a project."""
        scores = defaultdict(float)
        
        # Score based on content keywords
        content_text = f"{memory_block.summary} {memory_block.content} {' '.join(memory_block.topics)}".lower()
        
        for project, patterns in self.project_patterns.items():
            # Keyword matching
            for keyword in patterns['keywords']:
                if keyword in content_text:
                    scores[project] += 2.0
            
            # File pattern matching
            if memory_block.source_path:
                source_path = memory_block.source_path.lower()
                for pattern in patterns['file_patterns']:
                    if pattern in source_path:
                        scores[project] += 3.0
            
            # Archetype alignment
            if memory_block.archetype in patterns['archetypes']:
                scores[project] += 1.5
        
        # Additional scoring based on archetype-project mapping
        if memory_block.archetype in self.archetype_project_mapping:
            for project in self.archetype_project_mapping[memory_block.archetype]:
                scores[project] += 1.0
        
        # Return highest scoring project or default
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general.development'
    
    def get_project_hierarchy(self, project_id: str) -> Dict[str, str]:
        """Get project hierarchy information."""
        hierarchy_map = {
            'muse.pantheon': {
                'domain': 'muse',
                'category': 'pantheon',
                'subcategory': 'core',
                'description': 'Core MUSE Pantheon memory system'
            },
            'blog.agent': {
                'domain': 'blog',
                'category': 'agent',
                'subcategory': 'content',
                'description': 'Blog content generation and management'
            },
            'warden.system': {
                'domain': 'warden',
                'category': 'system',
                'subcategory': 'security',
                'description': 'Security and monitoring system'
            },
            'ai.assistant': {
                'domain': 'ai',
                'category': 'assistant',
                'subcategory': 'interaction',
                'description': 'AI assistant and agent systems'
            },
            'legal.documentation': {
                'domain': 'legal',
                'category': 'documentation',
                'subcategory': 'compliance',
                'description': 'Legal documents and compliance'
            },
            'data.analytics': {
                'domain': 'data',
                'category': 'analytics',
                'subcategory': 'insights',
                'description': 'Data analysis and insights'
            },
            'creative.content': {
                'domain': 'creative',
                'category': 'content',
                'subcategory': 'media',
                'description': 'Creative and media content'
            },
            'development.tools': {
                'domain': 'development',
                'category': 'tools',
                'subcategory': 'utilities',
                'description': 'Development tools and utilities'
            },
            'general.development': {
                'domain': 'general',
                'category': 'development',
                'subcategory': 'misc',
                'description': 'General development and miscellaneous'
            }
        }
        
        return hierarchy_map.get(project_id, {
            'domain': 'unknown',
            'category': 'unknown',
            'subcategory': 'unknown',
            'description': 'Unknown project category'
        })


class ProjectAssignmentEngine:
    """Main engine for project assignment."""
    
    def __init__(self):
        """Initialize the assignment engine."""
        self.classifier = ProjectClassifier()
        self.assignment_stats = defaultdict(int)
    
    def assign_projects(self, memory_blocks: List[MemoryBlock]) -> List[MemoryBlock]:
        """Assign projects to all memory blocks."""
        logger.info(f"Assigning projects to {len(memory_blocks)} memory blocks")
        
        updated_blocks = []
        
        for block in memory_blocks:
            # Classify the block
            new_project = self.classifier.classify_memory_block(block)
            
            # Update project if different
            if block.project != new_project:
                logger.debug(f"Updated project for {block.id_hash}: {block.project} -> {new_project}")
                block.project = new_project
                block.metadata['project_assigned_by'] = 'auto_classifier'
                block.metadata['original_project'] = block.project if hasattr(block, 'project') else 'unknown'
            
            # Add project hierarchy information
            hierarchy = self.classifier.get_project_hierarchy(new_project)
            block.metadata['project_hierarchy'] = hierarchy
            
            self.assignment_stats[new_project] += 1
            updated_blocks.append(block)
        
        self._log_assignment_stats()
        return updated_blocks
    
    def assign_projects_with_clusters(self, memory_blocks: List[MemoryBlock], 
                                    cluster_labels: List[int]) -> List[MemoryBlock]:
        """Assign projects considering cluster information."""
        logger.info("Assigning projects with cluster context")
        
        # First, do standard assignment
        updated_blocks = self.assign_projects(memory_blocks)
        
        # Then, refine based on cluster patterns
        cluster_project_patterns = self._analyze_cluster_patterns(updated_blocks, cluster_labels)
        
        # Apply cluster-based refinements
        for i, (block, cluster_label) in enumerate(zip(updated_blocks, cluster_labels)):
            if cluster_label in cluster_project_patterns:
                cluster_info = cluster_project_patterns[cluster_label]
                
                # If cluster has a dominant project, consider switching
                dominant_project = cluster_info['dominant_project']
                if (cluster_info['confidence'] > 0.7 and 
                    block.project != dominant_project):
                    
                    logger.debug(f"Cluster-based refinement for {block.id_hash}: {block.project} -> {dominant_project}")
                    block.project = dominant_project
                    block.metadata['project_refined_by_cluster'] = True
                    block.metadata['cluster_id'] = cluster_label
        
        return updated_blocks
    
    def _analyze_cluster_patterns(self, memory_blocks: List[MemoryBlock], 
                                cluster_labels: List[int]) -> Dict[int, Dict[str, Any]]:
        """Analyze project patterns within clusters."""
        cluster_projects = defaultdict(lambda: defaultdict(int))
        cluster_sizes = defaultdict(int)
        
        # Count projects per cluster
        for block, cluster_label in zip(memory_blocks, cluster_labels):
            cluster_projects[cluster_label][block.project] += 1
            cluster_sizes[cluster_label] += 1
        
        # Analyze patterns
        patterns = {}
        for cluster_id, project_counts in cluster_projects.items():
            total_blocks = cluster_sizes[cluster_id]
            dominant_project = max(project_counts.items(), key=lambda x: x[1])
            confidence = dominant_project[1] / total_blocks
            
            patterns[cluster_id] = {
                'dominant_project': dominant_project[0],
                'confidence': confidence,
                'total_blocks': total_blocks,
                'project_distribution': dict(project_counts)
            }
        
        return patterns
    
    def _log_assignment_stats(self):
        """Log assignment statistics."""
        logger.info("Project assignment statistics:")
        for project, count in sorted(self.assignment_stats.items()):
            logger.info(f"  {project}: {count} blocks")


class ProjectReportGenerator:
    """Generates comprehensive project assignment reports."""
    
    def generate_report(self, memory_blocks: List[MemoryBlock], 
                       output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive project assignment report."""
        # Analyze project distribution
        project_stats = defaultdict(lambda: {
            'count': 0,
            'archetypes': defaultdict(int),
            'file_types': defaultdict(int),
            'blocks': []
        })
        
        for block in memory_blocks:
            project = block.project
            project_stats[project]['count'] += 1
            project_stats[project]['archetypes'][block.archetype] += 1
            project_stats[project]['file_types'][block.file_type or 'unknown'] += 1
            project_stats[project]['blocks'].append({
                'id_hash': block.id_hash,
                'summary': block.summary,
                'archetype': block.archetype,
                'source_path': block.source_path
            })
        
        # Generate project hierarchy visualization
        hierarchy_data = self._generate_hierarchy_data(project_stats)
        
        # Create detailed report
        report = {
            'project_assignment_summary': {
                'total_blocks': len(memory_blocks),
                'total_projects': len(project_stats),
                'assignment_timestamp': memory_blocks[0].created_at if memory_blocks else None
            },
            'project_statistics': {
                project: {
                    'count': stats['count'],
                    'percentage': (stats['count'] / len(memory_blocks)) * 100,
                    'top_archetype': max(stats['archetypes'].items(), key=lambda x: x[1])[0] if stats['archetypes'] else 'unknown',
                    'top_file_type': max(stats['file_types'].items(), key=lambda x: x[1])[0] if stats['file_types'] else 'unknown',
                    'archetype_distribution': dict(stats['archetypes']),
                    'file_type_distribution': dict(stats['file_types'])
                }
                for project, stats in project_stats.items()
            },
            'project_hierarchy': hierarchy_data,
            'detailed_assignments': {
                project: stats['blocks'][:10]  # First 10 blocks per project
                for project, stats in project_stats.items()
            }
        }
        
        # Save report
        report_file = output_dir / "project_assignment_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable summary
        self._generate_human_readable_summary(report, output_dir)
        
        logger.info(f"Project assignment report saved to: {report_file}")
        return report
    
    def _generate_hierarchy_data(self, project_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate project hierarchy visualization data."""
        classifier = ProjectClassifier()
        hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for project in project_stats.keys():
            hierarchy_info = classifier.get_project_hierarchy(project)
            domain = hierarchy_info['domain']
            category = hierarchy_info['category']
            subcategory = hierarchy_info['subcategory']
            
            hierarchy[domain][category][subcategory].append({
                'project_id': project,
                'count': project_stats[project]['count'],
                'description': hierarchy_info['description']
            })
        
        return dict(hierarchy)
    
    def _generate_human_readable_summary(self, report: Dict[str, Any], output_dir: Path):
        """Generate human-readable project summary."""
        summary_lines = []
        summary_lines.append("MUSE Pantheon Project Assignment Summary")
        summary_lines.append("=" * 50)
        summary_lines.append("")
        
        # Overall stats
        total_blocks = report['project_assignment_summary']['total_blocks']
        total_projects = report['project_assignment_summary']['total_projects']
        
        summary_lines.append(f"Total Memory Blocks: {total_blocks}")
        summary_lines.append(f"Total Projects: {total_projects}")
        summary_lines.append("")
        
        # Project breakdown
        summary_lines.append("Project Distribution:")
        summary_lines.append("-" * 20)
        
        for project, stats in sorted(report['project_statistics'].items(), 
                                   key=lambda x: x[1]['count'], reverse=True):
            percentage = stats['percentage']
            count = stats['count']
            archetype = stats['top_archetype']
            
            summary_lines.append(f"  {project}: {count} blocks ({percentage:.1f}%) - Primary archetype: {archetype}")
        
        summary_lines.append("")
        summary_lines.append("Project Hierarchy:")
        summary_lines.append("-" * 15)
        
        # Hierarchy breakdown
        for domain, categories in report['project_hierarchy'].items():
            summary_lines.append(f"  Domain: {domain}")
            for category, subcategories in categories.items():
                summary_lines.append(f"    Category: {category}")
                for subcategory, projects in subcategories.items():
                    for project_info in projects:
                        summary_lines.append(f"      - {project_info['project_id']} ({project_info['count']} blocks)")
        
        # Save summary
        summary_file = output_dir / "project_assignment_summary.txt"
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Human-readable summary saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Project Assignment for MUSE Pantheon")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for project assignments')
    parser.add_argument('--clusters-file', 
                       help='Optional clusters file for cluster-aware assignment')
    
    args = parser.parse_args()
    
    blocks_dir = Path(args.blocks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load memory blocks
    logger.info(f"Loading memory blocks from: {blocks_dir}")
    memory_blocks = []
    
    for json_file in blocks_dir.glob("*.json"):
        # Skip non-memory block files
        if not json_file.name.startswith("memory_block_"):
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                block = MemoryBlock.from_dict(data)
                memory_blocks.append(block)
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {str(e)}")
    
    if not memory_blocks:
        logger.error("No memory blocks found!")
        return 1
    
    logger.info(f"Loaded {len(memory_blocks)} memory blocks")
    
    # Load clusters if provided
    cluster_labels = None
    if args.clusters_file:
        try:
            with open(args.clusters_file, 'r') as f:
                cluster_data = json.load(f)
                cluster_labels = cluster_data.get('clustering_results', {}).get('cluster_labels', [])
                logger.info(f"Loaded cluster labels for {len(cluster_labels)} blocks")
        except Exception as e:
            logger.warning(f"Failed to load clusters file: {str(e)}")
    
    # Assign projects
    assignment_engine = ProjectAssignmentEngine()
    
    if cluster_labels and len(cluster_labels) == len(memory_blocks):
        updated_blocks = assignment_engine.assign_projects_with_clusters(memory_blocks, cluster_labels)
    else:
        updated_blocks = assignment_engine.assign_projects(memory_blocks)
    
    # Save updated memory blocks
    logger.info("Saving updated memory blocks...")
    for block in updated_blocks:
        block.save(output_dir)
    
    # Generate comprehensive report
    report_generator = ProjectReportGenerator()
    report = report_generator.generate_report(updated_blocks, output_dir)
    
    logger.info(f"Project assignment complete. Updated {len(updated_blocks)} blocks.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())