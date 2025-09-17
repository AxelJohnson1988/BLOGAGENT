#!/usr/bin/env python3
"""
MUSE Pantheon Project Assignment System
Auto-assigns project IDs based on content analysis and archetype mapping.
"""
import os
import sys
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from memory_block import MemoryBlock, ARCHETYPES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectMatcher:
    """Intelligent project assignment based on content analysis"""
    
    def __init__(self):
        self.project_patterns = self._initialize_project_patterns()
        self.archetype_themes = self._initialize_archetype_themes()
        self.legal_patterns = self._initialize_legal_patterns()
        
    def _initialize_project_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize project identification patterns"""
        return {
            'legal.ruby_tuesday': {
                'keywords': ['kaj hospitality', 'ruby tuesday', 'paystub', 'earnings', 'payroll'],
                'patterns': [r'kaj\s+hospitality', r'ruby\s+tuesday', r'earnings\s+statement'],
                'confidence_boost': 0.9,
                'archetype': 'Guardian'
            },
            'legal.employment': {
                'keywords': ['employment', 'hr', 'payroll', 'benefits', 'salary'],
                'patterns': [r'employment\s+record', r'pay\s+statement', r'w-?2'],
                'confidence_boost': 0.8,
                'archetype': 'Guardian'
            },
            'mindprint.core': {
                'keywords': ['mindprint', 'memory block', 'semantic', 'embedding'],
                'patterns': [r'memory\s+block', r'mindprint', r'semantic\s+search'],
                'confidence_boost': 0.9,
                'archetype': 'Memory'
            },
            'muse.pantheon': {
                'keywords': ['muse', 'pantheon', 'archetype', 'warden', 'pipeline'],
                'patterns': [r'muse\s+pantheon', r'archetype', r'universal\s+ingest'],
                'confidence_boost': 0.9,
                'archetype': 'Warden'
            },
            'vision.ocr': {
                'keywords': ['ocr', 'image', 'text extraction', 'visual'],
                'patterns': [r'ocr\s+extracted', r'image\s+file', r'tesseract'],
                'confidence_boost': 0.8,
                'archetype': 'Vision'
            },
            'code.python': {
                'keywords': ['python', 'def', 'class', 'import'],
                'patterns': [r'def\s+\w+', r'class\s+\w+', r'import\s+\w+'],
                'confidence_boost': 0.7,
                'archetype': 'Builder'
            },
            'docs.technical': {
                'keywords': ['documentation', 'readme', 'api', 'guide'],
                'patterns': [r'# \w+', r'## \w+', r'```'],
                'confidence_boost': 0.6,
                'archetype': 'Scribe'
            },
            'data.analysis': {
                'keywords': ['analysis', 'data', 'csv', 'statistics', 'report'],
                'patterns': [r'data\s+analysis', r'\.csv', r'statistics'],
                'confidence_boost': 0.7,
                'archetype': 'Analyst'
            },
            'media.content': {
                'keywords': ['video', 'audio', 'media', 'content'],
                'patterns': [r'\.(mp4|mov|mp3|wav)', r'media\s+file'],
                'confidence_boost': 0.6,
                'archetype': 'Vision'
            },
            'archive.storage': {
                'keywords': ['archive', 'backup', 'storage', 'compressed'],
                'patterns': [r'\.(zip|tar|gz)', r'archive\s+file'],
                'confidence_boost': 0.5,
                'archetype': 'Explorer'
            }
        }
    
    def _initialize_archetype_themes(self) -> Dict[str, List[str]]:
        """Initialize archetype-specific themes"""
        return {
            'Guardian': ['security', 'validation', 'ethics', 'legal', 'compliance', 'protection'],
            'Vision': ['image', 'visual', 'ocr', 'media', 'graphics', 'perception'],
            'Warden': ['orchestration', 'pipeline', 'management', 'coordination', 'workflow'],
            'Memory': ['storage', 'retrieval', 'embedding', 'search', 'database', 'memory'],
            'Scribe': ['documentation', 'writing', 'notes', 'text', 'content', 'communication'],
            'Analyst': ['analysis', 'data', 'statistics', 'metrics', 'research', 'insights'],
            'Builder': ['code', 'development', 'construction', 'programming', 'implementation'],
            'Explorer': ['discovery', 'research', 'investigation', 'exploration', 'unknown'],
            'Mentor': ['teaching', 'guidance', 'education', 'learning', 'knowledge'],
            'Connector': ['integration', 'connection', 'linking', 'communication', 'bridge'],
            'Sage': ['wisdom', 'philosophy', 'deep', 'knowledge', 'understanding'],
            'Creator': ['innovation', 'art', 'creative', 'design', 'imagination']
        }
    
    def _initialize_legal_patterns(self) -> Dict[str, List[str]]:
        """Initialize legal document patterns for IP proof"""
        return {
            'paystub': ['pay period', 'gross pay', 'net pay', 'deductions', 'earnings'],
            'contract': ['agreement', 'contract', 'terms', 'conditions', 'parties'],
            'invoice': ['invoice', 'bill', 'amount due', 'payment', 'charges'],
            'receipt': ['receipt', 'transaction', 'paid', 'purchase', 'payment'],
            'correspondence': ['email', 'letter', 'communication', 'message', 'response']
        }
    
    def analyze_content(self, block: MemoryBlock) -> Dict[str, Any]:
        """Analyze MemoryBlock content for project assignment"""
        content = (block.content + " " + block.summary + " " + " ".join(block.topics)).lower()
        
        analysis = {
            'project_matches': [],
            'archetype_scores': defaultdict(float),
            'legal_type': None,
            'theme_analysis': {},
            'confidence_factors': []
        }
        
        # Project pattern matching
        for project_id, project_info in self.project_patterns.items():
            score = self._calculate_project_score(content, project_info)
            if score > 0.1:  # Minimum threshold
                analysis['project_matches'].append({
                    'project_id': project_id,
                    'score': score,
                    'archetype': project_info['archetype'],
                    'reason': f"Matched {len([k for k in project_info['keywords'] if k in content])} keywords"
                })
        
        # Archetype analysis
        for archetype, themes in self.archetype_themes.items():
            score = sum(1 for theme in themes if theme in content) / len(themes)
            analysis['archetype_scores'][archetype] = score
        
        # Legal document type detection
        for legal_type, patterns in self.legal_patterns.items():
            if any(pattern in content for pattern in patterns):
                analysis['legal_type'] = legal_type
                break
        
        # File path analysis
        if block.source and block.source.file_path:
            path_analysis = self._analyze_file_path(block.source.file_path)
            analysis['confidence_factors'].extend(path_analysis)
        
        return analysis
    
    def _calculate_project_score(self, content: str, project_info: Dict[str, Any]) -> float:
        """Calculate project matching score"""
        score = 0.0
        
        # Keyword matching
        keyword_matches = sum(1 for keyword in project_info['keywords'] if keyword in content)
        keyword_score = (keyword_matches / len(project_info['keywords'])) * 0.6
        
        # Pattern matching
        pattern_matches = sum(1 for pattern in project_info['patterns'] if re.search(pattern, content))
        pattern_score = (pattern_matches / len(project_info['patterns'])) * 0.4
        
        base_score = keyword_score + pattern_score
        
        # Apply confidence boost
        final_score = base_score * project_info.get('confidence_boost', 1.0)
        
        return min(final_score, 1.0)
    
    def _analyze_file_path(self, file_path: str) -> List[str]:
        """Analyze file path for additional context"""
        path_lower = file_path.lower()
        factors = []
        
        if 'legal' in path_lower:
            factors.append("File in legal directory")
        if 'document' in path_lower:
            factors.append("File in documents directory")
        if 'image' in path_lower or any(ext in path_lower for ext in ['.jpg', '.png', '.gif']):
            factors.append("Image file type")
        if 'code' in path_lower or any(ext in path_lower for ext in ['.py', '.js', '.java']):
            factors.append("Code file type")
        
        return factors
    
    def suggest_project(self, block: MemoryBlock) -> Tuple[str, str, float, str]:
        """Suggest best project assignment for a MemoryBlock"""
        analysis = self.analyze_content(block)
        
        # Find best project match
        if analysis['project_matches']:
            best_match = max(analysis['project_matches'], key=lambda x: x['score'])
            return (
                best_match['project_id'],
                best_match['archetype'],
                best_match['score'],
                best_match['reason']
            )
        
        # Fallback to archetype-based assignment
        if analysis['archetype_scores']:
            best_archetype = max(analysis['archetype_scores'].items(), key=lambda x: x[1])
            archetype_name = best_archetype[0]
            archetype_score = best_archetype[1]
            
            if archetype_score > 0.2:
                # Create theme-based project ID
                main_topics = block.topics[:2] if block.topics else ['general']
                topic_id = '_'.join(main_topics).replace(' ', '_')
                project_id = f"{archetype_name.lower()}.{topic_id}"
                
                return (
                    project_id,
                    archetype_name,
                    archetype_score * 0.8,  # Lower confidence for fallback
                    f"Archetype-based assignment based on content themes"
                )
        
        # Final fallback
        file_ext = Path(block.source.file_path).suffix.lower() if block.source else '.unknown'
        fallback_archetype = self._get_fallback_archetype(file_ext)
        return (
            f"general.{fallback_archetype.lower()}",
            fallback_archetype,
            0.3,
            "Default assignment based on file type"
        )
    
    def _get_fallback_archetype(self, file_ext: str) -> str:
        """Get fallback archetype based on file extension"""
        ext_mapping = {
            '.py': 'Builder',
            '.js': 'Builder',
            '.md': 'Scribe',
            '.txt': 'Scribe',
            '.pdf': 'Scribe',
            '.jpg': 'Vision',
            '.png': 'Vision',
            '.mp4': 'Vision',
            '.json': 'Memory',
            '.csv': 'Analyst'
        }
        return ext_mapping.get(file_ext, 'Explorer')


class ProjectHierarchyBuilder:
    """Builds project hierarchies and relationships"""
    
    def __init__(self):
        self.project_tree = defaultdict(lambda: defaultdict(list))
        self.project_metadata = {}
    
    def add_project_assignment(self, project_id: str, archetype: str, block_id: str, confidence: float):
        """Add a project assignment to the hierarchy"""
        parts = project_id.split('.')
        if len(parts) >= 2:
            category = parts[0]
            subcategory = '.'.join(parts[1:])
            self.project_tree[category][subcategory].append({
                'block_id': block_id,
                'archetype': archetype,
                'confidence': confidence
            })
        
        # Update project metadata
        if project_id not in self.project_metadata:
            self.project_metadata[project_id] = {
                'primary_archetype': archetype,
                'total_blocks': 0,
                'avg_confidence': 0.0,
                'archetypes': Counter()
            }
        
        metadata = self.project_metadata[project_id]
        metadata['total_blocks'] += 1
        metadata['archetypes'][archetype] += 1
        
        # Update average confidence
        old_avg = metadata['avg_confidence']
        old_count = metadata['total_blocks'] - 1
        metadata['avg_confidence'] = (old_avg * old_count + confidence) / metadata['total_blocks']
        
        # Update primary archetype
        metadata['primary_archetype'] = metadata['archetypes'].most_common(1)[0][0]
    
    def get_hierarchy(self) -> Dict[str, Any]:
        """Get the complete project hierarchy"""
        return {
            'hierarchy': dict(self.project_tree),
            'metadata': dict(self.project_metadata),
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate hierarchy summary statistics"""
        total_projects = len(self.project_metadata)
        total_blocks = sum(meta['total_blocks'] for meta in self.project_metadata.values())
        
        archetype_distribution = Counter()
        for meta in self.project_metadata.values():
            archetype_distribution[meta['primary_archetype']] += 1
        
        category_distribution = Counter()
        for project_id in self.project_metadata.keys():
            category = project_id.split('.')[0]
            category_distribution[category] += 1
        
        return {
            'total_projects': total_projects,
            'total_blocks': total_blocks,
            'avg_blocks_per_project': total_blocks / total_projects if total_projects > 0 else 0,
            'archetype_distribution': dict(archetype_distribution),
            'category_distribution': dict(category_distribution)
        }


class ProjectAssignmentPipeline:
    """Main pipeline for project assignment"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.matcher = ProjectMatcher()
        self.hierarchy_builder = ProjectHierarchyBuilder()
        self.stats = {
            'processed': 0,
            'assignments': defaultdict(int),
            'confidence_distribution': []
        }
    
    def load_memory_blocks(self, blocks_dir: Path) -> List[MemoryBlock]:
        """Load MemoryBlocks from directory"""
        blocks = []
        for json_file in blocks_dir.glob("*.json"):
            try:
                block = MemoryBlock.load_from_file(json_file)
                blocks.append(block)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(blocks)} MemoryBlocks")
        return blocks
    
    def load_cluster_results(self, clusters_file: Path) -> Optional[Dict[str, Any]]:
        """Load clustering results if available"""
        try:
            with open(clusters_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cluster results: {e}")
            return None
    
    def assign_projects(self, blocks: List[MemoryBlock], cluster_results: Optional[Dict[str, Any]] = None):
        """Assign projects to all MemoryBlocks"""
        logger.info("Starting project assignment")
        
        # Create cluster-to-project mapping if cluster results available
        cluster_projects = {}
        if cluster_results:
            cluster_projects = self._create_cluster_project_mapping(cluster_results)
        
        for block in blocks:
            # Try cluster-based assignment first
            project_assigned = False
            if cluster_results:
                project_assigned = self._assign_from_cluster(block, cluster_projects)
            
            # Fallback to content-based assignment
            if not project_assigned:
                self._assign_from_content(block)
            
            self.stats['processed'] += 1
        
        logger.info(f"Assigned projects to {self.stats['processed']} blocks")
    
    def _create_cluster_project_mapping(self, cluster_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create mapping from cluster to project information"""
        cluster_projects = {}
        
        for cluster_id, summary in cluster_results.get('cluster_summaries', {}).items():
            archetype = summary['primary_archetype']
            top_topics = summary['top_topics'][:2]
            
            # Create project ID from cluster info
            topic_str = '_'.join(top_topics).replace(' ', '_') if top_topics else 'general'
            project_id = f"cluster.{archetype.lower()}.{topic_str}"
            
            cluster_projects[cluster_id] = {
                'project_id': project_id,
                'archetype': archetype,
                'confidence': 0.8,  # High confidence for cluster-based assignments
                'reason': f"Cluster-based assignment from {summary['size']} similar blocks"
            }
        
        return cluster_projects
    
    def _assign_from_cluster(self, block: MemoryBlock, cluster_projects: Dict[str, Dict[str, Any]]) -> bool:
        """Try to assign project based on cluster membership"""
        # This would require cluster membership info in the block
        # For now, return False to fallback to content-based assignment
        return False
    
    def _assign_from_content(self, block: MemoryBlock):
        """Assign project based on content analysis"""
        project_id, archetype, confidence, reason = self.matcher.suggest_project(block)
        
        # Assign to block
        block.assign_project(
            project_id=project_id,
            archetype=archetype,
            confidence=confidence,
            reason=reason
        )
        
        # Update hierarchy and stats
        self.hierarchy_builder.add_project_assignment(
            project_id, archetype, block.id_hash, confidence
        )
        
        self.stats['assignments'][project_id] += 1
        self.stats['confidence_distribution'].append(confidence)
    
    def save_results(self, blocks: List[MemoryBlock]):
        """Save assignment results"""
        # Save updated blocks
        updated_blocks_dir = self.output_dir / "updated_blocks"
        updated_blocks_dir.mkdir(exist_ok=True)
        
        for block in blocks:
            filename = f"memory_block_{block.id_hash.split(':')[1]}.json"
            block.save_to_file(updated_blocks_dir / filename)
        
        # Save project hierarchy
        hierarchy = self.hierarchy_builder.get_hierarchy()
        hierarchy_file = self.output_dir / "project_hierarchy.json"
        with open(hierarchy_file, 'w') as f:
            json.dump(hierarchy, f, indent=2)
        
        # Save assignment statistics
        stats_file = self.output_dir / "assignment_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Create human-readable project summary
        self._create_project_summary(hierarchy)
        
        logger.info(f"Assignment results saved to {self.output_dir}")
    
    def _create_project_summary(self, hierarchy: Dict[str, Any]):
        """Create human-readable project summary"""
        summary_file = self.output_dir / "project_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# MUSE Pantheon Project Assignment Summary\n\n")
            
            # Overall statistics
            summary = hierarchy['summary']
            f.write(f"## Overview\n\n")
            f.write(f"- **Total Projects**: {summary['total_projects']}\n")
            f.write(f"- **Total Blocks**: {summary['total_blocks']}\n")
            f.write(f"- **Average Blocks per Project**: {summary['avg_blocks_per_project']:.1f}\n\n")
            
            # Archetype distribution
            f.write(f"## Archetype Distribution\n\n")
            for archetype, count in summary['archetype_distribution'].items():
                f.write(f"- **{archetype}**: {count} projects\n")
            f.write("\n")
            
            # Category distribution
            f.write(f"## Category Distribution\n\n")
            for category, count in summary['category_distribution'].items():
                f.write(f"- **{category}**: {count} projects\n")
            f.write("\n")
            
            # Detailed project breakdown
            f.write(f"## Project Details\n\n")
            for project_id, metadata in hierarchy['metadata'].items():
                f.write(f"### {project_id}\n")
                f.write(f"- **Primary Archetype**: {metadata['primary_archetype']}\n")
                f.write(f"- **Total Blocks**: {metadata['total_blocks']}\n")
                f.write(f"- **Average Confidence**: {metadata['avg_confidence']:.2f}\n")
                f.write(f"- **Archetype Mix**: {dict(metadata['archetypes'])}\n\n")
    
    def run_pipeline(self, blocks_dir: Path, clusters_file: Optional[Path] = None):
        """Run the complete project assignment pipeline"""
        logger.info("Starting project assignment pipeline")
        
        # Load data
        blocks = self.load_memory_blocks(blocks_dir)
        cluster_results = self.load_cluster_results(clusters_file) if clusters_file else None
        
        if not blocks:
            logger.error("No MemoryBlocks found")
            return
        
        # Assign projects
        self.assign_projects(blocks, cluster_results)
        
        # Save results
        self.save_results(blocks)
        
        logger.info("Project assignment pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Project Assignment Pipeline")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for assignment results')
    parser.add_argument('--clusters-file', 
                       help='Optional clusters.json file from clustering pipeline')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = ProjectAssignmentPipeline(args.output_dir)
    clusters_file = Path(args.clusters_file) if args.clusters_file else None
    pipeline.run_pipeline(Path(args.blocks_dir), clusters_file)


if __name__ == "__main__":
    main()