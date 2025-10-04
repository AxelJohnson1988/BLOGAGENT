#!/usr/bin/env python3
"""
MUSE Pantheon Project Assignment System
Auto-assigns project IDs and hierarchies based on content analysis
"""
import os
import sys
import json
import argparse
import pathlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from warden.schema.memory_block import MemoryBlock
except ImportError:
    # Try direct import if running from repository root
    sys.path.insert(0, os.getcwd())
    from warden.schema.memory_block import MemoryBlock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectAssignmentEngine:
    """Assigns project IDs and creates project hierarchies for MemoryBlocks"""
    
    def __init__(self):
        # Predefined project patterns and rules
        self.project_patterns = {
            'legal': {
                'keywords': ['legal', 'contract', 'agreement', 'law', 'court', 'attorney', 'litigation'],
                'file_patterns': ['contract', 'agreement', 'legal'],
                'base_project': 'legal'
            },
            'financial': {
                'keywords': ['payment', 'invoice', 'tax', 'salary', 'budget', 'financial', 'money', 'cost'],
                'file_patterns': ['invoice', 'payment', 'tax', 'salary', 'budget'],
                'base_project': 'financial'
            },
            'development': {
                'keywords': ['code', 'programming', 'development', 'software', 'api', 'database'],
                'file_patterns': ['.py', '.js', '.html', '.css', '.java', '.cpp'],
                'base_project': 'development'
            },
            'research': {
                'keywords': ['research', 'study', 'analysis', 'experiment', 'hypothesis', 'academic'],
                'file_patterns': ['research', 'study', 'analysis', 'paper'],
                'base_project': 'research'
            },
            'communications': {
                'keywords': ['email', 'message', 'communication', 'meeting', 'call', 'discussion'],
                'file_patterns': ['.eml', 'email', 'message'],
                'base_project': 'communications'
            },
            'media': {
                'keywords': ['image', 'video', 'photo', 'picture', 'visual', 'media', 'graphics'],
                'file_patterns': ['.jpg', '.png', '.gif', '.mp4', '.mov', '.avi'],
                'base_project': 'media'
            },
            'documentation': {
                'keywords': ['documentation', 'manual', 'guide', 'readme', 'instructions'],
                'file_patterns': ['.md', 'readme', 'manual', 'guide'],
                'base_project': 'knowledge'
            },
            'personal': {
                'keywords': ['personal', 'diary', 'journal', 'private', 'family', 'friend'],
                'file_patterns': ['diary', 'journal', 'personal'],
                'base_project': 'personal'
            }
        }
        
        # MUSE Archetype to project mappings
        self.archetype_projects = {
            'Builder': 'development',
            'Vision': 'media',
            'Guardian': 'security',
            'Scholar': 'research',
            'Scribe': 'knowledge',
            'Herald': 'communications',
            'Analyst': 'analytics',
            'Keeper': 'personal',
            'Muse': 'creative',
            'Warden': 'system',
            'Voice': 'media',
            'unknown': 'general'
        }
    
    def load_memory_blocks(self, blocks_dir: pathlib.Path) -> List[MemoryBlock]:
        """Load all MemoryBlocks from directory"""
        memory_blocks = []
        
        for json_file in blocks_dir.glob('*.json'):
            # Skip summary files
            if json_file.name == 'ingestion_summary.json':
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                memory_block = MemoryBlock.from_dict(data)
                memory_blocks.append(memory_block)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(memory_blocks)} MemoryBlocks")
        return memory_blocks
    
    def load_cluster_assignments(self, clusters_file: pathlib.Path) -> Dict[str, int]:
        """Load cluster assignments if available"""
        cluster_map = {}
        
        if clusters_file.exists():
            try:
                with open(clusters_file, 'r') as f:
                    cluster_data = json.load(f)
                
                for assignment in cluster_data:
                    cluster_map[assignment['id_hash']] = assignment.get('cluster_id', 0)
                
                logger.info(f"Loaded cluster assignments for {len(cluster_map)} blocks")
            except Exception as e:
                logger.error(f"Failed to load cluster assignments: {e}")
        
        return cluster_map
    
    def analyze_content_themes(self, memory_block: MemoryBlock) -> Dict[str, float]:
        """Analyze content themes and return confidence scores"""
        themes = {}
        
        # Combine all text content for analysis
        text_content = f"{memory_block.summary} {memory_block.content} {' '.join(memory_block.topics)}"
        text_lower = text_content.lower()
        file_path_lower = memory_block.source_file.lower()
        
        # Score each project pattern
        for theme, pattern in self.project_patterns.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in pattern['keywords'] if keyword in text_lower)
            score += keyword_matches * 0.3
            
            # File pattern matching
            file_matches = sum(1 for pattern_str in pattern['file_patterns'] if pattern_str in file_path_lower)
            score += file_matches * 0.5
            
            # Topic matching
            topic_matches = sum(1 for topic in memory_block.topics if topic.lower() in pattern['keywords'])
            score += topic_matches * 0.4
            
            themes[theme] = score
        
        return themes
    
    def assign_primary_project(self, memory_block: MemoryBlock, themes: Dict[str, float]) -> str:
        """Assign primary project based on theme analysis"""
        # Find highest scoring theme
        if themes:
            best_theme = max(themes.items(), key=lambda x: x[1])
            if best_theme[1] > 0.5:  # Minimum confidence threshold
                return self.project_patterns[best_theme[0]]['base_project']
        
        # Fallback to archetype-based assignment
        archetype_project = self.archetype_projects.get(memory_block.archetype, 'general')
        return archetype_project
    
    def generate_project_hierarchy(self, memory_block: MemoryBlock, primary_project: str, themes: Dict[str, float]) -> Dict[str, str]:
        """Generate hierarchical project structure"""
        # Extract file type and date for sub-categorization
        file_type = memory_block.source_type.lstrip('.')
        date_str = memory_block.date[:7]  # YYYY-MM format
        
        # Create hierarchy
        hierarchy = {
            'domain': primary_project,
            'category': self._determine_category(memory_block, themes),
            'subcategory': self._determine_subcategory(memory_block, file_type),
            'timeline': date_str
        }
        
        # Generate full project ID
        project_id = f"{hierarchy['domain']}.{hierarchy['category']}.{hierarchy['subcategory']}"
        hierarchy['full_id'] = project_id
        
        return hierarchy
    
    def _determine_category(self, memory_block: MemoryBlock, themes: Dict[str, float]) -> str:
        """Determine project category based on content analysis"""
        # Sort themes by score
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        
        # Use top theme for category, or archetype as fallback
        if sorted_themes and sorted_themes[0][1] > 0.3:
            return sorted_themes[0][0]
        
        # Archetype-based categories
        archetype_categories = {
            'Builder': 'code',
            'Vision': 'visual',
            'Guardian': 'compliance',
            'Scholar': 'analysis',
            'Scribe': 'documents',
            'Herald': 'communications',
            'Analyst': 'data',
            'Keeper': 'archive',
            'Muse': 'creative',
            'Warden': 'system',
            'Voice': 'audio'
        }
        
        return archetype_categories.get(memory_block.archetype, 'misc')
    
    def _determine_subcategory(self, memory_block: MemoryBlock, file_type: str) -> str:
        """Determine subcategory based on file type and content"""
        # File type mappings
        type_mappings = {
            'py': 'python',
            'js': 'javascript',
            'html': 'web',
            'css': 'styles',
            'md': 'markdown',
            'pdf': 'documents',
            'jpg': 'images',
            'png': 'images',
            'mp4': 'videos',
            'mp3': 'audio',
            'json': 'data',
            'csv': 'spreadsheets',
            'txt': 'text'
        }
        
        subcategory = type_mappings.get(file_type, file_type or 'unknown')
        
        # Enhance with topic information
        if memory_block.topics:
            primary_topic = memory_block.topics[0].lower()
            if primary_topic in ['config', 'setup', 'installation']:
                subcategory = 'configuration'
            elif primary_topic in ['test', 'testing', 'unittest']:
                subcategory = 'testing'
            elif primary_topic in ['api', 'endpoint', 'service']:
                subcategory = 'api'
        
        return subcategory
    
    def create_project_metadata(self, project_id: str, memory_blocks: List[MemoryBlock]) -> Dict[str, Any]:
        """Create metadata for a project based on its MemoryBlocks"""
        if not memory_blocks:
            return {}
        
        # Aggregate metadata
        all_topics = []
        archetypes = []
        file_types = []
        dates = []
        
        for block in memory_blocks:
            all_topics.extend(block.topics)
            archetypes.append(block.archetype)
            file_types.append(block.source_type)
            dates.append(block.date)
        
        # Calculate statistics
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        archetype_counts = {}
        for archetype in archetypes:
            archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
        
        # Create metadata
        metadata = {
            'project_id': project_id,
            'block_count': len(memory_blocks),
            'date_range': {
                'earliest': min(dates) if dates else None,
                'latest': max(dates) if dates else None
            },
            'top_topics': sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'archetypes': archetype_counts,
            'file_types': list(set(file_types)),
            'created_at': datetime.utcnow().isoformat() + "Z"
        }
        
        return metadata
    
    def assign_projects_to_blocks(self, memory_blocks: List[MemoryBlock], 
                                cluster_assignments: Dict[str, int] = None) -> Dict[str, Any]:
        """Assign projects to all MemoryBlocks"""
        project_assignments = {}
        project_hierarchies = {}
        project_metadata = {}
        
        # Group blocks by cluster if available
        cluster_groups = {}
        if cluster_assignments:
            for block in memory_blocks:
                cluster_id = cluster_assignments.get(block.id_hash, 0)
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(block)
        else:
            # Single group if no clustering
            cluster_groups[0] = memory_blocks
        
        # Process each cluster/group
        for cluster_id, blocks in cluster_groups.items():
            logger.info(f"Processing cluster {cluster_id} with {len(blocks)} blocks")
            
            for block in blocks:
                # Analyze content themes
                themes = self.analyze_content_themes(block)
                
                # Assign primary project
                primary_project = self.assign_primary_project(block, themes)
                
                # Generate hierarchy
                hierarchy = self.generate_project_hierarchy(block, primary_project, themes)
                
                # Store assignments
                project_assignments[block.id_hash] = {
                    'project_id': hierarchy['full_id'],
                    'domain': hierarchy['domain'],
                    'category': hierarchy['category'],
                    'subcategory': hierarchy['subcategory'],
                    'timeline': hierarchy['timeline'],
                    'confidence_scores': themes,
                    'cluster_id': cluster_id
                }
                
                project_hierarchies[hierarchy['full_id']] = hierarchy
        
        # Create project metadata
        projects_blocks = {}
        for block in memory_blocks:
            project_id = project_assignments[block.id_hash]['project_id']
            if project_id not in projects_blocks:
                projects_blocks[project_id] = []
            projects_blocks[project_id].append(block)
        
        for project_id, blocks in projects_blocks.items():
            project_metadata[project_id] = self.create_project_metadata(project_id, blocks)
        
        return {
            'assignments': project_assignments,
            'hierarchies': project_hierarchies,
            'metadata': project_metadata
        }
    
    def update_memory_blocks_with_projects(self, memory_blocks: List[MemoryBlock], 
                                         project_assignments: Dict[str, Any],
                                         output_dir: pathlib.Path) -> int:
        """Update MemoryBlocks with project assignments and save"""
        updated_count = 0
        
        for block in memory_blocks:
            assignment = project_assignments.get(block.id_hash)
            if assignment:
                # Update block with project information
                block.project = assignment['domain']
                block.project_id = assignment['project_id']
                
                # Add assignment metadata
                block.metadata.update({
                    'project_hierarchy': {
                        'domain': assignment['domain'],
                        'category': assignment['category'],
                        'subcategory': assignment['subcategory'],
                        'timeline': assignment['timeline']
                    },
                    'project_confidence': assignment['confidence_scores'],
                    'cluster_id': assignment.get('cluster_id', 0)
                })
                
                # Save updated block
                output_file = output_dir / f"{block.id_hash}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(block.to_json())
                
                updated_count += 1
        
        logger.info(f"Updated {updated_count} MemoryBlocks with project assignments")
        return updated_count
    
    def save_project_assignments(self, output_dir: pathlib.Path, results: Dict[str, Any]) -> Dict[str, str]:
        """Save project assignment results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        # Save assignments
        assignments_file = output_dir / 'project_assignments.json'
        with open(assignments_file, 'w') as f:
            json.dump(results['assignments'], f, indent=2)
        file_paths['assignments'] = str(assignments_file)
        
        # Save hierarchies
        hierarchies_file = output_dir / 'project_hierarchies.json'
        with open(hierarchies_file, 'w') as f:
            json.dump(results['hierarchies'], f, indent=2)
        file_paths['hierarchies'] = str(hierarchies_file)
        
        # Save metadata
        metadata_file = output_dir / 'project_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        file_paths['metadata'] = str(metadata_file)
        
        # Save summary
        summary = {
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'total_assignments': len(results['assignments']),
            'unique_projects': len(results['hierarchies']),
            'project_domains': list(set(h['domain'] for h in results['hierarchies'].values())),
            'files_created': file_paths
        }
        
        summary_file = output_dir / 'assignment_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        file_paths['summary'] = str(summary_file)
        
        logger.info(f"Project assignment results saved to {output_dir}")
        return file_paths


def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Project Assignment System")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for updated blocks and assignment data')
    parser.add_argument('--clusters-file', 
                       help='Optional cluster assignments file from clustering phase')
    parser.add_argument('--update-blocks', action='store_true',
                       help='Update original MemoryBlocks with project assignments')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize assignment engine
    engine = ProjectAssignmentEngine()
    
    # Load MemoryBlocks
    blocks_dir = pathlib.Path(args.blocks_dir)
    if not blocks_dir.exists():
        logger.error(f"Blocks directory does not exist: {blocks_dir}")
        return 1
    
    memory_blocks = engine.load_memory_blocks(blocks_dir)
    if not memory_blocks:
        logger.error("No MemoryBlocks found")
        return 1
    
    # Load cluster assignments if available
    cluster_assignments = {}
    if args.clusters_file:
        clusters_file = pathlib.Path(args.clusters_file)
        cluster_assignments = engine.load_cluster_assignments(clusters_file)
    
    # Assign projects
    results = engine.assign_projects_to_blocks(memory_blocks, cluster_assignments)
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    
    # Save results
    file_paths = engine.save_project_assignments(output_dir, results)
    
    # Update MemoryBlocks if requested
    updated_count = 0
    if args.update_blocks:
        updated_blocks_dir = output_dir / 'updated_blocks'
        updated_blocks_dir.mkdir(exist_ok=True)
        updated_count = engine.update_memory_blocks_with_projects(
            memory_blocks, results['assignments'], updated_blocks_dir
        )
    
    # Print summary
    print("✅ Project assignment complete")
    print(f"📊 Processed {len(memory_blocks)} MemoryBlocks")
    print(f"🎯 Created {len(results['hierarchies'])} unique projects")
    print(f"📁 Results saved to: {output_dir}")
    if updated_count > 0:
        print(f"🔄 Updated {updated_count} MemoryBlocks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())