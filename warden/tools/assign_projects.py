#!/usr/bin/env python3
"""
Project Assignment System for BLOGAGENT
Maps MemoryBlocks to project hierarchies and MUSE archetypes
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

# Add parent directories to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_block import MemoryBlock, load_all_memory_blocks, save_memory_block


class ProjectMapper:
    """Maps content to project hierarchies based on themes and context"""
    
    # Project hierarchy templates
    PROJECT_TEMPLATES = {
        'ai': {
            'prefix': 'ai',
            'keywords': ['ai', 'artificial intelligence', 'machine learning', 'neural', 'model', 'algorithm', 'agent'],
            'subcategories': ['research', 'development', 'testing', 'deployment']
        },
        'blog': {
            'prefix': 'blog',
            'keywords': ['blog', 'article', 'post', 'content', 'writing', 'publish'],
            'subcategories': ['drafts', 'published', 'ideas', 'analytics']
        },
        'legal': {
            'prefix': 'legal',
            'keywords': ['legal', 'contract', 'agreement', 'law', 'regulation', 'compliance', 'payroll', 'earnings'],
            'subcategories': ['contracts', 'payroll', 'compliance', 'documentation']
        },
        'business': {
            'prefix': 'business',
            'keywords': ['business', 'company', 'strategy', 'market', 'finance', 'revenue', 'client'],
            'subcategories': ['strategy', 'operations', 'finance', 'marketing']
        },
        'development': {
            'prefix': 'dev',
            'keywords': ['code', 'development', 'programming', 'software', 'api', 'database', 'framework'],
            'subcategories': ['frontend', 'backend', 'database', 'testing']
        },
        'research': {
            'prefix': 'research',
            'keywords': ['research', 'study', 'analysis', 'data', 'experiment', 'hypothesis'],
            'subcategories': ['literature', 'experiments', 'data', 'results']
        },
        'documentation': {
            'prefix': 'docs',
            'keywords': ['documentation', 'manual', 'guide', 'readme', 'instructions', 'help'],
            'subcategories': ['user_guides', 'technical', 'api', 'tutorials']
        }
    }
    
    def __init__(self):
        self.project_assignments = {}
        self.project_stats = defaultdict(int)
    
    def analyze_content_themes(self, content: str, topics: List[str]) -> Dict[str, float]:
        """Analyze content to determine theme scores"""
        content_lower = content.lower()
        combined_text = f"{content_lower} {' '.join(topics)}"
        
        theme_scores = {}
        
        for theme, template in self.PROJECT_TEMPLATES.items():
            score = 0.0
            keyword_matches = 0
            
            for keyword in template['keywords']:
                if keyword in combined_text:
                    keyword_matches += 1
                    # Weight longer keywords more heavily
                    score += len(keyword.split())
            
            # Normalize score
            if template['keywords']:
                theme_scores[theme] = score / len(template['keywords'])
            else:
                theme_scores[theme] = 0.0
        
        return theme_scores
    
    def determine_subcategory(self, content: str, main_theme: str) -> str:
        """Determine subcategory within main theme"""
        if main_theme not in self.PROJECT_TEMPLATES:
            return 'general'
        
        content_lower = content.lower()
        template = self.PROJECT_TEMPLATES[main_theme]
        
        # Check for subcategory-specific keywords
        for subcategory in template['subcategories']:
            if subcategory in content_lower:
                return subcategory
        
        # Content-based subcategory inference
        if main_theme == 'legal':
            if any(word in content_lower for word in ['payroll', 'earnings', 'salary', 'wage']):
                return 'payroll'
            elif any(word in content_lower for word in ['contract', 'agreement']):
                return 'contracts'
            else:
                return 'documentation'
        
        elif main_theme == 'ai':
            if any(word in content_lower for word in ['research', 'paper', 'study']):
                return 'research'
            elif any(word in content_lower for word in ['test', 'testing', 'validation']):
                return 'testing'
            elif any(word in content_lower for word in ['deploy', 'production', 'live']):
                return 'deployment'
            else:
                return 'development'
        
        elif main_theme == 'development':
            if any(word in content_lower for word in ['frontend', 'ui', 'interface', 'css', 'html']):
                return 'frontend'
            elif any(word in content_lower for word in ['backend', 'server', 'api', 'database']):
                return 'backend'
            elif any(word in content_lower for word in ['test', 'testing', 'unit', 'integration']):
                return 'testing'
            else:
                return 'general'
        
        # Default to first subcategory
        return template['subcategories'][0] if template['subcategories'] else 'general'
    
    def assign_project_id(self, memory_block: MemoryBlock) -> str:
        """Assign project ID to a MemoryBlock"""
        
        # Analyze content themes
        theme_scores = self.analyze_content_themes(memory_block.content, memory_block.topics)
        
        # Find dominant theme
        if theme_scores:
            dominant_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
            max_score = theme_scores[dominant_theme]
        else:
            dominant_theme = 'general'
            max_score = 0.0
        
        # If no clear theme, use file-based inference
        if max_score < 0.5:
            dominant_theme = self._infer_theme_from_file(memory_block)
        
        # Determine subcategory
        subcategory = self.determine_subcategory(memory_block.content, dominant_theme)
        
        # Get project prefix
        prefix = self.PROJECT_TEMPLATES.get(dominant_theme, {}).get('prefix', dominant_theme)
        
        # Create project ID
        project_id = f"{prefix}.{subcategory}"
        
        # Update statistics
        self.project_stats[project_id] += 1
        
        return project_id
    
    def _infer_theme_from_file(self, memory_block: MemoryBlock) -> str:
        """Infer theme from file path and type"""
        source_path = memory_block.metadata.source_file.lower()
        file_type = memory_block.metadata.file_type.lower()
        
        # Path-based inference
        if 'legal' in source_path or 'contract' in source_path:
            return 'legal'
        elif 'blog' in source_path or 'article' in source_path:
            return 'blog'
        elif 'ai' in source_path or 'ml' in source_path:
            return 'ai'
        elif 'business' in source_path or 'company' in source_path:
            return 'business'
        elif 'research' in source_path or 'study' in source_path:
            return 'research'
        elif 'doc' in source_path or 'readme' in source_path:
            return 'documentation'
        
        # File type-based inference
        if file_type in ['.py', '.js', '.ts', '.java', '.cpp']:
            return 'development'
        elif file_type in ['.md', '.txt', '.rst']:
            return 'documentation'
        elif file_type in ['.pdf', '.doc', '.docx']:
            return 'documentation'
        elif file_type in ['.jpg', '.png', '.gif']:
            return 'documentation'  # Assume images are for docs unless proven otherwise
        
        return 'general'


class ArchetypeAnalyzer:
    """Analyzes and refines MUSE archetype assignments"""
    
    # Enhanced archetype definitions
    ARCHETYPE_PROFILES = {
        'Guardian': {
            'keywords': ['protect', 'security', 'safety', 'guard', 'defend', 'shield', 'privacy', 'ethics'],
            'content_patterns': ['compliance', 'regulation', 'policy', 'privacy', 'ethics', 'security'],
            'file_patterns': ['security', 'privacy', 'compliance', 'ethics']
        },
        'Visionary': {
            'keywords': ['future', 'vision', 'strategy', 'plan', 'roadmap', 'innovation', 'direction'],
            'content_patterns': ['strategy', 'planning', 'vision', 'future', 'roadmap', 'goals'],
            'file_patterns': ['strategy', 'vision', 'roadmap', 'planning']
        },
        'Analyst': {
            'keywords': ['analyze', 'data', 'metrics', 'statistics', 'research', 'study', 'evaluate'],
            'content_patterns': ['analysis', 'data', 'metrics', 'statistics', 'research', 'findings'],
            'file_patterns': ['analysis', 'data', 'metrics', 'research', 'stats']
        },
        'Creator': {
            'keywords': ['create', 'build', 'design', 'develop', 'make', 'construct', 'generate'],
            'content_patterns': ['development', 'creation', 'building', 'design', 'implementation'],
            'file_patterns': ['create', 'build', 'design', 'develop']
        },
        'Connector': {
            'keywords': ['connect', 'link', 'integrate', 'bridge', 'network', 'relationship', 'join'],
            'content_patterns': ['integration', 'connection', 'networking', 'linking', 'bridging'],
            'file_patterns': ['integration', 'connector', 'bridge', 'link']
        },
        'Optimizer': {
            'keywords': ['optimize', 'improve', 'enhance', 'efficiency', 'performance', 'streamline'],
            'content_patterns': ['optimization', 'improvement', 'efficiency', 'performance', 'enhancement'],
            'file_patterns': ['optimize', 'improve', 'performance', 'efficiency']
        },
        'Explorer': {
            'keywords': ['explore', 'discover', 'investigate', 'search', 'find', 'uncover', 'experiment'],
            'content_patterns': ['exploration', 'discovery', 'investigation', 'research', 'experimentation'],
            'file_patterns': ['explore', 'discover', 'research', 'experiment']
        },
        'Synthesizer': {
            'keywords': ['combine', 'merge', 'integrate', 'synthesize', 'unify', 'consolidate'],
            'content_patterns': ['synthesis', 'integration', 'combination', 'merging', 'consolidation'],
            'file_patterns': ['synthesis', 'merge', 'integrate', 'combine']
        },
        'Protector': {
            'keywords': ['protect', 'preserve', 'maintain', 'backup', 'save', 'archive', 'conserve'],
            'content_patterns': ['protection', 'preservation', 'maintenance', 'backup', 'archival'],
            'file_patterns': ['backup', 'archive', 'preserve', 'maintain']
        },
        'Innovator': {
            'keywords': ['innovate', 'new', 'novel', 'creative', 'breakthrough', 'revolutionary'],
            'content_patterns': ['innovation', 'creativity', 'novelty', 'breakthrough', 'new approach'],
            'file_patterns': ['innovation', 'new', 'creative', 'novel']
        }
    }
    
    def analyze_archetype(self, memory_block: MemoryBlock) -> str:
        """Analyze and potentially refine archetype assignment"""
        content_lower = memory_block.content.lower()
        source_path = memory_block.metadata.source_file.lower()
        
        archetype_scores = {}
        
        for archetype, profile in self.ARCHETYPE_PROFILES.items():
            score = 0.0
            
            # Keyword matching in content
            for keyword in profile['keywords']:
                if keyword in content_lower:
                    score += 2.0
            
            # Content pattern matching
            for pattern in profile['content_patterns']:
                if pattern in content_lower:
                    score += 1.5
            
            # File path pattern matching
            for pattern in profile['file_patterns']:
                if pattern in source_path:
                    score += 1.0
            
            archetype_scores[archetype] = score
        
        # Find best matching archetype
        if archetype_scores:
            best_archetype = max(archetype_scores.items(), key=lambda x: x[1])[0]
            best_score = archetype_scores[best_archetype]
            
            # Only override if we have a strong match
            if best_score >= 2.0:
                return best_archetype
        
        # Return original archetype if no strong match
        return memory_block.archetype


class ProjectAssignmentPipeline:
    """Main pipeline for project assignment and archetype refinement"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.project_mapper = ProjectMapper()
        self.archetype_analyzer = ArchetypeAnalyzer()
    
    def load_cluster_data(self, clusters_file: Path) -> Optional[Dict[str, Any]]:
        """Load cluster assignments if available"""
        if not clusters_file or not clusters_file.exists():
            return None
        
        try:
            with open(clusters_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cluster data: {e}")
            return None
    
    def assign_projects(
        self, 
        memory_blocks: List[MemoryBlock], 
        cluster_data: Optional[Dict[str, Any]] = None
    ) -> List[MemoryBlock]:
        """Assign projects to MemoryBlocks and refine archetypes"""
        
        updated_blocks = []
        
        print(f"Assigning projects to {len(memory_blocks)} MemoryBlocks...")
        
        for i, block in enumerate(memory_blocks):
            if i % 100 == 0 and i > 0:
                print(f"Progress: {i}/{len(memory_blocks)} blocks processed")
            
            # Create a copy of the block for modification
            updated_block = MemoryBlock(
                id_hash=block.id_hash,
                summary=block.summary,
                content=block.content,
                topics=block.topics,
                skills=block.skills,
                project=block.project,  # Will be updated
                archetype=block.archetype,  # Will be refined
                created_at=block.created_at,
                ethics=block.ethics,
                metadata=block.metadata,
                links=block.links
            )
            
            # Assign project ID
            new_project_id = self.project_mapper.assign_project_id(updated_block)
            updated_block.project = new_project_id
            
            # Refine archetype
            refined_archetype = self.archetype_analyzer.analyze_archetype(updated_block)
            updated_block.archetype = refined_archetype
            
            updated_blocks.append(updated_block)
        
        return updated_blocks
    
    def generate_project_hierarchy(self, updated_blocks: List[MemoryBlock]) -> Dict[str, Any]:
        """Generate project hierarchy and statistics"""
        
        project_hierarchy = defaultdict(lambda: defaultdict(list))
        archetype_distribution = Counter()
        project_stats = Counter()
        
        for block in updated_blocks:
            # Parse project ID
            project_parts = block.project.split('.')
            main_project = project_parts[0] if project_parts else 'general'
            sub_project = project_parts[1] if len(project_parts) > 1 else 'general'
            
            # Build hierarchy
            project_hierarchy[main_project][sub_project].append({
                'id_hash': block.id_hash,
                'summary': block.summary[:100] + "..." if len(block.summary) > 100 else block.summary,
                'archetype': block.archetype,
                'source_file': block.metadata.source_file
            })
            
            # Update statistics
            archetype_distribution[block.archetype] += 1
            project_stats[block.project] += 1
        
        # Convert defaultdict to regular dict for JSON serialization
        hierarchy_dict = {}
        for main_project, sub_projects in project_hierarchy.items():
            hierarchy_dict[main_project] = dict(sub_projects)
        
        return {
            'hierarchy': hierarchy_dict,
            'archetype_distribution': dict(archetype_distribution),
            'project_statistics': dict(project_stats),
            'total_blocks': len(updated_blocks)
        }
    
    def run_assignment(
        self, 
        blocks_dir: Path, 
        clusters_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run complete project assignment pipeline"""
        
        print(f"Loading MemoryBlocks from {blocks_dir}")
        memory_blocks = load_all_memory_blocks(blocks_dir)
        
        if not memory_blocks:
            print("No MemoryBlocks found to process")
            return {'status': 'no_blocks'}
        
        print(f"Loaded {len(memory_blocks)} MemoryBlocks")
        
        # Load cluster data if available
        cluster_data = self.load_cluster_data(clusters_file) if clusters_file else None
        if cluster_data:
            print("Using cluster data for enhanced project assignment")
        
        # Assign projects and refine archetypes
        updated_blocks = self.assign_projects(memory_blocks, cluster_data)
        
        # Generate project hierarchy
        print("Generating project hierarchy...")
        project_hierarchy = self.generate_project_hierarchy(updated_blocks)
        
        # Save updated MemoryBlocks
        updated_blocks_dir = self.output_dir / "project_assignments" / "updated_memory_blocks"
        updated_blocks_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving updated MemoryBlocks to {updated_blocks_dir}")
        saved_count = 0
        for block in updated_blocks:
            try:
                save_memory_block(block, updated_blocks_dir)
                saved_count += 1
            except Exception as e:
                print(f"Error saving updated block {block.id_hash}: {e}")
        
        # Save project hierarchy
        assignments_dir = self.output_dir / "project_assignments"
        assignments_dir.mkdir(parents=True, exist_ok=True)
        
        hierarchy_file = assignments_dir / "project_hierarchy.json"
        with open(hierarchy_file, 'w') as f:
            json.dump(project_hierarchy, f, indent=2, default=str)
        
        # Save assignment summary
        summary = {
            'status': 'completed',
            'memory_blocks_processed': len(memory_blocks),
            'memory_blocks_updated': saved_count,
            'unique_projects': len(project_hierarchy['project_statistics']),
            'archetype_distribution': project_hierarchy['archetype_distribution'],
            'project_statistics': project_hierarchy['project_statistics'],
            'hierarchy_file': str(hierarchy_file),
            'updated_blocks_dir': str(updated_blocks_dir)
        }
        
        summary_file = assignments_dir / "assignment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*50)
        print("PROJECT ASSIGNMENT SUMMARY")
        print("="*50)
        print(f"MemoryBlocks processed: {summary['memory_blocks_processed']}")
        print(f"MemoryBlocks updated: {summary['memory_blocks_updated']}")
        print(f"Unique projects: {summary['unique_projects']}")
        print(f"Results saved to: {assignments_dir}")
        
        # Print top projects and archetypes
        print("\nTop Projects:")
        for project, count in sorted(project_hierarchy['project_statistics'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {project}: {count} blocks")
        
        print("\nArchetype Distribution:")
        for archetype, count in sorted(project_hierarchy['archetype_distribution'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {archetype}: {count} blocks")
        
        return summary


def main():
    """Main entry point for project assignment pipeline"""
    parser = argparse.ArgumentParser(description="Project Assignment Pipeline")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', default='./_work/pipeline_output',
                       help='Output directory for assignment results')
    parser.add_argument('--clusters-file', 
                       help='Optional cluster assignments file for enhanced mapping')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProjectAssignmentPipeline(args.output_dir)
    
    # Run assignment
    summary = pipeline.run_assignment(
        Path(args.blocks_dir), 
        Path(args.clusters_file) if args.clusters_file else None
    )
    
    # Save summary
    summary_path = Path(args.output_dir) / "assignment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAssignment summary saved to: {summary_path}")
    
    # Exit with appropriate code
    if summary['status'] == 'completed':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()