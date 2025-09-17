#!/usr/bin/env python3
"""
MUSE Pantheon Project Assignment System
Auto-assigns project IDs to MemoryBlocks based on content themes and archetype mapping
"""
import json
import pathlib
import argparse
from typing import Dict, List, Any, Set
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectAssigner:
    """Assigns project IDs to MemoryBlocks based on content analysis and MUSE archetypes"""
    
    def __init__(self, output_dir: str):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_blocks = []
        self.clusters = {}
        self.project_hierarchy = {}
        self.archetype_projects = {
            'Vision': ['media.visual_assets', 'design.ui_ux', 'analysis.image_data'],
            'Builder': ['development.codebase', 'engineering.tools', 'infrastructure.systems'],
            'Sage': ['knowledge.documentation', 'research.studies', 'wisdom.insights'],
            'Guardian': ['security.protocols', 'compliance.legal', 'ethics.guidelines'],
            'Connector': ['integration.apis', 'communication.channels', 'network.services'],
            'default': ['general.miscellaneous', 'uncategorized.items']
        }
        
    def load_memory_blocks(self, blocks_dir: str) -> int:
        """Load all MemoryBlocks from JSON files"""
        blocks_path = pathlib.Path(blocks_dir)
        if not blocks_path.exists():
            logger.error(f"Blocks directory not found: {blocks_dir}")
            return 0
            
        json_files = list(blocks_path.glob("*.json"))
        logger.info(f"Loading {len(json_files)} MemoryBlocks for project assignment...")
        
        loaded_count = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    block_data = json.load(f)
                    self.memory_blocks.append(block_data)
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Successfully loaded {loaded_count} MemoryBlocks")
        return loaded_count
    
    def load_clusters(self, clusters_file: str) -> bool:
        """Load cluster information if available"""
        clusters_path = pathlib.Path(clusters_file)
        if not clusters_path.exists():
            logger.warning(f"Clusters file not found: {clusters_file}")
            return False
            
        try:
            with open(clusters_path, 'r') as f:
                self.clusters = json.load(f)
            logger.info(f"Loaded {len(self.clusters)} clusters")
            return True
        except Exception as e:
            logger.error(f"Error loading clusters: {e}")
            return False
    
    def analyze_content_themes(self, block: Dict[str, Any]) -> Set[str]:
        """Analyze content to extract themes for project assignment"""
        themes = set()
        
        # Analyze content text
        content = (block.get('content', '') + ' ' + block.get('summary', '')).lower()
        
        # Theme classification rules
        theme_keywords = {
            'legal': ['contract', 'agreement', 'legal', 'terms', 'compliance', 'regulation'],
            'financial': ['payment', 'invoice', 'cost', 'budget', 'financial', 'money', 'revenue'],
            'technical': ['code', 'function', 'class', 'api', 'algorithm', 'technical', 'programming'],
            'documentation': ['manual', 'guide', 'documentation', 'readme', 'instructions'],
            'design': ['design', 'ui', 'ux', 'interface', 'mockup', 'prototype'],
            'analysis': ['analysis', 'report', 'data', 'metrics', 'statistics', 'insights'],
            'communication': ['email', 'message', 'communication', 'correspondence'],
            'media': ['image', 'video', 'audio', 'media', 'visual', 'graphic'],
            'project_management': ['project', 'milestone', 'timeline', 'deliverable', 'task'],
            'research': ['research', 'study', 'investigation', 'experiment', 'hypothesis']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content for keyword in keywords):
                themes.add(theme)
        
        # Analyze file type
        file_type = block.get('metadata', {}).get('file_type', '').lower()
        file_type_themes = {
            '.py': 'technical',
            '.js': 'technical',
            '.ts': 'technical',
            '.md': 'documentation',
            '.pdf': 'documentation',
            '.jpg': 'media',
            '.jpeg': 'media',
            '.png': 'media',
            '.json': 'technical',
            '.csv': 'analysis'
        }
        
        if file_type in file_type_themes:
            themes.add(file_type_themes[file_type])
        
        # Analyze topics
        for topic in block.get('topics', []):
            if topic in theme_keywords:
                themes.add(topic)
        
        return themes
    
    def suggest_project_id(self, block: Dict[str, Any], themes: Set[str]) -> str:
        """Suggest a project ID based on themes and archetype"""
        archetype = block.get('archetype', 'default')
        
        # Priority-based project assignment
        project_rules = {
            'legal': 'compliance.legal_documents',
            'financial': 'business.financial_records',
            'technical': 'development.technical_assets',
            'documentation': 'knowledge.documentation',
            'design': 'creative.design_assets',
            'analysis': 'insights.data_analysis',
            'communication': 'operations.communications',
            'media': 'content.media_library',
            'project_management': 'operations.project_tracking',
            'research': 'knowledge.research_studies'
        }
        
        # Find highest priority theme
        for theme in ['legal', 'financial', 'technical', 'design', 'analysis']:
            if theme in themes:
                return project_rules[theme]
        
        # Fall back to other themes
        for theme in themes:
            if theme in project_rules:
                return project_rules[theme]
        
        # Fall back to archetype-based assignment
        if archetype in self.archetype_projects:
            return self.archetype_projects[archetype][0]
        
        return 'general.uncategorized'
    
    def create_project_hierarchy(self) -> Dict[str, Any]:
        """Create a hierarchical project structure"""
        hierarchy = {
            'compliance': {
                'description': 'Legal, regulatory, and compliance materials',
                'projects': ['legal_documents', 'regulatory_filings', 'contracts']
            },
            'business': {
                'description': 'Business operations and financial records',
                'projects': ['financial_records', 'business_plans', 'operational_docs']
            },
            'development': {
                'description': 'Technical development and engineering',
                'projects': ['technical_assets', 'codebase', 'system_architecture']
            },
            'knowledge': {
                'description': 'Documentation, research, and knowledge base',
                'projects': ['documentation', 'research_studies', 'knowledge_base']
            },
            'creative': {
                'description': 'Design, media, and creative assets',
                'projects': ['design_assets', 'media_library', 'brand_materials']
            },
            'insights': {
                'description': 'Analysis, reporting, and data insights',
                'projects': ['data_analysis', 'reports', 'metrics_tracking']
            },
            'operations': {
                'description': 'Day-to-day operations and communications',
                'projects': ['communications', 'project_tracking', 'administrative']
            },
            'content': {
                'description': 'Content creation and management',
                'projects': ['media_library', 'written_content', 'multimedia']
            },
            'general': {
                'description': 'Miscellaneous and uncategorized items',
                'projects': ['uncategorized', 'temporary', 'archived']
            }
        }
        
        self.project_hierarchy = hierarchy
        return hierarchy
    
    def assign_projects_to_blocks(self) -> Dict[str, Any]:
        """Assign project IDs to all memory blocks"""
        logger.info("Starting project assignment...")
        
        # Create project hierarchy
        self.create_project_hierarchy()
        
        assignment_stats = {
            'total_blocks': len(self.memory_blocks),
            'assignments': {},
            'themes_found': set(),
            'archetypes_used': set()
        }
        
        # Process each memory block
        for i, block in enumerate(self.memory_blocks):
            # Analyze themes
            themes = self.analyze_content_themes(block)
            assignment_stats['themes_found'].update(themes)
            
            # Get archetype
            archetype = block.get('archetype', 'default')
            assignment_stats['archetypes_used'].add(archetype)
            
            # Suggest project ID
            suggested_project = self.suggest_project_id(block, themes)
            
            # Update block with new project assignment
            block['project_suggestion'] = {
                'match_type': 'auto_assigned',
                'project_id': suggested_project,
                'themes': list(themes),
                'confidence': self._calculate_confidence(themes, archetype)
            }
            
            # Track assignment stats
            if suggested_project not in assignment_stats['assignments']:
                assignment_stats['assignments'][suggested_project] = 0
            assignment_stats['assignments'][suggested_project] += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(self.memory_blocks)} blocks...")
        
        # Convert sets to lists for JSON serialization
        assignment_stats['themes_found'] = list(assignment_stats['themes_found'])
        assignment_stats['archetypes_used'] = list(assignment_stats['archetypes_used'])
        
        logger.info(f"Project assignment complete. {len(assignment_stats['assignments'])} projects assigned.")
        
        return assignment_stats
    
    def _calculate_confidence(self, themes: Set[str], archetype: str) -> float:
        """Calculate confidence score for project assignment"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on number of themes
        confidence += min(len(themes) * 0.1, 0.3)
        
        # Increase confidence for specific archetypes
        if archetype in ['Vision', 'Builder', 'Sage', 'Guardian']:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def update_memory_blocks_with_projects(self, blocks_dir: str) -> int:
        """Update the original MemoryBlock files with project assignments"""
        logger.info("Updating MemoryBlock files with project assignments...")
        
        updated_count = 0
        blocks_path = pathlib.Path(blocks_dir)
        
        for block in self.memory_blocks:
            # Find the corresponding JSON file
            block_hash = block.get('id_hash', '').replace(':', '_')
            json_file = blocks_path / f"{block_hash}.json"
            
            if json_file.exists():
                try:
                    with open(json_file, 'w') as f:
                        json.dump(block, f, indent=2)
                    updated_count += 1
                except Exception as e:
                    logger.error(f"Error updating {json_file}: {e}")
        
        logger.info(f"Updated {updated_count} MemoryBlock files")
        return updated_count
    
    def save_results(self, assignment_stats: Dict[str, Any]) -> Dict[str, str]:
        """Save project assignment results"""
        results = {}
        
        # Save project hierarchy
        hierarchy_file = self.output_dir / "project_hierarchy.json"
        with open(hierarchy_file, 'w') as f:
            json.dump(self.project_hierarchy, f, indent=2)
        results["hierarchy_file"] = str(hierarchy_file)
        
        # Save assignment statistics
        stats_file = self.output_dir / "assignment_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(assignment_stats, f, indent=2)
        results["stats_file"] = str(stats_file)
        
        # Save updated memory blocks
        updated_blocks_file = self.output_dir / "updated_memory_blocks.json"
        with open(updated_blocks_file, 'w') as f:
            json.dump(self.memory_blocks, f, indent=2, default=str)
        results["updated_blocks_file"] = str(updated_blocks_file)
        
        # Save summary report
        summary_file = self.output_dir / "assignment_summary.json"
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_blocks_processed": assignment_stats['total_blocks'],
            "unique_projects_assigned": len(assignment_stats['assignments']),
            "themes_discovered": len(assignment_stats['themes_found']),
            "archetypes_used": len(assignment_stats['archetypes_used']),
            "top_projects": sorted(assignment_stats['assignments'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        results["summary_file"] = str(summary_file)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Project Assignment System")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for assignment results')
    parser.add_argument('--clusters-file',
                       help='Optional clusters file for enhanced assignment')
    parser.add_argument('--update-blocks', action='store_true',
                       help='Update original MemoryBlock files with assignments')
    
    args = parser.parse_args()
    
    # Initialize assigner
    assigner = ProjectAssigner(args.output_dir)
    
    # Load memory blocks
    loaded_count = assigner.load_memory_blocks(args.blocks_dir)
    if loaded_count == 0:
        logger.error("No MemoryBlocks loaded. Exiting.")
        return
    
    # Load clusters if available
    if args.clusters_file:
        assigner.load_clusters(args.clusters_file)
    
    # Assign projects
    assignment_stats = assigner.assign_projects_to_blocks()
    
    # Update original files if requested
    if args.update_blocks:
        updated_count = assigner.update_memory_blocks_with_projects(args.blocks_dir)
        assignment_stats['updated_files'] = updated_count
    
    # Save results
    output_files = assigner.save_results(assignment_stats)
    
    print(f"✅ Project assignment complete!")
    print(f"📊 Processed {assignment_stats['total_blocks']} MemoryBlocks")
    print(f"📁 Assigned to {len(assignment_stats['assignments'])} projects")
    print(f"🏷️ Found {len(assignment_stats['themes_found'])} unique themes")
    print(f"📄 Results saved to: {args.output_dir}")
    print(f"📈 Top projects: {dict(sorted(assignment_stats['assignments'].items(), key=lambda x: x[1], reverse=True)[:5])}")

if __name__ == "__main__":
    main()