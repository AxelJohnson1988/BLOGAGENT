#!/usr/bin/env python3
"""
MUSE Pantheon Clustering & Embeddings System
Semantic clustering and visualization of MemoryBlocks
"""
import os
import sys
import json
import argparse
import pathlib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
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


class MemoryBlockClusterer:
    """Performs semantic clustering and embedding of MemoryBlocks"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.vectorizer = None
        self.clusterer = None
        self.pca = None
    
    def load_memory_blocks(self, blocks_dir: pathlib.Path) -> List[MemoryBlock]:
        """Load all MemoryBlocks from directory"""
        memory_blocks = []
        
        for json_file in blocks_dir.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                memory_block = MemoryBlock.from_dict(data)
                memory_blocks.append(memory_block)
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
        
        logger.info(f"Loaded {len(memory_blocks)} MemoryBlocks")
        return memory_blocks
    
    def create_feature_vectors(self, memory_blocks: List[MemoryBlock]) -> Tuple[np.ndarray, List[str]]:
        """Create feature vectors from MemoryBlock content"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Combine summary, content, and topics for feature extraction
        documents = []
        for block in memory_blocks:
            # Combine text fields
            text_content = f"{block.summary} {block.content} {' '.join(block.topics)} {block.archetype}"
            documents.append(text_content)
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        feature_vectors = self.vectorizer.fit_transform(documents)
        feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Created feature vectors: {feature_vectors.shape}")
        return feature_vectors, feature_names
    
    def perform_clustering(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Perform K-means clustering"""
        from sklearn.cluster import KMeans
        
        # Adjust number of clusters based on data size
        n_samples = feature_vectors.shape[0]
        actual_clusters = min(self.n_clusters, max(2, n_samples // 5))
        
        self.clusterer = KMeans(
            n_clusters=actual_clusters,
            random_state=42,
            n_init=10
        )
        
        cluster_labels = self.clusterer.fit_predict(feature_vectors)
        
        logger.info(f"Clustering complete: {actual_clusters} clusters")
        return cluster_labels
    
    def reduce_dimensions(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Reduce dimensions for visualization using PCA"""
        from sklearn.decomposition import PCA
        
        # Reduce to 2D for visualization
        n_components = min(2, feature_vectors.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        
        reduced_vectors = self.pca.fit_transform(feature_vectors.toarray())
        
        logger.info(f"Dimension reduction complete: {reduced_vectors.shape}")
        return reduced_vectors
    
    def analyze_clusters(self, memory_blocks: List[MemoryBlock], cluster_labels: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Analyze cluster characteristics"""
        cluster_analysis = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_blocks = [block for i, block in enumerate(memory_blocks) if cluster_labels[i] == cluster_id]
            
            # Analyze topics
            all_topics = []
            for block in cluster_blocks:
                all_topics.extend(block.topics)
            
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Sort topics by frequency
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Analyze archetypes
            archetypes = [block.archetype for block in cluster_blocks]
            archetype_counts = {}
            for archetype in archetypes:
                archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
            
            # Analyze file types
            file_types = [block.source_type for block in cluster_blocks]
            type_counts = {}
            for file_type in file_types:
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster_blocks),
                'top_topics': top_topics,
                'archetypes': archetype_counts,
                'file_types': type_counts,
                'sample_blocks': [
                    {
                        'id_hash': block.id_hash,
                        'summary': block.summary[:100] + "..." if len(block.summary) > 100 else block.summary,
                        'archetype': block.archetype
                    }
                    for block in cluster_blocks[:3]  # First 3 blocks as samples
                ]
            }
        
        return cluster_analysis
    
    def create_cluster_visualization(self, memory_blocks: List[MemoryBlock], 
                                   reduced_vectors: np.ndarray, 
                                   cluster_labels: np.ndarray,
                                   output_path: pathlib.Path) -> str:
        """Create interactive cluster visualization"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.offline import plot
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'x': reduced_vectors[:, 0],
                'y': reduced_vectors[:, 1] if reduced_vectors.shape[1] > 1 else np.zeros(len(reduced_vectors)),
                'cluster': cluster_labels,
                'id_hash': [block.id_hash for block in memory_blocks],
                'summary': [block.summary[:100] + "..." if len(block.summary) > 100 else block.summary 
                           for block in memory_blocks],
                'archetype': [block.archetype for block in memory_blocks],
                'topics': [", ".join(block.topics[:3]) for block in memory_blocks],
                'source_file': [pathlib.Path(block.source_file).name for block in memory_blocks]
            })
            
            # Create scatter plot
            fig = px.scatter(
                df, x='x', y='y', color='cluster',
                hover_data=['id_hash', 'archetype', 'topics', 'source_file'],
                title='MUSE Pantheon Memory Blocks - Semantic Clustering',
                labels={'x': 'PC1', 'y': 'PC2', 'cluster': 'Cluster'}
            )
            
            # Customize layout
            fig.update_layout(
                width=1200,
                height=800,
                showlegend=True,
                title_font_size=16
            )
            
            # Save interactive plot
            html_file = output_path / 'cluster_visualization.html'
            plot(fig, filename=str(html_file), auto_open=False)
            
            logger.info(f"Cluster visualization saved: {html_file}")
            return str(html_file)
        
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            return ""
    
    def suggest_project_assignments(self, cluster_analysis: Dict[int, Dict[str, Any]]) -> Dict[int, str]:
        """Suggest project assignments based on cluster analysis"""
        project_suggestions = {}
        
        for cluster_id, analysis in cluster_analysis.items():
            top_topics = [topic for topic, count in analysis['top_topics']]
            archetypes = list(analysis['archetypes'].keys())
            
            # Project assignment logic based on content patterns
            if any(topic in top_topics for topic in ['legal', 'contract', 'law']):
                project_suggestions[cluster_id] = 'legal.documents'
            elif any(topic in top_topics for topic in ['financial', 'payment', 'tax', 'invoice']):
                project_suggestions[cluster_id] = 'financial.records'
            elif any(topic in top_topics for topic in ['code', 'programming', 'technical']):
                project_suggestions[cluster_id] = 'development.codebase'
            elif any(topic in top_topics for topic in ['research', 'study', 'analysis']):
                project_suggestions[cluster_id] = 'research.analysis'
            elif any(topic in top_topics for topic in ['communication', 'email', 'message']):
                project_suggestions[cluster_id] = 'communications.archive'
            elif any(topic in top_topics for topic in ['media', 'image', 'video', 'visual']):
                project_suggestions[cluster_id] = 'media.collection'
            elif any(topic in top_topics for topic in ['personal', 'diary', 'journal']):
                project_suggestions[cluster_id] = 'personal.journal'
            elif 'documentation' in top_topics:
                project_suggestions[cluster_id] = 'knowledge.base'
            else:
                # Default assignment based on archetype
                if 'Builder' in archetypes:
                    project_suggestions[cluster_id] = 'development.general'
                elif 'Vision' in archetypes:
                    project_suggestions[cluster_id] = 'media.visual'
                elif 'Guardian' in archetypes:
                    project_suggestions[cluster_id] = 'security.compliance'
                elif 'Scholar' in archetypes:
                    project_suggestions[cluster_id] = 'research.knowledge'
                else:
                    project_suggestions[cluster_id] = 'general.miscellaneous'
        
        return project_suggestions
    
    def save_clustering_results(self, output_dir: pathlib.Path, 
                              memory_blocks: List[MemoryBlock],
                              cluster_labels: np.ndarray,
                              cluster_analysis: Dict[int, Dict[str, Any]],
                              project_suggestions: Dict[int, str]) -> Dict[str, Any]:
        """Save clustering results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster assignments
        cluster_assignments = []
        for i, block in enumerate(memory_blocks):
            cluster_assignments.append({
                'id_hash': block.id_hash,
                'cluster_id': int(cluster_labels[i]),
                'suggested_project': project_suggestions.get(cluster_labels[i], 'general.miscellaneous')
            })
        
        assignments_file = output_dir / 'cluster_assignments.json'
        with open(assignments_file, 'w') as f:
            json.dump(cluster_assignments, f, indent=2)
        
        # Save cluster analysis
        analysis_file = output_dir / 'cluster_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(cluster_analysis, f, indent=2)
        
        # Save summary
        summary = {
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'total_blocks': len(memory_blocks),
            'num_clusters': len(np.unique(cluster_labels)),
            'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in np.unique(cluster_labels)],
            'project_suggestions': project_suggestions
        }
        
        summary_file = output_dir / 'clustering_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Clustering results saved to {output_dir}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Clustering & Embeddings System")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for clustering results')
    parser.add_argument('--clusters', type=int, default=5,
                       help='Number of clusters (default: 5)')
    parser.add_argument('--assign-projects', action='store_true',
                       help='Generate project assignment suggestions')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize clusterer
    clusterer = MemoryBlockClusterer(n_clusters=args.clusters)
    
    # Load MemoryBlocks
    blocks_dir = pathlib.Path(args.blocks_dir)
    if not blocks_dir.exists():
        logger.error(f"Blocks directory does not exist: {blocks_dir}")
        return 1
    
    memory_blocks = clusterer.load_memory_blocks(blocks_dir)
    if not memory_blocks:
        logger.error("No MemoryBlocks found")
        return 1
    
    # Create feature vectors
    feature_vectors, feature_names = clusterer.create_feature_vectors(memory_blocks)
    
    # Perform clustering
    cluster_labels = clusterer.perform_clustering(feature_vectors)
    
    # Reduce dimensions for visualization
    reduced_vectors = clusterer.reduce_dimensions(feature_vectors)
    
    # Analyze clusters
    cluster_analysis = clusterer.analyze_clusters(memory_blocks, cluster_labels)
    
    # Generate project suggestions
    project_suggestions = {}
    if args.assign_projects:
        project_suggestions = clusterer.suggest_project_assignments(cluster_analysis)
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    
    # Create visualization
    viz_file = clusterer.create_cluster_visualization(
        memory_blocks, reduced_vectors, cluster_labels, output_dir
    )
    
    # Save results
    summary = clusterer.save_clustering_results(
        output_dir, memory_blocks, cluster_labels, cluster_analysis, project_suggestions
    )
    
    # Print summary
    print("✅ Clustering analysis complete")
    print(f"📊 Processed {summary['total_blocks']} MemoryBlocks")
    print(f"🎯 Created {summary['num_clusters']} clusters")
    print(f"📁 Results saved to: {output_dir}")
    if viz_file:
        print(f"📈 Visualization: {viz_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())