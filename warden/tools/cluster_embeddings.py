#!/usr/bin/env python3
"""
MUSE Pantheon Clustering and Embeddings System
Semantic clustering using TF-IDF vectorization and K-Means clustering.
"""
import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from memory_block import MemoryBlock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional dependencies with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. Clustering will use basic text similarity.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available. No interactive visualizations will be generated.")


class BasicTextSimilarity:
    """Fallback clustering when sklearn is not available"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.clusters = {}
    
    def fit_predict(self, texts: List[str]) -> List[int]:
        """Basic clustering using text similarity"""
        # Simple clustering based on text length and common words
        clusters = []
        cluster_centers = []
        
        for i, text in enumerate(texts):
            words = set(text.lower().split())
            assigned = False
            
            # Try to assign to existing cluster
            for j, center_words in enumerate(cluster_centers):
                overlap = len(words.intersection(center_words))
                if overlap > 3:  # Simple threshold
                    clusters.append(j)
                    assigned = True
                    break
            
            # Create new cluster if needed
            if not assigned:
                if len(cluster_centers) < self.n_clusters:
                    cluster_centers.append(words)
                    clusters.append(len(cluster_centers) - 1)
                else:
                    # Assign to random existing cluster
                    clusters.append(i % self.n_clusters)
        
        return clusters


class MemoryBlockClusterer:
    """Semantic clustering for MemoryBlocks"""
    
    def __init__(self, n_clusters: int = 8, use_advanced: bool = True):
        self.n_clusters = n_clusters
        self.use_advanced = use_advanced and SKLEARN_AVAILABLE
        self.vectorizer = None
        self.clusterer = None
        self.pca = None
        self.cluster_labels = None
        self.embeddings = None
        
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
    
    def prepare_texts(self, blocks: List[MemoryBlock]) -> List[str]:
        """Prepare text content for clustering"""
        texts = []
        for block in blocks:
            # Combine summary, content, and topics for clustering
            text_parts = [
                block.summary,
                block.content[:500],  # Limit content to avoid very long texts
                ' '.join(block.topics)
            ]
            combined_text = ' '.join(filter(None, text_parts))
            texts.append(combined_text)
        
        return texts
    
    def cluster_blocks(self, blocks: List[MemoryBlock]) -> Dict[str, Any]:
        """Perform clustering on MemoryBlocks"""
        logger.info("Starting clustering process")
        
        if not blocks:
            logger.warning("No blocks to cluster")
            return self._empty_result()
        
        texts = self.prepare_texts(blocks)
        
        if self.use_advanced:
            return self._advanced_clustering(blocks, texts)
        else:
            return self._basic_clustering(blocks, texts)
    
    def _advanced_clustering(self, blocks: List[MemoryBlock], texts: List[str]) -> Dict[str, Any]:
        """Advanced clustering using sklearn"""
        logger.info("Using advanced clustering with sklearn")
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"TF-IDF vectorization failed: {e}")
            return self._basic_clustering(blocks, texts)
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(tfidf_matrix)
        if optimal_k:
            self.n_clusters = optimal_k
        
        # K-Means Clustering
        self.clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        try:
            self.cluster_labels = self.clusterer.fit_predict(tfidf_matrix)
            logger.info(f"Clustering complete with {self.n_clusters} clusters")
        except Exception as e:
            logger.error(f"K-means clustering failed: {e}")
            return self._basic_clustering(blocks, texts)
        
        # Dimensionality reduction for visualization
        self.pca = PCA(n_components=2, random_state=42)
        self.embeddings = self.pca.fit_transform(tfidf_matrix.toarray())
        
        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(tfidf_matrix, self.cluster_labels)
            logger.info(f"Silhouette score: {silhouette_avg:.3f}")
        except:
            silhouette_avg = 0.0
        
        return self._format_clustering_results(blocks, silhouette_avg)
    
    def _basic_clustering(self, blocks: List[MemoryBlock], texts: List[str]) -> Dict[str, Any]:
        """Basic clustering fallback"""
        logger.info("Using basic clustering")
        
        basic_clusterer = BasicTextSimilarity(self.n_clusters)
        self.cluster_labels = basic_clusterer.fit_predict(texts)
        
        # Generate random 2D positions for visualization
        np.random.seed(42)
        self.embeddings = np.random.randn(len(blocks), 2)
        
        return self._format_clustering_results(blocks, 0.0)
    
    def _find_optimal_clusters(self, tfidf_matrix, max_k: int = 15) -> Optional[int]:
        """Find optimal number of clusters using elbow method"""
        if tfidf_matrix.shape[0] < 4:
            return None
        
        max_k = min(max_k, tfidf_matrix.shape[0] - 1)
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(tfidf_matrix)
                inertias.append(kmeans.inertia_)
            except:
                break
        
        if len(inertias) < 3:
            return None
        
        # Simple elbow detection
        deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        elbow_idx = deltas.index(max(deltas))
        optimal_k = k_range[elbow_idx]
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def _format_clustering_results(self, blocks: List[MemoryBlock], silhouette_score: float) -> Dict[str, Any]:
        """Format clustering results"""
        # Group blocks by cluster
        clusters = defaultdict(list)
        for i, (block, label) in enumerate(zip(blocks, self.cluster_labels)):
            cluster_info = {
                'block_id': block.id_hash,
                'summary': block.summary,
                'topics': block.topics,
                'archetype': block.project.archetype if block.project else 'Unknown',
                'file_path': block.source.file_path if block.source else 'Unknown',
                'embedding': self.embeddings[i].tolist() if self.embeddings is not None else [0, 0]
            }
            clusters[f"cluster_{label}"].append(cluster_info)
        
        # Generate cluster summaries
        cluster_summaries = {}
        for cluster_id, cluster_blocks in clusters.items():
            topics = []
            archetypes = []
            for block in cluster_blocks:
                topics.extend(block['topics'])
                archetypes.append(block['archetype'])
            
            # Most common topics and archetypes
            topic_counts = defaultdict(int)
            for topic in topics:
                topic_counts[topic] += 1
            
            archetype_counts = defaultdict(int)
            for archetype in archetypes:
                archetype_counts[archetype] += 1
            
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top_archetype = max(archetype_counts.items(), key=lambda x: x[1])[0] if archetype_counts else 'Unknown'
            
            cluster_summaries[cluster_id] = {
                'size': len(cluster_blocks),
                'primary_archetype': top_archetype,
                'top_topics': [topic for topic, count in top_topics],
                'description': f"Cluster of {len(cluster_blocks)} {top_archetype} blocks focusing on {', '.join([t for t, c in top_topics[:3]])}"
            }
        
        return {
            'clusters': dict(clusters),
            'cluster_summaries': cluster_summaries,
            'metadata': {
                'total_blocks': len(blocks),
                'num_clusters': self.n_clusters,
                'silhouette_score': silhouette_score,
                'clustering_method': 'advanced' if self.use_advanced else 'basic'
            }
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty clustering result"""
        return {
            'clusters': {},
            'cluster_summaries': {},
            'metadata': {
                'total_blocks': 0,
                'num_clusters': 0,
                'silhouette_score': 0.0,
                'clustering_method': 'none'
            }
        }
    
    def generate_visualizations(self, clustering_results: Dict[str, Any], output_dir: Path):
        """Generate interactive visualizations"""
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping visualizations.")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for visualization
        plot_data = []
        clusters = clustering_results['clusters']
        
        for cluster_id, blocks in clusters.items():
            for block in blocks:
                plot_data.append({
                    'x': block['embedding'][0],
                    'y': block['embedding'][1],
                    'cluster': cluster_id,
                    'archetype': block['archetype'],
                    'summary': block['summary'][:100] + "..." if len(block['summary']) > 100 else block['summary'],
                    'topics': ', '.join(block['topics'][:5])
                })
        
        if not plot_data:
            logger.warning("No data for visualization")
            return
        
        # Create cluster scatter plot
        fig = px.scatter(
            plot_data,
            x='x', y='y',
            color='cluster',
            symbol='archetype',
            hover_data=['summary', 'topics'],
            title='MemoryBlock Clustering Visualization',
            labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
        )
        
        fig.update_layout(
            width=1000,
            height=700,
            title_x=0.5
        )
        
        # Save interactive HTML
        html_path = output_dir / "cluster_visualization.html"
        fig.write_html(str(html_path))
        logger.info(f"Interactive visualization saved to {html_path}")
        
        # Create cluster summary visualization
        summary_data = []
        for cluster_id, summary in clustering_results['cluster_summaries'].items():
            summary_data.append({
                'cluster': cluster_id,
                'size': summary['size'],
                'archetype': summary['primary_archetype'],
                'description': summary['description']
            })
        
        if summary_data:
            fig_summary = px.bar(
                summary_data,
                x='cluster',
                y='size',
                color='archetype',
                title='Cluster Sizes by Primary Archetype',
                hover_data=['description']
            )
            
            summary_html_path = output_dir / "cluster_summary.html"
            fig_summary.write_html(str(summary_html_path))
            logger.info(f"Cluster summary visualization saved to {summary_html_path}")


class ClusteringPipeline:
    """Main clustering pipeline orchestrator"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clusterer = MemoryBlockClusterer()
    
    def run_clustering(self, blocks_dir: Path, assign_projects: bool = False):
        """Run the complete clustering pipeline"""
        logger.info("Starting clustering pipeline")
        
        # Load MemoryBlocks
        blocks = self.clusterer.load_memory_blocks(blocks_dir)
        if not blocks:
            logger.error("No MemoryBlocks found to cluster")
            return
        
        # Perform clustering
        clustering_results = self.clusterer.cluster_blocks(blocks)
        
        # Update blocks with cluster assignments if requested
        if assign_projects:
            self._assign_cluster_projects(blocks, clustering_results)
        
        # Save results
        self._save_results(clustering_results, blocks if assign_projects else None)
        
        # Generate visualizations
        self.clusterer.generate_visualizations(clustering_results, self.output_dir)
        
        logger.info("Clustering pipeline complete")
    
    def _assign_cluster_projects(self, blocks: List[MemoryBlock], clustering_results: Dict[str, Any]):
        """Assign cluster-based project IDs to blocks"""
        logger.info("Assigning cluster-based projects")
        
        # Create mapping from block ID to cluster
        block_to_cluster = {}
        for cluster_id, cluster_blocks in clustering_results['clusters'].items():
            for block_info in cluster_blocks:
                block_to_cluster[block_info['block_id']] = cluster_id
        
        # Update blocks with cluster project assignments
        for block in blocks:
            cluster_id = block_to_cluster.get(block.id_hash)
            if cluster_id:
                cluster_summary = clustering_results['cluster_summaries'][cluster_id]
                
                # Create project ID based on cluster archetype and topics
                archetype = cluster_summary['primary_archetype'].lower()
                top_topic = cluster_summary['top_topics'][0] if cluster_summary['top_topics'] else 'general'
                project_id = f"cluster.{archetype}.{top_topic}"
                
                block.assign_project(
                    project_id=project_id,
                    archetype=cluster_summary['primary_archetype'],
                    confidence=0.7,
                    reason=f"Clustered with {cluster_summary['size']} similar blocks"
                )
    
    def _save_results(self, clustering_results: Dict[str, Any], updated_blocks: Optional[List[MemoryBlock]] = None):
        """Save clustering results"""
        # Save clustering results
        results_file = self.output_dir / "clusters.json"
        with open(results_file, 'w') as f:
            json.dump(clustering_results, f, indent=2)
        
        # Save updated blocks if provided
        if updated_blocks:
            updated_blocks_dir = self.output_dir / "updated_blocks"
            updated_blocks_dir.mkdir(exist_ok=True)
            
            for block in updated_blocks:
                filename = f"memory_block_{block.id_hash.split(':')[1]}.json"
                block.save_to_file(updated_blocks_dir / filename)
        
        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MemoryBlock Clustering Pipeline")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for clustering results')
    parser.add_argument('--num-clusters', type=int, default=8,
                       help='Number of clusters (default: 8)')
    parser.add_argument('--assign-projects', action='store_true',
                       help='Update MemoryBlocks with cluster-based project assignments')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = ClusteringPipeline(args.output_dir)
    pipeline.clusterer.n_clusters = args.num_clusters
    pipeline.run_clustering(Path(args.blocks_dir), args.assign_projects)


if __name__ == "__main__":
    main()