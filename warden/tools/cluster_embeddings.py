#!/usr/bin/env python3
"""
Clustering and Embeddings System for BLOGAGENT
Groups MemoryBlocks semantically using TF-IDF, K-Means, and PCA
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add parent directories to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_block import MemoryBlock, load_all_memory_blocks


class EmbeddingGenerator:
    """Generate embeddings for MemoryBlocks using TF-IDF"""
    
    def __init__(self):
        self.vectorizer = None
        self.feature_names = None
    
    def generate_embeddings(self, memory_blocks: List[MemoryBlock]) -> Tuple[np.ndarray, List[str]]:
        """Generate TF-IDF embeddings for memory blocks"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError("scikit-learn is required for embeddings. Install with: pip install scikit-learn")
        
        if not memory_blocks:
            return np.array([]), []
        
        # Combine content and topics for richer embeddings
        documents = []
        for block in memory_blocks:
            combined_text = f"{block.summary} {block.content} {' '.join(block.topics)}"
            documents.append(combined_text)
        
        # Generate TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        embeddings = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        return embeddings.toarray(), self.feature_names


class SemanticClusterer:
    """Cluster MemoryBlocks semantically using K-Means"""
    
    def __init__(self):
        self.cluster_model = None
        self.n_clusters = None
    
    def determine_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        """Determine optimal number of clusters using elbow method"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("scikit-learn is required for clustering. Install with: pip install scikit-learn")
        
        if len(embeddings) < 2:
            return 1
        
        max_clusters = min(max_clusters, len(embeddings) - 1)
        
        if max_clusters < 2:
            return 1
        
        # Try different numbers of clusters
        silhouette_scores = []
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find the number of clusters with the highest silhouette score
        best_idx = np.argmax(silhouette_scores)
        best_n_clusters = list(cluster_range)[best_idx]
        
        print(f"Optimal number of clusters: {best_n_clusters} (silhouette score: {silhouette_scores[best_idx]:.3f})")
        
        return best_n_clusters
    
    def cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """Cluster embeddings using K-Means"""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn is required for clustering. Install with: pip install scikit-learn")
        
        if len(embeddings) == 0:
            return np.array([])
        
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters(embeddings)
        
        self.n_clusters = n_clusters
        
        # Perform K-Means clustering
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.cluster_model.fit_predict(embeddings)
        
        return cluster_labels


class DimensionalityReducer:
    """Reduce dimensionality for visualization using PCA"""
    
    def __init__(self):
        self.pca_model = None
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce embeddings to 2D/3D for visualization"""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("scikit-learn is required for PCA. Install with: pip install scikit-learn")
        
        if len(embeddings) == 0:
            return np.array([])
        
        # Ensure we don't try to reduce to more components than we have features
        n_components = min(n_components, embeddings.shape[1], embeddings.shape[0])
        
        self.pca_model = PCA(n_components=n_components)
        reduced_embeddings = self.pca_model.fit_transform(embeddings)
        
        print(f"Reduced embeddings from {embeddings.shape[1]} to {n_components} dimensions")
        print(f"Explained variance ratio: {self.pca_model.explained_variance_ratio_}")
        
        return reduced_embeddings


class ClusterAnalyzer:
    """Analyze and characterize clusters"""
    
    def analyze_clusters(
        self, 
        memory_blocks: List[MemoryBlock], 
        cluster_labels: np.ndarray,
        feature_names: List[str],
        embeddings: np.ndarray
    ) -> Dict[int, Dict[str, Any]]:
        """Analyze characteristics of each cluster"""
        
        cluster_analysis = {}
        
        # Group blocks by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append((i, memory_blocks[i]))
        
        for cluster_id, block_indices_and_blocks in clusters.items():
            indices = [idx for idx, _ in block_indices_and_blocks]
            blocks = [block for _, block in block_indices_and_blocks]
            
            # Analyze cluster characteristics
            analysis = self._analyze_single_cluster(cluster_id, blocks, indices, feature_names, embeddings)
            cluster_analysis[cluster_id] = analysis
        
        return cluster_analysis
    
    def _analyze_single_cluster(
        self, 
        cluster_id: int, 
        blocks: List[MemoryBlock], 
        indices: List[int],
        feature_names: List[str],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze a single cluster"""
        
        # Common topics
        all_topics = []
        for block in blocks:
            all_topics.extend(block.topics)
        
        topic_counts = defaultdict(int)
        for topic in all_topics:
            topic_counts[topic] += 1
        
        common_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Common archetypes
        archetype_counts = defaultdict(int)
        for block in blocks:
            archetype_counts[block.archetype] += 1
        
        dominant_archetype = max(archetype_counts.items(), key=lambda x: x[1])[0]
        
        # Common projects
        project_counts = defaultdict(int)
        for block in blocks:
            project_counts[block.project] += 1
        
        # File types
        file_type_counts = defaultdict(int)
        for block in blocks:
            file_type_counts[block.metadata.file_type] += 1
        
        # Top TF-IDF features for this cluster
        cluster_embeddings = embeddings[indices]
        mean_embedding = np.mean(cluster_embeddings, axis=0)
        top_features_indices = np.argsort(mean_embedding)[-10:][::-1]
        top_features = [(feature_names[i], mean_embedding[i]) for i in top_features_indices if mean_embedding[i] > 0]
        
        # Generate cluster summary
        cluster_summary = self._generate_cluster_summary(blocks, common_topics, dominant_archetype)
        
        return {
            'cluster_id': cluster_id,
            'size': len(blocks),
            'summary': cluster_summary,
            'dominant_archetype': dominant_archetype,
            'common_topics': [topic for topic, count in common_topics],
            'top_tfidf_features': top_features[:5],
            'project_distribution': dict(project_counts),
            'file_type_distribution': dict(file_type_counts),
            'sample_blocks': [
                {
                    'id_hash': block.id_hash,
                    'summary': block.summary[:100] + "..." if len(block.summary) > 100 else block.summary,
                    'source': block.metadata.source_file
                } for block in blocks[:3]
            ]
        }
    
    def _generate_cluster_summary(
        self, 
        blocks: List[MemoryBlock], 
        common_topics: List[Tuple[str, int]], 
        dominant_archetype: str
    ) -> str:
        """Generate a descriptive summary for the cluster"""
        
        size = len(blocks)
        topics_str = ", ".join([topic for topic, _ in common_topics[:3]])
        
        # Sample file types
        file_types = list(set(block.metadata.file_type for block in blocks))
        file_types_str = ", ".join(file_types[:3])
        
        summary = f"Cluster of {size} files with {dominant_archetype} archetype. "
        summary += f"Common themes: {topics_str}. "
        summary += f"File types: {file_types_str}."
        
        return summary


class VisualizationGenerator:
    """Generate interactive visualizations for clusters"""
    
    def create_cluster_visualization(
        self,
        memory_blocks: List[MemoryBlock],
        reduced_embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_analysis: Dict[int, Dict[str, Any]],
        output_path: Path
    ):
        """Create interactive cluster visualization using Plotly"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.offline import plot
        except ImportError:
            print("Plotly not available for visualization. Install with: pip install plotly")
            return
        
        if len(reduced_embeddings) == 0:
            print("No data to visualize")
            return
        
        # Prepare data for visualization
        x_coords = reduced_embeddings[:, 0]
        y_coords = reduced_embeddings[:, 1] if reduced_embeddings.shape[1] > 1 else np.zeros(len(x_coords))
        
        # Create hover text
        hover_texts = []
        for i, block in enumerate(memory_blocks):
            hover_text = f"<b>{block.metadata.source_file}</b><br>"
            hover_text += f"Archetype: {block.archetype}<br>"
            hover_text += f"Project: {block.project}<br>"
            hover_text += f"Topics: {', '.join(block.topics[:3])}<br>"
            hover_text += f"Summary: {block.summary[:150]}..."
            hover_texts.append(hover_text)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add points colored by cluster
        unique_clusters = np.unique(cluster_labels)
        colors = px.colors.qualitative.Set3[:len(unique_clusters)]
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_name = f"Cluster {cluster_id}"
            
            if cluster_id in cluster_analysis:
                cluster_name += f": {cluster_analysis[cluster_id]['summary'][:50]}..."
            
            fig.add_trace(go.Scatter(
                x=x_coords[cluster_mask],
                y=y_coords[cluster_mask],
                mode='markers',
                name=cluster_name,
                text=[hover_texts[j] for j in range(len(hover_texts)) if cluster_mask[j]],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title="MemoryBlock Semantic Clusters",
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            hovermode='closest',
            width=1000,
            height=700
        )
        
        # Save visualization
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot(fig, filename=str(output_path), auto_open=False)
        print(f"Cluster visualization saved to: {output_path}")


class ClusteringPipeline:
    """Main clustering pipeline"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.embedder = EmbeddingGenerator()
        self.clusterer = SemanticClusterer()
        self.reducer = DimensionalityReducer()
        self.analyzer = ClusterAnalyzer()
        self.visualizer = VisualizationGenerator()
    
    def run_clustering(self, blocks_dir: Path, assign_projects: bool = False) -> Dict[str, Any]:
        """Run complete clustering pipeline"""
        
        print(f"Loading MemoryBlocks from {blocks_dir}")
        memory_blocks = load_all_memory_blocks(blocks_dir)
        
        if not memory_blocks:
            print("No MemoryBlocks found to cluster")
            return {'status': 'no_blocks'}
        
        print(f"Loaded {len(memory_blocks)} MemoryBlocks")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings, feature_names = self.embedder.generate_embeddings(memory_blocks)
        
        if len(embeddings) == 0:
            print("No embeddings generated")
            return {'status': 'no_embeddings'}
        
        # Perform clustering
        print("Performing clustering...")
        cluster_labels = self.clusterer.cluster_embeddings(embeddings)
        
        # Reduce dimensions for visualization
        print("Reducing dimensions for visualization...")
        reduced_embeddings = self.reducer.reduce_dimensions(embeddings, n_components=2)
        
        # Analyze clusters
        print("Analyzing clusters...")
        cluster_analysis = self.analyzer.analyze_clusters(
            memory_blocks, cluster_labels, feature_names, embeddings
        )
        
        # Save results
        results_dir = self.output_dir / "clustering_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster analysis (convert numpy types for JSON serialization)
        analysis_file = results_dir / "cluster_analysis.json"
        serializable_analysis = {}
        for k, v in cluster_analysis.items():
            serializable_analysis[str(k)] = v
        with open(analysis_file, 'w') as f:
            json.dump(serializable_analysis, f, indent=2, default=str)
        
        # Save cluster assignments
        cluster_assignments = []
        for i, (block, label) in enumerate(zip(memory_blocks, cluster_labels)):
            cluster_assignments.append({
                'id_hash': block.id_hash,
                'cluster_id': int(label),
                'source_file': block.metadata.source_file
            })
        
        assignments_file = results_dir / "cluster_assignments.json"
        with open(assignments_file, 'w') as f:
            json.dump(cluster_assignments, f, indent=2)
        
        # Generate visualization
        print("Generating visualization...")
        viz_file = results_dir / "cluster_visualization.html"
        self.visualizer.create_cluster_visualization(
            memory_blocks, reduced_embeddings, cluster_labels, cluster_analysis, viz_file
        )
        
        # Summary
        summary = {
            'status': 'completed',
            'memory_blocks_processed': len(memory_blocks),
            'clusters_created': len(cluster_analysis),
            'embedding_dimensions': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'cluster_analysis_file': str(analysis_file),
            'cluster_assignments_file': str(assignments_file),
            'visualization_file': str(viz_file) if viz_file.exists() else None
        }
        
        print("\n" + "="*50)
        print("CLUSTERING SUMMARY")
        print("="*50)
        print(f"MemoryBlocks processed: {summary['memory_blocks_processed']}")
        print(f"Clusters created: {summary['clusters_created']}")
        print(f"Embedding dimensions: {summary['embedding_dimensions']}")
        print(f"Results saved to: {results_dir}")
        
        return summary


def main():
    """Main entry point for clustering pipeline"""
    parser = argparse.ArgumentParser(description="MemoryBlock Clustering Pipeline")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', default='./_work/pipeline_output',
                       help='Output directory for clustering results')
    parser.add_argument('--assign-projects', action='store_true',
                       help='Also assign project IDs based on clusters')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ClusteringPipeline(args.output_dir)
    
    # Run clustering
    summary = pipeline.run_clustering(Path(args.blocks_dir), args.assign_projects)
    
    # Save summary
    summary_path = Path(args.output_dir) / "clustering_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nClustering summary saved to: {summary_path}")
    
    # Exit with appropriate code
    if summary['status'] == 'completed':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()