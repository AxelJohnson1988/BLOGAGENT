#!/usr/bin/env python3
"""
MUSE Pantheon Clustering & Embeddings System
Performs semantic clustering and generates embeddings for MemoryBlocks
"""
import json
import pathlib
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import plotly.graph_objects as go
    import plotly.express as px
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn or plotly not available. Some features will be limited.")
    SKLEARN_AVAILABLE = False

class MemoryBlockClusterer:
    """Clusters MemoryBlocks based on semantic similarity"""
    
    def __init__(self, output_dir: str):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_blocks = []
        self.vectorizer = None
        self.vectors = None
        self.clusters = None
        
    def load_memory_blocks(self, blocks_dir: str) -> int:
        """Load all MemoryBlocks from JSON files"""
        blocks_path = pathlib.Path(blocks_dir)
        if not blocks_path.exists():
            logger.error(f"Blocks directory not found: {blocks_dir}")
            return 0
            
        json_files = list(blocks_path.glob("*.json"))
        logger.info(f"Loading {len(json_files)} MemoryBlocks...")
        
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
    
    def create_text_vectors(self) -> np.ndarray:
        """Create TF-IDF vectors from MemoryBlock content"""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available for vectorization")
            return None
            
        # Combine summary and content for vectorization
        texts = []
        for block in self.memory_blocks:
            text = f"{block.get('summary', '')} {block.get('content', '')} {' '.join(block.get('topics', []))}"
            texts.append(text)
        
        logger.info("Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        self.vectors = self.vectorizer.fit_transform(texts)
        logger.info(f"Created vectors with shape: {self.vectors.shape}")
        
        return self.vectors
    
    def perform_clustering(self, n_clusters: int = None) -> Dict[str, Any]:
        """Perform K-means clustering on the vectors"""
        if not SKLEARN_AVAILABLE or self.vectors is None:
            logger.error("Vectors not available for clustering")
            return {}
            
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            n_clusters = min(8, max(2, len(self.memory_blocks) // 5))
        
        logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.vectors)
        
        # Calculate silhouette score
        try:
            silhouette_avg = silhouette_score(self.vectors, cluster_labels)
            logger.info(f"Silhouette score: {silhouette_avg:.3f}")
        except:
            silhouette_avg = 0.0
        
        # Assign clusters to memory blocks
        clusters = {}
        for i, block in enumerate(self.memory_blocks):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "cluster_id": cluster_id,
                    "blocks": [],
                    "topics": [],
                    "archetypes": []
                }
            
            clusters[cluster_id]["blocks"].append(block)
            clusters[cluster_id]["topics"].extend(block.get('topics', []))
            clusters[cluster_id]["archetypes"].append(block.get('archetype', 'default'))
        
        # Generate cluster summaries
        for cluster_id, cluster_data in clusters.items():
            cluster_data["summary"] = self._generate_cluster_summary(cluster_data)
            cluster_data["dominant_topics"] = self._get_dominant_topics(cluster_data["topics"])
            cluster_data["dominant_archetype"] = max(set(cluster_data["archetypes"]), 
                                                   key=cluster_data["archetypes"].count)
        
        self.clusters = clusters
        
        return {
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_avg,
            "clusters": clusters
        }
    
    def _generate_cluster_summary(self, cluster_data: Dict[str, Any]) -> str:
        """Generate a summary for a cluster"""
        blocks = cluster_data["blocks"]
        if not blocks:
            return "Empty cluster"
        
        # Extract common themes
        summaries = [block.get('summary', '') for block in blocks]
        common_words = self._extract_common_words(summaries)
        
        return f"Cluster of {len(blocks)} blocks with themes: {', '.join(common_words[:3])}"
    
    def _get_dominant_topics(self, topics: List[str]) -> List[str]:
        """Get the most common topics in a cluster"""
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:5]
    
    def _extract_common_words(self, texts: List[str]) -> List[str]:
        """Extract common words from a list of texts"""
        all_words = []
        for text in texts:
            words = text.lower().split()
            all_words.extend([word for word in words if len(word) > 3])
        
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        return sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)
    
    def create_visualizations(self) -> Dict[str, str]:
        """Create interactive visualizations of the clusters"""
        if not SKLEARN_AVAILABLE or self.vectors is None or self.clusters is None:
            logger.warning("Cannot create visualizations without vectors and clusters")
            return {}
        
        logger.info("Creating cluster visualizations...")
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2, random_state=42)
        vectors_2d = pca.fit_transform(self.vectors.toarray())
        
        # Prepare data for plotting
        cluster_labels = []
        for block in self.memory_blocks:
            for cluster_id, cluster_data in self.clusters.items():
                if block in cluster_data["blocks"]:
                    cluster_labels.append(cluster_id)
                    break
        
        # Create scatter plot
        fig = go.Figure()
        
        for cluster_id in sorted(self.clusters.keys()):
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_x = [vectors_2d[i][0] for i in cluster_indices]
            cluster_y = [vectors_2d[i][1] for i in cluster_indices]
            
            cluster_info = self.clusters[cluster_id]
            hover_text = [f"Cluster {cluster_id}<br>Topics: {', '.join(cluster_info['dominant_topics'][:3])}<br>Summary: {self.memory_blocks[i].get('summary', '')[:100]}..." 
                         for i in cluster_indices]
            
            fig.add_trace(go.Scatter(
                x=cluster_x,
                y=cluster_y,
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                marker=dict(size=8, opacity=0.7)
            ))
        
        fig.update_layout(
            title='MUSE Pantheon Memory Block Clusters',
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            hovermode='closest'
        )
        
        # Save visualization
        viz_file = self.output_dir / "cluster_visualization.html"
        fig.write_html(str(viz_file))
        
        # Create topic distribution chart
        topic_counts = {}
        for cluster_data in self.clusters.values():
            for topic in cluster_data["dominant_topics"]:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if topic_counts:
            topics = list(topic_counts.keys())[:10]
            counts = [topic_counts[topic] for topic in topics]
            
            topic_fig = go.Figure(data=[go.Bar(x=topics, y=counts)])
            topic_fig.update_layout(
                title='Most Common Topics Across Clusters',
                xaxis_title='Topics',
                yaxis_title='Frequency'
            )
            
            topic_viz_file = self.output_dir / "topic_distribution.html"
            topic_fig.write_html(str(topic_viz_file))
        
        return {
            "cluster_plot": str(viz_file),
            "topic_plot": str(topic_viz_file) if topic_counts else None
        }
    
    def save_results(self) -> Dict[str, str]:
        """Save clustering results to JSON files"""
        results = {}
        
        # Save cluster assignments
        if self.clusters:
            clusters_file = self.output_dir / "clusters.json"
            with open(clusters_file, 'w') as f:
                json.dump(self.clusters, f, indent=2, default=str)
            results["clusters_file"] = str(clusters_file)
        
        # Save feature names
        if self.vectorizer:
            features_file = self.output_dir / "feature_names.json"
            with open(features_file, 'w') as f:
                json.dump(self.vectorizer.get_feature_names_out().tolist(), f, indent=2)
            results["features_file"] = str(features_file)
        
        # Save summary report
        summary_file = self.output_dir / "clustering_summary.json"
        summary = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_blocks": len(self.memory_blocks),
            "n_clusters": len(self.clusters) if self.clusters else 0,
            "cluster_summaries": {}
        }
        
        if self.clusters:
            for cluster_id, cluster_data in self.clusters.items():
                summary["cluster_summaries"][cluster_id] = {
                    "block_count": len(cluster_data["blocks"]),
                    "dominant_topics": cluster_data["dominant_topics"],
                    "dominant_archetype": cluster_data["dominant_archetype"],
                    "summary": cluster_data["summary"]
                }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        results["summary_file"] = str(summary_file)
        
        return results

def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Clustering & Embeddings")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for clustering results')
    parser.add_argument('--clusters', type=int,
                       help='Number of clusters (auto-determined if not specified)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create interactive visualizations')
    
    args = parser.parse_args()
    
    # Initialize clusterer
    clusterer = MemoryBlockClusterer(args.output_dir)
    
    # Load memory blocks
    loaded_count = clusterer.load_memory_blocks(args.blocks_dir)
    if loaded_count == 0:
        logger.error("No MemoryBlocks loaded. Exiting.")
        return
    
    # Create vectors
    vectors = clusterer.create_text_vectors()
    if vectors is None:
        logger.error("Failed to create vectors. Exiting.")
        return
    
    # Perform clustering
    cluster_results = clusterer.perform_clustering(args.clusters)
    if not cluster_results:
        logger.error("Clustering failed. Exiting.")
        return
    
    # Create visualizations if requested
    viz_files = {}
    if args.visualize:
        viz_files = clusterer.create_visualizations()
    
    # Save results
    output_files = clusterer.save_results()
    
    print(f"✅ Clustering complete!")
    print(f"📊 Processed {loaded_count} MemoryBlocks into {cluster_results['n_clusters']} clusters")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Results: {output_files}")
    if viz_files:
        print(f"📈 Visualizations: {viz_files}")

if __name__ == "__main__":
    main()