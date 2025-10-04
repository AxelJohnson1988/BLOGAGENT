#!/usr/bin/env python3
"""
Clustering and Embeddings System for MUSE Pantheon
Groups MemoryBlocks semantically and generates embeddings
"""
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Simple numpy replacement for basic operations
    class np:
        @staticmethod
        def array(data):
            return data
        
        @staticmethod
        def zeros(shape):
            if isinstance(shape, tuple):
                if len(shape) == 2:
                    return [[0.0] * shape[1] for _ in range(shape[0])]
                else:
                    return [0.0] * shape[0]
            else:
                return [0.0] * shape
        
        @staticmethod
        def mean(data, axis=None):
            if axis is None:
                flat = [item for sublist in data for item in (sublist if isinstance(sublist, list) else [sublist])]
                return sum(flat) / len(flat) if flat else 0
            elif axis == 0:
                if not data or not data[0]:
                    return []
                return [sum(col) / len(data) for col in zip(*data)]
            else:
                return [sum(row) / len(row) if row else 0 for row in data]
        
        @staticmethod
        def sqrt(x):
            return x ** 0.5
        
        @staticmethod
        def sum(data):
            if isinstance(data, list) and isinstance(data[0], list):
                return sum(sum(row) for row in data)
            else:
                return sum(data)
        
        @staticmethod
        def argmin(data, axis=None):
            if axis == 1:
                return [min(range(len(row)), key=lambda i: row[i]) for row in data]
            else:
                flat = [item for sublist in data for item in (sublist if isinstance(sublist, list) else [sublist])]
                return min(range(len(flat)), key=lambda i: flat[i])
        
        @staticmethod
        def any(mask):
            return any(mask)
        
        @staticmethod
        def allclose(a, b, rtol=1e-4):
            if len(a) != len(b):
                return False
            for i in range(len(a)):
                if isinstance(a[i], list):
                    if not np.allclose(a[i], b[i], rtol):
                        return False
                else:
                    if abs(a[i] - b[i]) > rtol * max(abs(a[i]), abs(b[i])):
                        return False
            return True

from collections import defaultdict
import math
import random

# Add common directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "common"))
from memory_block import MemoryBlock

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for MemoryBlocks using TF-IDF."""
    
    def __init__(self):
        """Initialize the embedding generator."""
        self.vocabulary = {}
        self.idf_scores = {}
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
    
    def fit_transform(self, memory_blocks: List[MemoryBlock]) -> List[List[float]]:
        """Fit the model and transform memory blocks to embeddings."""
        # Build vocabulary and compute IDF scores
        self._build_vocabulary(memory_blocks)
        
        # Generate TF-IDF vectors
        embeddings = []
        for block in memory_blocks:
            embedding = self._generate_tfidf_vector(block)
            embeddings.append(embedding)
        
        return embeddings
    
    def _build_vocabulary(self, memory_blocks: List[MemoryBlock]):
        """Build vocabulary and compute IDF scores."""
        # Collect all documents
        documents = []
        for block in memory_blocks:
            text = f"{block.summary} {block.content} {' '.join(block.topics)}"
            documents.append(self._preprocess_text(text))
        
        # Build vocabulary
        word_counts = defaultdict(int)
        for doc in documents:
            for word in set(doc):  # Use set to count document frequency, not word frequency
                word_counts[word] += 1
        
        # Filter vocabulary (remove very rare and very common words)
        min_df = max(1, len(documents) // 100)  # Minimum document frequency
        max_df = len(documents) * 0.8  # Maximum document frequency (80%)
        
        self.vocabulary = {}
        vocab_index = 0
        for word, count in word_counts.items():
            if min_df <= count <= max_df:
                self.vocabulary[word] = vocab_index
                vocab_index += 1
        
        # Compute IDF scores
        self.idf_scores = {}
        for word, index in self.vocabulary.items():
            df = word_counts[word]
            idf = math.log(len(documents) / df)
            self.idf_scores[word] = idf
        
        logger.info(f"Built vocabulary with {len(self.vocabulary)} words")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for vectorization."""
        # Simple preprocessing
        text = text.lower()
        words = []
        
        current_word = ""
        for char in text:
            if char.isalnum():
                current_word += char
            else:
                if current_word and len(current_word) > 2 and current_word not in self.stop_words:
                    words.append(current_word)
                current_word = ""
        
        # Don't forget the last word
        if current_word and len(current_word) > 2 and current_word not in self.stop_words:
            words.append(current_word)
        
        return words
    
    def _generate_tfidf_vector(self, memory_block: MemoryBlock) -> List[float]:
        """Generate TF-IDF vector for a memory block."""
        text = f"{memory_block.summary} {memory_block.content} {' '.join(memory_block.topics)}"
        words = self._preprocess_text(text)
        
        # Count word frequencies
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Generate TF-IDF vector
        vector = [0.0] * len(self.vocabulary)
        
        for word, freq in word_freq.items():
            if word in self.vocabulary:
                tf = freq / len(words) if words else 0
                idf = self.idf_scores[word]
                tfidf = tf * idf
                vector[self.vocabulary[word]] = tfidf
        
        return vector


class SemanticClustering:
    """Clusters MemoryBlocks based on semantic similarity."""
    
    def __init__(self, n_clusters: int = None):
        """Initialize the clustering system."""
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.labels = None
    
    def fit_predict(self, embeddings: List[List[float]], memory_blocks: List[MemoryBlock]) -> List[int]:
        """Fit clustering model and predict cluster labels."""
        if self.n_clusters is None:
            # Automatic cluster number selection
            self.n_clusters = max(2, min(10, len(memory_blocks) // 5))
        
        logger.info(f"Clustering {len(memory_blocks)} blocks into {self.n_clusters} clusters")
        
        # Simple K-means implementation
        self.labels = self._kmeans_clustering(embeddings)
        
        return self.labels
    
    def _kmeans_clustering(self, embeddings: List[List[float]], max_iterations: int = 100) -> List[int]:
        """Simple K-means clustering implementation."""
        n_samples = len(embeddings)
        n_features = len(embeddings[0]) if embeddings else 0
        
        if n_samples == 0 or n_features == 0:
            return []
        
        # Initialize centroids randomly
        centroids = []
        for k in range(self.n_clusters):
            centroid = [random.random() for _ in range(n_features)]
            centroids.append(centroid)
        
        for iteration in range(max_iterations):
            # Assign points to closest centroid
            distances = self._calculate_distances(embeddings, centroids)
            new_labels = [min(range(len(distances[i])), key=lambda k: distances[i][k]) for i in range(len(distances))]
            
            # Update centroids
            new_centroids = []
            for k in range(self.n_clusters):
                # Find points assigned to this cluster
                cluster_points = [embeddings[i] for i in range(len(embeddings)) if new_labels[i] == k]
                
                if cluster_points:
                    # Calculate mean of cluster points
                    centroid = [0.0] * n_features
                    for point in cluster_points:
                        for j in range(n_features):
                            centroid[j] += point[j]
                    centroid = [c / len(cluster_points) for c in centroid]
                else:
                    # Keep old centroid if no points assigned
                    centroid = centroids[k]
                
                new_centroids.append(centroid)
            
            # Check for convergence
            if iteration > 0 and self._centroids_equal(centroids, new_centroids):
                logger.info(f"Clustering converged after {iteration + 1} iterations")
                break
            
            centroids = new_centroids
        
        self.cluster_centers = centroids
        return new_labels
    
    def _centroids_equal(self, a: List[List[float]], b: List[List[float]], rtol: float = 1e-4) -> bool:
        """Check if centroids are approximately equal."""
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if len(a[i]) != len(b[i]):
                return False
            for j in range(len(a[i])):
                if abs(a[i][j] - b[i][j]) > rtol * max(abs(a[i][j]), abs(b[i][j]), 1e-8):
                    return False
        return True
    
    def _calculate_distances(self, points: List[List[float]], centroids: List[List[float]]) -> List[List[float]]:
        """Calculate Euclidean distances between points and centroids."""
        distances = []
        
        for point in points:
            point_distances = []
            for centroid in centroids:
                distance = 0.0
                for i in range(len(point)):
                    distance += (point[i] - centroid[i]) ** 2
                distance = distance ** 0.5
                point_distances.append(distance)
            distances.append(point_distances)
        
        return distances


class ClusterAnalyzer:
    """Analyzes and describes clusters."""
    
    def analyze_clusters(self, memory_blocks: List[MemoryBlock], labels: List[int]) -> Dict[int, Dict[str, Any]]:
        """Analyze clusters and generate descriptions."""
        cluster_analysis = defaultdict(lambda: {
            'blocks': [],
            'archetypes': defaultdict(int),
            'projects': defaultdict(int),
            'topics': defaultdict(int),
            'file_types': defaultdict(int),
            'description': '',
            'representative_block': None
        })
        
        # Group blocks by cluster
        for block, label in zip(memory_blocks, labels):
            cluster_analysis[label]['blocks'].append(block)
            cluster_analysis[label]['archetypes'][block.archetype] += 1
            cluster_analysis[label]['projects'][block.project] += 1
            cluster_analysis[label]['file_types'][block.file_type or 'unknown'] += 1
            
            for topic in block.topics:
                cluster_analysis[label]['topics'][topic] += 1
        
        # Generate descriptions and find representative blocks
        for cluster_id, data in cluster_analysis.items():
            blocks = data['blocks']
            
            # Find most common elements
            top_archetype = max(data['archetypes'].items(), key=lambda x: x[1])[0]
            top_project = max(data['projects'].items(), key=lambda x: x[1])[0]
            top_topics = sorted(data['topics'].items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Generate description
            data['description'] = f"Cluster of {len(blocks)} blocks, primarily {top_archetype} archetype in {top_project} project. Key topics: {', '.join([topic for topic, count in top_topics])}"
            
            # Find representative block (longest content)
            data['representative_block'] = max(blocks, key=lambda b: len(b.content))
        
        return dict(cluster_analysis)


class ClusterVisualizer:
    """Creates visualizations of clusters."""
    
    def create_cluster_visualization(self, embeddings: List[List[float]], labels: List[int], 
                                   memory_blocks: List[MemoryBlock], output_dir: Path):
        """Create cluster visualization using dimensionality reduction."""
        try:
            # Simple 2D projection using PCA
            reduced_embeddings = self._simple_pca(embeddings, n_components=2)
            
            # Create HTML visualization
            self._create_html_plot(reduced_embeddings, labels, memory_blocks, output_dir)
            
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
    
    def _simple_pca(self, data: List[List[float]], n_components: int = 2) -> List[List[float]]:
        """Simple PCA implementation."""
        if not data or not data[0]:
            return []
        
        n_samples = len(data)
        n_features = len(data[0])
        
        # Center the data
        mean = [sum(data[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        centered_data = [[data[i][j] - mean[j] for j in range(n_features)] for i in range(n_samples)]
        
        # For simplicity, just project onto first 2 dimensions
        # In a real implementation, we'd compute eigenvalues/eigenvectors
        if n_features >= 2:
            reduced_data = [[row[0], row[1]] for row in centered_data]
        elif n_features == 1:
            reduced_data = [[row[0], 0.0] for row in centered_data]
        else:
            reduced_data = [[0.0, 0.0] for _ in range(n_samples)]
        
        return reduced_data
    
    def _create_html_plot(self, embeddings_2d: List[List[float]], labels: List[int], 
                         memory_blocks: List[MemoryBlock], output_dir: Path):
        """Create interactive HTML plot."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>MUSE Pantheon Cluster Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>MUSE Pantheon Memory Block Clusters</h1>
    <div id="plot" style="width:100%;height:600px;"></div>
    <script>
"""
        
        # Prepare data for each cluster
        cluster_data = defaultdict(lambda: {'x': [], 'y': [], 'text': [], 'name': ''})
        
        for i, (embedding, label, block) in enumerate(zip(embeddings_2d, labels, memory_blocks)):
            cluster_data[label]['x'].append(embedding[0])
            cluster_data[label]['y'].append(embedding[1])
            cluster_data[label]['text'].append(f"{block.summary[:100]}...")
            cluster_data[label]['name'] = f"Cluster {label}"
        
        # Generate Plotly traces
        traces = []
        for cluster_id, data in cluster_data.items():
            color = colors[cluster_id % len(colors)]
            trace = f"""{{
                x: {data['x']},
                y: {data['y']},
                text: {json.dumps(data['text'])},
                mode: 'markers',
                type: 'scatter',
                name: '{data['name']}',
                marker: {{
                    color: '{color}',
                    size: 10
                }},
                hovertemplate: '%{{text}}<extra></extra>'
            }}"""
            traces.append(trace)
        
        html_content += f"""
        var data = [{', '.join(traces)}];
        
        var layout = {{
            title: 'Memory Block Clusters (PCA Projection)',
            xaxis: {{ title: 'PC1' }},
            yaxis: {{ title: 'PC2' }},
            hovermode: 'closest'
        }};
        
        Plotly.newPlot('plot', data, layout);
    </script>
</body>
</html>"""
        
        output_file = output_dir / "cluster_visualization.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Cluster visualization saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clustering and Embeddings for MUSE Pantheon")
    parser.add_argument('--blocks-dir', required=True,
                       help='Directory containing MemoryBlock JSON files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for clustering results')
    parser.add_argument('--n-clusters', type=int,
                       help='Number of clusters (auto-detected if not specified)')
    parser.add_argument('--assign-projects', action='store_true',
                       help='Also run project assignment based on clusters')
    
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
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.fit_transform(memory_blocks)
    
    # Perform clustering
    logger.info("Performing clustering...")
    clustering = SemanticClustering(n_clusters=args.n_clusters)
    labels = clustering.fit_predict(embeddings, memory_blocks)
    
    # Analyze clusters
    logger.info("Analyzing clusters...")
    analyzer = ClusterAnalyzer()
    cluster_analysis = analyzer.analyze_clusters(memory_blocks, labels)
    
    # Create visualization
    logger.info("Creating visualization...")
    visualizer = ClusterVisualizer()
    visualizer.create_cluster_visualization(embeddings, labels, memory_blocks, output_dir)
    
    # Save results
    results = {
        'clustering_results': {
            'n_clusters': len(set(labels)),
            'cluster_labels': labels,
            'cluster_analysis': {
                str(k): {
                    'description': v['description'],
                    'n_blocks': len(v['blocks']),
                    'top_archetype': max(v['archetypes'].items(), key=lambda x: x[1])[0],
                    'top_project': max(v['projects'].items(), key=lambda x: x[1])[0],
                    'representative_block_id': v['representative_block'].id_hash
                }
                for k, v in cluster_analysis.items()
            }
        },
        'memory_blocks_with_clusters': [
            {
                'id_hash': block.id_hash,
                'cluster_label': label,
                'summary': block.summary,
                'archetype': block.archetype,
                'project': block.project
            }
            for block, label in zip(memory_blocks, labels)
        ]
    }
    
    # Save clustering results
    results_file = output_dir / "clustering_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Clustering results saved to: {results_file}")
    logger.info(f"Found {len(set(labels))} clusters")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())