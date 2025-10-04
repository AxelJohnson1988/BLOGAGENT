# MUSE Pantheon Universal Ingest Pipeline

A comprehensive system for converting any file format into structured, semantic MemoryBlocks for AI agent memory systems.

## Quick Start

```bash
# Process files in current directory
./run_pipeline.py --scan-root . --apply

# Process specific directory with custom types  
./run_pipeline.py --scan-root /path/to/files --types .py .md .json --apply

# View interactive results
open _work/pipeline_output/clustering_results/cluster_visualization.html
```

## Features

- **Universal File Processing**: Handles text, code, images, PDFs, videos, archives
- **Semantic Clustering**: Groups similar content using TF-IDF + K-means  
- **Project Assignment**: Auto-categorizes content into project hierarchies
- **Ethics Tracking**: Built-in PII redaction and consent logging
- **Interactive Visualization**: Explore clusters in 3D semantic space

## Components

- `common/memory_block.py` - Core MemoryBlock schema with 12 archetypes
- `warden/tools/universal_ingest.py` - Universal file processor
- `warden/tools/cluster_embeddings.py` - Semantic clustering engine
- `warden/tools/assign_projects.py` - Project classification system
- `warden/tools/run_full_pipeline.py` - Complete pipeline orchestrator

## Documentation

See [PIPELINE_DOCS.md](PIPELINE_DOCS.md) for complete documentation, API reference, and integration examples.

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for AI agent guidance.
