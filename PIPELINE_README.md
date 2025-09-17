# MUSE Pantheon Universal Ingest Pipeline

A comprehensive universal ingest pipeline that processes any file format into structured MemoryBlocks for intellectual property proof, semantic search, and agent memory integration.

## Overview

The MUSE Pantheon Universal Ingest Pipeline converts diverse file formats (images, videos, documents, code, etc.) into normalized MemoryBlocks with proper metadata, ethics tracking, and project assignments.

## Quick Start

### Simple Usage
```bash
# Process current directory with all file types
./run_pipeline.py
```

### Advanced Usage
```bash
# Process specific directories
python warden/tools/run_full_pipeline.py --roots /path/to/docs /path/to/code

# Process only specific file types
python warden/tools/run_full_pipeline.py --roots /path/to/files --types .py .md .json .pdf

# Custom workspace
python warden/tools/run_full_pipeline.py --roots /data --workspace /custom/workspace
```

## Pipeline Components

### 1. Universal Ingestion (`universal_ingest.py`)
- **Purpose**: Extracts content from any file format
- **Supported Formats**: 
  - Text files (.txt, .md, .py, .js, .json, .csv, .html)
  - PDFs (via PyMuPDF)
  - Images (OCR placeholder)
  - Email files (.eml)
  - ZIP archives
- **Output**: MemoryBlock JSON files with standardized schema

### 2. Clustering & Embeddings (`cluster_embeddings.py`)
- **Purpose**: Groups semantically similar content
- **Features**:
  - TF-IDF vectorization
  - K-means clustering with auto-optimization
  - PCA dimensionality reduction
  - Interactive Plotly visualizations
- **Output**: Cluster assignments and interactive charts

### 3. Project Assignment (`assign_projects.py`)
- **Purpose**: Auto-assigns project IDs based on content themes
- **Features**:
  - Theme extraction (legal, financial, technical, etc.)
  - MUSE archetype mapping (Vision, Builder, Sage, Guardian, Connector)
  - Hierarchical project structure
  - Confidence scoring
- **Output**: Updated MemoryBlocks with project assignments

### 4. Pipeline Orchestrator (`run_full_pipeline.py`)
- **Purpose**: Coordinates the complete pipeline
- **Features**:
  - Sequential phase execution
  - Error handling and recovery
  - Progress reporting
  - Comprehensive summaries
- **Output**: Complete pipeline results and reports

## MemoryBlock Schema

Each processed file becomes a MemoryBlock with the following structure:

```json
{
  "id_hash": "sha256:abc123...",
  "summary": "Brief description of content",
  "content": "Extracted text content (truncated)",
  "topics": ["keyword1", "keyword2"],
  "skills": ["nano_script1.py", "nano_script2.py"],
  "date": "YYYY-MM-DD",
  "project_suggestion": {
    "match_type": "existing|new",
    "project_id": "category.subcategory"
  },
  "archetype": "Vision|Builder|Sage|Guardian|Connector",
  "created_at": "ISO timestamp",
  "links": [],
  "metadata": {
    "source_file": "/path/to/file",
    "file_size": 1024,
    "file_type": ".py",
    "validation_status": "passed"
  },
  "ethics": {
    "pii_redacted": true,
    "consent_logged": true
  }
}
```

## Output Structure

```
_work/pipeline_output/
├── memory_blocks/              # MemoryBlock JSON files
│   ├── sha256_abc123.json
│   └── sha256_def456.json
├── clustering_results/         # Clustering analysis
│   ├── clusters.json
│   ├── cluster_visualization.html
│   ├── topic_distribution.html
│   └── clustering_summary.json
├── project_assignments/        # Project assignments
│   ├── project_hierarchy.json
│   ├── assignment_stats.json
│   └── assignment_summary.json
└── pipeline_summary.txt        # Human-readable summary
```

## MUSE Archetype System

The pipeline uses the MUSE Pantheon archetype system for content classification:

- **Vision**: Images, visual design, UI/UX materials
- **Builder**: Code, technical documentation, system architecture
- **Sage**: Knowledge documents, research, insights
- **Guardian**: Security, legal, compliance materials
- **Connector**: APIs, integrations, communication channels

## Project Hierarchy

Auto-generated project structure:

- **compliance/**: Legal, regulatory, compliance materials
- **business/**: Financial records, business operations
- **development/**: Technical assets, codebase
- **knowledge/**: Documentation, research studies
- **creative/**: Design assets, media library
- **insights/**: Data analysis, reports, metrics
- **operations/**: Communications, project tracking
- **content/**: Media library, written content
- **general/**: Miscellaneous, uncategorized items

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Core dependencies:
- numpy, scikit-learn (clustering)
- plotly (visualizations)
- PyMuPDF (PDF processing)
- pillow (image processing)
- python-magic (file type detection)

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Install with `pip install -r requirements.txt`
2. **PDF processing fails**: Install PyMuPDF: `pip install PyMuPDF`
3. **OCR not working**: Install Tesseract and pytesseract (optional)
4. **Large file timeouts**: Adjust timeout values in scripts

### Error Handling

The pipeline is designed to be resilient:
- Individual file processing errors don't stop the pipeline
- Errors are logged and reported in summary files
- Partial results are saved even if some phases fail

### Performance Tips

- Use `--types` to process only specific file formats
- Process directories in smaller batches for very large datasets
- Monitor memory usage with large image/video files

## Integration

### Vector Store Integration
MemoryBlocks are designed for easy integration with vector databases:

```python
# Example: Load MemoryBlocks for vector store
import json
import pathlib

def load_memory_blocks(blocks_dir):
    blocks = []
    for json_file in pathlib.Path(blocks_dir).glob("*.json"):
        with open(json_file) as f:
            blocks.append(json.load(f))
    return blocks
```

### Agent Memory Integration
MemoryBlocks follow the MUSE Pantheon schema for direct agent integration:

- Standardized metadata for retrieval
- Ethics tracking for compliance
- Project hierarchies for organization
- Archetype mapping for agent specialization

## Next Steps

After running the pipeline:

1. **Review Results**: Check `pipeline_summary.txt` for overview
2. **Explore Clusters**: Open visualization HTML files in browser
3. **Validate Projects**: Review project assignments in JSON files
4. **Integrate**: Load MemoryBlocks into your vector store/agent system
5. **Iterate**: Refine project rules or clustering parameters as needed

## Support

For issues or enhancements:
1. Check error logs in pipeline output
2. Review this documentation
3. Validate file paths and permissions
4. Ensure all dependencies are installed