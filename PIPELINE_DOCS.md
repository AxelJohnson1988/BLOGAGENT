# MUSE Pantheon Universal Ingest Pipeline

## Overview

The MUSE Pantheon Universal Ingest Pipeline is a comprehensive system for converting any file format into structured, semantic MemoryBlocks. It processes files, extracts content, generates embeddings, clusters similar content, and assigns project hierarchies - all while maintaining ethics tracking and consent logging.

## Architecture

### Core Components

1. **MemoryBlock System** (`common/memory_block.py`)
   - Atomic, immutable, sovereign, and semantic memory blocks
   - 12 archetypes: Discoverer, Guardian, Alchemist, Oracle, Sage, Shaman, Visionary, Architect, Weaver, Navigator, Storyteller, Scribe
   - Built-in ethics tracking: PII redaction, consent logging, validation

2. **Universal Ingest** (`warden/tools/universal_ingest.py`)
   - Processes any file format: text, code, images, PDFs, videos, archives
   - OCR for images, PDF parsing, metadata extraction
   - Graceful degradation when dependencies are missing

3. **Clustering & Embeddings** (`warden/tools/cluster_embeddings.py`)
   - TF-IDF vectorization with custom preprocessing
   - K-means clustering with automatic cluster detection
   - Interactive Plotly visualization with PCA projection

4. **Project Assignment** (`warden/tools/assign_projects.py`)
   - Content-based project classification
   - Archetype-aware assignment
   - Hierarchical project organization

5. **Pipeline Orchestrator** (`warden/tools/run_full_pipeline.py`)
   - Coordinates all phases with error handling
   - Comprehensive reporting and statistics
   - Environment validation

## Installation

### Prerequisites

The pipeline works with just Python standard library, but additional features require:

```bash
# Optional: For enhanced processing
pip install pytesseract pillow  # OCR for images
pip install PyPDF2 pdfplumber   # PDF processing
pip install opencv-python       # Video metadata
pip install mutagen            # Audio metadata
```

### Setup

1. Clone the repository
2. Ensure Python 3.7+ is available
3. Make scripts executable: `chmod +x run_pipeline.py warden/tools/*.py`

## Usage

### Quick Start

```bash
# Process current directory with common file types
./run_pipeline.py --scan-root . --apply

# Process specific directory with custom types
./run_pipeline.py --scan-root /path/to/files --types .py .md .json --apply

# Dry run to see what would be processed
./run_pipeline.py --scan-root /path/to/files
```

### Advanced Usage

```bash
# Full pipeline with all options
python warden/tools/run_full_pipeline.py \
  --roots /path/to/files1 /path/to/files2 \
  --types .py .md .txt .json .jpg .pdf \
  --workspace /custom/workspace

# Individual phases
python warden/tools/universal_ingest.py --roots /path --output /output
python warden/tools/cluster_embeddings.py --blocks-dir /blocks --output-dir /clusters
python warden/tools/assign_projects.py --blocks-dir /blocks --output-dir /projects

# Environment validation
python warden/tools/run_full_pipeline.py --validate-only
```

### Supported File Types

- **Text**: .txt, .md, .py, .js, .ts, .json, .csv, .xml, .html
- **Documents**: .pdf, .doc, .docx, .rtf
- **Images**: .jpg, .jpeg, .png, .gif, .bmp, .tiff (with OCR)
- **Media**: .mp4, .avi, .mov, .mp3, .wav (metadata extraction)
- **Archives**: .zip, .tar, .gz, .rar (file listing)

## Output Structure

```
_work/pipeline_output/
├── memory_blocks/                    # Individual MemoryBlock JSON files
│   ├── memory_block_abc123.json
│   ├── memory_block_def456.json
│   └── ingest_report.json           # Ingestion statistics
├── clustering_results/              # Semantic clustering results
│   ├── clustering_results.json     # Cluster assignments
│   └── cluster_visualization.html  # Interactive visualization
├── project_assignments/             # Project assignment results
│   ├── memory_block_abc123.json    # Updated blocks with projects
│   ├── project_assignment_report.json
│   └── project_assignment_summary.txt
└── pipeline_summary.txt             # Overall execution summary
```

## MemoryBlock Schema

```json
{
  "id_hash": "abc123def456",
  "summary": "Brief description of content",
  "content": "Full extracted content",
  "topics": ["keyword1", "keyword2"],
  "skills": ["nano_warden_processing.py"],
  "date": "2024-12-17",
  "project": "muse.pantheon",
  "archetype": "Guardian",
  "created_at": "2024-12-17T10:30:00Z",
  "source_path": "/path/to/original/file.py",
  "file_type": ".py",
  "pii_redacted": false,
  "consent_logged": true,
  "ethics_review": "passed",
  "links": [],
  "parent_blocks": [],
  "metadata": {
    "file_size": 1024,
    "source": "file_ingest",
    "validation_status": "passed"
  }
}
```

## Project Classification

The system automatically assigns projects based on content analysis:

### Project Categories

- **muse.pantheon**: Core memory system, archetypes, foundational components
- **blog.agent**: Content generation, writing, publishing
- **warden.system**: Security, monitoring, ethics enforcement
- **ai.assistant**: AI agents, assistants, interaction systems
- **legal.documentation**: Contracts, compliance, legal content
- **data.analytics**: Data analysis, reporting, insights
- **creative.content**: Design, media, artistic content
- **development.tools**: Code tools, utilities, automation

### Archetype-Project Mapping

- **Guardian/Warden**: Security and monitoring systems
- **Oracle/Sage**: Knowledge and AI systems
- **Scribe/Storyteller**: Content and documentation
- **Alchemist/Architect**: Development and building
- **Visionary**: Creative and design systems

## Error Handling

The pipeline is designed for resilience:

- **Individual file failures** don't stop processing
- **Missing dependencies** result in graceful degradation
- **Comprehensive logging** with error details
- **Partial results** are still useful and saved

## Integration

### Loading MemoryBlocks

```python
import json
from pathlib import Path
from common.memory_block import MemoryBlock

# Load all memory blocks
blocks = []
for json_file in Path("_work/pipeline_output/memory_blocks").glob("memory_block_*.json"):
    with open(json_file) as f:
        data = json.load(f)
        block = MemoryBlock.from_dict(data)
        blocks.append(block)
```

### Querying by Project

```python
# Filter by project
muse_blocks = [b for b in blocks if b.project == "muse.pantheon"]

# Filter by archetype
guardian_blocks = [b for b in blocks if b.archetype == "Guardian"]

# Filter by topics
memory_blocks = [b for b in blocks if "memory" in b.topics]
```

### Using Clustering Results

```python
# Load clustering results
with open("_work/pipeline_output/clustering_results/clustering_results.json") as f:
    clustering = json.load(f)

# Get blocks in same cluster
cluster_0_blocks = [
    blocks[i] for i, label in enumerate(clustering["clustering_results"]["cluster_labels"])
    if label == 0
]
```

## Troubleshooting

### Common Issues

1. **No memory blocks created**
   - Check if input directory exists and contains supported files
   - Verify file permissions
   - Check logs for encoding errors

2. **Clustering fails**
   - Ensure at least 2 memory blocks were created
   - Check for extremely short content (less than 10 words)

3. **Missing visualizations**
   - Install optional dependencies for better processing
   - Check browser compatibility for HTML visualization

4. **Permission errors**
   - Ensure write permissions to output directory
   - Check that scripts are executable

### Debugging

```bash
# Enable verbose logging
export PYTHONPATH=/path/to/BLOGAGENT:$PYTHONPATH
python -m logging DEBUG warden/tools/run_full_pipeline.py --roots /path

# Validate environment
python warden/tools/run_full_pipeline.py --validate-only

# Test with single file type
python warden/tools/universal_ingest.py --roots /path --types .txt --output /tmp/test
```

## Performance

### Optimization Tips

- **Limit file types** to reduce processing time
- **Use SSD storage** for better I/O performance
- **Increase RAM** for large document processing
- **Process in batches** for very large datasets

### Scalability

- **Memory usage**: ~1MB per 1000 files processed
- **Processing speed**: ~100 files/second for text files
- **Storage**: ~10KB per MemoryBlock on average

## Contributing

### Adding New File Processors

1. Extend `UniversalFileProcessor._extract_content()` in `universal_ingest.py`
2. Add file extension to `supported_types`
3. Implement extraction logic with graceful error handling
4. Test with sample files

### Extending Project Classification

1. Add new project patterns to `ProjectClassifier.project_patterns`
2. Update archetype mappings as needed
3. Test classification accuracy with sample content

### Custom Archetypes

1. Add to `MemoryBlockFactory.ARCHETYPES` in `memory_block.py`
2. Update archetype determination logic
3. Adjust project mappings accordingly

## License

This project is part of the MUSE Pantheon system for AI agent memory and knowledge management.