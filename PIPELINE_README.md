# MUSE Pantheon Universal Ingest Pipeline

A comprehensive file ingestion system that converts various file formats into structured MemoryBlocks for intellectual property proof, semantic search, and AI agent memory systems.

## Quick Start

### Simple Launch
```bash
# Run with sensible defaults
python run_pipeline.py run
```

### Custom Configuration
```bash
# Process specific directories and file types
python run_pipeline.py custom --roots /path/to/files --types .py .md .json

# Full control with the main orchestrator
python warden/tools/run_full_pipeline.py --roots ./documents ./code --types .pdf .py .md
```

## Pipeline Overview

The pipeline consists of four main phases:

1. **🚀 Universal Ingestion** - Extracts content from various file formats
2. **🧠 Clustering & Embeddings** - Groups similar content semantically  
3. **🎯 Project Assignment** - Auto-assigns project IDs based on themes
4. **📊 Reporting** - Generates comprehensive analysis and visualizations

## Supported File Types

- **Code**: `.py`, `.js`, `.html`, `.css`, `.java`, `.cpp`, `.c`
- **Documents**: `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.md`, `.txt`
- **Images**: `.jpg`, `.png`, `.gif` (with OCR extraction)
- **Media**: `.mp4`, `.mov`, `.avi`, `.mp3`, `.wav`
- **Data**: `.json`, `.csv`, `.xml`, `.yaml`
- **Archives**: `.zip` (content listing)
- **Email**: `.eml` files
- **Web**: `.html`, `.xml`

## Architecture

### MemoryBlock Schema
Each processed file becomes a MemoryBlock with:

```json
{
  "id_hash": "unique_identifier", 
  "summary": "Brief content description",
  "content": "Extracted text/data",
  "topics": ["keywords", "themes"],
  "archetype": "MUSE_archetype",
  "project": "assigned_project",
  "ethics_status": "approved",
  "source_file": "/path/to/original",
  "metadata": {...}
}
```

### MUSE Archetypes
Files are classified into archetypes:

- **Builder**: Code and development files
- **Vision**: Images, videos, visual content
- **Guardian**: Legal, financial, compliance docs  
- **Scholar**: Research, analysis, academic content
- **Scribe**: Documentation, manuals, guides
- **Herald**: Communications, emails, messages
- **Analyst**: Data files, spreadsheets, metrics
- **Keeper**: Personal, private content
- **Muse**: Creative content, art, stories
- **Warden**: System, config, admin files

## Components

### Core Tools

- **`universal_ingest.py`** - Multi-format file content extractor
- **`cluster_embeddings.py`** - Semantic clustering with TF-IDF and K-means
- **`assign_projects.py`** - Project hierarchy and theme-based assignment
- **`run_full_pipeline.py`** - Complete pipeline orchestrator

### Schema & Memory

- **`warden/schema/memory_block.py`** - MemoryBlock data structure and builder
- **`warden/tools/`** - Processing pipeline components

## Usage Examples

### Basic Processing
```bash
# Process current directory with defaults
python run_pipeline.py run

# Check pipeline status  
python run_pipeline.py status

# Clean previous results
python run_pipeline.py clean
```

### Advanced Usage
```bash
# Process specific document directory
python warden/tools/run_full_pipeline.py \
  --roots /Users/username/Documents \
  --types .pdf .docx .jpg \
  --workspace /project/root

# Run only ingestion phase
python warden/tools/universal_ingest.py \
  --roots ./data \
  --output ./output/blocks \
  --types .json .csv

# Run clustering on existing blocks
python warden/tools/cluster_embeddings.py \
  --blocks-dir ./output/blocks \
  --output-dir ./output/clusters \
  --clusters 8
```

## Output Structure

```
_work/pipeline_output/
├── memory_blocks/           # Individual MemoryBlock JSON files
├── clustering_results/      # Cluster analysis and visualizations
├── project_assignments/     # Project hierarchies and assignments
└── pipeline_summary.txt     # Human-readable execution report
```

## Key Features

### Content Extraction
- **OCR**: Text extraction from images using Tesseract
- **PDF**: Text and metadata extraction with PyMuPDF
- **Office**: Word, Excel, PowerPoint content extraction
- **Code**: Language detection and pattern analysis
- **Email**: Full message parsing with headers and body

### Semantic Analysis
- **Topic Extraction**: Keyword-based theme identification
- **Clustering**: TF-IDF vectorization with K-means clustering
- **Embeddings**: Dimensionality reduction for visualization
- **Project Assignment**: Theme-based project hierarchy creation

### Ethics & Privacy
- **PII Redaction**: Automatic detection and redaction flags
- **Consent Tracking**: Built-in consent logging for each block
- **Ethics Status**: Approval tracking for sensitive content

### Integration Ready
- **Vector Store**: Embedding-ready for semantic search
- **JSON Export**: Standard format for system integration  
- **Graph Relationships**: Link tracking between related blocks
- **Timeline**: Temporal organization of content

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- **ML/NLP**: scikit-learn, numpy, pandas
- **Visualization**: plotly for interactive cluster plots
- **OCR**: pytesseract, opencv-python, Pillow
- **Documents**: PyMuPDF, python-docx, openpyxl
- **Web**: beautifulsoup4, requests, selenium

## Troubleshooting

### Common Issues

**Import Errors**: Ensure you're running from the repository root and dependencies are installed:
```bash
pip install -r requirements.txt
python -c "import sklearn, pandas, numpy; print('Dependencies OK')"
```

**OCR Failures**: Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS  
brew install tesseract
```

**Large File Handling**: The pipeline automatically skips files >100MB. To process larger files, modify the size limit in `universal_ingest.py`.

**Memory Issues**: For large datasets, increase available RAM or process in smaller batches by using specific file type filters.

### Performance Tips

- Use `--types` to filter to specific file formats for faster processing
- Process images separately if OCR is slow
- Clean output directory between runs for accurate statistics
- Use `--verbose` flag for detailed logging during debugging

## Integration

### With AI Agents
MemoryBlocks are designed for direct integration with AI agent memory systems:

```python
from warden.schema.memory_block import MemoryBlock

# Load blocks for agent context
blocks = []
for file_path in Path("_work/pipeline_output/memory_blocks").glob("*.json"):
    with open(file_path) as f:
        block = MemoryBlock.from_json(f.read())
        blocks.append(block)

# Use in agent queries
relevant_blocks = [b for b in blocks if "keyword" in b.topics]
```

### With Vector Databases
```python
# Extract embeddings for vector store
embeddings = [block.embeddings for block in blocks if block.embeddings]
metadata = [block.to_dict() for block in blocks]

# Insert into vector database
vector_db.insert(embeddings, metadata)
```

This pipeline provides a complete foundation for transforming unstructured files into structured, semantically rich MemoryBlocks ready for AI agent integration and semantic search systems.