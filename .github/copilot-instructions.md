# MUSE Pantheon Universal Ingest Pipeline - AI Agent Instructions

## System Architecture

**BLOGAGENT** implements the MUSE Pantheon Universal Ingest Pipeline - a comprehensive system that processes any file format into structured MemoryBlocks for intellectual property proof, semantic search, and agent memory integration.

### Core Components

- **Universal Ingester** (`warden/tools/universal_ingest.py`): Extracts content from any file type into standardized MemoryBlocks
- **Clustering Engine** (`warden/tools/cluster_embeddings.py`): Groups content semantically using TF-IDF and K-means
- **Project Assigner** (`warden/tools/assign_projects.py`): Auto-assigns project IDs based on content themes and MUSE archetypes
- **Pipeline Orchestrator** (`warden/tools/run_full_pipeline.py`): Coordinates complete pipeline execution

### MUSE Archetype System

Content is classified using 5 archetypes:
- **Vision**: Images, visual design, UI/UX materials
- **Builder**: Code, technical docs, system architecture  
- **Sage**: Knowledge documents, research, insights
- **Guardian**: Security, legal, compliance materials
- **Connector**: APIs, integrations, communication channels

## Critical Workflows

### Running the Pipeline

**Quick start** (process current directory):
```bash
./run_pipeline.py
```

**Full pipeline** (custom roots and file types):
```bash
python warden/tools/run_full_pipeline.py --roots /path/to/files --types .py .md .json
```

**Individual components**:
```bash
# Ingestion only
python warden/tools/universal_ingest.py --roots /data --output _work/blocks

# Clustering only  
python warden/tools/cluster_embeddings.py --blocks-dir _work/blocks --output-dir _work/clusters

# Project assignment only
python warden/tools/assign_projects.py --blocks-dir _work/blocks --output-dir _work/projects
```

## MemoryBlock Schema

Every processed file becomes a MemoryBlock with this structure:
```json
{
  "id_hash": "sha256:abc123",
  "summary": "Brief content description", 
  "content": "Extracted text (truncated)",
  "topics": ["keyword1", "keyword2"],
  "skills": ["nano_script.py"],
  "date": "YYYY-MM-DD",
  "project_suggestion": {"match_type": "existing", "project_id": "category.subcategory"},
  "archetype": "Vision|Builder|Sage|Guardian|Connector",
  "metadata": {"source_file": "/path", "file_type": ".py"},
  "ethics": {"pii_redacted": true, "consent_logged": true}
}
```

## Project-Specific Patterns

### File Processing Strategy
- **Text files**: Direct content extraction with encoding fallback
- **PDFs**: PyMuPDF extraction (fallback to filename if library missing)
- **Images**: OCR placeholder (easily extensible with Tesseract)
- **ZIP files**: Contents listing for nested processing
- **Email (.eml)**: Headers + body extraction

### Error Resilience
- Individual file failures don't stop pipeline
- All errors logged to summary reports
- Graceful degradation when optional dependencies missing

### Project Assignment Logic
Theme-based classification with fallback hierarchy:
1. Content keyword analysis (legal, financial, technical, etc.)
2. File type mapping (.py → technical, .jpg → media)
3. Archetype-based assignment
4. Default to "general.uncategorized"

## Integration Points

### Output Structure
```
_work/pipeline_output/
├── memory_blocks/           # Core MemoryBlock JSON files
├── clustering_results/      # Semantic groupings + visualizations
├── project_assignments/     # Auto-generated project hierarchy
└── pipeline_summary.txt     # Human-readable execution report
```

### Vector Store Integration
MemoryBlocks are designed for direct vector database ingestion:
- Standardized schema for consistent retrieval
- Content truncated to manageable size (2000 chars)
- Metadata preserved for filtering and faceting

### Dependencies
- **Core**: pathlib, json, hashlib (built-in Python)
- **ML Features**: scikit-learn, numpy (clustering/embeddings)
- **Visualizations**: plotly (interactive cluster charts)
- **PDF Processing**: PyMuPDF (optional, graceful fallback)

## Key Reference Files

- `PIPELINE_README.md`: Comprehensive usage documentation
- `requirements.txt`: All Python dependencies
- `warden/tools/`: All pipeline components
- `run_pipeline.py`: Quick launcher with sensible defaults

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Quick test**: `./run_pipeline.py` (processes current directory)
3. **Review outputs**: Check `_work/pipeline_output/pipeline_summary.txt`
4. **Integrate**: Load MemoryBlocks from `memory_blocks/` directory into your vector store

Focus on the pipeline's modular design - each component can be run independently or as part of the full orchestrated workflow. The system is built for resilience and extensibility.