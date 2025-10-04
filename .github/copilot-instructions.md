# MUSE Pantheon AI Agent Instructions

## System Overview

This is the **MUSE Pantheon Universal Ingest Pipeline** - a comprehensive AI coding agent system designed for universal file ingestion, semantic clustering, and structured memory creation. The system transforms any file format into immutable MemoryBlock objects for AI agent memory systems, IP proof documentation, and semantic search.

## Architecture

### Core Components
- **Universal Ingest Pipeline**: Processes any file format (images, PDFs, videos, documents, code) into structured MemoryBlocks
- **Semantic Clustering**: Groups similar content using TF-IDF vectorization and K-Means clustering  
- **Project Assignment**: Auto-assigns project IDs based on content analysis and archetype mapping
- **Memory Block System**: Immutable, atomic data structures with ethics tracking and consent logging

### Key Directories
```
warden/tools/           # Core pipeline scripts
_work/pipeline_output/  # Generated output and reports
memory_blocks/          # Raw MemoryBlock JSON files
clustering_results/     # Visualizations and cluster data
project_assignments/    # Project hierarchies and mappings
```

## Essential Workflows

### Running the Complete Pipeline
```bash
# Quick start with defaults
./run_pipeline.py

# Custom execution
python warden/tools/run_full_pipeline.py --roots /path/to/files --types .py .md .json .pdf .jpg

# Target specific directory (like MUSE_UNIFIED)
python warden/tools/run_full_pipeline.py --roots /Users/jakobaxelpaper/MUSE_UNIFIED
```

### Individual Pipeline Phases
```bash
# 1. Ingestion only
python warden/tools/universal_ingest.py --roots ./data --output ./memory_blocks

# 2. Clustering only  
python warden/tools/cluster_embeddings.py --blocks-dir ./memory_blocks --output-dir ./clusters

# 3. Project assignment only
python warden/tools/assign_projects.py --blocks-dir ./memory_blocks --output-dir ./projects
```

## MemoryBlock Schema

All processed files become MemoryBlock objects with this structure:
```json
{
  "id_hash": "sha256:first_12_chars",
  "summary": "2-4 sentence standalone summary",
  "content": "extracted_text_or_metadata", 
  "topics": ["lowercase", "keywords"],
  "skills": ["nano_script_references"],
  "project": "assigned_project_id",
  "archetype": "Guardian|Vision|Warden|etc",
  "created_at": "2024-09-17T09:00:00Z",
  "ethics": {
    "pii_redacted": true,
    "consent_logged": true,
    "validation_status": "passed"
  },
  "source": {
    "file_path": "original_file_location",
    "file_type": "detected_format"
  }
}
```

## Project-Specific Patterns

### File Type Handlers
- **Images**: OCR extraction using Tesseract, archetype = "Vision"
- **PDFs**: Text extraction with PyMuPDF, preserve metadata
- **Code files**: Syntax analysis, dependency detection
- **Videos**: Whisper transcription for audio content
- **Archives**: Recursive unpacking and nested processing

### Ethics and Consent Framework
- **PII Detection**: Automatic redaction of personal information
- **Consent Tracking**: Every MemoryBlock includes consent metadata
- **Validation Pipeline**: Ethics checks before MemoryBlock creation

### Archetype System
Maps content to MUSE archetypes for project organization:
- `Guardian`: Security, ethics, validation content
- `Vision`: Images, visual content, OCR results  
- `Warden`: Orchestration, pipeline management
- `Memory`: Storage, retrieval, embedding systems

## Integration Points

### External Dependencies
- **Ollama**: Local LLM for embeddings and content analysis
- **Tesseract**: OCR for image text extraction
- **Whisper**: Audio/video transcription
- **Plotly**: Interactive clustering visualizations

### Output Formats
- **JSON Lines**: Streaming MemoryBlock output
- **SQLite**: Local storage for semantic search
- **Interactive HTML**: Cluster visualization dashboards
- **CSV**: Human-readable summaries and reports

## Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run quick test**: `./run_pipeline.py --help`
3. **Process sample data**: `python warden/tools/run_full_pipeline.py --roots ./sample_data`
4. **Review outputs**: Check `_work/pipeline_output/` for results

## Key Files for AI Agents

- `warden/tools/run_full_pipeline.py`: Main orchestrator entry point
- `warden/tools/universal_ingest.py`: File format detection and processing
- `memory_block.py`: Core MemoryBlock data structure definition
- `run_pipeline.py`: Quick launcher with sensible defaults

## Troubleshooting

- **Missing OCR**: Install Tesseract: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Linux)
- **Memory issues**: Use `--batch-size` parameter for large file sets
- **Clustering failures**: Check TF-IDF vectorization parameters in cluster_embeddings.py
- **Permission errors**: Ensure write access to `_work/pipeline_output/` directory

This system enables comprehensive file ingestion for AI memory systems, IP documentation, and semantic knowledge graphs.