# MUSE Pantheon Universal Ingest Pipeline

## Overview

AI coding agents working in this repository should understand that this is the **MUSE Pantheon Universal Ingest Pipeline** - a comprehensive system for converting any file format into structured, semantic MemoryBlocks for AI agent memory systems.

## Core Architecture

### Memory Block System
- **Atomic, immutable, sovereign, semantic** memory blocks are the foundation
- Schema defined in `common/memory_block.py` with 12 archetypes: Discoverer, Guardian, Alchemist, Oracle, Sage, Shaman, Visionary, Architect, Weaver, Navigator, Storyteller, Scribe
- Ethics tracking built-in: `pii_redacted`, `consent_logged`, `ethics_review`
- Each block has: `id_hash`, `summary`, `content`, `topics`, `skills`, `archetype`, `project`

### Pipeline Flow
1. **Universal Ingest** (`warden/tools/universal_ingest.py`) - Processes any file type
2. **Clustering & Embeddings** (`warden/tools/cluster_embeddings.py`) - TF-IDF + K-means 
3. **Project Assignment** (`warden/tools/assign_projects.py`) - Auto-assigns based on content/archetype
4. **Orchestration** (`warden/tools/run_full_pipeline.py`) - Coordinates all phases

## Key Workflows

### Running the Pipeline
```bash
# Quick start (recommended)
./run_pipeline.py --scan-root /path/to/files --apply

# Full control
python warden/tools/run_full_pipeline.py --roots /path/to/files --types .py .md .json

# Validation only
python warden/tools/run_full_pipeline.py --validate-only
```

### Output Structure
```
_work/pipeline_output/
├── memory_blocks/           # Individual MemoryBlock JSON files
├── clustering_results/      # Semantic clusters + interactive visualization
├── project_assignments/     # Project hierarchies + updated blocks
└── pipeline_summary.txt     # Human-readable execution report
```

## Project-Specific Patterns

### File Processing Extensibility
- `UniversalFileProcessor` in `universal_ingest.py` supports OCR (images), PDF parsing, archive extraction
- Missing dependencies gracefully degrade (e.g., no Tesseract = metadata-only for images)
- Each processor creates MemoryBlocks via `MemoryBlockFactory.create_from_file()`

### Archetype-Project Mapping
- **Guardian/Warden**: Security, monitoring, ethics systems
- **Oracle/Sage**: AI assistants, data analytics, knowledge systems  
- **Scribe/Storyteller**: Blog content, documentation, creative work
- **Alchemist/Architect**: Development tools, code analysis, system building
- **Visionary**: Creative content, media processing, design systems

### Error Handling Philosophy
- **Resilient processing**: Individual file failures don't stop the pipeline
- **Graceful degradation**: Missing dependencies still produce metadata MemoryBlocks
- **Comprehensive logging**: All errors logged but pipeline continues

## Integration Points

### External Dependencies (Optional)
- **OCR**: `pytesseract` + `PIL` for image text extraction
- **PDF**: `PyPDF2` or `pdfplumber` for document processing  
- **Video**: `cv2` for metadata extraction
- **Audio**: `mutagen` for metadata extraction

### Memory System Integration
- MemoryBlocks are immutable once created (identified by `id_hash`)
- Clustering creates semantic relationships between blocks
- Project assignment enables multi-domain organization
- All output is JSON-serializable for downstream AI agent consumption

## Getting Started for AI Agents

1. **Validate environment**: `python warden/tools/run_full_pipeline.py --validate-only`
2. **Test with sample data**: `./run_pipeline.py --scan-root . --types .py .md`
3. **View results**: Check `_work/pipeline_output/cluster_visualization.html`
4. **Integrate**: Load MemoryBlocks from `_work/pipeline_output/memory_blocks/*.json`

The system is designed for **immediate productivity** - run the pipeline on any directory and get structured, semantic memory blocks ready for AI agent ingestion and retrieval.