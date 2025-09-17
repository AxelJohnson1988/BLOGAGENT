# BLOGAGENT Copilot Instructions

## Project Overview

BLOGAGENT is a universal ingest pipeline inspired by the MUSE Pantheon system that processes any file format into structured MemoryBlocks for semantic search, AI agent memory, and IP proof documentation.

## Architecture

### Core Components
- **Universal Ingest** (`warden/tools/universal_ingest.py`): Processes all file types into MemoryBlocks
- **Clustering Engine** (`warden/tools/cluster_embeddings.py`): Groups content semantically using TF-IDF and K-Means
- **Project Assignment** (`warden/tools/assign_projects.py`): Maps content to project hierarchies and MUSE archetypes
- **Pipeline Orchestrator** (`warden/tools/run_full_pipeline.py`): Coordinates the complete pipeline

### Data Structures
- **MemoryBlock**: Core immutable data structure with hash, summary, topics, skills, dates, and ethics tracking
- **Project Mapping**: Hierarchical project assignment based on content themes and archetype analysis
- **Embeddings**: Vector representations for semantic clustering and search

### Key Patterns
- **Ethics-First Design**: All processing includes PII redaction and consent logging
- **Archetype Mapping**: Content mapped to MUSE archetypes (Guardian, Visionary, etc.)
- **Immutable Blocks**: Each file becomes a hashed, timestamped, immutable memory block
- **Modular Pipeline**: Each component can run independently or as part of full orchestration

## Development Workflows

### Running the Pipeline
```bash
# Full pipeline with all file types
python warden/tools/run_full_pipeline.py --roots ./documents --types .py .md .json .pdf .jpg

# Quick start with defaults
./run_pipeline.sh

# Individual components
python warden/tools/universal_ingest.py --roots ./data --output ./_work/memory_blocks
python warden/tools/cluster_embeddings.py --blocks-dir ./_work/memory_blocks
python warden/tools/assign_projects.py --blocks-dir ./_work/memory_blocks
```

### Output Structure
```
_work/pipeline_output/
├── memory_blocks/           # Raw MemoryBlock JSON files
├── clustering_results/      # Interactive visualizations & cluster data
├── project_assignments/     # Project hierarchies & updated blocks
└── pipeline_summary.txt     # Human-readable execution report
```

### MemoryBlock Schema
```json
{
  "id_hash": "sha256:first_12_chars",
  "summary": "2-4 sentence standalone summary",
  "content": "extracted_content",
  "topics": ["lowercase", "keywords"],
  "skills": ["nano_script_references"],
  "project": "project.hierarchy.id",
  "archetype": "Guardian|Visionary|etc",
  "created_at": "ISO8601_timestamp",
  "ethics": {
    "pii_redacted": true,
    "consent_logged": true,
    "privacy_level": "public|private|sensitive"
  },
  "metadata": {
    "source_file": "original_path",
    "file_type": "extension",
    "size_bytes": 12345
  }
}
```

## Project Conventions

### File Processing
- **OCR for Images**: Uses Tesseract for image-to-text extraction
- **PDF Processing**: PyMuPDF for text and metadata extraction  
- **Code Analysis**: AST parsing for Python, basic text extraction for others
- **Error Resilience**: Individual file failures don't stop pipeline execution

### Dependencies
- Core: `json`, `pathlib`, `hashlib`, `datetime`
- ML: `sklearn`, `numpy`, `plotly` for clustering and visualization
- File Processing: `PyMuPDF`, `pytesseract`, `Pillow`

### Testing
- Test with sample files in `/tmp/test_data/`
- Validate MemoryBlock schema compliance
- Check clustering output and project assignments
- Verify ethics tracking and PII redaction

## Integration Points

### Memory System Integration
- Output MemoryBlocks are ready for vector database ingestion
- Supports semantic search via embeddings
- Compatible with agent memory and retrieval systems

### External Dependencies
- **Tesseract**: For OCR processing of images
- **Ollama**: Optional for advanced embeddings (if configured)
- **Vector Stores**: Output format compatible with standard vector databases

## Key Commands for AI Agents

When working with this codebase:
1. Always run from repository root
2. Use `python -m` pattern for module execution
3. Check `_work/pipeline_output/pipeline_summary.txt` for execution results
4. Monitor ethics compliance in all MemoryBlock outputs
5. Test with small file sets first, then scale up

This system transforms any file collection into a structured, searchable knowledge base while maintaining ethical AI practices and immutable audit trails.