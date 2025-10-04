# MUSE Pantheon Universal Ingest Pipeline - AI Coding Assistant Instructions

## Project Overview
This repository implements a comprehensive universal file ingest pipeline that converts various file formats into structured MemoryBlocks for IP proof, semantic search, and AI agent memory systems.

## Architecture & Key Components

### Core Pipeline Flow
1. **Universal Ingestion** (`warden/tools/universal_ingest.py`) - Extracts content from 15+ file formats (code, docs, images, PDFs, media)
2. **Semantic Clustering** (`warden/tools/cluster_embeddings.py`) - Groups content using TF-IDF + K-means clustering
3. **Project Assignment** (`warden/tools/assign_projects.py`) - Auto-assigns project hierarchies based on content themes
4. **Pipeline Orchestration** (`warden/tools/run_full_pipeline.py`) - Coordinates all phases with error handling

### MemoryBlock Schema
Central data structure in `warden/schema/memory_block.py`:
- **Identity**: `id_hash`, `summary`, `content`
- **Semantics**: `topics`, `skills`, `archetype` (12 MUSE archetypes: Builder, Vision, Guardian, etc.)
- **Ethics**: `consent_logged`, `pii_redacted`, `ethics_status`
- **Lineage**: `source_file`, `links`, `embeddings`

### MUSE Archetypes System
Files are classified into 12 archetypes:
- **Builder**: Code files (.py, .js, .java)
- **Vision**: Images/videos (OCR extraction)
- **Guardian**: Legal/financial docs
- **Scholar**: Research/analysis content
- Content-based assignment in `get_archetype_for_content()`

## Critical Workflows

### Quick Launch
```bash
python run_pipeline.py run                    # Process current directory with defaults
python run_pipeline.py custom --roots /path   # Custom directory processing
python run_pipeline.py status                 # Check pipeline status
```

### Full Pipeline Execution
```bash
python warden/tools/run_full_pipeline.py --roots /Users/username/Documents --types .pdf .py .md
```

### Individual Phase Execution
```bash
# Ingestion only
python warden/tools/universal_ingest.py --roots ./data --output ./blocks

# Clustering existing blocks  
python warden/tools/cluster_embeddings.py --blocks-dir ./blocks --output-dir ./clusters

# Project assignment
python warden/tools/assign_projects.py --blocks-dir ./blocks --output-dir ./assignments
```

## Project-Specific Patterns

### File Processing Patterns
- **OCR Integration**: Images processed via Tesseract in `_extract_image_ocr()`
- **PDF Handling**: PyMuPDF extraction limited to 50 pages for performance
- **Code Analysis**: Language detection + pattern extraction in `_extract_code_topics()`
- **Error Resilience**: Continue processing on individual file failures

### Output Structure
```
_work/pipeline_output/
├── memory_blocks/           # Individual MemoryBlock JSON files (atomic units)
├── clustering_results/      # Interactive Plotly visualizations + cluster analysis
├── project_assignments/     # Project hierarchies + updated blocks
└── pipeline_summary.txt     # Human-readable execution summary
```

### Ethics & Consent Tracking
- All MemoryBlocks have `ethics_status`, `consent_logged`, `pii_redacted` fields
- Builder pattern enforces validation: `MemoryBlockBuilder().with_ethics()`
- IP proof focus: immutable blocks with source hashing

## Integration Points

### Dependencies (requirements.txt)
- **ML/NLP**: scikit-learn, numpy, pandas (clustering)
- **OCR**: pytesseract, opencv-python (image text extraction)
- **Documents**: PyMuPDF, python-docx, openpyxl (office docs)
- **Visualization**: plotly (interactive cluster plots)

### External Service Integration
- **Ollama**: Ready for LLM embeddings (commented in clustering)
- **Whisper**: Audio/video transcription hooks (basic metadata only currently)
- **Vector Stores**: MemoryBlocks export embeddings for semantic search

### Cross-Component Communication
- **JSON Lines**: MemoryBlock serialization format
- **File-based**: Phases communicate via JSON files in `_work/`
- **Subprocess Orchestration**: Main orchestrator uses subprocess calls

## Development Patterns

### Adding New File Types
1. Add extractor method to `UniversalFileExtractor` class
2. Update `supported_types` mapping in constructor
3. Add archetype mapping in `MUSE_ARCHETYPES` dict
4. Test with sample files

### Extending Project Assignment
1. Add patterns to `project_patterns` in `ProjectAssignmentEngine`
2. Update `_determine_category()` logic
3. Add archetype mappings in `archetype_projects`

### Error Handling Convention
- Log errors but continue processing: `logger.error()` + return empty results
- Graceful degradation: Fall back to generic extraction
- Phase isolation: Individual phase failures don't break pipeline

## Key Reference Files

- **Core Schema**: `warden/schema/memory_block.py` - MemoryBlock definition & builders
- **Main Orchestrator**: `warden/tools/run_full_pipeline.py` - Complete pipeline control
- **Quick Launcher**: `run_pipeline.py` - Simplified interface for common operations
- **Documentation**: `PIPELINE_README.md` - Comprehensive usage guide

## Testing & Validation

### Environment Validation
```bash
python warden/tools/run_full_pipeline.py --validate-only
```

### Sample Data Testing
- Create small test dataset in `test_data/`
- Run pipeline on test data to validate output format
- Check MemoryBlock JSON schema compliance

## Common Patterns for AI Agents

1. **Loading MemoryBlocks**: Use `MemoryBlock.from_json()` for deserialization
2. **Filtering by Archetype**: `blocks = [b for b in blocks if b.archetype == 'Builder']`
3. **Project-based Queries**: Filter by `project_id` for thematic content
4. **Embedding Integration**: Access `block.embeddings` for vector operations
5. **Ethics Compliance**: Check `block.ethics_status` before content access