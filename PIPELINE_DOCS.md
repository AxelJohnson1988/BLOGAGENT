# MUSE Pantheon Universal Ingest Pipeline

## 🌟 Overview

The **MUSE Pantheon Universal Ingest Pipeline** is a comprehensive AI coding agent system designed for universal file ingestion, semantic clustering, and structured memory creation. It transforms any file format into immutable MemoryBlock objects for AI agent memory systems, IP proof documentation, and semantic search.

## 🎯 Key Features

- **Universal File Support**: Processes any file format (images, videos, documents, code, archives)
- **Semantic Clustering**: Groups similar content using advanced ML techniques
- **Project Assignment**: Auto-assigns project IDs based on content analysis
- **Memory Block System**: Creates immutable, atomic data structures with ethics tracking
- **Interactive Visualizations**: Generates cluster exploration dashboards
- **IP Proof Documentation**: Structured timeline for intellectual property evidence

## 🏗️ Architecture

### Core Components

```
📁 MUSE Pantheon System
├── 📋 memory_block.py              # Core MemoryBlock schema
├── 🔧 warden/tools/                # Pipeline components
│   ├── universal_ingest.py         # File extraction & MemoryBlock creation
│   ├── cluster_embeddings.py       # Semantic clustering & visualization  
│   ├── assign_projects.py          # Project assignment & hierarchy
│   └── run_full_pipeline.py        # Main orchestrator
├── 🚀 run_pipeline.py              # Quick launcher
├── 📖 .github/copilot-instructions.md  # AI agent guidance
└── 📊 _work/pipeline_output/       # Generated results
```

### Data Flow

```
📄 Files → 🔍 Extract → 🧠 MemoryBlocks → 🎯 Cluster → 📋 Projects → 📊 Reports
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Minimal install (basic functionality)
pip install numpy

# Full install (all features)
pip install -r requirements.txt

# Optional external tools for enhanced processing
# macOS:
brew install tesseract poppler
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr poppler-utils
```

### 2. Run the Pipeline

**Quick Launch (Recommended):**
```bash
# Process current directory
./run_pipeline.py

# Process specific directory (like MUSE_UNIFIED from problem statement)
./run_pipeline.py --target /Users/jakobaxelpaper/MUSE_UNIFIED

# Process with file filters
./run_pipeline.py --target ./documents --filter-docs
```

**Advanced Usage:**
```bash
# Full control over pipeline
python warden/tools/run_full_pipeline.py --roots /path/to/files --types .py .md .pdf

# Individual pipeline phases
python warden/tools/universal_ingest.py --roots ./data --output ./memory_blocks
python warden/tools/cluster_embeddings.py --blocks-dir ./memory_blocks --output-dir ./clusters
python warden/tools/assign_projects.py --blocks-dir ./memory_blocks --output-dir ./projects
```

### 3. Review Results

```bash
# Human-readable summary
cat _work/pipeline_output/pipeline_summary.txt

# Interactive cluster visualization
open _work/pipeline_output/clustering_results/cluster_visualization.html

# Project assignments
cat _work/pipeline_output/project_assignments/project_summary.md
```

## 📋 MemoryBlock Schema

Every processed file becomes a MemoryBlock with this structure:

```json
{
  "id_hash": "sha256:first_12_chars",
  "summary": "2-4 sentence standalone summary",
  "content": "extracted_text_or_metadata", 
  "topics": ["lowercase", "keywords"],
  "skills": ["nano_script_references"],
  "project": {
    "project_id": "assigned_project_id",
    "archetype": "Guardian|Vision|Warden|etc",
    "confidence_score": 0.8,
    "assignment_reason": "explanation"
  },
  "created_at": "2024-09-17T09:00:00Z",
  "ethics": {
    "pii_redacted": true,
    "consent_logged": true,
    "validation_status": "passed"
  },
  "source": {
    "file_path": "original_file_location",
    "file_type": "detected_format",
    "file_size": 1024,
    "created_at": "2024-09-17T08:00:00Z"
  },
  "links": [],
  "metadata": {}
}
```

## 🎭 MUSE Archetype System

Content is automatically mapped to one of 12 archetypes:

- **Guardian**: Security, ethics, validation, legal documents
- **Vision**: Images, visual content, OCR results, media
- **Warden**: Orchestration, pipeline management, workflows
- **Memory**: Storage, retrieval, embedding systems, databases
- **Scribe**: Documentation, writing, content creation
- **Analyst**: Data analysis, clustering, insights, reports
- **Builder**: Code, construction, development, programming
- **Explorer**: Discovery, research, investigation, archives
- **Mentor**: Teaching, guidance, knowledge transfer
- **Connector**: Integration, communication, linking systems
- **Sage**: Wisdom, deep knowledge, philosophy
- **Creator**: Innovation, art, creative expression

## 📁 File Type Support

### Text & Documents
- **Code files**: `.py`, `.js`, `.ts`, `.java`, etc. → Extract functions, classes, imports
- **Documents**: `.txt`, `.md`, `.pdf`, `.docx` → Extract text content and structure
- **Data files**: `.json`, `.yaml`, `.csv` → Parse structure and content

### Media & Visual
- **Images**: `.jpg`, `.png`, `.gif` → OCR text extraction (requires tesseract)
- **Videos**: `.mp4`, `.mov`, `.avi` → Metadata extraction (future: transcription)
- **Audio**: `.mp3`, `.wav`, `.m4a` → Metadata extraction (future: transcription)

### Archives & Special
- **Archives**: `.zip`, `.tar`, `.gz` → Content listing (future: recursive extraction)
- **Web files**: `.html`, `.xml` → Text extraction and structure parsing

## 🔧 Pipeline Phases

### Phase 1: Universal Ingestion
- Scans directories recursively
- Detects file types automatically
- Extracts content using appropriate parsers
- Creates MemoryBlocks with metadata

### Phase 2: Semantic Clustering
- Uses TF-IDF vectorization for text analysis
- Applies K-Means clustering (with automatic cluster count optimization)
- Generates 2D embeddings via PCA for visualization
- Creates interactive Plotly dashboards

### Phase 3: Project Assignment
- Analyzes content themes and patterns
- Matches against predefined project templates
- Assigns archetype-based categorization
- Builds project hierarchy and relationships

### Phase 4: Reporting & Visualization
- Generates human-readable summaries
- Creates machine-readable JSON reports
- Produces interactive cluster visualizations
- Builds project assignment documentation

## 🎛️ Configuration Options

### File Type Filtering
```bash
# Document files only
./run_pipeline.py --filter-docs

# Code files only  
./run_pipeline.py --filter-code

# Media files only
./run_pipeline.py --filter-media

# Custom types
python warden/tools/run_full_pipeline.py --types .py .md .json
```

### Output Customization
```bash
# Custom workspace
python warden/tools/run_full_pipeline.py --workspace /custom/path

# Custom cluster count
python warden/tools/cluster_embeddings.py --num-clusters 12
```

## 🔍 Troubleshooting

### Common Issues

**"No MemoryBlocks found"**
- Check that target directory exists and contains supported file types
- Verify file permissions allow reading

**"Clustering failed"**
- Install scikit-learn: `pip install scikit-learn`
- For minimal systems, basic clustering fallback is used automatically

**"OCR not working"**
- Install tesseract: `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Linux)
- Images will still be processed but without text extraction

**"PDF extraction limited"**
- Install poppler: `brew install poppler` (macOS) or `apt-get install poppler-utils` (Linux)
- PDFs will still be processed but with basic metadata only

### Dependency Status Check
```bash
# Check what's available
./run_pipeline.py --check-deps
python warden/tools/run_full_pipeline.py --check-deps
```

## 📊 Output Structure

```
_work/pipeline_output/
├── memory_blocks/                   # Individual MemoryBlock JSON files
│   ├── memory_block_abc123.json     # Generated MemoryBlocks
│   ├── memory_blocks.jsonl          # JSON Lines format
│   ├── ingest_stats.json            # Ingestion statistics
│   └── collection_summary.json      # Collection overview
├── clustering_results/              # Clustering analysis
│   ├── clusters.json                # Cluster assignments & metadata
│   ├── cluster_visualization.html   # Interactive scatter plot
│   └── cluster_summary.html         # Cluster size breakdown
├── project_assignments/             # Project categorization
│   ├── updated_blocks/              # MemoryBlocks with project assignments
│   ├── project_hierarchy.json       # Project structure & relationships
│   ├── assignment_stats.json        # Assignment statistics
│   └── project_summary.md           # Human-readable project breakdown
├── pipeline_summary.txt             # Human-readable execution summary
└── pipeline_report.json             # Machine-readable execution report
```

## 🎯 Use Cases

### IP Proof Documentation
Convert files into timestamped, immutable MemoryBlocks for intellectual property evidence:
```python
# Week journal format for IP proof
block.get_week_journal_format()
```

### AI Agent Memory Systems
Feed MemoryBlocks directly into AI agent memory for retrieval and reasoning:
```python
collection = MemoryBlockCollection()
collection.load_from_directory("memory_blocks/")
```

### Content Discovery & Organization
Use semantic clustering to discover themes and organize large file collections automatically.

### Research & Analysis
Query and analyze patterns across diverse file types with unified MemoryBlock interface.

## 🤝 Integration with AI Agents

The system is designed for seamless integration with AI coding agents like GitHub Copilot, VS Code extensions, and custom agents. See `.github/copilot-instructions.md` for detailed guidance.

## 📈 Extending the System

### Adding New File Types
1. Add extraction logic to `UniversalFileExtractor` in `universal_ingest.py`
2. Map file extension to archetype in `FILE_TYPE_ARCHETYPES`
3. Update documentation

### Custom Project Patterns
1. Add patterns to `ProjectMatcher._initialize_project_patterns()` in `assign_projects.py`
2. Define matching keywords and confidence scores
3. Test with sample files

### New Archetypes
1. Add to `ARCHETYPES` dictionary in `memory_block.py`
2. Define themes in `ProjectMatcher._initialize_archetype_themes()`
3. Update file type mappings

---

For questions or issues, refer to the `.github/copilot-instructions.md` for AI agent guidance or check the troubleshooting section above.