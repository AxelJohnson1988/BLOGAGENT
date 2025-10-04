# BLOGAGENT Universal Ingest Pipeline

## Overview

BLOGAGENT is a universal ingest pipeline inspired by the MUSE Pantheon system that processes any file format into structured MemoryBlocks for semantic search, AI agent memory, and IP proof documentation.

## Features

- **Universal File Processing**: Handles code files (.py, .js, .ts, etc.), documents (.md, .txt, .pdf), images, and structured data
- **MemoryBlock Creation**: Converts all content into standardized, immutable memory blocks with ethics tracking
- **Semantic Clustering**: Groups similar content using TF-IDF embeddings and K-Means clustering
- **Project Assignment**: Automatically maps content to project hierarchies and MUSE archetypes
- **Interactive Visualizations**: Generates Plotly-based cluster visualizations
- **Ethics-First Design**: Built-in PII redaction and consent tracking

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AxelJohnson1988/BLOGAGENT.git
cd BLOGAGENT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For enhanced file processing (optional):
```bash
# PDF processing
pip install PyMuPDF pdfplumber

# Image OCR (requires Tesseract)
pip install pytesseract Pillow

# Word documents
pip install python-docx
```

### Running the Pipeline

**Option 1: Quick Start (Recommended)**
```bash
./run_pipeline.sh
```

**Option 2: Custom Configuration**
```bash
python warden/tools/run_full_pipeline.py --roots ./documents ./code --types .py .md .json .pdf
```

**Option 3: Individual Components**
```bash
# Just ingestion
python warden/tools/universal_ingest.py --roots ./data --output ./_work/pipeline_output

# Just clustering
python warden/tools/cluster_embeddings.py --blocks-dir ./_work/pipeline_output/memory_blocks

# Just project assignment
python warden/tools/assign_projects.py --blocks-dir ./_work/pipeline_output/memory_blocks
```

## Output Structure

After running the pipeline, you'll find results in `_work/pipeline_output/`:

```
_work/pipeline_output/
├── memory_blocks/                    # Individual MemoryBlock JSON files
│   ├── sha256_abc123def.json
│   └── sha256_xyz789ghi.json
├── clustering_results/
│   ├── cluster_analysis.json         # Cluster characteristics and summaries
│   ├── cluster_assignments.json     # Block-to-cluster mappings
│   └── cluster_visualization.html   # Interactive cluster plot
├── project_assignments/
│   ├── project_hierarchy.json       # Project structure and statistics
│   ├── assignment_summary.json      # Assignment results
│   └── updated_memory_blocks/       # MemoryBlocks with updated projects
└── pipeline_summary.txt             # Human-readable execution summary
```

## MemoryBlock Schema

Each processed file becomes a MemoryBlock with this structure:

```json
{
  "id_hash": "sha256:abc123def456",
  "summary": "Concise 2-4 sentence summary of content",
  "content": "Full extracted text content",
  "topics": ["keyword1", "keyword2", "keyword3"],
  "skills": ["nano_warden_script_reference.py"],
  "project": "project.subcategory",
  "archetype": "Guardian|Visionary|Analyst|Creator|etc",
  "created_at": "2024-01-15T10:30:00Z",
  "ethics": {
    "pii_redacted": true,
    "consent_logged": true,
    "privacy_level": "public|private|sensitive"
  },
  "metadata": {
    "source_file": "/path/to/original/file.py",
    "file_type": ".py",
    "size_bytes": 4096,
    "last_modified": "2024-01-15T10:30:00Z",
    "encoding": "utf-8"
  },
  "links": []
}
```

## MUSE Archetypes

The system maps content to 10 MUSE Pantheon archetypes:

- **Guardian**: Security, privacy, compliance, protection
- **Visionary**: Strategy, planning, future-focused content
- **Analyst**: Data analysis, research, metrics, evaluation
- **Creator**: Development, building, design, implementation
- **Connector**: Integration, networking, relationships
- **Optimizer**: Performance, efficiency, improvement
- **Explorer**: Research, experimentation, discovery
- **Synthesizer**: Combining, merging, consolidating ideas
- **Protector**: Backup, preservation, maintenance
- **Innovator**: New approaches, creativity, breakthroughs

## Project Categories

Content is automatically categorized into project hierarchies:

- `ai.*` - Artificial intelligence and machine learning
- `blog.*` - Blog posts, articles, content writing
- `legal.*` - Legal documents, contracts, compliance
- `business.*` - Business strategy, operations, finance
- `dev.*` - Software development, programming
- `research.*` - Research projects, studies, analysis
- `docs.*` - Documentation, guides, manuals

## Configuration

### Supported File Types

By default, the pipeline processes:
- Code: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.h`
- Text: `.md`, `.txt`, `.rst`
- Data: `.json`, `.yaml`, `.yml`, `.xml`, `.csv`
- Web: `.html`, `.htm`
- Images: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp` (with OCR)
- Documents: `.pdf`, `.docx`, `.doc`

### Customizing Processing

You can specify which file types to process:

```bash
# Only Python and Markdown files
./run_pipeline.sh --types .py .md

# Only PDFs from Documents folder
python warden/tools/run_full_pipeline.py --roots ~/Documents --types .pdf
```

### Ethics and Privacy

The system automatically:
- Redacts potential PII (Social Security Numbers, passwords, etc.)
- Sets privacy levels based on content analysis
- Logs consent status for each processed file
- Maintains audit trails with immutable hashes

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies with `pip install -r requirements.txt`

2. **OCR not working**: Install Tesseract:
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   ```

3. **PDF processing fails**: Install PDF libraries:
   ```bash
   pip install PyMuPDF pdfplumber
   ```

4. **Memory issues with large files**: The pipeline automatically skips files >10MB. Adjust in `universal_ingest.py` if needed.

5. **No clusters created**: Ensure you have enough diverse content. The system needs at least 2 documents to cluster.

### Debugging

Check the pipeline summary for detailed results:
```bash
cat _work/pipeline_output/pipeline_summary.txt
```

Individual component logs are printed to console during execution.

## Integration

### With Vector Databases

MemoryBlocks are designed for easy vector database integration:

```python
from warden.memory_block import load_all_memory_blocks

# Load processed blocks
blocks = load_all_memory_blocks(Path("_work/pipeline_output/memory_blocks"))

# Extract content for embedding
texts = [block.content for block in blocks]
metadata = [block.to_dict() for block in blocks]

# Insert into your vector database
# vector_db.insert(texts, metadata)
```

### With AI Agents

Use MemoryBlocks for agent memory and retrieval:

```python
# Search by project
legal_blocks = [b for b in blocks if b.project.startswith("legal")]

# Search by archetype
guardian_blocks = [b for b in blocks if b.archetype == "Guardian"]

# Search by topics
security_blocks = [b for b in blocks if "security" in b.topics]
```

## Examples

### Process All Files in Current Directory
```bash
./run_pipeline.sh --roots .
```

### Process Specific File Types from Multiple Directories
```bash
python warden/tools/run_full_pipeline.py \
  --roots ./src ./docs ./examples \
  --types .py .md .rst .json
```

### Extract Only Images with OCR
```bash
python warden/tools/universal_ingest.py \
  --roots ./screenshots \
  --types .png .jpg .jpeg \
  --output ./_work/image_extraction
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

## Support

- Check the troubleshooting section above
- Review pipeline logs and summaries
- Open an issue on GitHub with reproduction steps

---

**Happy ingesting!** 🚀