# MUSE Pantheon Universal Ingest Pipeline

🌟 **Transform any file format into immutable MemoryBlocks for AI agent memory systems, IP proof documentation, and semantic search.**

## 🚀 Quick Start

```bash
# Install basic dependencies (numpy only required)
pip install numpy

# Process current directory
./run_pipeline.py

# Process specific directory (like MUSE_UNIFIED from the problem statement)
./run_pipeline.py --target /Users/jakobaxelpaper/MUSE_UNIFIED

# Check what's available
./run_pipeline.py --check-deps
```

## ✨ What This System Does

The **MUSE Pantheon Universal Ingest Pipeline** is a comprehensive AI coding agent system that:

- 📁 **Processes ANY file format** - images, videos, documents, code, archives, PDFs
- 🧠 **Creates structured MemoryBlocks** - immutable, atomic data with full metadata  
- 🎯 **Auto-assigns projects** - intelligent categorization based on content analysis
- 🔍 **Semantic clustering** - groups similar content using ML techniques
- 📊 **Interactive visualizations** - explore your data relationships
- 🛡️ **Ethics & consent tracking** - built-in PII redaction and validation
- 📝 **IP proof documentation** - timestamped, immutable records for legal evidence

## 🏗️ What Gets Created

Every file becomes a **MemoryBlock** like this:

```json
{
  "id_hash": "sha256:abc123...",
  "summary": "Code file with 3 classes, 8 functions, 5 imports", 
  "content": "extracted file content",
  "topics": ["python", "agent", "memory"],
  "project": {
    "project_id": "mindprint.core",
    "archetype": "Builder", 
    "confidence_score": 0.9
  },
  "ethics": {"pii_redacted": true, "consent_logged": true},
  "source": {"file_path": "/path/to/file.py", "file_size": 1024}
}
```

## 🎭 MUSE Archetype System

Content automatically maps to **12 archetypes**:

| Archetype | Purpose | File Types |
|-----------|---------|------------|
| **Guardian** | Security, legal, validation | Contracts, paystubs, legal docs |
| **Vision** | Images, visual content | .jpg, .png, .mp4, OCR results |
| **Memory** | Storage, databases | .json, .sqlite, config files |
| **Builder** | Code, development | .py, .js, .java, .cpp |
| **Scribe** | Documentation | .md, .txt, .pdf, .docx |
| **Analyst** | Data analysis | .csv, reports, statistics |

## 📊 Real Results

**From our test run:**
- ✅ Processed 4 sample files (Python, JSON, Markdown, Text)
- ✅ Generated 4 MemoryBlocks with complete metadata
- ✅ Auto-assigned archetypes: Memory, Builder, Scribe
- ✅ Created searchable topic index with 38 unique terms
- ✅ Generated comprehensive statistics and reports

## 🎯 Perfect For

### **AI Agent Memory Systems**
```python
# Load MemoryBlocks for AI agent retrieval
collection = MemoryBlockCollection() 
collection.load_from_directory("memory_blocks/")
agent.load_memory(collection)
```

### **IP Proof Documentation** 
```bash
# Convert founder journal entries to legal timeline
./run_pipeline.py --target ~/Documents/founder_journal
# Creates timestamped, immutable records for IP evidence
```

### **Content Discovery & Organization**
```bash
# Automatically organize large file collections
./run_pipeline.py --target ~/Documents --filter-docs
# Groups similar content, finds themes, assigns projects
```

## 🛠️ Advanced Usage

```bash
# Full control over pipeline
python warden/tools/run_full_pipeline.py --roots /path/to/files --types .py .md .pdf

# Individual phases  
python warden/tools/universal_ingest.py --roots ./data --output ./memory_blocks
python warden/tools/cluster_embeddings.py --blocks-dir ./memory_blocks --output-dir ./clusters
python warden/tools/assign_projects.py --blocks-dir ./memory_blocks --output-dir ./projects

# With file type filters
./run_pipeline.py --target ./src --filter-code     # Code files only
./run_pipeline.py --target ./docs --filter-docs    # Documents only  
./run_pipeline.py --target ./media --filter-media  # Images/videos only
```

## 📈 Enhanced Features (Optional)

Install full dependencies for advanced features:

```bash
pip install -r requirements.txt
```

**Unlocks:**
- 🧠 **Advanced clustering** - K-means with automatic optimization
- 📊 **Interactive visualizations** - Plotly dashboards 
- 🔍 **OCR text extraction** - Extract text from images
- 📄 **Enhanced PDF processing** - Better text extraction

## 📁 Output Structure

```
_work/pipeline_output/
├── memory_blocks/                   # MemoryBlock JSON files + stats
├── clustering_results/              # Interactive visualizations
├── project_assignments/             # Project hierarchies  
├── pipeline_summary.txt             # Human-readable report
└── pipeline_report.json             # Machine-readable data
```

## 🤖 AI Agent Integration

This system is designed for **GitHub Copilot, VS Code extensions, and custom AI agents**. 

👀 **See `.github/copilot-instructions.md`** for detailed AI agent guidance.

## 📖 Documentation

- **📋 [PIPELINE_DOCS.md](PIPELINE_DOCS.md)** - Comprehensive technical documentation
- **🤖 [.github/copilot-instructions.md](.github/copilot-instructions.md)** - AI agent guidance
- **⚙️ [requirements.txt](requirements.txt)** - Dependencies and installation

## 🎉 Ready to Transform Your Files?

```bash
# Start with your own data
./run_pipeline.py --target /path/to/your/files

# Or try the sample data
./run_pipeline.py --target ./sample_data
```

**Transform any file collection into structured, searchable, AI-ready MemoryBlocks in minutes!**
