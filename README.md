# BLOGAGENT - MUSE Pantheon Universal Ingest Pipeline

A comprehensive universal ingest pipeline that processes any file format into structured MemoryBlocks for intellectual property proof, semantic search, and agent memory integration.

## Quick Start

### Simple Usage
```bash
# Process current directory with all file types
./run_pipeline.py
```

### Advanced Usage
```bash
# Process specific directories
python warden/tools/run_full_pipeline.py --roots /path/to/docs /path/to/code

# Process only specific file types
python warden/tools/run_full_pipeline.py --roots /path/to/files --types .py .md .json .pdf
```

## Features

- **Universal File Processing**: Handles code, documents, images, PDFs, emails, and more
- **Structured MemoryBlocks**: Standardized format with metadata, ethics tracking, and project assignments
- **Semantic Clustering**: Groups similar content using TF-IDF and K-means
- **Auto Project Assignment**: Maps content to project hierarchies using MUSE archetypes
- **Interactive Visualizations**: Plotly-based cluster exploration and topic analysis
- **Error Resilience**: Continues processing even if individual files fail

## Documentation

- See [PIPELINE_README.md](PIPELINE_README.md) for comprehensive usage documentation
- See [.github/copilot-instructions.md](.github/copilot-instructions.md) for AI agent guidance

## Installation

```bash
pip install -r requirements.txt
```

## MUSE Archetype System

Content is classified using 5 archetypes:
- **Vision**: Images, visual design, UI/UX materials
- **Builder**: Code, technical docs, system architecture  
- **Sage**: Knowledge documents, research, insights
- **Guardian**: Security, legal, compliance materials
- **Connector**: APIs, integrations, communication channels
