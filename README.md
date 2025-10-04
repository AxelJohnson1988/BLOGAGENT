# BLOGAGENT - Universal Ingest Pipeline

A comprehensive universal ingest pipeline inspired by the MUSE Pantheon system that processes any file format into structured MemoryBlocks for semantic search, AI agent memory, and IP proof documentation.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/AxelJohnson1988/BLOGAGENT.git
cd BLOGAGENT
pip install -r requirements.txt

# Run the full pipeline
./run_pipeline.sh --roots ./documents --types .py .md .json .pdf
```

## 📋 Features

- **Universal File Processing**: Code, documents, images, PDFs, structured data
- **MemoryBlock Creation**: Standardized, immutable blocks with ethics tracking
- **Semantic Clustering**: TF-IDF embeddings with K-Means clustering
- **Project Assignment**: Automatic mapping to MUSE archetypes and project hierarchies
- **Interactive Visualizations**: Plotly-based cluster exploration
- **Ethics-First Design**: Built-in PII redaction and consent tracking

## 📊 Output

The pipeline generates:
- **MemoryBlocks**: Individual JSON files with structured content
- **Cluster Analysis**: Semantic groupings with interactive visualizations
- **Project Assignments**: Hierarchical categorization and archetype mapping
- **Comprehensive Reports**: Human-readable summaries and statistics

## 📚 Documentation

See [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for detailed usage, configuration, and integration instructions.

## 🎯 Architecture

Built with modular components following MUSE Pantheon patterns:
- Universal ingestion with ethics tracking
- Semantic clustering and embeddings
- Project assignment with archetype mapping
- Comprehensive reporting and visualization

Perfect for AI agents, knowledge management, and IP documentation workflows.
