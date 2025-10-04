#!/bin/bash
"""
Quick launcher for BLOGAGENT Universal Ingest Pipeline
Runs the complete pipeline with sensible defaults
"""

# Default configuration
DEFAULT_ROOTS=("." "./docs" "./examples")
DEFAULT_TYPES=(".py" ".md" ".json" ".txt" ".html")

# Parse command line arguments
ROOTS=()
TYPES=()
WORKSPACE="."

while [[ $# -gt 0 ]]; do
    case $1 in
        --roots)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ROOTS+=("$1")
                shift
            done
            ;;
        --types)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TYPES+=("$1")
                shift
            done
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --help|-h)
            echo "BLOGAGENT Universal Ingest Pipeline Quick Launcher"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --roots DIR1 DIR2 ...    Root directories to scan (default: . ./docs ./examples)"
            echo "  --types .ext1 .ext2 ...  File types to process (default: .py .md .json .txt .html)"
            echo "  --workspace DIR          Workspace directory (default: .)"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use defaults"
            echo "  $0 --roots ./src ./docs              # Scan specific directories"
            echo "  $0 --types .py .md                   # Process only Python and Markdown files"
            echo "  $0 --roots ~/Documents --types .pdf  # Scan Documents for PDFs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Use defaults if not specified
if [ ${#ROOTS[@]} -eq 0 ]; then
    ROOTS=("${DEFAULT_ROOTS[@]}")
fi

if [ ${#TYPES[@]} -eq 0 ]; then
    TYPES=("${DEFAULT_TYPES[@]}")
fi

# Change to workspace directory
cd "$WORKSPACE" || {
    echo "Error: Cannot change to workspace directory: $WORKSPACE"
    exit 1
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not found"
    exit 1
fi

# Check if the pipeline script exists
PIPELINE_SCRIPT="warden/tools/run_full_pipeline.py"
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo "Error: Pipeline script not found: $PIPELINE_SCRIPT"
    echo "Make sure you're running this from the BLOGAGENT repository root"
    exit 1
fi

# Display configuration
echo "🎯 BLOGAGENT Universal Ingest Pipeline"
echo "======================================"
echo "Workspace: $WORKSPACE"
echo "Scanning roots: ${ROOTS[*]}"
echo "File types: ${TYPES[*]}"
echo ""

# Build command
CMD=(python3 "$PIPELINE_SCRIPT" --workspace "$WORKSPACE" --roots "${ROOTS[@]}")

if [ ${#TYPES[@]} -gt 0 ]; then
    CMD+=(--types "${TYPES[@]}")
fi

# Run the pipeline
echo "Executing: ${CMD[*]}"
echo ""

exec "${CMD[@]}"