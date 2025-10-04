#!/usr/bin/env python3
"""
Universal File Ingest System for BLOGAGENT
Processes any file format into normalized MemoryBlocks
"""

import os
import sys
import json
import argparse
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_block import MemoryBlock, MemoryBlockFactory, save_memory_block


class UniversalFileProcessor:
    """Processes various file types into structured content"""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',  # Code files
        '.md', '.txt', '.rst',  # Text files
        '.json', '.yaml', '.yml', '.xml', '.csv',  # Data files
        '.html', '.htm',  # Web files
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',  # Images (for OCR)
        '.pdf',  # PDF files
        '.docx', '.doc',  # Word documents
    }
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        self.errors = []
    
    def process_file(self, file_path: Path) -> Optional[MemoryBlock]:
        """Process a single file into a MemoryBlock"""
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = file_path.suffix.lower()
            file_size = file_path.stat().st_size
            
            # Skip very large files (>10MB) to prevent memory issues
            if file_size > 10 * 1024 * 1024:
                print(f"Skipping large file ({file_size} bytes): {file_path}")
                return None
            
            # Extract content based on file type
            content = self._extract_content(file_path, file_ext)
            
            if not content or len(content.strip()) < 10:
                print(f"Skipping file with insufficient content: {file_path}")
                return None
            
            # Create MemoryBlock
            memory_block = MemoryBlockFactory.create_memory_block(
                content=content,
                source_path=str(file_path),
                file_type=file_ext,
                size_bytes=file_size
            )
            
            self.processed_count += 1
            return memory_block
            
        except Exception as e:
            self.error_count += 1
            error_msg = f"Error processing {file_path}: {e}"
            self.errors.append(error_msg)
            print(f"Warning: {error_msg}")
            return None
    
    def _extract_content(self, file_path: Path, file_ext: str) -> str:
        """Extract text content from file based on extension"""
        
        if file_ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']:
            return self._extract_code_content(file_path, file_ext)
        elif file_ext in ['.md', '.txt', '.rst']:
            return self._extract_text_content(file_path)
        elif file_ext in ['.json', '.yaml', '.yml']:
            return self._extract_structured_data(file_path, file_ext)
        elif file_ext in ['.html', '.htm']:
            return self._extract_html_content(file_path)
        elif file_ext in ['.csv']:
            return self._extract_csv_content(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return self._extract_image_content(file_path)
        elif file_ext == '.pdf':
            return self._extract_pdf_content(file_path)
        elif file_ext in ['.docx', '.doc']:
            return self._extract_word_content(file_path)
        else:
            # Try as plain text
            return self._extract_text_content(file_path)
    
    def _extract_code_content(self, file_path: Path, file_ext: str) -> str:
        """Extract content from code files with syntax analysis"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # For Python files, extract docstrings and comments
            if file_ext == '.py':
                try:
                    tree = ast.parse(content)
                    extracted = []
                    
                    # Extract module docstring
                    if (ast.get_docstring(tree)):
                        extracted.append(f"Module: {ast.get_docstring(tree)}")
                    
                    # Extract function and class docstrings
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            docstring = ast.get_docstring(node)
                            if docstring:
                                extracted.append(f"{type(node).__name__} {node.name}: {docstring}")
                    
                    # Combine with original content
                    if extracted:
                        content = "\n".join(extracted) + "\n\n" + content
                        
                except SyntaxError:
                    # If parsing fails, use raw content
                    pass
            
            return content
            
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except:
                return f"Binary file content from {file_path}"
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except:
                return f"Unable to decode text file: {file_path}"
    
    def _extract_structured_data(self, file_path: Path, file_ext: str) -> str:
        """Extract content from structured data files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if file_ext == '.json':
                try:
                    # Parse and pretty-print JSON
                    data = json.loads(content)
                    return f"JSON structure:\n{json.dumps(data, indent=2)}"
                except json.JSONDecodeError:
                    return content
            else:
                return content
                
        except Exception as e:
            return f"Error reading structured data from {file_path}: {e}"
    
    def _extract_html_content(self, file_path: Path) -> str:
        """Extract text content from HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Simple HTML tag removal
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            return clean_content
            
        except Exception as e:
            return f"Error processing HTML file {file_path}: {e}"
    
    def _extract_csv_content(self, file_path: Path) -> str:
        """Extract content from CSV files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Limit to first 50 lines for summary
            preview_lines = lines[:50]
            content = ''.join(preview_lines)
            
            if len(lines) > 50:
                content += f"\n... (truncated, total {len(lines)} lines)"
            
            return f"CSV data preview:\n{content}"
            
        except Exception as e:
            return f"Error processing CSV file {file_path}: {e}"
    
    def _extract_image_content(self, file_path: Path) -> str:
        """Extract text from images using OCR (if available)"""
        try:
            # Try to use pytesseract for OCR if available
            try:
                import pytesseract
                from PIL import Image
                
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                
                if text.strip():
                    return f"OCR extracted text from {file_path.name}:\n{text}"
                else:
                    return f"Image file {file_path.name} (no text detected via OCR)"
                    
            except ImportError:
                # OCR libraries not available
                return f"Image file {file_path.name} (OCR not available - install pytesseract and Pillow for text extraction)"
                
        except Exception as e:
            return f"Image file {file_path.name} (error: {e})"
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text from PDF files"""
        try:
            # Try PyMuPDF first
            try:
                import fitz  # PyMuPDF
                
                doc = fitz.open(file_path)
                text = ""
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
                
                doc.close()
                
                return f"PDF content from {file_path.name}:\n{text}"
                
            except ImportError:
                # Try pdfplumber as alternative
                try:
                    import pdfplumber
                    
                    text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    
                    return f"PDF content from {file_path.name}:\n{text}"
                    
                except ImportError:
                    return f"PDF file {file_path.name} (PDF processing libraries not available - install PyMuPDF or pdfplumber)"
                    
        except Exception as e:
            return f"PDF file {file_path.name} (error: {e})"
    
    def _extract_word_content(self, file_path: Path) -> str:
        """Extract text from Word documents"""
        try:
            # Try python-docx
            try:
                from docx import Document
                
                doc = Document(file_path)
                text = ""
                
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                
                return f"Word document content from {file_path.name}:\n{text}"
                
            except ImportError:
                return f"Word document {file_path.name} (python-docx not available for text extraction)"
                
        except Exception as e:
            return f"Word document {file_path.name} (error: {e})"


class UniversalIngestPipeline:
    """Main pipeline for universal file ingestion"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.processor = UniversalFileProcessor()
        self.memory_blocks = []
    
    def scan_directory(self, root_path: Path, file_types: Optional[List[str]] = None) -> List[Path]:
        """Scan directory for supported files"""
        files = []
        
        if not root_path.exists():
            print(f"Warning: Directory does not exist: {root_path}")
            return files
        
        # Determine which extensions to process
        extensions_to_process = set()
        if file_types:
            extensions_to_process.update(file_types)
        else:
            extensions_to_process = self.processor.SUPPORTED_EXTENSIONS
        
        print(f"Scanning {root_path} for files with extensions: {extensions_to_process}")
        
        # Walk through directory
        for file_path in root_path.rglob('*'):
            if file_path.is_file():
                if file_path.suffix.lower() in extensions_to_process:
                    files.append(file_path)
        
        print(f"Found {len(files)} files to process")
        return files
    
    def process_files(self, files: List[Path]) -> List[MemoryBlock]:
        """Process list of files into MemoryBlocks"""
        memory_blocks = []
        
        print(f"Processing {len(files)} files...")
        
        for i, file_path in enumerate(files, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(files)} files processed")
            
            memory_block = self.processor.process_file(file_path)
            if memory_block:
                memory_blocks.append(memory_block)
        
        return memory_blocks
    
    def save_memory_blocks(self, memory_blocks: List[MemoryBlock]) -> List[Path]:
        """Save MemoryBlocks to output directory"""
        saved_files = []
        blocks_dir = self.output_dir / "memory_blocks"
        
        print(f"Saving {len(memory_blocks)} MemoryBlocks to {blocks_dir}")
        
        for block in memory_blocks:
            try:
                saved_path = save_memory_block(block, blocks_dir)
                saved_files.append(saved_path)
            except Exception as e:
                print(f"Error saving MemoryBlock {block.id_hash}: {e}")
        
        return saved_files
    
    def run_ingestion(self, root_paths: List[str], file_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run complete ingestion pipeline"""
        
        # Collect all files from root paths
        all_files = []
        for root_path_str in root_paths:
            root_path = Path(root_path_str).expanduser().resolve()
            files = self.scan_directory(root_path, file_types)
            all_files.extend(files)
        
        if not all_files:
            print("No files found to process")
            return {'status': 'no_files', 'memory_blocks': 0}
        
        # Process files into MemoryBlocks
        memory_blocks = self.process_files(all_files)
        
        # Save MemoryBlocks
        saved_files = self.save_memory_blocks(memory_blocks)
        
        # Generate summary
        summary = {
            'status': 'completed',
            'total_files_found': len(all_files),
            'memory_blocks_created': len(memory_blocks),
            'memory_blocks_saved': len(saved_files),
            'errors': self.processor.error_count,
            'error_messages': self.processor.errors[:10],  # First 10 errors
            'output_directory': str(self.output_dir)
        }
        
        print("\n" + "="*50)
        print("INGESTION SUMMARY")
        print("="*50)
        print(f"Files found: {summary['total_files_found']}")
        print(f"MemoryBlocks created: {summary['memory_blocks_created']}")
        print(f"MemoryBlocks saved: {summary['memory_blocks_saved']}")
        print(f"Errors: {summary['errors']}")
        print(f"Output directory: {summary['output_directory']}")
        
        return summary


def main():
    """Main entry point for universal ingestion"""
    parser = argparse.ArgumentParser(description="Universal File Ingest Pipeline")
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--types', nargs='+',
                       help='Specific file types to process (e.g., .py .md .json)')
    parser.add_argument('--output', default='./_work/pipeline_output',
                       help='Output directory for MemoryBlocks')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UniversalIngestPipeline(args.output)
    
    # Run ingestion
    summary = pipeline.run_ingestion(args.roots, args.types)
    
    # Save summary
    summary_path = Path(args.output) / "ingestion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nIngestion summary saved to: {summary_path}")
    
    # Exit with appropriate code
    if summary['status'] == 'completed' and summary['memory_blocks_created'] > 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()