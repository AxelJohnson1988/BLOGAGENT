#!/usr/bin/env python3
"""
MUSE Pantheon Universal File Ingest Pipeline
Processes any file format into structured MemoryBlock objects.
"""
import os
import sys
import argparse
import json
import hashlib
import mimetypes
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
import logging

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from memory_block import MemoryBlock, MemoryBlockCollection, FILE_TYPE_ARCHETYPES, ARCHETYPES

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UniversalFileExtractor:
    """Universal file content extractor supporting multiple formats"""
    
    def __init__(self):
        self.supported_extractors = {
            'text': self._extract_text,
            'code': self._extract_code,
            'image': self._extract_image,
            'pdf': self._extract_pdf,
            'html': self._extract_html,
            'json': self._extract_json,
            'zip': self._extract_zip,
            'video': self._extract_video,
            'audio': self._extract_audio
        }
    
    def detect_file_type(self, file_path: Path) -> str:
        """Detect file type for appropriate extraction method"""
        suffix = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Code files
        if suffix in ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.cpp', '.c', '.java']:
            return 'code'
        
        # Text files
        if suffix in ['.txt', '.md', '.rst', '.log']:
            return 'text'
        
        # Images
        if suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            return 'image'
        
        # PDFs
        if suffix == '.pdf':
            return 'pdf'
        
        # HTML/XML
        if suffix in ['.html', '.htm', '.xml']:
            return 'html'
        
        # JSON/YAML
        if suffix in ['.json', '.yaml', '.yml']:
            return 'json'
        
        # Archives
        if suffix in ['.zip', '.tar', '.gz', '.rar']:
            return 'zip'
        
        # Video
        if suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return 'video'
        
        # Audio
        if suffix in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            return 'audio'
        
        # Default to text for unknown types
        return 'text'
    
    def extract_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from file based on detected type"""
        try:
            file_type = self.detect_file_type(file_path)
            extractor = self.supported_extractors.get(file_type, self._extract_text)
            
            result = extractor(file_path)
            result['file_type'] = file_type
            result['original_path'] = str(file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract from {file_path}: {e}")
            return {
                'content': f"Failed to extract content: {e}",
                'summary': f"Error processing {file_path.name}",
                'topics': ['error', 'extraction_failed'],
                'file_type': 'error',
                'original_path': str(file_path)
            }
    
    def _extract_text(self, file_path: Path) -> Dict[str, Any]:
        """Extract from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Generate summary (first few lines or truncated content)
            lines = content.split('\n')
            summary = ' '.join(lines[:3]).strip()
            if len(summary) > 200:
                summary = summary[:200] + "..."
            
            # Extract basic topics from filename and content
            topics = self._extract_topics_from_text(content, file_path.name)
            
            return {
                'content': content,
                'summary': summary or f"Text file: {file_path.name}",
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"Text extraction failed: {e}")
    
    def _extract_code(self, file_path: Path) -> Dict[str, Any]:
        """Extract from source code files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Extract functions, classes, imports
            summary = self._analyze_code_structure(content, file_path.suffix)
            
            # Code-specific topics
            topics = self._extract_code_topics(content, file_path)
            
            return {
                'content': content,
                'summary': summary,
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"Code extraction failed: {e}")
    
    def _extract_image(self, file_path: Path) -> Dict[str, Any]:
        """Extract from images using OCR (if available)"""
        try:
            # Try OCR with tesseract if available
            ocr_content = self._try_ocr(file_path)
            
            if ocr_content:
                summary = f"Image with OCR text: {ocr_content[:100]}..."
                topics = self._extract_topics_from_text(ocr_content, file_path.name)
                content = f"Image file: {file_path.name}\nOCR extracted text:\n{ocr_content}"
            else:
                summary = f"Image file: {file_path.name}"
                topics = ['image', 'visual', file_path.stem.lower()]
                content = f"Image file: {file_path.name}\nNo OCR text extracted."
            
            return {
                'content': content,
                'summary': summary,
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"Image extraction failed: {e}")
    
    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract from PDF files"""
        try:
            # Try PyMuPDF first, fallback to basic extraction
            content = self._try_pdf_extraction(file_path)
            
            if not content:
                content = f"PDF file: {file_path.name}\nContent extraction not available."
            
            # Generate summary from first paragraph
            lines = content.split('\n')
            summary_lines = [line.strip() for line in lines if line.strip()][:3]
            summary = ' '.join(summary_lines)[:200]
            
            topics = self._extract_topics_from_text(content, file_path.name)
            
            return {
                'content': content,
                'summary': summary or f"PDF document: {file_path.name}",
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"PDF extraction failed: {e}")
    
    def _extract_html(self, file_path: Path) -> Dict[str, Any]:
        """Extract from HTML/XML files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Extract text content (basic HTML stripping)
            text_content = self._strip_html_tags(content)
            
            summary = f"HTML document: {file_path.name}"
            topics = self._extract_topics_from_text(text_content, file_path.name)
            
            return {
                'content': text_content,
                'summary': summary,
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"HTML extraction failed: {e}")
    
    def _extract_json(self, file_path: Path) -> Dict[str, Any]:
        """Extract from JSON/YAML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    # Basic YAML parsing (without PyYAML dependency)
                    content = f.read()
                    data_summary = "YAML configuration file"
                else:
                    data = json.load(f)
                    content = json.dumps(data, indent=2)
                    data_summary = f"JSON with {len(data)} top-level keys" if isinstance(data, dict) else "JSON data"
            
            summary = f"{data_summary}: {file_path.name}"
            topics = self._extract_topics_from_text(content, file_path.name)
            
            return {
                'content': content,
                'summary': summary,
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"JSON extraction failed: {e}")
    
    def _extract_zip(self, file_path: Path) -> Dict[str, Any]:
        """Extract from archive files"""
        try:
            # List archive contents without extracting
            content = f"Archive file: {file_path.name}\nContents listing not available without extraction."
            summary = f"Archive file: {file_path.name}"
            topics = ['archive', 'compressed', file_path.stem.lower()]
            
            return {
                'content': content,
                'summary': summary,
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"Archive extraction failed: {e}")
    
    def _extract_video(self, file_path: Path) -> Dict[str, Any]:
        """Extract from video files"""
        try:
            # Video metadata extraction (basic)
            content = f"Video file: {file_path.name}\nAudio transcription not available."
            summary = f"Video file: {file_path.name}"
            topics = ['video', 'media', file_path.stem.lower()]
            
            return {
                'content': content,
                'summary': summary,
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"Video extraction failed: {e}")
    
    def _extract_audio(self, file_path: Path) -> Dict[str, Any]:
        """Extract from audio files"""
        try:
            # Audio metadata extraction (basic)
            content = f"Audio file: {file_path.name}\nTranscription not available."
            summary = f"Audio file: {file_path.name}"
            topics = ['audio', 'media', file_path.stem.lower()]
            
            return {
                'content': content,
                'summary': summary,
                'topics': topics
            }
        except Exception as e:
            raise Exception(f"Audio extraction failed: {e}")
    
    def _try_ocr(self, file_path: Path) -> Optional[str]:
        """Try OCR extraction using tesseract"""
        try:
            result = subprocess.run(
                ['tesseract', str(file_path), 'stdout'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def _try_pdf_extraction(self, file_path: Path) -> Optional[str]:
        """Try PDF text extraction"""
        try:
            # Try using pdftotext if available
            result = subprocess.run(
                ['pdftotext', str(file_path), '-'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None
    
    def _analyze_code_structure(self, content: str, file_ext: str) -> str:
        """Analyze code structure for summary"""
        lines = content.split('\n')
        
        functions = []
        classes = []
        imports = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('def ') and file_ext == '.py':
                functions.append(line.split('(')[0].replace('def ', ''))
            elif line.startswith('class ') and file_ext == '.py':
                classes.append(line.split('(')[0].replace('class ', '').replace(':', ''))
            elif line.startswith('import ') or line.startswith('from '):
                imports.append(line.split()[1])
        
        summary_parts = []
        if classes:
            summary_parts.append(f"{len(classes)} classes")
        if functions:
            summary_parts.append(f"{len(functions)} functions")
        if imports:
            summary_parts.append(f"{len(imports)} imports")
        
        return f"Code file with {', '.join(summary_parts) if summary_parts else 'code content'}"
    
    def _extract_topics_from_text(self, text: str, filename: str) -> List[str]:
        """Extract topics from text content"""
        topics = []
        
        # Add filename-based topics
        name_parts = filename.lower().replace('_', ' ').replace('-', ' ').split()
        topics.extend([part for part in name_parts if len(part) > 2])
        
        # Add content-based topics (simple keyword extraction)
        words = text.lower().split()
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add most frequent meaningful words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics.extend([word for word, count in top_words if count > 1])
        
        return list(set(topics))[:10]  # Limit to 10 unique topics
    
    def _extract_code_topics(self, content: str, file_path: Path) -> List[str]:
        """Extract code-specific topics"""
        topics = []
        
        # Language-specific topics
        ext = file_path.suffix.lower()
        if ext in ['.py']:
            topics.append('python')
        elif ext in ['.js', '.jsx']:
            topics.append('javascript')
        elif ext in ['.ts', '.tsx']:
            topics.append('typescript')
        
        # Add function/class names as topics
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('def ') or line.startswith('class '):
                name = line.split('(')[0].split(':')[0].split()[-1]
                topics.append(name.lower())
        
        return topics
    
    def _strip_html_tags(self, html_content: str) -> str:
        """Basic HTML tag stripping"""
        import re
        clean = re.sub('<.*?>', '', html_content)
        return ' '.join(clean.split())


class UniversalIngestPipeline:
    """Main pipeline for universal file ingestion"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = UniversalFileExtractor()
        self.collection = MemoryBlockCollection()
        self.stats = {
            'processed': 0,
            'failed': 0,
            'file_types': {},
            'errors': []
        }
    
    def scan_directories(self, root_paths: List[str], file_types: List[str] = None) -> Iterator[Path]:
        """Scan directories for files to process"""
        for root_path in root_paths:
            root = Path(root_path).expanduser()
            if not root.exists():
                logger.warning(f"Path does not exist: {root}")
                continue
            
            if root.is_file():
                yield root
            else:
                for file_path in root.rglob('*'):
                    if file_path.is_file():
                        if file_types:
                            if file_path.suffix.lower() in file_types:
                                yield file_path
                        else:
                            yield file_path
    
    def process_file(self, file_path: Path) -> Optional[MemoryBlock]:
        """Process a single file into a MemoryBlock"""
        try:
            logger.info(f"Processing: {file_path}")
            
            # Extract content
            extracted = self.extractor.extract_content(file_path)
            
            # Determine archetype
            archetype = FILE_TYPE_ARCHETYPES.get(file_path.suffix.lower(), 'Explorer')
            
            # Create MemoryBlock
            memory_block = MemoryBlock.create_from_file(
                file_path=file_path,
                content=extracted['content'],
                summary=extracted['summary'],
                topics=extracted['topics'],
                skills=[f"nano_ingest_{archetype.lower()}.py"]
            )
            
            # Assign project (basic assignment)
            memory_block.assign_project(
                project_id=f"ingest.{archetype.lower()}",
                archetype=archetype,
                confidence=0.8,
                reason="Automatic assignment based on file type"
            )
            
            self.stats['processed'] += 1
            file_type = extracted['file_type']
            self.stats['file_types'][file_type] = self.stats['file_types'].get(file_type, 0) + 1
            
            return memory_block
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.stats['failed'] += 1
            self.stats['errors'].append(f"{file_path}: {e}")
            return None
    
    def run_pipeline(self, root_paths: List[str], file_types: List[str] = None):
        """Run the complete ingestion pipeline"""
        logger.info(f"Starting universal ingest pipeline")
        logger.info(f"Root paths: {root_paths}")
        logger.info(f"File types filter: {file_types}")
        
        for file_path in self.scan_directories(root_paths, file_types):
            memory_block = self.process_file(file_path)
            if memory_block:
                self.collection.add_block(memory_block)
        
        # Save results
        self.save_results()
        
        logger.info(f"Pipeline complete. Processed: {self.stats['processed']}, Failed: {self.stats['failed']}")
    
    def save_results(self):
        """Save all generated MemoryBlocks and statistics"""
        # Save individual MemoryBlocks
        self.collection.save_all(self.output_dir)
        
        # Save as JSON Lines
        self.collection.to_jsonl(self.output_dir / "memory_blocks.jsonl")
        
        # Save statistics
        stats_file = self.output_dir / "ingest_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save collection summary
        collection_stats = self.collection.get_stats()
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(collection_stats, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Universal File Ingest Pipeline")
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories or files to process')
    parser.add_argument('--output', required=True,
                       help='Output directory for MemoryBlocks')
    parser.add_argument('--types', nargs='+',
                       help='File extensions to include (e.g., .txt .pdf .jpg)')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = UniversalIngestPipeline(args.output)
    pipeline.run_pipeline(args.roots, args.types)


if __name__ == "__main__":
    main()