#!/usr/bin/env python3
"""
Universal File Ingest System for MUSE Pantheon
Processes any file format and converts to MemoryBlocks
"""
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes
import zipfile
import hashlib

# Add common directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "common"))
from memory_block import MemoryBlock, MemoryBlockFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UniversalFileProcessor:
    """Processes files of any format into MemoryBlocks."""
    
    def __init__(self):
        """Initialize the file processor."""
        self.supported_types = {
            '.txt', '.md', '.py', '.js', '.ts', '.json', '.csv', '.xml', '.html',
            '.pdf', '.doc', '.docx', '.rtf',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
            '.mp4', '.avi', '.mov', '.wmv', '.flv', '.mp3', '.wav', '.m4a',
            '.zip', '.tar', '.gz', '.rar'
        }
        
    def can_process(self, filepath: Path) -> bool:
        """Check if file can be processed."""
        return filepath.suffix.lower() in self.supported_types
    
    def process_file(self, filepath: Path) -> Optional[MemoryBlock]:
        """Process a single file into a MemoryBlock."""
        try:
            if not filepath.exists() or not filepath.is_file():
                logger.warning(f"File not found or not a file: {filepath}")
                return None
                
            if not self.can_process(filepath):
                logger.warning(f"Unsupported file type: {filepath.suffix}")
                return None
            
            # Extract content based on file type
            content = self._extract_content(filepath)
            if not content:
                logger.warning(f"No content extracted from: {filepath}")
                return None
            
            # Create MemoryBlock
            memory_block = MemoryBlockFactory.create_from_file(filepath, content)
            
            # Validate and set status
            if memory_block.validate():
                memory_block.metadata["validation_status"] = "passed"
                logger.info(f"Successfully processed: {filepath.name}")
            else:
                memory_block.metadata["validation_status"] = "failed"
                logger.warning(f"Validation failed for: {filepath.name}")
            
            return memory_block
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return None
    
    def _extract_content(self, filepath: Path) -> str:
        """Extract content from file based on type."""
        file_ext = filepath.suffix.lower()
        
        try:
            if file_ext in {'.txt', '.md', '.py', '.js', '.ts', '.json', '.csv', '.xml', '.html'}:
                return self._extract_text_content(filepath)
            elif file_ext == '.pdf':
                return self._extract_pdf_content(filepath)
            elif file_ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}:
                return self._extract_image_content(filepath)
            elif file_ext in {'.mp4', '.avi', '.mov', '.wmv', '.flv'}:
                return self._extract_video_content(filepath)
            elif file_ext in {'.mp3', '.wav', '.m4a'}:
                return self._extract_audio_content(filepath)
            elif file_ext in {'.zip', '.tar', '.gz', '.rar'}:
                return self._extract_archive_content(filepath)
            else:
                # Fallback to text extraction
                return self._extract_text_content(filepath)
                
        except Exception as e:
            logger.error(f"Content extraction failed for {filepath}: {str(e)}")
            return f"Content extraction failed: {str(e)}"
    
    def _extract_text_content(self, filepath: Path) -> str:
        """Extract text content from text-based files."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read as binary and decode with errors='ignore'
            with open(filepath, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
            return content
            
        except Exception as e:
            return f"Text extraction failed: {str(e)}"
    
    def _extract_pdf_content(self, filepath: Path) -> str:
        """Extract text from PDF files."""
        try:
            # Try to import PDF processing libraries
            try:
                import PyPDF2
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                pass
            
            try:
                import pdfplumber
                with pdfplumber.open(filepath) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                pass
            
            # Fallback: return file metadata
            return f"PDF file: {filepath.name} (size: {filepath.stat().st_size} bytes)"
            
        except Exception as e:
            return f"PDF extraction failed: {str(e)}"
    
    def _extract_image_content(self, filepath: Path) -> str:
        """Extract metadata and OCR from images."""
        try:
            # Try OCR extraction
            try:
                import pytesseract
                from PIL import Image
                
                image = Image.open(filepath)
                text = pytesseract.image_to_string(image)
                
                if text.strip():
                    return f"OCR extracted text: {text}"
                else:
                    return f"Image file: {filepath.name} (size: {filepath.stat().st_size} bytes)"
                    
            except ImportError:
                pass
            
            # Fallback: return image metadata
            try:
                from PIL import Image
                with Image.open(filepath) as img:
                    return f"Image: {filepath.name}, format: {img.format}, size: {img.size}, mode: {img.mode}"
            except ImportError:
                pass
            
            return f"Image file: {filepath.name} (size: {filepath.stat().st_size} bytes)"
            
        except Exception as e:
            return f"Image processing failed: {str(e)}"
    
    def _extract_video_content(self, filepath: Path) -> str:
        """Extract metadata from video files."""
        try:
            # Try to extract video metadata
            try:
                import cv2
                cap = cv2.VideoCapture(str(filepath))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()
                
                return f"Video: {filepath.name}, duration: {duration:.2f}s, resolution: {int(width)}x{int(height)}, fps: {fps}"
            except ImportError:
                pass
            
            # Fallback: return basic file info
            return f"Video file: {filepath.name} (size: {filepath.stat().st_size} bytes)"
            
        except Exception as e:
            return f"Video processing failed: {str(e)}"
    
    def _extract_audio_content(self, filepath: Path) -> str:
        """Extract metadata from audio files."""
        try:
            # Try to extract audio metadata
            try:
                import mutagen
                audiofile = mutagen.File(filepath)
                if audiofile:
                    info = audiofile.info
                    return f"Audio: {filepath.name}, duration: {info.length:.2f}s, bitrate: {info.bitrate}"
            except ImportError:
                pass
            
            # Fallback: return basic file info
            return f"Audio file: {filepath.name} (size: {filepath.stat().st_size} bytes)"
            
        except Exception as e:
            return f"Audio processing failed: {str(e)}"
    
    def _extract_archive_content(self, filepath: Path) -> str:
        """Extract file list from archives."""
        try:
            if filepath.suffix.lower() == '.zip':
                with zipfile.ZipFile(filepath, 'r') as zip_file:
                    file_list = zip_file.namelist()
                    return f"ZIP archive: {filepath.name}, contains {len(file_list)} files: {', '.join(file_list[:10])}{'...' if len(file_list) > 10 else ''}"
            else:
                return f"Archive file: {filepath.name} (size: {filepath.stat().st_size} bytes)"
                
        except Exception as e:
            return f"Archive processing failed: {str(e)}"


class UniversalIngest:
    """Main ingest orchestrator."""
    
    def __init__(self, output_dir: Path):
        """Initialize the ingest system."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = UniversalFileProcessor()
        self.stats = {
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'total_files': 0
        }
    
    def ingest_directory(self, root_dir: Path, file_types: List[str] = None) -> List[MemoryBlock]:
        """Ingest all files from a directory."""
        logger.info(f"Starting ingest of directory: {root_dir}")
        
        memory_blocks = []
        
        # Get all files
        if file_types:
            files = []
            for file_type in file_types:
                files.extend(root_dir.rglob(f"*{file_type}"))
        else:
            files = [f for f in root_dir.rglob("*") if f.is_file()]
        
        self.stats['total_files'] = len(files)
        logger.info(f"Found {len(files)} files to process")
        
        for filepath in files:
            try:
                memory_block = self.processor.process_file(filepath)
                
                if memory_block:
                    # Save the memory block
                    output_path = memory_block.save(self.output_dir)
                    memory_blocks.append(memory_block)
                    self.stats['processed'] += 1
                    
                    if self.stats['processed'] % 100 == 0:
                        logger.info(f"Processed {self.stats['processed']} files...")
                else:
                    self.stats['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {str(e)}")
                self.stats['failed'] += 1
        
        self._save_ingest_report(memory_blocks)
        logger.info(f"Ingest complete. Processed: {self.stats['processed']}, Failed: {self.stats['failed']}")
        
        return memory_blocks
    
    def _save_ingest_report(self, memory_blocks: List[MemoryBlock]):
        """Save ingest report with statistics."""
        report = {
            'ingest_stats': self.stats,
            'memory_blocks_created': len(memory_blocks),
            'output_directory': str(self.output_dir),
            'block_summaries': [
                {
                    'id_hash': block.id_hash,
                    'summary': block.summary,
                    'archetype': block.archetype,
                    'project': block.project,
                    'source_path': block.source_path
                }
                for block in memory_blocks[:50]  # First 50 blocks
            ]
        }
        
        report_path = self.output_dir / "ingest_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Ingest report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Universal File Ingest for MUSE Pantheon")
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (e.g., .txt .pdf .jpg)')
    parser.add_argument('--output', required=True,
                       help='Output directory for memory blocks')
    
    args = parser.parse_args()
    
    # Initialize ingest system
    ingest_system = UniversalIngest(args.output)
    
    # Process each root directory
    all_blocks = []
    for root in args.roots:
        root_path = Path(root)
        if root_path.exists():
            blocks = ingest_system.ingest_directory(root_path, args.types)
            all_blocks.extend(blocks)
        else:
            logger.warning(f"Root directory does not exist: {root}")
    
    logger.info(f"Total memory blocks created: {len(all_blocks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())