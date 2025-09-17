#!/usr/bin/env python3
"""
MUSE Pantheon Universal File Ingest System
Extracts content from various file formats and creates MemoryBlocks
"""
import os
import sys
import argparse
import json
import pathlib
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from warden.schema.memory_block import MemoryBlock, MemoryBlockBuilder, get_archetype_for_content
except ImportError:
    # Try direct import if running from repository root
    sys.path.insert(0, os.getcwd())
    from warden.schema.memory_block import MemoryBlock, MemoryBlockBuilder, get_archetype_for_content

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UniversalFileExtractor:
    """Extracts content from various file formats"""
    
    def __init__(self):
        self.supported_types = {
            '.txt': self._extract_text,
            '.md': self._extract_markdown,
            '.py': self._extract_code,
            '.js': self._extract_code,
            '.json': self._extract_json,
            '.csv': self._extract_csv,
            '.html': self._extract_html,
            '.xml': self._extract_html,
            '.pdf': self._extract_pdf,
            '.docx': self._extract_docx,
            '.pptx': self._extract_pptx,
            '.xlsx': self._extract_excel,
            '.jpg': self._extract_image_ocr,
            '.jpeg': self._extract_image_ocr,
            '.png': self._extract_image_ocr,
            '.gif': self._extract_image_ocr,
            '.mp4': self._extract_video,
            '.mov': self._extract_video,
            '.avi': self._extract_video,
            '.mp3': self._extract_audio,
            '.wav': self._extract_audio,
            '.zip': self._extract_archive,
            '.eml': self._extract_email
        }
    
    def extract_file_content(self, file_path: pathlib.Path) -> Tuple[str, str, List[str], Dict[str, Any]]:
        """
        Extract content from a file
        Returns: (summary, content, topics, metadata)
        """
        try:
            file_ext = file_path.suffix.lower()
            file_size = file_path.stat().st_size
            
            # Skip very large files (>100MB)
            if file_size > 100 * 1024 * 1024:
                logger.warning(f"Skipping large file: {file_path} ({file_size} bytes)")
                return "", "", [], {"error": "file_too_large", "size": file_size}
            
            # Try to extract using appropriate method
            if file_ext in self.supported_types:
                extractor = self.supported_types[file_ext]
                content, topics, metadata = extractor(file_path)
            else:
                # Fallback to text extraction
                content, topics, metadata = self._extract_generic(file_path)
            
            # Generate summary
            summary = self._generate_summary(content, file_path)
            
            return summary, content, topics, metadata
            
        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {e}")
            return "", "", [], {"error": str(e)}
    
    def _extract_text(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from text files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            topics = self._extract_topics_from_text(content)
            metadata = {"type": "text", "encoding": "utf-8"}
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_markdown(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            topics = self._extract_topics_from_text(content)
            topics.append("documentation")
            metadata = {"type": "markdown", "format": "md"}
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_code(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from code files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract code-specific topics
            topics = self._extract_code_topics(content, file_path.suffix)
            
            metadata = {
                "type": "code",
                "language": self._detect_language(file_path.suffix),
                "lines": len(content.split('\n'))
            }
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_json(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            content = json.dumps(json_data, indent=2)
            topics = ["data", "json", "configuration"]
            
            # Extract topics from keys
            if isinstance(json_data, dict):
                topics.extend(list(json_data.keys())[:10])  # Limit to first 10 keys
            
            metadata = {"type": "json", "format": "json"}
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_csv(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from CSV files"""
        try:
            import pandas as pd
            df = pd.read_csv(file_path, nrows=1000)  # Limit rows for large files
            
            content = df.to_string()
            topics = ["data", "csv", "spreadsheet"] + list(df.columns)[:10]
            
            metadata = {
                "type": "csv",
                "rows": len(df),
                "columns": list(df.columns)
            }
            return content, topics, metadata
        except Exception as e:
            # Fallback to text extraction
            return self._extract_text(file_path)
    
    def _extract_html(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from HTML/XML files"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract text content
            content = soup.get_text()
            
            # Extract topics from title, headings, meta tags
            topics = ["web", "html"]
            if soup.title:
                topics.extend(self._extract_topics_from_text(soup.title.string or ""))
            
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                if tag.string:
                    topics.extend(self._extract_topics_from_text(tag.string))
            
            metadata = {"type": "html", "format": "html"}
            return content, topics, metadata
        except Exception as e:
            return self._extract_text(file_path)
    
    def _extract_pdf(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from PDF files"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            content = ""
            
            for page_num in range(min(doc.page_count, 50)):  # Limit to first 50 pages
                page = doc.load_page(page_num)
                content += page.get_text()
            
            doc.close()
            
            topics = self._extract_topics_from_text(content)
            topics.append("document")
            
            metadata = {
                "type": "pdf",
                "pages": doc.page_count,
                "format": "pdf"
            }
            return content, topics, metadata
        except Exception as e:
            logger.warning(f"PDF extraction failed for {file_path}: {e}")
            return "", [], {"error": str(e)}
    
    def _extract_docx(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from DOCX files"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            topics = self._extract_topics_from_text(content)
            topics.append("document")
            
            metadata = {"type": "docx", "format": "docx"}
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_pptx(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from PPTX files"""
        try:
            from pptx import Presentation
            
            prs = Presentation(file_path)
            content = ""
            
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        content += shape.text + "\n"
            
            topics = self._extract_topics_from_text(content)
            topics.append("presentation")
            
            metadata = {"type": "pptx", "slides": len(prs.slides)}
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_excel(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from Excel files"""
        try:
            import pandas as pd
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            content = ""
            
            for sheet_name in excel_file.sheet_names[:5]:  # Limit to first 5 sheets
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=100)
                content += f"Sheet: {sheet_name}\n{df.to_string()}\n\n"
            
            topics = ["data", "spreadsheet", "excel"] + list(df.columns)[:10]
            
            metadata = {
                "type": "excel",
                "sheets": excel_file.sheet_names,
                "format": "xlsx"
            }
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_image_ocr(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract text from images using OCR"""
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(file_path)
            content = pytesseract.image_to_string(image)
            
            topics = self._extract_topics_from_text(content) if content.strip() else ["image"]
            topics.append("visual")
            
            metadata = {
                "type": "image",
                "format": file_path.suffix.lower(),
                "size": f"{image.width}x{image.height}",
                "ocr_extracted": bool(content.strip())
            }
            return content, topics, metadata
        except Exception as e:
            # If OCR fails, return basic image info
            topics = ["image", "visual"]
            metadata = {"type": "image", "format": file_path.suffix.lower(), "ocr_error": str(e)}
            return "", topics, metadata
    
    def _extract_video(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from video files (metadata only for now)"""
        try:
            # For now, just extract metadata
            # In a full implementation, you'd use whisper for audio transcription
            file_size = file_path.stat().st_size
            
            content = f"Video file: {file_path.name}"
            topics = ["video", "media", "visual"]
            
            metadata = {
                "type": "video",
                "format": file_path.suffix.lower(),
                "size_bytes": file_size,
                "transcription_available": False
            }
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_audio(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from audio files (metadata only for now)"""
        try:
            # For now, just extract metadata
            # In a full implementation, you'd use whisper for transcription
            file_size = file_path.stat().st_size
            
            content = f"Audio file: {file_path.name}"
            topics = ["audio", "media", "sound"]
            
            metadata = {
                "type": "audio",
                "format": file_path.suffix.lower(),
                "size_bytes": file_size,
                "transcription_available": False
            }
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_archive(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from archive files"""
        try:
            import zipfile
            
            content = f"Archive file: {file_path.name}\n"
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()[:50]  # Limit to first 50 files
                content += "Contents:\n" + "\n".join(file_list)
            
            topics = ["archive", "compressed", "zip"]
            
            metadata = {
                "type": "archive",
                "format": "zip",
                "file_count": len(file_list),
                "files": file_list
            }
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_email(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Extract content from email files"""
        try:
            import email
            
            with open(file_path, 'rb') as f:
                msg = email.message_from_binary_file(f)
            
            # Extract email content
            subject = msg.get('Subject', '')
            sender = msg.get('From', '')
            recipient = msg.get('To', '')
            date = msg.get('Date', '')
            
            # Get body content
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            content = f"Subject: {subject}\nFrom: {sender}\nTo: {recipient}\nDate: {date}\n\n{body}"
            
            topics = self._extract_topics_from_text(content)
            topics.extend(["email", "communication"])
            
            metadata = {
                "type": "email",
                "subject": subject,
                "sender": sender,
                "recipient": recipient,
                "date": date
            }
            return content, topics, metadata
        except Exception as e:
            return "", [], {"error": str(e)}
    
    def _extract_generic(self, file_path: pathlib.Path) -> Tuple[str, List[str], Dict[str, Any]]:
        """Generic extraction for unknown file types"""
        try:
            # Try to read as text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB
            
            topics = self._extract_topics_from_text(content)
            metadata = {"type": "generic", "format": file_path.suffix.lower()}
            return content, topics, metadata
        except Exception as e:
            # If all else fails, return file info
            content = f"File: {file_path.name}"
            topics = ["file", "unknown"]
            metadata = {"type": "unknown", "error": str(e)}
            return content, topics, metadata
    
    def _generate_summary(self, content: str, file_path: pathlib.Path) -> str:
        """Generate a concise summary of the content"""
        if not content or len(content.strip()) == 0:
            return f"File: {file_path.name} (no extractable content)"
        
        # Simple summary generation
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if not non_empty_lines:
            return f"File: {file_path.name} (empty content)"
        
        # Take first meaningful line and file info
        first_line = non_empty_lines[0][:100]
        file_type = file_path.suffix.lower()
        
        return f"{file_type.upper()} file containing: {first_line}..."
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text content using simple keyword analysis"""
        if not text:
            return []
        
        # Simple topic extraction based on common keywords
        topic_keywords = {
            'financial': ['payment', 'invoice', 'money', 'cost', 'price', 'budget', 'tax', 'salary'],
            'legal': ['contract', 'agreement', 'law', 'legal', 'court', 'attorney', 'rights'],
            'technical': ['code', 'algorithm', 'function', 'variable', 'database', 'api', 'software'],
            'business': ['meeting', 'project', 'deadline', 'client', 'customer', 'sales', 'marketing'],
            'personal': ['family', 'friend', 'personal', 'private', 'diary', 'journal'],
            'research': ['study', 'analysis', 'research', 'data', 'experiment', 'hypothesis'],
            'communication': ['email', 'message', 'call', 'meeting', 'discussion', 'conversation'],
            'media': ['image', 'video', 'audio', 'photo', 'picture', 'sound', 'music']
        }
        
        text_lower = text.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_code_topics(self, content: str, file_ext: str) -> List[str]:
        """Extract topics specific to code files"""
        topics = ["code", "programming"]
        
        # Language-specific topics
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        
        if file_ext in lang_map:
            topics.append(lang_map[file_ext])
        
        # Code pattern detection
        content_lower = content.lower()
        code_patterns = {
            'api': ['api', 'endpoint', 'request', 'response'],
            'database': ['sql', 'database', 'query', 'table'],
            'web': ['http', 'html', 'css', 'web', 'browser'],
            'ai': ['machine learning', 'neural', 'model', 'training'],
            'testing': ['test', 'assert', 'unittest', 'pytest'],
            'config': ['config', 'settings', 'environment', 'setup']
        }
        
        for topic, patterns in code_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                topics.append(topic)
        
        return topics
    
    def _detect_language(self, file_ext: str) -> str:
        """Detect programming language from file extension"""
        lang_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.html': 'HTML',
            '.css': 'CSS',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.ts': 'TypeScript'
        }
        return lang_map.get(file_ext, 'Unknown')


class MemoryBlockIngestor:
    """Creates MemoryBlocks from extracted file content"""
    
    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = UniversalFileExtractor()
    
    def ingest_file(self, file_path: pathlib.Path) -> Optional[MemoryBlock]:
        """Process a single file and create a MemoryBlock"""
        try:
            logger.info(f"Processing: {file_path}")
            
            # Extract content
            summary, content, topics, metadata = self.extractor.extract_file_content(file_path)
            
            if not summary and not content:
                logger.warning(f"No content extracted from {file_path}")
                return None
            
            # Determine archetype
            archetype = get_archetype_for_content(str(file_path), content, topics)
            
            # Build MemoryBlock
            builder = MemoryBlockBuilder()
            memory_block = (builder
                          .with_content(summary, content)
                          .with_source(str(file_path), file_path.suffix.lower())
                          .with_topics(topics)
                          .with_archetype(archetype)
                          .with_ethics(pii_redacted=True, consent_logged=True)
                          .with_metadata(metadata)
                          .build())
            
            # Save to file
            output_file = self.output_dir / f"{memory_block.id_hash}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(memory_block.to_json())
            
            logger.info(f"Created MemoryBlock: {memory_block.id_hash}")
            return memory_block
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return None
    
    def ingest_directory(self, root_path: pathlib.Path, file_types: Optional[List[str]] = None) -> List[MemoryBlock]:
        """Process all files in a directory"""
        memory_blocks = []
        processed_count = 0
        
        for file_path in root_path.rglob('*'):
            if file_path.is_file():
                # Filter by file types if specified
                if file_types and file_path.suffix.lower() not in file_types:
                    continue
                
                # Skip hidden files and common build artifacts
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                
                if any(pattern in str(file_path) for pattern in ['__pycache__', 'node_modules', '.git', '.venv']):
                    continue
                
                memory_block = self.ingest_file(file_path)
                if memory_block:
                    memory_blocks.append(memory_block)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} files...")
        
        logger.info(f"Completed ingestion: {len(memory_blocks)} MemoryBlocks created from {processed_count} files")
        return memory_blocks


def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Universal File Ingest System")
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--output', required=True,
                       help='Output directory for MemoryBlocks')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (e.g., .txt .pdf .jpg)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize ingestor
    output_path = pathlib.Path(args.output)
    ingestor = MemoryBlockIngestor(output_path)
    
    # Process each root directory
    all_memory_blocks = []
    for root in args.roots:
        root_path = pathlib.Path(root)
        if not root_path.exists():
            logger.error(f"Root path does not exist: {root}")
            continue
        
        logger.info(f"Processing root: {root}")
        memory_blocks = ingestor.ingest_directory(root_path, args.types)
        all_memory_blocks.extend(memory_blocks)
    
    # Create summary report
    summary = {
        'total_blocks': len(all_memory_blocks),
        'timestamp': datetime.utcnow().isoformat() + "Z",
        'roots_processed': args.roots,
        'output_directory': str(output_path)
    }
    
    summary_file = output_path / 'ingestion_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Ingestion complete: {len(all_memory_blocks)} MemoryBlocks created")
    print(f"📁 Output: {output_path}")
    print(f"📄 Summary: {summary_file}")


if __name__ == "__main__":
    main()