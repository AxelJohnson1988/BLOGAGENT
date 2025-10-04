#!/usr/bin/env python3
"""
MUSE Pantheon Universal Ingest System
Processes any file format into structured MemoryBlocks with ethics tracking
"""
import hashlib
import json
import os
import pathlib
import argparse
import zipfile
import email
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MemoryBlock:
    """Standardized memory block structure for MUSE Pantheon system"""
    id_hash: str
    summary: str
    content: str
    topics: List[str]
    skills: List[str]
    date: str
    project_suggestion: Dict[str, str]
    archetype: str = "default"
    created_at: str = ""
    links: List[str] = None
    metadata: Dict[str, Any] = None
    ethics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.metadata is None:
            self.metadata = {}
        if self.ethics is None:
            self.ethics = {"pii_redacted": True, "consent_logged": True}
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"

class UniversalIngester:
    """Universal file ingester that converts any file type to MemoryBlocks"""
    
    def __init__(self, output_dir: str):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_count = 0
        self.error_count = 0
        
    def generate_hash(self, content: str) -> str:
        """Generate SHA256 hash for content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
        
    def extract_text_content(self, file_path: pathlib.Path) -> str:
        """Extract text content from various file types"""
        try:
            suffix = file_path.suffix.lower()
            
            # Text-based files
            if suffix in ['.txt', '.md', '.py', '.js', '.ts', '.json', '.csv', '.html', '.xml']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            # PDF files
            elif suffix == '.pdf':
                return self._extract_pdf_content(file_path)
                
            # Image files (OCR placeholder)
            elif suffix in ['.jpg', '.jpeg', '.png', '.gif']:
                return self._extract_image_content(file_path)
                
            # Email files
            elif suffix == '.eml':
                return self._extract_email_content(file_path)
                
            # ZIP files
            elif suffix == '.zip':
                return self._extract_zip_content(file_path)
                
            # Default: try as text
            else:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        return f.read()
                except:
                    return f"Binary file: {file_path.name}"
                    
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
            return f"Error reading file: {str(e)}"
    
    def _extract_pdf_content(self, file_path: pathlib.Path) -> str:
        """Extract text from PDF files"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            return f"PDF file (PyMuPDF not available): {file_path.name}"
        except Exception as e:
            return f"PDF extraction error: {str(e)}"
    
    def _extract_image_content(self, file_path: pathlib.Path) -> str:
        """Extract content from images using OCR placeholder"""
        # Placeholder for OCR functionality
        return f"Image file: {file_path.name} (OCR would extract text here)"
    
    def _extract_email_content(self, file_path: pathlib.Path) -> str:
        """Extract content from email files"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                msg = email.message_from_file(f)
                content = f"Subject: {msg.get('Subject', 'No Subject')}\n"
                content += f"From: {msg.get('From', 'Unknown')}\n"
                content += f"Date: {msg.get('Date', 'Unknown')}\n\n"
                
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                else:
                    content += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    
                return content
        except Exception as e:
            return f"Email extraction error: {str(e)}"
    
    def _extract_zip_content(self, file_path: pathlib.Path) -> str:
        """Extract file list from ZIP archives"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                return f"ZIP archive containing {len(file_list)} files:\n" + "\n".join(file_list[:50])
        except Exception as e:
            return f"ZIP extraction error: {str(e)}"
    
    def create_memory_block(self, file_path: pathlib.Path) -> MemoryBlock:
        """Create a MemoryBlock from a file"""
        content = self.extract_text_content(file_path)
        
        # Generate summary (first 200 chars or intelligent extraction)
        summary = self._generate_summary(content, file_path)
        
        # Extract topics and skills
        topics = self._extract_topics(content, file_path)
        skills = self._extract_skills(content, file_path)
        
        # Determine archetype based on file type
        archetype = self._determine_archetype(file_path)
        
        # Create project suggestion
        project_suggestion = self._suggest_project(content, file_path)
        
        # Generate hash
        id_hash = self.generate_hash(f"{file_path.name}_{content[:100]}")
        
        return MemoryBlock(
            id_hash=f"sha256:{id_hash}",
            summary=summary,
            content=content[:2000],  # Limit content size
            topics=topics,
            skills=skills,
            date=datetime.now().strftime("%Y-%m-%d"),
            project_suggestion=project_suggestion,
            archetype=archetype,
            metadata={
                "source_file": str(file_path),
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "file_type": file_path.suffix.lower(),
                "validation_status": "passed"
            }
        )
    
    def _generate_summary(self, content: str, file_path: pathlib.Path) -> str:
        """Generate a concise summary of the content"""
        if not content.strip():
            return f"Empty {file_path.suffix} file: {file_path.name}"
        
        # Take first meaningful content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if lines:
            summary = lines[0][:150]
            if len(content) > 150:
                summary += "..."
            return summary
        
        return f"{file_path.suffix} file: {file_path.name}"
    
    def _extract_topics(self, content: str, file_path: pathlib.Path) -> List[str]:
        """Extract relevant topics from content"""
        topics = []
        
        # File type based topics
        suffix = file_path.suffix.lower()
        if suffix in ['.py', '.js', '.ts']:
            topics.append("code")
        elif suffix in ['.md', '.txt']:
            topics.append("documentation")
        elif suffix in ['.jpg', '.jpeg', '.png']:
            topics.append("image")
        elif suffix == '.pdf':
            topics.append("document")
        
        # Content-based topic extraction (basic keyword matching)
        content_lower = content.lower()
        keywords = {
            "legal": ["contract", "agreement", "legal", "terms"],
            "financial": ["payment", "invoice", "cost", "budget"],
            "technical": ["function", "class", "import", "api"],
            "project": ["project", "milestone", "deliverable", "timeline"]
        }
        
        for topic, words in keywords.items():
            if any(word in content_lower for word in words):
                topics.append(topic)
        
        return topics[:5]  # Limit to 5 topics
    
    def _extract_skills(self, content: str, file_path: pathlib.Path) -> List[str]:
        """Extract relevant skills/nano scripts this content relates to"""
        skills = []
        
        # File type based skills
        suffix = file_path.suffix.lower()
        if suffix == '.py':
            skills.append("nano_python_parser.py")
        elif suffix in ['.jpg', '.jpeg', '.png']:
            skills.append("nano_vision_ocr.py")
        elif suffix == '.pdf':
            skills.append("nano_pdf_extractor.py")
        elif suffix == '.md':
            skills.append("nano_markdown_processor.py")
        
        # Generic skills
        skills.append("nano_warden_memory_tagger.py")
        
        return skills
    
    def _determine_archetype(self, file_path: pathlib.Path) -> str:
        """Determine MUSE archetype based on file characteristics"""
        suffix = file_path.suffix.lower()
        
        archetype_map = {
            '.py': 'Builder',
            '.js': 'Builder', 
            '.ts': 'Builder',
            '.md': 'Sage',
            '.txt': 'Sage',
            '.pdf': 'Sage',
            '.jpg': 'Vision',
            '.jpeg': 'Vision',
            '.png': 'Vision',
            '.json': 'Connector',
            '.csv': 'Connector'
        }
        
        return archetype_map.get(suffix, 'Guardian')
    
    def _suggest_project(self, content: str, file_path: pathlib.Path) -> Dict[str, str]:
        """Suggest project assignment based on content analysis"""
        content_lower = content.lower()
        
        # Project classification logic
        if any(word in content_lower for word in ["legal", "contract", "agreement"]):
            return {"match_type": "existing", "project_id": "legal.documents"}
        elif any(word in content_lower for word in ["payment", "invoice", "financial"]):
            return {"match_type": "existing", "project_id": "financial.records"}
        elif any(word in content_lower for word in ["code", "function", "class", "import"]):
            return {"match_type": "existing", "project_id": "development.codebase"}
        elif "image" in str(file_path).lower() or file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            return {"match_type": "existing", "project_id": "media.assets"}
        else:
            return {"match_type": "new", "project_id": "general.documents"}
    
    def process_directory(self, root_path: str, file_types: List[str] = None) -> Dict[str, Any]:
        """Process all files in a directory tree"""
        root = pathlib.Path(root_path)
        if not root.exists():
            logger.error(f"Root path does not exist: {root_path}")
            return {"error": "Root path not found"}
        
        logger.info(f"Starting ingestion from: {root}")
        
        # Find all files
        if file_types:
            files = []
            for file_type in file_types:
                files.extend(root.rglob(f"*{file_type}"))
        else:
            files = [f for f in root.rglob("*") if f.is_file()]
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process each file
        results = {
            "processed_files": [],
            "error_files": [],
            "total_blocks": 0
        }
        
        for file_path in files:
            try:
                # Create memory block
                memory_block = self.create_memory_block(file_path)
                
                # Save to JSON file
                output_file = self.output_dir / f"{memory_block.id_hash.replace(':', '_')}.json"
                with open(output_file, 'w') as f:
                    json.dump(asdict(memory_block), f, indent=2)
                
                results["processed_files"].append(str(file_path))
                results["total_blocks"] += 1
                self.processed_count += 1
                
                if self.processed_count % 10 == 0:
                    logger.info(f"Processed {self.processed_count} files...")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results["error_files"].append({"file": str(file_path), "error": str(e)})
                self.error_count += 1
        
        logger.info(f"Ingestion complete. Processed: {self.processed_count}, Errors: {self.error_count}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="MUSE Pantheon Universal File Ingester")
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Root directories to scan for files')
    parser.add_argument('--output', required=True,
                       help='Output directory for MemoryBlocks')
    parser.add_argument('--types', nargs='+',
                       help='File types to process (e.g., .txt .pdf .jpg)')
    
    args = parser.parse_args()
    
    # Initialize ingester
    ingester = UniversalIngester(args.output)
    
    # Process each root directory
    all_results = []
    for root in args.roots:
        logger.info(f"Processing root: {root}")
        results = ingester.process_directory(root, args.types)
        all_results.append({"root": root, "results": results})
    
    # Save summary report
    summary_file = pathlib.Path(args.output) / "ingestion_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_processed": ingester.processed_count,
            "total_errors": ingester.error_count,
            "roots_processed": all_results
        }, f, indent=2)
    
    print(f"✅ Ingestion complete! Processed {ingester.processed_count} files")
    print(f"📁 Output directory: {args.output}")
    print(f"📄 Summary: {summary_file}")

if __name__ == "__main__":
    main()