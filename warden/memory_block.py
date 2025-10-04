#!/usr/bin/env python3
"""
Universal Ingest Pipeline - Core Data Structures
Defines MemoryBlock schema and utilities for the BLOGAGENT universal ingest system
"""

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class EthicsMetadata:
    """Ethics and privacy tracking for MemoryBlocks"""
    pii_redacted: bool = True
    consent_logged: bool = True
    privacy_level: str = "public"  # public, private, sensitive
    redaction_reason: Optional[str] = None


@dataclass
class FileMetadata:
    """Source file metadata"""
    source_file: str
    file_type: str
    size_bytes: int
    last_modified: Optional[str] = None
    encoding: Optional[str] = None


@dataclass
class MemoryBlock:
    """
    Core immutable data structure for BLOGAGENT universal ingest system.
    Based on MUSE Pantheon MemoryBlock schema.
    """
    id_hash: str
    summary: str
    content: str
    topics: List[str]
    skills: List[str]
    project: str
    archetype: str
    created_at: str
    ethics: EthicsMetadata
    metadata: FileMetadata
    links: List[str] = None
    
    def __post_init__(self):
        if self.links is None:
            self.links = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MemoryBlock to dictionary for JSON serialization"""
        return {
            'id_hash': self.id_hash,
            'summary': self.summary,
            'content': self.content,
            'topics': self.topics,
            'skills': self.skills,
            'project': self.project,
            'archetype': self.archetype,
            'created_at': self.created_at,
            'ethics': asdict(self.ethics),
            'metadata': asdict(self.metadata),
            'links': self.links
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        """Create MemoryBlock from dictionary"""
        ethics = EthicsMetadata(**data['ethics'])
        metadata = FileMetadata(**data['metadata'])
        
        return cls(
            id_hash=data['id_hash'],
            summary=data['summary'],
            content=data['content'],
            topics=data['topics'],
            skills=data['skills'],
            project=data['project'],
            archetype=data['archetype'],
            created_at=data['created_at'],
            ethics=ethics,
            metadata=metadata,
            links=data.get('links', [])
        )


class MemoryBlockFactory:
    """Factory for creating MemoryBlocks from various file types"""
    
    # MUSE Pantheon Archetypes
    ARCHETYPES = [
        "Guardian", "Visionary", "Analyst", "Creator", "Connector",
        "Optimizer", "Explorer", "Synthesizer", "Protector", "Innovator"
    ]
    
    @staticmethod
    def generate_hash(content: str, source_path: str) -> str:
        """Generate SHA256 hash for content identification"""
        combined = f"{content}:{source_path}"
        hash_obj = hashlib.sha256(combined.encode('utf-8'))
        return f"sha256:{hash_obj.hexdigest()[:12]}"
    
    @staticmethod
    def extract_topics(content: str) -> List[str]:
        """Extract key topics from content using simple keyword extraction"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words (alphanumeric, length > 2)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out stop words and get unique terms
        topics = list(set(word for word in words if word not in stop_words))
        
        # Limit to top 10 most frequent topics
        from collections import Counter
        word_freq = Counter(words)
        frequent_topics = [word for word, _ in word_freq.most_common(10) if word not in stop_words]
        
        return frequent_topics[:10]
    
    @staticmethod
    def infer_archetype(content: str, file_type: str) -> str:
        """Infer MUSE archetype based on content and file type"""
        content_lower = content.lower()
        
        # Keyword-based archetype inference
        if any(word in content_lower for word in ['analyze', 'data', 'statistics', 'metrics']):
            return "Analyst"
        elif any(word in content_lower for word in ['create', 'design', 'build', 'develop']):
            return "Creator"
        elif any(word in content_lower for word in ['vision', 'future', 'strategy', 'plan']):
            return "Visionary"
        elif any(word in content_lower for word in ['security', 'protect', 'guard', 'safety']):
            return "Guardian"
        elif any(word in content_lower for word in ['connect', 'link', 'relationship', 'network']):
            return "Connector"
        elif any(word in content_lower for word in ['optimize', 'improve', 'enhance', 'efficiency']):
            return "Optimizer"
        elif any(word in content_lower for word in ['explore', 'discover', 'research', 'investigate']):
            return "Explorer"
        elif any(word in content_lower for word in ['combine', 'merge', 'integrate', 'synthesize']):
            return "Synthesizer"
        elif any(word in content_lower for word in ['innovate', 'new', 'novel', 'creative']):
            return "Innovator"
        else:
            return "Protector"  # Default archetype
    
    @staticmethod
    def generate_summary(content: str, max_sentences: int = 3) -> str:
        """Generate a concise summary of the content"""
        # Simple sentence extraction for summary
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return "Content summary not available."
        
        # Take first few sentences or up to max_sentences
        summary_sentences = sentences[:max_sentences]
        summary = '. '.join(summary_sentences)
        
        # Ensure it ends with proper punctuation
        if not summary.endswith(('.', '!', '?')):
            summary += '.'
            
        return summary
    
    @staticmethod
    def infer_project_id(content: str, file_path: str) -> str:
        """Infer project ID based on content and file structure"""
        path_parts = Path(file_path).parts
        
        # Extract project hints from path
        if len(path_parts) > 1:
            potential_project = path_parts[-2].lower().replace(' ', '_')
        else:
            potential_project = "general"
        
        # Add content-based project classification
        content_lower = content.lower()
        if 'blog' in content_lower or 'article' in content_lower:
            return f"blog.{potential_project}"
        elif 'legal' in content_lower or 'contract' in content_lower:
            return f"legal.{potential_project}"
        elif 'payroll' in content_lower or 'earnings' in content_lower:
            return f"legal.payroll"
        elif 'agent' in content_lower or 'ai' in content_lower:
            return f"ai.{potential_project}"
        else:
            return f"general.{potential_project}"
    
    @classmethod
    def create_memory_block(
        cls,
        content: str,
        source_path: str,
        file_type: str,
        size_bytes: int
    ) -> MemoryBlock:
        """Create a MemoryBlock from content and metadata"""
        
        # Generate unique hash
        id_hash = cls.generate_hash(content, source_path)
        
        # Extract information from content
        summary = cls.generate_summary(content)
        topics = cls.extract_topics(content)
        archetype = cls.infer_archetype(content, file_type)
        project_id = cls.infer_project_id(content, source_path)
        
        # Create timestamp
        created_at = datetime.now().isoformat() + "Z"
        
        # Infer skills (nano scripts that might be relevant)
        skills = cls._infer_skills(content, file_type)
        
        # Create ethics metadata with PII redaction
        ethics = EthicsMetadata(
            pii_redacted=True,
            consent_logged=True,
            privacy_level=cls._determine_privacy_level(content)
        )
        
        # Create file metadata
        metadata = FileMetadata(
            source_file=source_path,
            file_type=file_type,
            size_bytes=size_bytes,
            last_modified=created_at,
            encoding="utf-8"
        )
        
        return MemoryBlock(
            id_hash=id_hash,
            summary=summary,
            content=content,
            topics=topics,
            skills=skills,
            project=project_id,
            archetype=archetype,
            created_at=created_at,
            ethics=ethics,
            metadata=metadata
        )
    
    @staticmethod
    def _infer_skills(content: str, file_type: str) -> List[str]:
        """Infer relevant nano skills/scripts based on content"""
        skills = []
        content_lower = content.lower()
        
        # File type based skills
        if file_type == '.py':
            skills.append("nano_warden_code_analyzer.py")
        elif file_type == '.md':
            skills.append("nano_warden_markdown_processor.py")
        elif file_type in ['.jpg', '.jpeg', '.png']:
            skills.append("nano_warden_vision_ocr.py")
        elif file_type == '.pdf':
            skills.append("nano_warden_pdf_extractor.py")
        
        # Content-based skills
        if 'memory' in content_lower:
            skills.append("nano_warden_memory_tagger.py")
        if 'legal' in content_lower:
            skills.append("nano_warden_legal_parser.py")
        if 'payroll' in content_lower:
            skills.append("nano_warden_payroll_parser.py")
        
        return skills if skills else ["nano_warden_general_processor.py"]
    
    @staticmethod
    def _determine_privacy_level(content: str) -> str:
        """Determine privacy level based on content"""
        content_lower = content.lower()
        
        # Check for sensitive information
        sensitive_keywords = ['ssn', 'social security', 'password', 'credit card', 'bank account', 'personal']
        if any(keyword in content_lower for keyword in sensitive_keywords):
            return "sensitive"
        
        # Check for private information
        private_keywords = ['earnings', 'salary', 'payroll', 'private', 'confidential']
        if any(keyword in content_lower for keyword in private_keywords):
            return "private"
        
        return "public"


def save_memory_block(block: MemoryBlock, output_dir: Path) -> Path:
    """Save MemoryBlock to JSON file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from hash
    filename = f"{block.id_hash.replace(':', '_')}.json"
    file_path = output_dir / filename
    
    # Save to JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(block.to_dict(), f, indent=2, ensure_ascii=False)
    
    return file_path


def load_memory_block(file_path: Path) -> MemoryBlock:
    """Load MemoryBlock from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return MemoryBlock.from_dict(data)


def load_all_memory_blocks(blocks_dir: Path) -> List[MemoryBlock]:
    """Load all MemoryBlocks from a directory"""
    blocks = []
    
    if not blocks_dir.exists():
        return blocks
    
    for json_file in blocks_dir.glob('*.json'):
        try:
            block = load_memory_block(json_file)
            blocks.append(block)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return blocks