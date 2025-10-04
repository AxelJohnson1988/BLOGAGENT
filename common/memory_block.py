#!/usr/bin/env python3
"""
MUSE Pantheon MemoryBlock Schema
Core data structure for immutable, atomic, sovereign, and semantic memory blocks.
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class MemoryBlock:
    """
    Atomic, immutable, sovereign, and semantic memory block.
    Core data structure for the MUSE Pantheon system.
    """
    # Core identification
    id_hash: str
    summary: str
    content: str
    
    # Metadata
    topics: List[str]
    skills: List[str]
    date: str
    project: str
    archetype: str
    
    # System fields
    created_at: str = ""
    source_path: Optional[str] = None
    file_type: Optional[str] = None
    
    # Ethics and consent tracking
    pii_redacted: bool = False
    consent_logged: bool = True
    ethics_review: str = "passed"
    
    # Relationships
    links: List[str] = None
    parent_blocks: List[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values and validate required fields."""
        if self.links is None:
            self.links = []
        if self.parent_blocks is None:
            self.parent_blocks = []
        if self.metadata is None:
            self.metadata = {}
            
        # Generate hash if not provided
        if not self.id_hash:
            self.id_hash = self.generate_hash()
            
        # Set created_at if not provided
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def generate_hash(self) -> str:
        """Generate SHA256 hash from content and metadata."""
        content_for_hash = f"{self.summary}:{self.content}:{self.date}"
        return hashlib.sha256(content_for_hash.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        """Create MemoryBlock from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryBlock':
        """Create MemoryBlock from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, output_dir: Path) -> Path:
        """Save memory block to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"memory_block_{self.id_hash}.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
            
        return filepath

    def validate(self) -> bool:
        """Validate memory block structure and content."""
        required_fields = ['id_hash', 'summary', 'content', 'topics', 'date', 'archetype']
        for field in required_fields:
            if not getattr(self, field):
                return False
        return True


class MemoryBlockFactory:
    """Factory for creating MemoryBlocks from various sources."""
    
    ARCHETYPES = [
        "Discoverer", "Guardian", "Alchemist", "Oracle", "Sage", "Shaman",
        "Visionary", "Architect", "Weaver", "Navigator", "Storyteller", "Scribe"
    ]
    
    @classmethod
    def create_from_file(cls, filepath: Path, content: str, summary: str = None) -> MemoryBlock:
        """Create MemoryBlock from file content."""
        # Generate summary if not provided
        if not summary:
            summary = cls._generate_summary(content, filepath)
        
        # Extract topics from content
        topics = cls._extract_topics(content, filepath)
        
        # Determine archetype based on content type
        archetype = cls._determine_archetype(filepath, content)
        
        # Generate skills based on file type and content
        skills = cls._generate_skills(filepath, content)
        
        # Determine project based on path and content
        project = cls._determine_project(filepath, content)
        
        return MemoryBlock(
            id_hash="",  # Will be generated in __post_init__
            summary=summary,
            content=content,
            topics=topics,
            skills=skills,
            date=datetime.now().strftime("%Y-%m-%d"),
            project=project,
            archetype=archetype,
            source_path=str(filepath),
            file_type=filepath.suffix.lower(),
            metadata={
                "file_size": filepath.stat().st_size if filepath.exists() else 0,
                "source": "file_ingest",
                "validation_status": "pending"
            }
        )
    
    @classmethod
    def _generate_summary(cls, content: str, filepath: Path) -> str:
        """Generate summary from content."""
        # Simple summary generation - first few sentences or truncated content
        sentences = content.split('. ')
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'
        elif len(content) > 200:
            return content[:200] + "..."
        else:
            return content
    
    @classmethod
    def _extract_topics(cls, content: str, filepath: Path) -> List[str]:
        """Extract topics from content and filename."""
        topics = []
        
        # Add file type as topic
        if filepath.suffix:
            topics.append(filepath.suffix.lower().replace('.', ''))
        
        # Add filename parts as topics
        name_parts = filepath.stem.lower().split('_')
        topics.extend([part for part in name_parts if len(part) > 2])
        
        # Simple keyword extraction from content
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = content.lower().split()
        word_freq = {}
        
        for word in words:
            word = word.strip('.,!?";()[]{}')
            if len(word) > 3 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top 5 most frequent words as topics
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics.extend([word for word, count in top_words])
        
        return list(set(topics))[:10]  # Limit to 10 unique topics
    
    @classmethod
    def _determine_archetype(cls, filepath: Path, content: str) -> str:
        """Determine archetype based on file type and content."""
        file_ext = filepath.suffix.lower()
        
        # Map file types to archetypes
        archetype_mapping = {
            '.py': 'Alchemist',
            '.js': 'Alchemist', 
            '.ts': 'Alchemist',
            '.md': 'Scribe',
            '.txt': 'Scribe',
            '.pdf': 'Sage',
            '.jpg': 'Visionary',
            '.jpeg': 'Visionary',
            '.png': 'Visionary',
            '.mp4': 'Storyteller',
            '.json': 'Oracle',
            '.csv': 'Oracle',
            '.zip': 'Guardian'
        }
        
        return archetype_mapping.get(file_ext, 'Discoverer')
    
    @classmethod
    def _generate_skills(cls, filepath: Path, content: str) -> List[str]:
        """Generate skills based on file type and content."""
        skills = []
        file_ext = filepath.suffix.lower()
        
        # Base skill from file processing
        skills.append(f"nano_warden_universal_ingest_{file_ext.replace('.', '')}.py")
        
        # Add content-specific skills
        if 'class ' in content or 'def ' in content:
            skills.append("nano_warden_code_analyzer.py")
        if 'import ' in content:
            skills.append("nano_warden_dependency_tracker.py")
        if any(word in content.lower() for word in ['memory', 'block', 'muse']):
            skills.append("nano_warden_memory_system.py")
        
        return skills
    
    @classmethod
    def _determine_project(cls, filepath: Path, content: str) -> str:
        """Determine project based on path and content."""
        path_parts = filepath.parts
        
        # Check for common project patterns
        if 'muse' in str(filepath).lower():
            return 'muse.pantheon'
        elif 'blog' in str(filepath).lower():
            return 'blog.agent'
        elif 'warden' in str(filepath).lower():
            return 'warden.system'
        elif any(part in ['test', 'tests'] for part in path_parts):
            return 'testing.framework'
        else:
            return 'general.development'