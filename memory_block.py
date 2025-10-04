"""
MUSE Pantheon MemoryBlock Core Schema
Atomic, immutable, sovereign, and semantic data structures for AI agent memory systems.
"""
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class EthicsMetadata:
    """Ethics and consent tracking for MemoryBlocks"""
    pii_redacted: bool = True
    consent_logged: bool = True
    validation_status: str = "passed"
    redaction_reason: Optional[str] = None


@dataclass
class SourceMetadata:
    """Source file information for MemoryBlocks"""
    file_path: str
    file_type: str
    file_size: Optional[int] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None


@dataclass
class ProjectMetadata:
    """Project assignment and archetype mapping"""
    project_id: str
    archetype: str
    confidence_score: float = 0.0
    assignment_reason: Optional[str] = None


@dataclass
class MemoryBlock:
    """
    Core MemoryBlock structure for MUSE Pantheon system.
    Atomic, immutable, sovereign, and semantic data structure.
    """
    # Core identifiers
    id_hash: str
    summary: str
    content: str
    
    # Semantic metadata
    topics: List[str]
    skills: List[str]
    
    # Project and archetype
    project: Optional[ProjectMetadata] = None
    
    # Timestamps
    created_at: str = ""
    
    # Ethics and source tracking
    ethics: Optional[EthicsMetadata] = None
    source: Optional[SourceMetadata] = None
    
    # Additional metadata
    links: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values and validate structure"""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        
        if self.links is None:
            self.links = []
            
        if self.metadata is None:
            self.metadata = {}
            
        if self.ethics is None:
            self.ethics = EthicsMetadata()
    
    @classmethod
    def create_from_file(cls, file_path: Path, content: str, summary: str, 
                        topics: List[str], skills: List[str] = None) -> 'MemoryBlock':
        """Create a MemoryBlock from file content"""
        
        # Generate hash from content
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        id_hash = f"sha256:{content_hash}"
        
        # Create source metadata
        source = SourceMetadata(
            file_path=str(file_path),
            file_type=file_path.suffix.lower(),
            file_size=file_path.stat().st_size if file_path.exists() else None,
            created_at=datetime.fromtimestamp(file_path.stat().st_ctime).isoformat() + "Z" if file_path.exists() else None,
            modified_at=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() + "Z" if file_path.exists() else None
        )
        
        return cls(
            id_hash=id_hash,
            summary=summary,
            content=content,
            topics=topics,
            skills=skills or [],
            source=source
        )
    
    def assign_project(self, project_id: str, archetype: str, 
                      confidence: float = 0.0, reason: str = None):
        """Assign project and archetype to this MemoryBlock"""
        self.project = ProjectMetadata(
            project_id=project_id,
            archetype=archetype,
            confidence_score=confidence,
            assignment_reason=reason
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MemoryBlock to dictionary for JSON serialization"""
        result = asdict(self)
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert MemoryBlock to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_to_file(self, output_path: Path):
        """Save MemoryBlock to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'MemoryBlock':
        """Load MemoryBlock from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert nested dicts back to dataclasses
        if 'ethics' in data and data['ethics']:
            data['ethics'] = EthicsMetadata(**data['ethics'])
        if 'source' in data and data['source']:
            data['source'] = SourceMetadata(**data['source'])
        if 'project' in data and data['project']:
            data['project'] = ProjectMetadata(**data['project'])
            
        return cls(**data)
    
    def get_week_journal_format(self) -> Dict[str, Any]:
        """Convert to week journal format for IP proof documentation"""
        return {
            "id_hash": self.id_hash,
            "summary": self.summary,
            "content": self.content,
            "topics": self.topics,
            "project": self.project.project_id if self.project else "unknown",
            "archetype": self.project.archetype if self.project else "unknown",
            "created_at": self.created_at,
            "links": self.links,
            "metadata": {
                **self.metadata,
                "source": "founder_journal" if "journal" in self.source.file_path.lower() else "system_ingest",
                "validation_status": self.ethics.validation_status if self.ethics else "unknown"
            }
        }


class MemoryBlockCollection:
    """Collection manager for multiple MemoryBlocks"""
    
    def __init__(self):
        self.blocks: List[MemoryBlock] = []
    
    def add_block(self, block: MemoryBlock):
        """Add a MemoryBlock to the collection"""
        self.blocks.append(block)
    
    def save_all(self, output_dir: Path):
        """Save all MemoryBlocks to individual JSON files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, block in enumerate(self.blocks):
            filename = f"memory_block_{block.id_hash.split(':')[1]}.json"
            block.save_to_file(output_dir / filename)
    
    def to_jsonl(self, output_path: Path):
        """Save all MemoryBlocks as JSON Lines format"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for block in self.blocks:
                f.write(block.to_json(indent=None) + '\n')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'total_blocks': len(self.blocks),
            'archetypes': list(set(b.project.archetype for b in self.blocks if b.project)),
            'projects': list(set(b.project.project_id for b in self.blocks if b.project)),
            'file_types': list(set(b.source.file_type for b in self.blocks if b.source)),
            'topics_frequency': self._get_topic_frequency()
        }
    
    def _get_topic_frequency(self) -> Dict[str, int]:
        """Calculate topic frequency across all blocks"""
        topic_counts = {}
        for block in self.blocks:
            for topic in block.topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        return dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))


# MUSE Pantheon Archetypes
ARCHETYPES = {
    'Guardian': 'Security, ethics, validation content',
    'Vision': 'Images, visual content, OCR results',
    'Warden': 'Orchestration, pipeline management', 
    'Memory': 'Storage, retrieval, embedding systems',
    'Scribe': 'Documentation, writing, content creation',
    'Analyst': 'Data analysis, clustering, insights',
    'Builder': 'Code, construction, development',
    'Explorer': 'Discovery, research, investigation',
    'Mentor': 'Teaching, guidance, knowledge transfer',
    'Connector': 'Integration, communication, linking',
    'Sage': 'Wisdom, deep knowledge, philosophy',
    'Creator': 'Innovation, art, creative expression'
}

# File type to archetype mapping
FILE_TYPE_ARCHETYPES = {
    '.py': 'Builder',
    '.js': 'Builder', 
    '.ts': 'Builder',
    '.md': 'Scribe',
    '.txt': 'Scribe',
    '.pdf': 'Scribe',
    '.jpg': 'Vision',
    '.jpeg': 'Vision',
    '.png': 'Vision',
    '.mp4': 'Vision',
    '.csv': 'Analyst',
    '.json': 'Memory',
    '.zip': 'Explorer'
}