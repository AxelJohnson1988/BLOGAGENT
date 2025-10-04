"""
MUSE Pantheon MemoryBlock Schema
Core data structure for universal ingest pipeline
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib
import json
import uuid


@dataclass
class MemoryBlock:
    """
    Atomic, immutable, sovereign, and semantic memory unit.
    Core data structure for the MUSE Pantheon system.
    """
    # Core Identity
    id_hash: str = field(default="")
    summary: str = field(default="")
    content: str = field(default="")
    
    # Semantic Classification
    topics: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    archetype: str = field(default="unknown")  # MUSE archetype (Guardian, Vision, etc.)
    
    # Temporal & Project Context
    created_at: str = field(default="")
    date: str = field(default="")
    project: str = field(default="")
    project_id: str = field(default="")
    
    # Ethics & Consent Tracking
    ethics_status: str = field(default="approved")
    consent_logged: bool = field(default=True)
    pii_redacted: bool = field(default=False)
    
    # Source & Lineage
    source_file: str = field(default="")
    source_type: str = field(default="")
    source_hash: str = field(default="")
    
    # Semantic Relations
    links: List[str] = field(default_factory=list)
    embeddings: Optional[List[float]] = field(default=None)
    
    # Metadata & Extensions
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_status: str = field(default="passed")
    emotion_score: float = field(default=0.0)
    
    def __post_init__(self):
        """Initialize computed fields after creation"""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.date:
            self.date = datetime.utcnow().strftime("%Y-%m-%d")
        if not self.id_hash:
            self.id_hash = self.generate_hash()
    
    def generate_hash(self) -> str:
        """Generate unique hash for this memory block"""
        content_str = f"{self.summary}:{self.content}:{self.source_file}:{self.created_at}"
        hash_obj = hashlib.sha256(content_str.encode('utf-8'))
        return hash_obj.hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id_hash': self.id_hash,
            'summary': self.summary,
            'content': self.content,
            'topics': self.topics,
            'skills': self.skills,
            'archetype': self.archetype,
            'created_at': self.created_at,
            'date': self.date,
            'project': self.project,
            'project_id': self.project_id,
            'ethics_status': self.ethics_status,
            'consent_logged': self.consent_logged,
            'pii_redacted': self.pii_redacted,
            'source_file': self.source_file,
            'source_type': self.source_type,
            'source_hash': self.source_hash,
            'links': self.links,
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'validation_status': self.validation_status,
            'emotion_score': self.emotion_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        """Create MemoryBlock from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryBlock':
        """Create MemoryBlock from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class MemoryBlockBuilder:
    """Builder pattern for creating MemoryBlocks with validation"""
    
    def __init__(self):
        self.data = {}
    
    def with_content(self, summary: str, content: str) -> 'MemoryBlockBuilder':
        """Set core content"""
        self.data['summary'] = summary
        self.data['content'] = content
        return self
    
    def with_source(self, file_path: str, file_type: str) -> 'MemoryBlockBuilder':
        """Set source information"""
        self.data['source_file'] = file_path
        self.data['source_type'] = file_type
        # Generate source hash
        if file_path:
            self.data['source_hash'] = hashlib.sha256(file_path.encode()).hexdigest()[:12]
        return self
    
    def with_topics(self, topics: List[str]) -> 'MemoryBlockBuilder':
        """Set semantic topics"""
        self.data['topics'] = [topic.lower().strip() for topic in topics]
        return self
    
    def with_archetype(self, archetype: str) -> 'MemoryBlockBuilder':
        """Set MUSE archetype"""
        self.data['archetype'] = archetype
        return self
    
    def with_project(self, project: str, project_id: str = "") -> 'MemoryBlockBuilder':
        """Set project context"""
        self.data['project'] = project
        self.data['project_id'] = project_id or project.lower().replace(' ', '_')
        return self
    
    def with_ethics(self, pii_redacted: bool = False, consent_logged: bool = True) -> 'MemoryBlockBuilder':
        """Set ethics and consent tracking"""
        self.data['pii_redacted'] = pii_redacted
        self.data['consent_logged'] = consent_logged
        self.data['ethics_status'] = "approved" if consent_logged else "pending"
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'MemoryBlockBuilder':
        """Add metadata"""
        self.data['metadata'] = metadata
        return self
    
    def build(self) -> MemoryBlock:
        """Build and validate the MemoryBlock"""
        # Validation
        if not self.data.get('summary'):
            raise ValueError("MemoryBlock requires a summary")
        if not self.data.get('content'):
            raise ValueError("MemoryBlock requires content")
        
        return MemoryBlock(**self.data)


# Archetype mappings for different content types
MUSE_ARCHETYPES = {
    'code': 'Builder',
    'documentation': 'Scribe', 
    'image': 'Vision',
    'video': 'Vision',
    'audio': 'Voice',
    'data': 'Analyst',
    'legal': 'Guardian',
    'financial': 'Guardian',
    'personal': 'Keeper',
    'research': 'Scholar',
    'creative': 'Muse',
    'communication': 'Herald',
    'system': 'Warden'
}

def get_archetype_for_content(file_type: str, content: str, topics: List[str]) -> str:
    """Determine appropriate MUSE archetype based on content analysis"""
    file_type = file_type.lower()
    content_lower = content.lower()
    topics_lower = [t.lower() for t in topics]
    
    # Direct file type mappings
    if any(ext in file_type for ext in ['.py', '.js', '.java', '.cpp', '.c', '.html', '.css']):
        return 'Builder'
    elif any(ext in file_type for ext in ['.jpg', '.png', '.gif', '.svg', '.jpeg']):
        return 'Vision'
    elif any(ext in file_type for ext in ['.mp4', '.avi', '.mov', '.mkv']):
        return 'Vision'
    elif any(ext in file_type for ext in ['.mp3', '.wav', '.flac', '.ogg']):
        return 'Voice'
    
    # Content-based analysis
    if any(topic in topics_lower for topic in ['legal', 'contract', 'agreement', 'law']):
        return 'Guardian'
    elif any(topic in topics_lower for topic in ['financial', 'payment', 'invoice', 'tax']):
        return 'Guardian'
    elif any(topic in topics_lower for topic in ['research', 'study', 'analysis', 'academic']):
        return 'Scholar'
    elif any(topic in topics_lower for topic in ['creative', 'art', 'design', 'story']):
        return 'Muse'
    elif any(topic in topics_lower for topic in ['communication', 'email', 'message', 'chat']):
        return 'Herald'
    elif any(topic in topics_lower for topic in ['system', 'config', 'setup', 'admin']):
        return 'Warden'
    elif any(topic in topics_lower for topic in ['documentation', 'readme', 'guide', 'manual']):
        return 'Scribe'
    elif any(topic in topics_lower for topic in ['data', 'statistics', 'metrics', 'analytics']):
        return 'Analyst'
    elif any(topic in topics_lower for topic in ['personal', 'diary', 'journal', 'private']):
        return 'Keeper'
    
    return 'unknown'