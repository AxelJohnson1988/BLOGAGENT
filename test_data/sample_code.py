#!/usr/bin/env python3
"""
Sample Python file for testing the BLOGAGENT universal ingest pipeline.
This file demonstrates code analysis and memory block creation.
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any


class ExampleClass:
    """An example class to demonstrate docstring extraction"""
    
    def __init__(self, name: str):
        """Initialize the example class with a name"""
        self.name = name
        self.created_at = datetime.now()
    
    def process_data(self, data: List[Dict[str, Any]]) -> str:
        """
        Process input data and return a summary.
        
        This method demonstrates the type of code that would be analyzed
        by the universal ingest pipeline.
        """
        if not data:
            return "No data to process"
        
        # Create a hash of the data
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        
        return f"Processed {len(data)} items (hash: {data_hash[:8]})"


def main():
    """Main function demonstrating the example class"""
    example = ExampleClass("test_pipeline")
    
    sample_data = [
        {"id": 1, "type": "document", "content": "Sample document content"},
        {"id": 2, "type": "image", "content": "Image analysis results"},
        {"id": 3, "type": "code", "content": "Python source code"}
    ]
    
    result = example.process_data(sample_data)
    print(f"Example result: {result}")


if __name__ == "__main__":
    main()