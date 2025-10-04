"""
Sample Python script for testing MUSE Pantheon ingest pipeline.
This demonstrates code file extraction and MemoryBlock creation.
"""

class SampleAgent:
    """Sample AI agent class for demonstration"""
    
    def __init__(self, name: str):
        self.name = name
        self.memory_blocks = []
    
    def process_data(self, data: str) -> dict:
        """Process input data and return structured result"""
        return {
            'processed': True,
            'agent': self.name,
            'data_length': len(data)
        }
    
    def add_memory(self, block):
        """Add a memory block to the agent's memory"""
        self.memory_blocks.append(block)


def main():
    """Main function demonstrating agent usage"""
    agent = SampleAgent("TestAgent")
    result = agent.process_data("sample data")
    print(f"Agent {agent.name} processed data: {result}")


if __name__ == "__main__":
    main()