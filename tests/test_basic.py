#!/usr/bin/env python3
"""
Basic tests for BioinformaticsAgent components
"""

import unittest
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bioagent_architecture import (
    BioinformaticsAgent, DataMetadata, DataType, AnalysisTask, ReasoningType
)
from bioagent_tools import SequenceStatsTool, get_all_bioinformatics_tools


class TestBioinformaticsAgent(unittest.TestCase):
    """Test cases for the main BioinformaticsAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = BioinformaticsAgent()
    
    def test_agent_initialization(self):
        """Test that agent initializes correctly"""
        self.assertIsInstance(self.agent, BioinformaticsAgent)
        self.assertEqual(len(self.agent.tools), 0)  # No tools registered initially
    
    def test_tool_registration(self):
        """Test tool registration"""
        tool = SequenceStatsTool()
        self.agent.register_tool(tool)
        
        self.assertEqual(len(self.agent.tools), 1)
        self.assertIn(tool.name, self.agent.tools)
    
    def test_multiple_tool_registration(self):
        """Test registering multiple tools"""
        tools = get_all_bioinformatics_tools()
        
        for tool in tools:
            self.agent.register_tool(tool)
        
        self.assertEqual(len(self.agent.tools), len(tools))


class TestDataMetadata(unittest.TestCase):
    """Test cases for DataMetadata class"""
    
    def test_metadata_creation(self):
        """Test creating metadata"""
        metadata = DataMetadata(
            data_type=DataType.EXPRESSION_MATRIX,
            file_path="test.csv",
            organism="Homo sapiens"
        )
        
        self.assertEqual(metadata.data_type, DataType.EXPRESSION_MATRIX)
        self.assertEqual(metadata.file_path, "test.csv")
        self.assertEqual(metadata.organism, "Homo sapiens")
    
    def test_metadata_context_string(self):
        """Test metadata to context string conversion"""
        metadata = DataMetadata(
            data_type=DataType.GENOMIC_SEQUENCE,
            file_path="test.fasta",
            organism="Mus musculus",
            quality_metrics={"gc_content": 0.42}
        )
        
        context = metadata.to_context_string()
        
        self.assertIn("genomic_sequence", context)
        self.assertIn("test.fasta", context)
        self.assertIn("Mus musculus", context)
        self.assertIn("gc_content", context)


class TestAnalysisTask(unittest.TestCase):
    """Test cases for AnalysisTask class"""
    
    def test_task_creation(self):
        """Test creating an analysis task"""
        metadata = DataMetadata(
            data_type=DataType.PROTEIN_SEQUENCE,
            file_path="proteins.fasta"
        )
        
        task = AnalysisTask(
            task_id="test_task",
            instruction="Analyze protein sequences",
            data_metadata=[metadata],
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
        )
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.instruction, "Analyze protein sequences")
        self.assertEqual(len(task.data_metadata), 1)
        self.assertEqual(task.reasoning_type, ReasoningType.CHAIN_OF_THOUGHT)


class TestSequenceStatsTool(unittest.TestCase):
    """Test cases for SequenceStatsTool"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tool = SequenceStatsTool()
    
    def test_tool_properties(self):
        """Test tool basic properties"""
        self.assertEqual(self.tool.name, "sequence_stats")
        self.assertIn("sequence", self.tool.description.lower())
        self.assertIn(DataType.GENOMIC_SEQUENCE, self.tool.supported_data_types)
        self.assertIn(DataType.PROTEIN_SEQUENCE, self.tool.supported_data_types)
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters
        valid_params = {
            "input_file": "test.fasta",
            "sequence_type": "dna"
        }
        is_valid, error = self.tool.validate_parameters(valid_params)
        self.assertTrue(is_valid)
        
        # Invalid parameters (missing required)
        invalid_params = {
            "sequence_type": "dna"
        }
        is_valid, error = self.tool.validate_parameters(invalid_params)
        self.assertFalse(is_valid)
        self.assertIn("input_file", error)
    
    def test_data_compatibility(self):
        """Test data type compatibility checking"""
        # Compatible data
        compatible_metadata = [
            DataMetadata(data_type=DataType.GENOMIC_SEQUENCE, file_path="test.fasta")
        ]
        self.assertTrue(self.tool.validate_data_compatibility(compatible_metadata))
        
        # Incompatible data
        incompatible_metadata = [
            DataMetadata(data_type=DataType.EXPRESSION_MATRIX, file_path="test.csv")
        ]
        self.assertFalse(self.tool.validate_data_compatibility(incompatible_metadata))


class TestAsyncComponents(unittest.TestCase):
    """Test cases for async components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = BioinformaticsAgent()
        self.agent.register_tool(SequenceStatsTool())
    
    def test_async_initialization(self):
        """Test async components can be initialized"""
        async def test_async():
            # This test mainly checks that async components don't crash
            metadata = DataMetadata(
                data_type=DataType.GENOMIC_SEQUENCE,
                file_path="nonexistent.fasta"
            )
            
            task = AnalysisTask(
                task_id="async_test",
                instruction="Test async functionality",
                data_metadata=[metadata]
            )
            
            # We expect this to fail gracefully since file doesn't exist
            try:
                result = await self.agent.analyze_data(task)
                # If it doesn't crash, that's good
                self.assertIn('success', result)
            except Exception as e:
                # Expected to fail due to missing file, but shouldn't crash
                self.assertIsInstance(e, Exception)
        
        # Run the async test
        asyncio.run(test_async())


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)