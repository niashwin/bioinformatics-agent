#!/usr/bin/env python3
"""
Quick Start Example for BioinformaticsAgent

This example demonstrates basic usage of the BioinformaticsAgent system.
"""

import asyncio
import sys
import os

# Add src to path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bioagent_architecture import (
    BioinformaticsAgent, DataMetadata, DataType, AnalysisTask, ReasoningType
)
from bioagent_tools import get_all_bioinformatics_tools


async def quick_start_example():
    """Quick start example for new users"""
    
    print("=" * 60)
    print("BioinformaticsAgent - Quick Start Example")
    print("=" * 60)
    
    # Initialize the agent
    print("\n1. Initializing BioinformaticsAgent...")
    agent = BioinformaticsAgent()
    
    # Register all available tools
    print("2. Registering bioinformatics tools...")
    tools = get_all_bioinformatics_tools()
    for tool in tools:
        agent.register_tool(tool)
    print(f"   Registered {len(tools)} tools")
    
    # Create sample data metadata
    print("3. Setting up analysis task...")
    metadata = DataMetadata(
        data_type=DataType.EXPRESSION_MATRIX,
        file_path="sample_rnaseq_data.csv",
        organism="Homo sapiens",
        tissue_type="liver",
        experimental_condition="drug_treatment_vs_control",
        sample_size=20,
        quality_metrics={
            "average_quality": 32,
            "total_reads": 25_000_000,
            "mapping_rate": 0.94
        }
    )
    
    # Create analysis task
    task = AnalysisTask(
        task_id="quick_start_example",
        instruction="""
        Perform RNA-seq differential expression analysis:
        1. Quality control assessment
        2. Differential expression analysis between conditions
        3. Multiple testing correction (FDR)
        4. Generate volcano plot and heatmap
        5. Provide biological interpretation
        """,
        data_metadata=[metadata],
        reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
    )
    
    # Perform analysis
    print("4. Running analysis...")
    print("   Note: This is a demonstration - actual file processing would require real data")
    
    try:
        result = await agent.analyze_data(task)
        
        print("\n5. Analysis Results:")
        print("-" * 40)
        if result.get('success'):
            print("✓ Analysis completed successfully!")
            print(f"✓ Reasoning steps: {len(result.get('reasoning_steps', []))}")
            print(f"✓ Iterations required: {result.get('iterations', 1)}")
            
            # Show generated code preview
            code = result.get('code', '')
            if code:
                print(f"\n6. Generated Analysis Code Preview ({len(code)} characters):")
                print("-" * 40)
                # Show first 300 characters
                preview = code[:300] + "..." if len(code) > 300 else code
                print(preview)
            
        else:
            print("✗ Analysis failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        print("This is expected in demo mode without actual data files")
    
    print("\n" + "=" * 60)
    print("Quick Start Example Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Prepare your own bioinformatics data")
    print("2. Update the DataMetadata with your file paths")
    print("3. Customize the analysis instruction")
    print("4. Run the analysis on real data")
    print("\nFor more examples, see the full example in bioagent_example.py")


if __name__ == "__main__":
    asyncio.run(quick_start_example())