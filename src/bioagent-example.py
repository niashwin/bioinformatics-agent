#!/usr/bin/env python3
"""
BioinformaticsAgent Example Implementation

This module demonstrates a complete example of using the BioinformaticsAgent system
for various common bioinformatics analyses:
- RNA-seq differential expression analysis
- Variant calling and annotation
- Protein structure analysis
- Multi-omics integration
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tempfile
import os

# Import all components of the BioinformaticsAgent system
from bioagent_architecture import (
    BioinformaticsAgent, DataMetadata, DataType, AnalysisTask, 
    ReasoningType, BioinformaticsHypothesis
)
from bioagent_tools import (
    get_all_bioinformatics_tools, SequenceStatsTool, 
    RNASeqDifferentialExpressionTool, VariantAnnotationTool,
    ProteinStructureAnalysisTool, BioinformaticsVisualizationTool,
    PhylogeneticAnalysisTool
)
from bioagent_reasoning import (
    BioinformaticsReasoningPattern, BioinformaticsChainOfThought,
    BioinformaticsReflectionEngine, HypothesisEngine,
    IterativeImprovementFramework
)
from bioagent_pipeline import (
    BioinformaticsPipeline, PipelineManager, PipelineStep,
    ExecutionMode, PipelineTemplateLibrary
)
from bioagent_prompts import BioinformaticsPromptConstructor
from bioagent_feedback import (
    FeedbackIntegrationEngine, IterativeImprovementSystem,
    UserFeedbackHandler, AdaptiveLearningSystem
)


# =================== Example Data Generation ===================

def generate_example_data():
    """Generate example data files for demonstration"""
    
    example_dir = Path("example_data")
    example_dir.mkdir(exist_ok=True)
    
    # 1. RNA-seq count matrix
    np.random.seed(42)
    genes = [f"GENE_{i:04d}" for i in range(1000)]
    samples = [f"Sample_{i}" for i in range(1, 13)]  # 6 control, 6 treatment
    
    # Generate count data with some differential expression
    base_expression = np.random.negative_binomial(10, 0.3, (1000, 12))
    
    # Add differential expression to some genes
    de_genes = np.random.choice(1000, 100, replace=False)
    for gene_idx in de_genes[:50]:  # Upregulated
        base_expression[gene_idx, 6:] = base_expression[gene_idx, 6:] * np.random.uniform(2, 5)
    for gene_idx in de_genes[50:]:  # Downregulated
        base_expression[gene_idx, 6:] = base_expression[gene_idx, 6:] * np.random.uniform(0.2, 0.5)
    
    count_df = pd.DataFrame(base_expression, index=genes, columns=samples)
    count_df.to_csv(example_dir / "rnaseq_counts.csv")
    
    # 2. Sample information
    sample_info = pd.DataFrame({
        'sample_id': samples,
        'condition': ['control'] * 6 + ['treatment'] * 6,
        'batch': ['batch1', 'batch1', 'batch2', 'batch2', 'batch3', 'batch3'] * 2,
        'age': np.random.randint(20, 60, 12),
        'sex': np.random.choice(['M', 'F'], 12)
    })
    sample_info.to_csv(example_dir / "sample_info.csv", index=False)
    
    # 3. Genomic sequences (FASTA)
    with open(example_dir / "sequences.fasta", "w") as f:
        sequences = [
            ("seq1_human", "ATGCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"),
            ("seq2_mouse", "ATGCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"),
            ("seq3_rat", "ATGCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"),
            ("seq4_chimp", "ATGCGATCGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC")
        ]
        for seq_id, seq in sequences:
            f.write(f">{seq_id}\n{seq}\n")
    
    # 4. Variant data (VCF-like)
    with open(example_dir / "variants.vcf", "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        variants = [
            ("chr1", "12345", ".", "A", "G", "100", "PASS", "DP=30"),
            ("chr1", "23456", ".", "C", "T", "150", "PASS", "DP=40"),
            ("chr2", "34567", ".", "G", "A", "200", "PASS", "DP=50"),
            ("chr3", "45678", ".", "T", "C", "120", "PASS", "DP=35")
        ]
        for var in variants:
            f.write("\t".join(map(str, var)) + "\n")
    
    # 5. Protein sequences
    with open(example_dir / "proteins.fasta", "w") as f:
        proteins = [
            ("PROT1", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"),
            ("PROT2", "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKNNREK"),
            ("PROT3", "MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDM")
        ]
        for prot_id, seq in proteins:
            f.write(f">{prot_id}\n{seq}\n")
    
    return example_dir


# =================== Example 1: RNA-seq Analysis ===================

async def example_rnaseq_analysis():
    """Complete RNA-seq differential expression analysis example"""
    
    print("=" * 50)
    print("Example 1: RNA-seq Differential Expression Analysis")
    print("=" * 50)
    
    # Generate example data
    data_dir = generate_example_data()
    
    # Initialize the agent
    agent = BioinformaticsAgent()
    
    # Register all tools
    for tool in get_all_bioinformatics_tools():
        agent.register_tool(tool)
    
    # Create data metadata
    metadata = DataMetadata(
        data_type=DataType.EXPRESSION_MATRIX,
        file_path=str(data_dir / "rnaseq_counts.csv"),
        organism="Homo sapiens",
        tissue_type="liver",
        experimental_condition="drug_treatment_vs_control",
        sequencing_platform="Illumina NovaSeq",
        sample_size=12,
        quality_metrics={
            "average_quality": 35,
            "total_reads": 30_000_000,
            "mapping_rate": 0.95
        }
    )
    
    # Create analysis task
    task = AnalysisTask(
        task_id="rnaseq_example_001",
        instruction="""
        Perform comprehensive RNA-seq differential expression analysis:
        1. Quality control and data exploration
        2. Differential expression between treatment and control
        3. Multiple testing correction
        4. Pathway enrichment analysis
        5. Generate visualizations (PCA, volcano plot, heatmap)
        Interpret results in biological context.
        """,
        data_metadata=[metadata],
        expected_outputs=[
            "de_results.csv", 
            "pathway_enrichment.csv", 
            "pca_plot.png", 
            "volcano_plot.png"
        ],
        reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
    )
    
    # Perform analysis
    print("\nStarting RNA-seq analysis...")
    result = await agent.analyze_data(task)
    
    # Display results
    print(f"\nAnalysis Status: {'Success' if result['success'] else 'Failed'}")
    if result['success']:
        print(f"Iterations required: {result['iterations']}")
        print(f"Generated code length: {len(result['code'])} characters")
        
        # Show code snippet
        print("\nGenerated Analysis Code (first 500 chars):")
        print("-" * 40)
        print(result['code'][:500] + "...")
        print("-" * 40)
    
    # Collect feedback
    feedback_engine = FeedbackIntegrationEngine()
    feedback_report = await feedback_engine.collect_comprehensive_feedback(
        result, {"analysis_id": task.task_id}
    )
    
    print(f"\nQuality Assessment:")
    print(f"Overall quality score: {feedback_report.overall_quality_score:.2f}")
    print(f"Critical issues: {len(feedback_report.get_critical_items())}")
    print(f"Improvement suggestions: {len(feedback_report.improvement_priority_list)}")
    
    return result


# =================== Example 2: Variant Analysis Pipeline ===================

async def example_variant_analysis():
    """Variant calling and annotation pipeline example"""
    
    print("\n" + "=" * 50)
    print("Example 2: Variant Analysis Pipeline")
    print("=" * 50)
    
    # Initialize pipeline manager
    pipeline_manager = PipelineManager()
    
    # Create variant analysis pipeline
    pipeline = BioinformaticsPipeline(
        pipeline_id="variant_pipeline_001",
        execution_mode=ExecutionMode.ADAPTIVE
    )
    
    # Add pipeline steps
    variant_tool = VariantAnnotationTool()
    stats_tool = SequenceStatsTool()
    viz_tool = BioinformaticsVisualizationTool()
    
    # Step 1: Quality control
    qc_step = PipelineStep(
        step_id="variant_qc",
        tool=stats_tool,
        parameters={
            "input_file": "example_data/sequences.fasta",
            "sequence_type": "dna",
            "output_format": "json"
        }
    )
    pipeline.add_step(qc_step)
    
    # Step 2: Variant annotation
    annotation_step = PipelineStep(
        step_id="variant_annotation",
        tool=variant_tool,
        parameters={
            "vcf_file": "example_data/variants.vcf",
            "genome_build": "hg38",
            "include_population_freq": True,
            "include_pathogenicity": True
        },
        dependencies=["variant_qc"]
    )
    pipeline.add_step(annotation_step)
    
    # Step 3: Visualization
    viz_step = PipelineStep(
        step_id="variant_visualization",
        tool=viz_tool,
        parameters={
            "data_file": "$output:variant_annotation:annotated_variants",
            "plot_type": "scatter",
            "output_format": "png"
        },
        dependencies=["variant_annotation"]
    )
    pipeline.add_step(viz_step)
    
    # Validate pipeline
    is_valid, errors = pipeline.validate_pipeline()
    print(f"\nPipeline validation: {'Passed' if is_valid else 'Failed'}")
    if errors:
        print(f"Errors: {errors}")
    
    # Create metadata
    metadata = DataMetadata(
        data_type=DataType.VARIANT_DATA,
        file_path="example_data/variants.vcf",
        organism="Homo sapiens",
        genome_build="hg38"
    )
    
    # Execute pipeline
    print("\nExecuting variant analysis pipeline...")
    try:
        pipeline_result = await pipeline.execute([metadata])
        print(f"Pipeline status: {pipeline_result.status}")
        print(f"Success rate: {pipeline_result.success_rate:.2%}")
        print(f"Total runtime: {pipeline_result.total_runtime}")
        
        # Show step results
        print("\nStep Results:")
        for step_id, result in pipeline_result.step_results.items():
            print(f"- {step_id}: {'Success' if result.success else 'Failed'}")
    
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
    
    return pipeline_result


# =================== Example 3: Protein Analysis with Reflection ===================

async def example_protein_analysis():
    """Protein structure and sequence analysis with reflection"""
    
    print("\n" + "=" * 50)
    print("Example 3: Protein Analysis with Reflection")
    print("=" * 50)
    
    # Initialize agent with enhanced reasoning
    agent = BioinformaticsAgent()
    
    # Register protein analysis tools
    agent.register_tool(SequenceStatsTool())
    agent.register_tool(ProteinStructureAnalysisTool())
    agent.register_tool(BioinformaticsVisualizationTool())
    
    # Create metadata
    metadata = DataMetadata(
        data_type=DataType.PROTEIN_SEQUENCE,
        file_path="example_data/proteins.fasta",
        organism="Homo sapiens",
        custom_metadata={
            "protein_family": "kinase",
            "experimental_method": "X-ray crystallography"
        }
    )
    
    # Create analysis task
    task = AnalysisTask(
        task_id="protein_analysis_001",
        instruction="""
        Analyze protein sequences and structures:
        1. Calculate sequence statistics and properties
        2. Identify functional domains and motifs
        3. Analyze physicochemical properties
        4. Compare sequences for evolutionary relationships
        5. Generate structural insights if possible
        """,
        data_metadata=[metadata],
        reasoning_type=ReasoningType.ITERATIVE_REFINEMENT
    )
    
    # Initialize reflection engine
    reflection_engine = BioinformaticsReflectionEngine()
    
    # Perform initial analysis
    print("\nPerforming initial protein analysis...")
    result = await agent.analyze_data(task)
    
    # Simulate reflection process
    from bioagent_architecture import ReflectionContext
    
    reflection_context = ReflectionContext(
        original_analysis=task.instruction,
        generated_code=result.get('code', ''),
        execution_results=str(result)
    )
    
    # Perform deep reflection
    print("\nPerforming reflection and quality assessment...")
    reflected_context = await reflection_engine.deep_reflection(
        reflection_context, task
    )
    
    # Display reflection results
    if hasattr(reflected_context, 'quality_assessments'):
        quality = reflected_context.quality_assessments[0]
        print(f"\nQuality Assessment Scores:")
        print(f"- Data Quality: {quality.data_quality_score:.2f}")
        print(f"- Methodological Soundness: {quality.methodological_soundness:.2f}")
        print(f"- Statistical Validity: {quality.statistical_validity:.2f}")
        print(f"- Biological Relevance: {quality.biological_relevance:.2f}")
        print(f"- Reproducibility: {quality.reproducibility_score:.2f}")
        print(f"- Overall Confidence: {quality.overall_confidence:.2f}")
    
    print(f"\nIdentified Issues: {len(reflected_context.identified_issues)}")
    for issue in reflected_context.identified_issues[:3]:
        print(f"- {issue}")
    
    print(f"\nImprovement Suggestions: {len(reflected_context.improvement_suggestions)}")
    for suggestion in reflected_context.improvement_suggestions[:3]:
        print(f"- {suggestion}")
    
    return result, reflected_context


# =================== Example 4: Multi-omics Integration ===================

async def example_multiomics_integration():
    """Multi-omics data integration example"""
    
    print("\n" + "=" * 50)
    print("Example 4: Multi-omics Integration Analysis")
    print("=" * 50)
    
    # Create multiple data types
    metadata_list = [
        DataMetadata(
            data_type=DataType.EXPRESSION_MATRIX,
            file_path="example_data/rnaseq_counts.csv",
            organism="Homo sapiens",
            tissue_type="liver"
        ),
        DataMetadata(
            data_type=DataType.PROTEOMICS_DATA,
            file_path="example_data/proteomics_intensities.csv",
            organism="Homo sapiens",
            tissue_type="liver"
        ),
        DataMetadata(
            data_type=DataType.VARIANT_DATA,
            file_path="example_data/variants.vcf",
            organism="Homo sapiens"
        )
    ]
    
    # Create integration task
    task = AnalysisTask(
        task_id="multiomics_integration_001",
        instruction="""
        Perform integrated multi-omics analysis:
        1. Correlate gene expression with protein abundance
        2. Identify variants affecting gene/protein expression
        3. Perform pathway-level integration
        4. Identify multi-omics biomarkers
        5. Create integrated visualization
        
        Focus on finding coherent biological patterns across data types.
        """,
        data_metadata=metadata_list,
        reasoning_type=ReasoningType.COMPARATIVE_ANALYSIS
    )
    
    # Initialize specialized components
    agent = BioinformaticsAgent()
    hypothesis_engine = HypothesisEngine()
    
    # Generate hypotheses
    print("\nGenerating biological hypotheses...")
    hypotheses = await hypothesis_engine.generate_hypotheses(task)
    
    print(f"Generated {len(hypotheses)} testable hypotheses:")
    for i, hypothesis in enumerate(hypotheses[:3], 1):
        print(f"\n{i}. {hypothesis.statement}")
        print(f"   Rationale: {hypothesis.biological_rationale}")
        print(f"   Required data: {[dt.value for dt in hypothesis.required_data_types]}")
    
    # Create pipeline for multi-omics
    pipeline = BioinformaticsPipeline(
        pipeline_id="multiomics_pipeline",
        execution_mode=ExecutionMode.PARALLEL
    )
    
    # Note: In a real implementation, we would add actual multi-omics tools
    print("\nMulti-omics integration pipeline created")
    print("Would perform:")
    print("- Cross-omics correlation analysis")
    print("- Multi-level pathway enrichment")
    print("- Integrated network analysis")
    print("- Machine learning for biomarker discovery")
    
    return hypotheses


# =================== Example 5: Adaptive Learning Demo ===================

async def example_adaptive_learning():
    """Demonstrate adaptive learning from feedback"""
    
    print("\n" + "=" * 50)
    print("Example 5: Adaptive Learning System")
    print("=" * 50)
    
    # Initialize learning system
    learning_system = AdaptiveLearningSystem()
    user_feedback_handler = UserFeedbackHandler()
    
    # Simulate analysis history with feedback
    print("\nSimulating analysis history...")
    
    # Successful analysis
    good_analysis = {
        "analysis_id": "learn_001",
        "quality_control": True,
        "multiple_testing_correction": "FDR",
        "pathway_analysis": True,
        "effect_sizes_reported": True
    }
    
    # Analysis with issues
    poor_analysis = {
        "analysis_id": "learn_002",
        "differential_expression": True,
        # Missing: multiple testing correction, pathway analysis
    }
    
    # Process user feedback
    feedback1 = await user_feedback_handler.process_user_feedback(
        "learn_001",
        "Great analysis! The pathway results are very insightful.",
        good_analysis
    )
    
    feedback2 = await user_feedback_handler.process_user_feedback(
        "learn_002",
        "Missing multiple testing correction and pathway analysis",
        poor_analysis
    )
    
    # Learn from history
    await learning_system.learn_from_history(
        [feedback1, feedback2],
        [good_analysis, poor_analysis]
    )
    
    # Get prevention recommendations
    planned_analysis = {
        "planned_analysis": "differential expression analysis"
    }
    
    recommendations = learning_system.get_prevention_recommendations(planned_analysis)
    
    print("\nLearned patterns:")
    print(f"Success patterns: {list(learning_system.success_patterns.keys())}")
    print(f"Failure patterns: {list(learning_system.failure_patterns.keys())}")
    
    print("\nPrevention recommendations for new analysis:")
    for rec in recommendations:
        print(f"- {rec}")
    
    # Show common issues
    common_issues = user_feedback_handler.get_common_issues()
    print("\nMost common user-reported issues:")
    for issue in common_issues[:3]:
        print(f"- {issue['issue']} (reported {issue['frequency']} times)")
    
    return learning_system


# =================== Main Example Runner ===================

async def run_all_examples():
    """Run all example demonstrations"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("BioinformaticsAgent System - Comprehensive Examples")
    print("=" * 70)
    print()
    print("This demonstration showcases the capabilities of the BioinformaticsAgent")
    print("system for various bioinformatics analyses with advanced features like")
    print("reflection, feedback integration, and adaptive learning.")
    print()
    
    # Run examples
    examples = [
        ("RNA-seq Analysis", example_rnaseq_analysis),
        ("Variant Analysis Pipeline", example_variant_analysis),
        ("Protein Analysis with Reflection", example_protein_analysis),
        ("Multi-omics Integration", example_multiomics_integration),
        ("Adaptive Learning", example_adaptive_learning)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\nRunning: {name}")
            result = await example_func()
            results[name] = {"status": "Success", "result": result}
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results[name] = {"status": "Failed", "error": str(e)}
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("Example Summary")
    print("=" * 70)
    
    for name, result in results.items():
        status = result["status"]
        print(f"{name}: {status}")
    
    print("\n" + "=" * 70)
    print("BioinformaticsAgent demonstration completed!")
    print("=" * 70)


# =================== Interactive Mode ===================

async def interactive_mode():
    """Run the agent in interactive mode"""
    
    print("=" * 70)
    print("BioinformaticsAgent - Interactive Mode")
    print("=" * 70)
    print()
    print("Welcome to the BioinformaticsAgent interactive mode!")
    print("You can describe your analysis needs, and the agent will help you.")
    print("Type 'exit' to quit, 'help' for commands.")
    print()
    
    # Initialize agent
    agent = BioinformaticsAgent()
    
    # Register all available tools
    for tool in get_all_bioinformatics_tools():
        agent.register_tool(tool)
    
    # Feedback systems
    feedback_engine = FeedbackIntegrationEngine()
    user_feedback_handler = UserFeedbackHandler()
    
    while True:
        try:
            # Get user input
            user_input = input("\nDescribe your analysis need: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- exit: Quit the program")
                print("- help: Show this help message")
                print("- list tools: Show available tools")
                print("- example: Run example analyses")
                print("\nOr describe your bioinformatics analysis need in natural language.")
                continue
            
            elif user_input.lower() == 'list tools':
                print("\nAvailable tools:")
                for tool_name in agent.tools.keys():
                    print(f"- {tool_name}")
                continue
            
            elif user_input.lower() == 'example':
                await run_all_examples()
                continue
            
            # Process analysis request
            print("\nProcessing your request...")
            
            # For demo, create a simple task
            # In real implementation, would parse user input more sophisticatedly
            task = AnalysisTask(
                task_id=f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                instruction=user_input,
                data_metadata=[],  # Would be populated based on user's data
                reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
            )
            
            # Note: In real implementation, would guide user through data upload
            print("\nNote: In a full implementation, you would be guided to:")
            print("1. Upload your data files")
            print("2. Specify data types and metadata")
            print("3. Set analysis parameters")
            print("\nFor now, showing what the agent would do:")
            
            # Show analysis plan
            print("\nAnalysis Plan:")
            print("-" * 40)
            print("1. Data validation and quality control")
            print("2. Appropriate statistical analysis")
            print("3. Biological interpretation")
            print("4. Visualization generation")
            print("5. Report creation")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again or type 'help' for assistance.")


# =================== Entry Point ===================

def main():
    """Main entry point for the example implementation"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(run_all_examples())


if __name__ == "__main__":
    main()