#!/usr/bin/env python3
"""
BioinformaticsAgent: A specialized AI agent for computational biology and bioinformatics analysis.

This system extends the Gemini CLI architecture with specialized components for biological data analysis,
including reflection loops, chain of thought reasoning, and extensible tool framework.

Key Features:
- Bioinformatics-specific tool ecosystem
- Data analysis pipeline orchestration
- Reflection and iterative improvement
- Chain of thought reasoning
- Extensible architecture for new tools
- Metadata-aware analysis planning
"""

import asyncio
import json
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from datetime import datetime
import uuid


# =================== Core Data Structures ===================

class DataType(Enum):
    """Supported biological data types"""
    GENOMIC_SEQUENCE = "genomic_sequence"
    PROTEIN_SEQUENCE = "protein_sequence"
    RNA_SEQUENCE = "rna_sequence"
    EXPRESSION_MATRIX = "expression_matrix"
    VARIANT_DATA = "variant_data"
    PHYLOGENETIC_TREE = "phylogenetic_tree"
    STRUCTURE_PDB = "structure_pdb"
    ANNOTATION_GFF = "annotation_gff"
    ALIGNMENT_SAM = "alignment_sam"
    FASTQ_READS = "fastq_reads"
    METHYLATION_DATA = "methylation_data"
    SINGLE_CELL_DATA = "single_cell_data"
    PROTEOMICS_DATA = "proteomics_data"
    METABOLOMICS_DATA = "metabolomics_data"


class ReasoningType(Enum):
    """Types of reasoning patterns"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REFLECTION = "reflection"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    ITERATIVE_REFINEMENT = "iterative_refinement"


@dataclass
class DataMetadata:
    """Comprehensive metadata for biological datasets"""
    data_type: DataType
    file_path: str
    organism: Optional[str] = None
    tissue_type: Optional[str] = None
    experimental_condition: Optional[str] = None
    sequencing_platform: Optional[str] = None
    genome_build: Optional[str] = None
    data_format: Optional[str] = None
    sample_size: Optional[int] = None
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_context_string(self) -> str:
        """Convert metadata to context string for prompts"""
        context = f"Data Type: {self.data_type.value}\n"
        context += f"File Path: {self.file_path}\n"
        
        if self.organism:
            context += f"Organism: {self.organism}\n"
        if self.tissue_type:
            context += f"Tissue: {self.tissue_type}\n"
        if self.experimental_condition:
            context += f"Condition: {self.experimental_condition}\n"
        if self.sequencing_platform:
            context += f"Platform: {self.sequencing_platform}\n"
        if self.genome_build:
            context += f"Genome Build: {self.genome_build}\n"
        if self.data_format:
            context += f"Format: {self.data_format}\n"
        if self.sample_size:
            context += f"Sample Size: {self.sample_size}\n"
            
        if self.quality_metrics:
            context += "Quality Metrics:\n"
            for metric, value in self.quality_metrics.items():
                context += f"  {metric}: {value}\n"
                
        return context


@dataclass
class AnalysisTask:
    """Represents an analysis task with context"""
    task_id: str
    instruction: str
    data_metadata: List[DataMetadata]
    expected_outputs: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    reasoning_type: ReasoningType = ReasoningType.CHAIN_OF_THOUGHT


@dataclass
class ReflectionContext:
    """Context for reflection and improvement"""
    original_analysis: str
    generated_code: str
    execution_results: Optional[str] = None
    user_feedback: Optional[str] = None
    identified_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    iteration_count: int = 0


@dataclass
class ChainOfThoughtStep:
    """Individual step in chain of thought reasoning"""
    step_id: str
    description: str
    reasoning: str
    action: Optional[str] = None
    expected_outcome: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


# =================== Tool Framework ===================

class BioToolResult:
    """Result from a bioinformatics tool execution"""
    def __init__(self, success: bool, output: Any = None, error: str = None, 
                 metadata: Dict[str, Any] = None, visualization_data: Any = None):
        self.success = success
        self.output = output
        self.error = error
        self.metadata = metadata or {}
        self.visualization_data = visualization_data
        self.timestamp = datetime.now()
        
    def to_llm_content(self) -> str:
        """Format result for LLM consumption"""
        if not self.success:
            return f"Tool execution failed: {self.error}"
        
        content = f"Tool executed successfully at {self.timestamp}\n"
        if self.output:
            content += f"Output: {str(self.output)[:1000]}...\n"  # Truncate long outputs
        if self.metadata:
            content += f"Metadata: {json.dumps(self.metadata, indent=2)}\n"
        return content


class BioinformaticsTool(ABC):
    """Abstract base class for all bioinformatics tools"""
    
    def __init__(self, name: str, description: str, supported_data_types: List[DataType]):
        self.name = name
        self.description = description
        self.supported_data_types = supported_data_types
        self.parameter_schema = self._define_parameter_schema()
        
    @abstractmethod
    def _define_parameter_schema(self) -> Dict[str, Any]:
        """Define JSON schema for tool parameters"""
        pass
    
    @abstractmethod
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute the tool with given parameters and data"""
        pass
    
    def validate_data_compatibility(self, data_metadata: List[DataMetadata]) -> bool:
        """Check if tool is compatible with provided data types"""
        for metadata in data_metadata:
            if metadata.data_type not in self.supported_data_types:
                return False
        return True
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, str]:
        """Validate parameters against schema"""
        # Simplified validation - in practice, use jsonschema library
        required_params = self.parameter_schema.get('required', [])
        for param in required_params:
            if param not in params:
                return False, f"Missing required parameter: {param}"
        return True, ""


# =================== Reflection and Reasoning System ===================

class ReflectionEngine:
    """Handles reflection loops and iterative improvement"""
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.reflection_history = []
    
    async def reflect_on_analysis(self, context: ReflectionContext) -> ReflectionContext:
        """Perform reflection on analysis and suggest improvements"""
        
        # Analyze the current state
        issues = await self._identify_issues(context)
        improvements = await self._generate_improvements(context, issues)
        
        context.identified_issues.extend(issues)
        context.improvement_suggestions.extend(improvements)
        context.iteration_count += 1
        
        self.reflection_history.append(context)
        return context
    
    async def _identify_issues(self, context: ReflectionContext) -> List[str]:
        """Identify potential issues in the analysis"""
        issues = []
        
        # Code quality checks
        if not context.generated_code.strip():
            issues.append("Generated code is empty")
        
        # Result validation
        if context.execution_results and "error" in context.execution_results.lower():
            issues.append("Execution resulted in errors")
        
        # User feedback analysis
        if context.user_feedback:
            if "incorrect" in context.user_feedback.lower():
                issues.append("User indicated incorrect results")
            if "missing" in context.user_feedback.lower():
                issues.append("User indicated missing analysis components")
        
        return issues
    
    async def _generate_improvements(self, context: ReflectionContext, 
                                   issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on identified issues"""
        improvements = []
        
        for issue in issues:
            if "empty" in issue:
                improvements.append("Generate comprehensive analysis code")
            elif "error" in issue:
                improvements.append("Debug and fix execution errors")
            elif "incorrect" in issue:
                improvements.append("Revise analysis methodology")
            elif "missing" in issue:
                improvements.append("Add missing analysis components")
        
        return improvements


class ChainOfThoughtReasoner:
    """Implements chain of thought reasoning for complex analyses"""
    
    def __init__(self):
        self.reasoning_chains = {}
    
    async def create_reasoning_chain(self, task: AnalysisTask) -> List[ChainOfThoughtStep]:
        """Create a chain of thought for a given analysis task"""
        
        steps = []
        
        # Step 1: Data Understanding
        step1 = ChainOfThoughtStep(
            step_id="data_understanding",
            description="Understand the provided data",
            reasoning=f"First, I need to understand the data types: {[m.data_type.value for m in task.data_metadata]}",
            action="examine_data_metadata",
            expected_outcome="Clear understanding of data structure and characteristics"
        )
        steps.append(step1)
        
        # Step 2: Analysis Planning
        step2 = ChainOfThoughtStep(
            step_id="analysis_planning",
            description="Plan the analysis approach",
            reasoning="Based on the data and instruction, determine the best analysis strategy",
            action="create_analysis_plan",
            expected_outcome="Detailed analysis plan with tool selection",
            dependencies=["data_understanding"]
        )
        steps.append(step2)
        
        # Step 3: Tool Selection
        step3 = ChainOfThoughtStep(
            step_id="tool_selection",
            description="Select appropriate bioinformatics tools",
            reasoning="Choose tools that are compatible with the data types and analysis goals",
            action="select_tools",
            expected_outcome="List of tools to be used in the analysis",
            dependencies=["analysis_planning"]
        )
        steps.append(step3)
        
        # Step 4: Code Generation
        step4 = ChainOfThoughtStep(
            step_id="code_generation",
            description="Generate analysis code",
            reasoning="Create executable code that implements the planned analysis",
            action="generate_code",
            expected_outcome="Complete, executable analysis code",
            dependencies=["tool_selection"]
        )
        steps.append(step4)
        
        # Step 5: Result Interpretation
        step5 = ChainOfThoughtStep(
            step_id="result_interpretation",
            description="Interpret and summarize results",
            reasoning="Analyze the output and provide biological insights",
            action="interpret_results",
            expected_outcome="Clear interpretation of analysis results",
            dependencies=["code_generation"]
        )
        steps.append(step5)
        
        self.reasoning_chains[task.task_id] = steps
        return steps


# =================== Pipeline Orchestration ===================

class AnalysisPipeline:
    """Orchestrates complex multi-step analyses"""
    
    def __init__(self):
        self.steps = []
        self.dependencies = {}
        self.results = {}
    
    def add_step(self, step_id: str, tool: BioinformaticsTool, params: Dict[str, Any], 
                 dependencies: List[str] = None):
        """Add a step to the pipeline"""
        self.steps.append({
            'id': step_id,
            'tool': tool,
            'params': params,
            'dependencies': dependencies or []
        })
        self.dependencies[step_id] = dependencies or []
    
    async def execute_pipeline(self, data_metadata: List[DataMetadata]) -> Dict[str, BioToolResult]:
        """Execute the entire pipeline"""
        results = {}
        executed = set()
        
        # Topological sort for dependency resolution
        while len(executed) < len(self.steps):
            for step in self.steps:
                step_id = step['id']
                if step_id in executed:
                    continue
                
                # Check if all dependencies are satisfied
                if all(dep in executed for dep in step['dependencies']):
                    # Execute step
                    tool_result = await step['tool'].execute(step['params'], data_metadata)
                    results[step_id] = tool_result
                    executed.add(step_id)
        
        return results


# =================== Core Agent Architecture ===================

class BioinformaticsAgent:
    """Main agent class for bioinformatics analysis"""
    
    def __init__(self, model_api_key: str = None):
        self.model_api_key = model_api_key
        self.tools = {}
        self.reflection_engine = ReflectionEngine()
        self.chain_of_thought = ChainOfThoughtReasoner()
        self.conversation_history = []
        self.analysis_history = []
        
        # Initialize with basic tools
        self._initialize_basic_tools()
    
    def _initialize_basic_tools(self):
        """Initialize the agent with basic bioinformatics tools"""
        # This would be expanded with actual tool implementations
        pass
    
    def register_tool(self, tool: BioinformaticsTool):
        """Register a new bioinformatics tool"""
        self.tools[tool.name] = tool
        logging.info(f"Registered tool: {tool.name}")
    
    async def analyze_data(self, task: AnalysisTask) -> Dict[str, Any]:
        """Main method to analyze biological data"""
        
        # Create reasoning chain
        reasoning_steps = await self.chain_of_thought.create_reasoning_chain(task)
        
        # Execute analysis with reflection
        result = await self._execute_with_reflection(task, reasoning_steps)
        
        # Store in history
        self.analysis_history.append({
            'task': task,
            'result': result,
            'timestamp': datetime.now()
        })
        
        return result
    
    async def _execute_with_reflection(self, task: AnalysisTask, 
                                     reasoning_steps: List[ChainOfThoughtStep]) -> Dict[str, Any]:
        """Execute analysis with reflection loops"""
        
        reflection_context = ReflectionContext(
            original_analysis=task.instruction,
            generated_code=""
        )
        
        for iteration in range(self.reflection_engine.max_iterations):
            try:
                # Generate analysis code
                code = await self._generate_analysis_code(task, reasoning_steps, reflection_context)
                reflection_context.generated_code = code
                
                # Execute code (simulated)
                execution_result = await self._execute_analysis_code(code, task.data_metadata)
                reflection_context.execution_results = str(execution_result)
                
                # Check if satisfactory
                if execution_result.success and not reflection_context.user_feedback:
                    return {
                        'success': True,
                        'code': code,
                        'results': execution_result,
                        'reasoning_steps': reasoning_steps,
                        'iterations': iteration + 1
                    }
                
                # Reflect and improve
                reflection_context = await self.reflection_engine.reflect_on_analysis(reflection_context)
                
            except Exception as e:
                reflection_context.identified_issues.append(f"Exception: {str(e)}")
        
        return {
            'success': False,
            'error': 'Max iterations reached without satisfactory result',
            'reflection_context': reflection_context
        }
    
    async def _generate_analysis_code(self, task: AnalysisTask, 
                                    reasoning_steps: List[ChainOfThoughtStep],
                                    reflection_context: ReflectionContext) -> str:
        """Generate analysis code based on task and reasoning"""
        
        # Build context for code generation
        context = self._build_code_generation_context(task, reasoning_steps, reflection_context)
        
        # This would call the LLM to generate code
        # For now, return a template
        return self._generate_template_code(task)
    
    def _build_code_generation_context(self, task: AnalysisTask,
                                     reasoning_steps: List[ChainOfThoughtStep],
                                     reflection_context: ReflectionContext) -> str:
        """Build comprehensive context for code generation"""
        
        context = f"# Bioinformatics Analysis Task\n\n"
        context += f"## Task Description\n{task.instruction}\n\n"
        
        context += "## Data Metadata\n"
        for i, metadata in enumerate(task.data_metadata):
            context += f"Dataset {i+1}:\n{metadata.to_context_string()}\n"
        
        context += "\n## Available Tools\n"
        for tool_name, tool in self.tools.items():
            context += f"- {tool_name}: {tool.description}\n"
        
        context += "\n## Reasoning Steps\n"
        for step in reasoning_steps:
            context += f"{step.step_id}: {step.description}\n"
            context += f"  Reasoning: {step.reasoning}\n"
        
        if reflection_context.identified_issues:
            context += "\n## Issues to Address\n"
            for issue in reflection_context.identified_issues:
                context += f"- {issue}\n"
        
        if reflection_context.improvement_suggestions:
            context += "\n## Improvement Suggestions\n"
            for suggestion in reflection_context.improvement_suggestions:
                context += f"- {suggestion}\n"
        
        return context
    
    def _generate_template_code(self, task: AnalysisTask) -> str:
        """Generate template code based on data types"""
        
        data_types = [m.data_type for m in task.data_metadata]
        
        if DataType.GENOMIC_SEQUENCE in data_types:
            return self._generate_genomics_template()
        elif DataType.EXPRESSION_MATRIX in data_types:
            return self._generate_expression_template()
        elif DataType.PROTEIN_SEQUENCE in data_types:
            return self._generate_protein_template()
        else:
            return self._generate_generic_template()
    
    def _generate_genomics_template(self) -> str:
        return """
import pandas as pd
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt

# Load genomic sequence data
def load_genomic_data(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append({
            'id': record.id,
            'sequence': str(record.seq),
            'length': len(record.seq)
        })
    return pd.DataFrame(sequences)

# Analyze sequence composition
def analyze_composition(sequences_df):
    results = {}
    for idx, row in sequences_df.iterrows():
        seq = row['sequence']
        gc_content = (seq.count('G') + seq.count('C')) / len(seq) * 100
        results[row['id']] = {
            'length': row['length'],
            'gc_content': gc_content
        }
    return results

# Main analysis
def main():
    # Load data
    data = load_genomic_data('input_file.fasta')
    
    # Perform analysis
    composition_results = analyze_composition(data)
    
    # Generate summary
    print("Genomic Sequence Analysis Results:")
    for seq_id, metrics in composition_results.items():
        print(f"{seq_id}: Length={metrics['length']}, GC%={metrics['gc_content']:.2f}")
    
    return composition_results

if __name__ == "__main__":
    results = main()
"""
    
    def _generate_expression_template(self) -> str:
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load expression matrix
def load_expression_data(file_path):
    return pd.read_csv(file_path, index_col=0)

# Perform differential expression analysis
def differential_expression_analysis(expr_data, conditions):
    # Simplified differential expression
    results = []
    for gene in expr_data.index:
        gene_expr = expr_data.loc[gene]
        fold_change = np.log2(gene_expr.mean() + 1)
        results.append({
            'gene': gene,
            'fold_change': fold_change,
            'mean_expression': gene_expr.mean()
        })
    return pd.DataFrame(results)

# PCA analysis
def perform_pca(expr_data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(expr_data.T)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    return pca_result, pca.explained_variance_ratio_

# Main analysis
def main():
    # Load expression data
    expr_data = load_expression_data('expression_matrix.csv')
    
    # Perform differential expression
    de_results = differential_expression_analysis(expr_data, None)
    
    # PCA analysis
    pca_coords, var_explained = perform_pca(expr_data)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_coords[:, 0], pca_coords[:, 1])
    plt.xlabel(f'PC1 ({var_explained[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({var_explained[1]:.2%} variance)')
    plt.title('PCA of Expression Data')
    
    plt.subplot(1, 2, 2)
    plt.hist(de_results['fold_change'], bins=30)
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Gene Expression Changes')
    
    plt.tight_layout()
    plt.savefig('expression_analysis.png')
    
    return {
        'de_results': de_results,
        'pca_coords': pca_coords,
        'variance_explained': var_explained
    }

if __name__ == "__main__":
    results = main()
"""
    
    def _generate_protein_template(self) -> str:
        return """
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils import ProtParam
import matplotlib.pyplot as plt

# Load protein sequences
def load_protein_data(file_path):
    proteins = []
    for record in SeqIO.parse(file_path, "fasta"):
        proteins.append({
            'id': record.id,
            'sequence': str(record.seq),
            'length': len(record.seq)
        })
    return pd.DataFrame(proteins)

# Analyze protein properties
def analyze_protein_properties(proteins_df):
    results = []
    for idx, row in proteins_df.iterrows():
        seq = row['sequence']
        analyzer = ProtParam.ProteinAnalysis(seq)
        
        results.append({
            'protein_id': row['id'],
            'length': row['length'],
            'molecular_weight': analyzer.molecular_weight(),
            'isoelectric_point': analyzer.isoelectric_point(),
            'instability_index': analyzer.instability_index(),
            'hydrophobicity': analyzer.gravy()
        })
    
    return pd.DataFrame(results)

# Main analysis
def main():
    # Load protein data
    protein_data = load_protein_data('proteins.fasta')
    
    # Analyze properties
    properties = analyze_protein_properties(protein_data)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(properties['molecular_weight'], bins=20)
    axes[0, 0].set_title('Molecular Weight Distribution')
    
    axes[0, 1].hist(properties['isoelectric_point'], bins=20)
    axes[0, 1].set_title('Isoelectric Point Distribution')
    
    axes[1, 0].scatter(properties['length'], properties['molecular_weight'])
    axes[1, 0].set_xlabel('Protein Length')
    axes[1, 0].set_ylabel('Molecular Weight')
    axes[1, 0].set_title('Length vs Molecular Weight')
    
    axes[1, 1].hist(properties['hydrophobicity'], bins=20)
    axes[1, 1].set_title('Hydrophobicity Distribution')
    
    plt.tight_layout()
    plt.savefig('protein_analysis.png')
    
    return properties

if __name__ == "__main__":
    results = main()
"""
    
    def _generate_generic_template(self) -> str:
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generic data loading function
def load_data(file_path, file_format='csv'):
    if file_format == 'csv':
        return pd.read_csv(file_path)
    elif file_format == 'tsv':
        return pd.read_csv(file_path, sep='\\t')
    else:
        raise ValueError(f"Unsupported format: {file_format}")

# Basic data analysis
def basic_analysis(data):
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'basic_stats': data.describe().to_dict()
    }
    return summary

# Main analysis
def main():
    # Load data
    data = load_data('input_data.csv')
    
    # Perform basic analysis
    analysis_results = basic_analysis(data)
    
    # Print summary
    print("Data Analysis Summary:")
    print(f"Shape: {analysis_results['shape']}")
    print(f"Columns: {analysis_results['columns']}")
    
    return analysis_results

if __name__ == "__main__":
    results = main()
"""
    
    async def _execute_analysis_code(self, code: str, 
                                   data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute the generated analysis code (simulated)"""
        
        # In a real implementation, this would:
        # 1. Create a safe execution environment
        # 2. Execute the code with proper data paths
        # 3. Capture outputs and results
        # 4. Handle errors gracefully
        
        # For now, simulate successful execution
        return BioToolResult(
            success=True,
            output="Analysis completed successfully",
            metadata={
                'execution_time': '2.5 seconds',
                'files_generated': ['results.csv', 'analysis_plot.png']
            }
        )
    
    async def provide_feedback(self, task_id: str, feedback: str) -> Dict[str, Any]:
        """Provide feedback on a previous analysis for improvement"""
        
        # Find the analysis in history
        target_analysis = None
        for analysis in self.analysis_history:
            if analysis['task'].task_id == task_id:
                target_analysis = analysis
                break
        
        if not target_analysis:
            return {'error': 'Analysis not found'}
        
        # Create reflection context with feedback
        reflection_context = ReflectionContext(
            original_analysis=target_analysis['task'].instruction,
            generated_code=target_analysis['result'].get('code', ''),
            execution_results=str(target_analysis['result'].get('results', '')),
            user_feedback=feedback
        )
        
        # Re-execute with reflection
        improved_result = await self._execute_with_reflection(
            target_analysis['task'], 
            target_analysis['result'].get('reasoning_steps', [])
        )
        
        return {
            'original_result': target_analysis['result'],
            'improved_result': improved_result,
            'feedback_incorporated': feedback
        }


# =================== Example Tool Implementations ===================

class FastQCTool(BioinformaticsTool):
    """Tool for quality control of sequencing data"""
    
    def __init__(self):
        super().__init__(
            name="fastqc",
            description="Quality control analysis for high throughput sequence data",
            supported_data_types=[DataType.FASTQ_READS]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_file": {"type": "string", "description": "Path to FASTQ file"},
                "output_dir": {"type": "string", "description": "Output directory"},
                "threads": {"type": "integer", "default": 1}
            },
            "required": ["input_file"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        # Simulate FastQC execution
        await asyncio.sleep(1)  # Simulate processing time
        
        return BioToolResult(
            success=True,
            output="FastQC analysis completed",
            metadata={
                "quality_score": 28.5,
                "total_sequences": 1000000,
                "poor_quality": 2.1
            },
            visualization_data="fastqc_report.html"
        )


class BlastTool(BioinformaticsTool):
    """Tool for sequence similarity search"""
    
    def __init__(self):
        super().__init__(
            name="blast",
            description="Basic Local Alignment Search Tool for sequence similarity",
            supported_data_types=[DataType.GENOMIC_SEQUENCE, DataType.PROTEIN_SEQUENCE]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query_file": {"type": "string", "description": "Query sequence file"},
                "database": {"type": "string", "description": "BLAST database"},
                "evalue": {"type": "number", "default": 0.001},
                "max_hits": {"type": "integer", "default": 10}
            },
            "required": ["query_file", "database"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        # Simulate BLAST execution
        await asyncio.sleep(2)  # Simulate processing time
        
        hits = [
            {"subject": "seq1", "evalue": 1e-50, "identity": 98.5},
            {"subject": "seq2", "evalue": 1e-30, "identity": 85.2}
        ]
        
        return BioToolResult(
            success=True,
            output=hits,
            metadata={
                "total_hits": len(hits),
                "database_searched": params["database"]
            }
        )


# =================== System Prompts ===================

BIOINFORMATICS_SYSTEM_PROMPT = """
You are BioAgent, an expert computational biologist and bioinformatics specialist with deep knowledge across:

## Core Expertise
- Genomics, transcriptomics, proteomics, and metabolomics
- Sequence analysis and alignment algorithms
- Statistical analysis and machine learning for biological data
- Data visualization and interpretation
- Experimental design and quality control

## Your Capabilities
- Analyze complex biological datasets with appropriate statistical methods
- Generate publication-quality code in Python, R, and shell scripts
- Create comprehensive visualizations and reports
- Provide biological context and interpretation for results
- Suggest follow-up experiments and analyses

## Analysis Approach
1. **Data Understanding**: Thoroughly examine data types, formats, and metadata
2. **Method Selection**: Choose appropriate tools and algorithms based on data characteristics
3. **Quality Control**: Implement proper validation and quality checks
4. **Statistical Rigor**: Apply correct statistical methods with appropriate corrections
5. **Biological Context**: Interpret results within relevant biological frameworks
6. **Reproducibility**: Generate well-documented, reproducible analysis pipelines

## Code Generation Guidelines
- Write clean, well-commented, and modular code
- Include error handling and input validation
- Generate appropriate visualizations
- Provide clear output summaries
- Follow best practices for each tool/package

## Available Tools
{tool_descriptions}

## Current Analysis Context
Data Types: {data_types}
Organism: {organism}
Experimental Context: {experimental_context}

## Reasoning Mode
You will use {reasoning_type} to approach this analysis:
- Break down complex problems into logical steps
- Explain your reasoning at each step  
- Consider alternative approaches
- Validate assumptions and results
- Reflect on outcomes and suggest improvements

Remember: Always prioritize scientific accuracy, statistical validity, and biological relevance in your analyses.
"""


# =================== Example Usage ===================

async def main_example():
    """Example usage of the BioinformaticsAgent"""
    
    # Initialize the agent
    agent = BioinformaticsAgent()
    
    # Register tools
    agent.register_tool(FastQCTool())
    agent.register_tool(BlastTool())
    
    # Define data metadata
    rna_seq_metadata = DataMetadata(
        data_type=DataType.FASTQ_READS,
        file_path="/data/rna_seq_samples.fastq",
        organism="Homo sapiens",
        tissue_type="liver",
        experimental_condition="drug_treatment_vs_control",
        sequencing_platform="Illumina HiSeq",
        sample_size=24,
        quality_metrics={"average_quality": 30, "total_reads": 50000000}
    )
    
    # Create analysis task
    task = AnalysisTask(
        task_id=str(uuid.uuid4()),
        instruction="""
        Analyze the RNA-seq data to identify differentially expressed genes 
        between drug treatment and control conditions. Perform quality control,
        read alignment, quantification, and statistical analysis. Generate
        visualizations including PCA plots, volcano plots, and heatmaps.
        Provide biological interpretation of the top differentially expressed genes.
        """,
        data_metadata=[rna_seq_metadata],
        expected_outputs=["differential_expression_results.csv", "quality_report.html", "pca_plot.png"],
        reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
    )
    
    # Perform analysis
    print("Starting bioinformatics analysis...")
    result = await agent.analyze_data(task)
    
    print(f"Analysis completed: {result['success']}")
    if result['success']:
        print(f"Generated code length: {len(result['code'])} characters")
        print(f"Reasoning steps: {len(result['reasoning_steps'])}")
        print(f"Iterations required: {result['iterations']}")
    
    # Simulate user feedback and improvement
    feedback = "The analysis looks good, but please add pathway enrichment analysis for the top genes"
    improved_result = await agent.provide_feedback(task.task_id, feedback)
    
    print("Feedback incorporated and analysis improved!")
    
    return result


if __name__ == "__main__":
    # Run example
    result = asyncio.run(main_example())
    print("Example completed successfully!")