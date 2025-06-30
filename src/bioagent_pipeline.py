#!/usr/bin/env python3
"""
BioinformaticsAgent Pipeline Orchestration: Advanced pipeline management and workflow orchestration
for complex bioinformatics analyses.

This module provides:
- Dynamic pipeline construction
- Dependency management
- Parallel execution
- Pipeline optimization
- Error handling and recovery
- Resource management
- Workflow templates
"""

import asyncio
import json
import logging
import networkx as nx
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, AsyncGenerator
from datetime import datetime, timedelta
import uuid
import concurrent.futures
from pathlib import Path

# Import base classes
from bioagent_architecture import (
    BioinformaticsTool, BioToolResult, DataType, DataMetadata, AnalysisTask
)


# =================== Pipeline Components ===================

class PipelineStepStatus(Enum):
    """Status of individual pipeline steps"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ExecutionMode(Enum):
    """Pipeline execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    DISTRIBUTED = "distributed"


@dataclass
class PipelineStep:
    """Individual step in a bioinformatics pipeline"""
    step_id: str
    tool: BioinformaticsTool
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    status: PipelineStepStatus = PipelineStepStatus.PENDING
    estimated_runtime: Optional[timedelta] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    cache_results: bool = True
    
    def __post_init__(self):
        if not self.resource_requirements:
            self.resource_requirements = {
                "cpu_cores": 1,
                "memory_gb": 4,
                "disk_gb": 10,
                "gpu_required": False
            }


@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    pipeline_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    step_results: Dict[str, BioToolResult] = field(default_factory=dict)
    execution_graph: Optional[nx.DiGraph] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    
    @property
    def total_runtime(self) -> Optional[timedelta]:
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        if not self.step_results:
            return 0.0
        successful = sum(1 for result in self.step_results.values() if result.success)
        return successful / len(self.step_results)


# =================== Pipeline Templates ===================

class PipelineTemplate:
    """Template for common bioinformatics workflows"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps = []
        self.template_parameters = {}
    
    def add_step(self, step: PipelineStep):
        """Add a step to the template"""
        self.steps.append(step)
    
    def instantiate(self, parameters: Dict[str, Any]) -> 'BioinformaticsPipeline':
        """Create a pipeline instance from the template"""
        pipeline = BioinformaticsPipeline(f"{self.name}_{uuid.uuid4().hex[:8]}")
        
        # Substitute template parameters
        for step in self.steps:
            instantiated_step = self._instantiate_step(step, parameters)
            pipeline.add_step(instantiated_step)
        
        return pipeline
    
    def _instantiate_step(self, step: PipelineStep, parameters: Dict[str, Any]) -> PipelineStep:
        """Instantiate a step with template parameters"""
        # Create a new step with substituted parameters
        instantiated_params = {}
        for key, value in step.parameters.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                param_name = value[2:-1]
                instantiated_params[key] = parameters.get(param_name, value)
            else:
                instantiated_params[key] = value
        
        return PipelineStep(
            step_id=step.step_id,
            tool=step.tool,
            parameters=instantiated_params,
            dependencies=step.dependencies.copy(),
            optional_dependencies=step.optional_dependencies.copy(),
            outputs=step.outputs.copy(),
            estimated_runtime=step.estimated_runtime,
            resource_requirements=step.resource_requirements.copy()
        )


# =================== Pipeline Orchestrator ===================

class BioinformaticsPipeline:
    """Main pipeline orchestration class"""
    
    def __init__(self, pipeline_id: str, execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE):
        self.pipeline_id = pipeline_id
        self.execution_mode = execution_mode
        self.steps: Dict[str, PipelineStep] = {}
        self.execution_graph = nx.DiGraph()
        self.result_cache: Dict[str, BioToolResult] = {}
        self.resource_manager = ResourceManager()
        self.step_executor = StepExecutor()
        
        # Pipeline metadata
        self.created_at = datetime.now()
        self.description = ""
        self.tags = []
        
        # Execution state
        self.is_running = False
        self.current_result: Optional[PipelineResult] = None
    
    def add_step(self, step: PipelineStep):
        """Add a step to the pipeline"""
        self.steps[step.step_id] = step
        self.execution_graph.add_node(step.step_id, step=step)
        
        # Add dependency edges
        for dep in step.dependencies:
            if dep in self.steps:
                self.execution_graph.add_edge(dep, step.step_id)
        
        # Optional dependencies (won't block execution)
        for opt_dep in step.optional_dependencies:
            if opt_dep in self.steps:
                self.execution_graph.add_edge(
                    opt_dep, step.step_id, 
                    edge_type="optional"
                )
    
    def remove_step(self, step_id: str):
        """Remove a step from the pipeline"""
        if step_id in self.steps:
            del self.steps[step_id]
            self.execution_graph.remove_node(step_id)
    
    def validate_pipeline(self) -> Tuple[bool, List[str]]:
        """Validate the pipeline structure"""
        errors = []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.execution_graph):
            cycles = list(nx.simple_cycles(self.execution_graph))
            errors.append(f"Pipeline contains cycles: {cycles}")
        
        # Check for missing dependencies
        for step_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in self.steps:
                    errors.append(f"Step {step_id} depends on missing step {dep}")
        
        # Check tool compatibility
        for step_id, step in self.steps.items():
            # This would check if the tool is compatible with input data
            # For now, just check if tool exists
            if not step.tool:
                errors.append(f"Step {step_id} has no tool assigned")
        
        return len(errors) == 0, errors
    
    async def execute(self, data_metadata: List[DataMetadata]) -> PipelineResult:
        """Execute the entire pipeline"""
        
        # Validate pipeline
        is_valid, errors = self.validate_pipeline()
        if not is_valid:
            raise ValueError(f"Pipeline validation failed: {errors}")
        
        # Initialize result
        self.current_result = PipelineResult(
            pipeline_id=self.pipeline_id,
            status="running",
            start_time=datetime.now(),
            execution_graph=self.execution_graph.copy()
        )
        
        self.is_running = True
        
        try:
            # Determine execution order
            execution_order = self._plan_execution()
            
            # Execute based on mode
            if self.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(execution_order, data_metadata)
            elif self.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(execution_order, data_metadata)
            elif self.execution_mode == ExecutionMode.ADAPTIVE:
                await self._execute_adaptive(execution_order, data_metadata)
            
            self.current_result.status = "completed"
            
        except Exception as e:
            self.current_result.status = "failed"
            self.current_result.error_log.append(str(e))
            logging.error(f"Pipeline {self.pipeline_id} failed: {e}")
            raise
        
        finally:
            self.is_running = False
            self.current_result.end_time = datetime.now()
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
        
        return self.current_result
    
    def _plan_execution(self) -> List[List[str]]:
        """Plan the execution order of steps"""
        # Topological sort to determine execution levels
        execution_levels = []
        
        # Create a copy of the graph for processing
        graph_copy = self.execution_graph.copy()
        
        while graph_copy.nodes():
            # Find nodes with no incoming edges (ready to execute)
            ready_nodes = [
                node for node in graph_copy.nodes() 
                if graph_copy.in_degree(node) == 0
            ]
            
            if not ready_nodes:
                # This shouldn't happen with a valid DAG
                raise ValueError("Pipeline contains unresolvable dependencies")
            
            execution_levels.append(ready_nodes)
            
            # Remove the ready nodes and their outgoing edges
            graph_copy.remove_nodes_from(ready_nodes)
        
        return execution_levels
    
    async def _execute_sequential(self, execution_order: List[List[str]], 
                                data_metadata: List[DataMetadata]):
        """Execute pipeline sequentially"""
        for level in execution_order:
            for step_id in level:
                await self._execute_step(step_id, data_metadata)
    
    async def _execute_parallel(self, execution_order: List[List[str]], 
                              data_metadata: List[DataMetadata]):
        """Execute pipeline with parallel steps within each level"""
        for level in execution_order:
            # Execute all steps in this level in parallel
            tasks = [
                self._execute_step(step_id, data_metadata) 
                for step_id in level
            ]
            await asyncio.gather(*tasks)
    
    async def _execute_adaptive(self, execution_order: List[List[str]], 
                              data_metadata: List[DataMetadata]):
        """Execute pipeline with adaptive resource management"""
        
        for level in execution_order:
            # Check available resources
            available_resources = await self.resource_manager.get_available_resources()
            
            # Determine optimal parallelization
            parallel_groups = self._optimize_parallel_execution(level, available_resources)
            
            for group in parallel_groups:
                tasks = [
                    self._execute_step(step_id, data_metadata) 
                    for step_id in group
                ]
                await asyncio.gather(*tasks)
    
    def _optimize_parallel_execution(self, step_ids: List[str], 
                                   available_resources: Dict[str, Any]) -> List[List[str]]:
        """Optimize parallel execution based on resource constraints"""
        
        # Simple bin-packing algorithm for resource allocation
        groups = []
        current_group = []
        current_resources = available_resources.copy()
        
        # Sort steps by resource requirements (largest first)
        sorted_steps = sorted(
            step_ids,
            key=lambda x: self.steps[x].resource_requirements.get('memory_gb', 1),
            reverse=True
        )
        
        for step_id in sorted_steps:
            step = self.steps[step_id]
            required_memory = step.resource_requirements.get('memory_gb', 1)
            required_cores = step.resource_requirements.get('cpu_cores', 1)
            
            # Check if step can fit in current group
            if (current_resources.get('memory_gb', 0) >= required_memory and
                current_resources.get('cpu_cores', 0) >= required_cores):
                
                current_group.append(step_id)
                current_resources['memory_gb'] -= required_memory
                current_resources['cpu_cores'] -= required_cores
            else:
                # Start new group
                if current_group:
                    groups.append(current_group)
                current_group = [step_id]
                current_resources = available_resources.copy()
                current_resources['memory_gb'] -= required_memory
                current_resources['cpu_cores'] -= required_cores
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    async def _execute_step(self, step_id: str, data_metadata: List[DataMetadata]):
        """Execute a single pipeline step"""
        step = self.steps[step_id]
        
        # Check if result is cached
        if step.cache_results and step_id in self.result_cache:
            self.current_result.step_results[step_id] = self.result_cache[step_id]
            return
        
        # Update step status
        step.status = PipelineStepStatus.RUNNING
        
        # Check dependencies
        dependency_results = {}
        for dep_id in step.dependencies:
            if dep_id in self.current_result.step_results:
                dependency_results[dep_id] = self.current_result.step_results[dep_id]
        
        try:
            # Execute the step
            result = await self.step_executor.execute_step(
                step, data_metadata, dependency_results
            )
            
            # Store result
            self.current_result.step_results[step_id] = result
            
            if step.cache_results:
                self.result_cache[step_id] = result
            
            step.status = PipelineStepStatus.COMPLETED
            
        except Exception as e:
            step.status = PipelineStepStatus.FAILED
            step.retry_count += 1
            
            # Create error result
            error_result = BioToolResult(
                success=False,
                error=f"Step execution failed: {str(e)}"
            )
            self.current_result.step_results[step_id] = error_result
            
            # Retry if possible
            if step.retry_count <= step.max_retries:
                logging.warning(f"Retrying step {step_id} (attempt {step.retry_count})")
                await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
                await self._execute_step(step_id, data_metadata)
            else:
                logging.error(f"Step {step_id} failed after {step.max_retries} retries")
                raise
    
    def _calculate_performance_metrics(self):
        """Calculate pipeline performance metrics"""
        if not self.current_result:
            return
        
        metrics = {}
        
        # Total runtime
        if self.current_result.total_runtime:
            metrics['total_runtime_seconds'] = self.current_result.total_runtime.total_seconds()
        
        # Success rate
        metrics['success_rate'] = self.current_result.success_rate
        
        # Step-level metrics
        step_runtimes = []
        for step_id, result in self.current_result.step_results.items():
            if hasattr(result, 'metadata') and 'execution_time' in result.metadata:
                step_runtimes.append(result.metadata['execution_time'])
        
        if step_runtimes:
            metrics['avg_step_runtime'] = np.mean(step_runtimes)
            metrics['max_step_runtime'] = np.max(step_runtimes)
        
        # Resource utilization
        total_cpu_hours = sum(
            self.steps[step_id].resource_requirements.get('cpu_cores', 1) * 
            self.steps[step_id].estimated_runtime.total_seconds() / 3600
            for step_id in self.steps
            if self.steps[step_id].estimated_runtime
        )
        metrics['estimated_cpu_hours'] = total_cpu_hours
        
        self.current_result.performance_metrics = metrics


# =================== Resource Management ===================

class ResourceManager:
    """Manages computational resources for pipeline execution"""
    
    def __init__(self):
        self.total_resources = {
            'cpu_cores': 8,
            'memory_gb': 32,
            'disk_gb': 1000,
            'gpu_count': 0
        }
        self.allocated_resources = {
            'cpu_cores': 0,
            'memory_gb': 0,
            'disk_gb': 0,
            'gpu_count': 0
        }
    
    async def get_available_resources(self) -> Dict[str, Any]:
        """Get currently available resources"""
        available = {}
        for resource, total in self.total_resources.items():
            allocated = self.allocated_resources.get(resource, 0)
            available[resource] = max(0, total - allocated)
        return available
    
    async def allocate_resources(self, requirements: Dict[str, Any]) -> bool:
        """Allocate resources for a step"""
        available = await self.get_available_resources()
        
        # Check if resources are available
        for resource, required in requirements.items():
            if available.get(resource, 0) < required:
                return False
        
        # Allocate resources
        for resource, required in requirements.items():
            self.allocated_resources[resource] = \
                self.allocated_resources.get(resource, 0) + required
        
        return True
    
    async def release_resources(self, requirements: Dict[str, Any]):
        """Release allocated resources"""
        for resource, amount in requirements.items():
            current = self.allocated_resources.get(resource, 0)
            self.allocated_resources[resource] = max(0, current - amount)


# =================== Step Executor ===================

class StepExecutor:
    """Executes individual pipeline steps"""
    
    async def execute_step(self, step: PipelineStep, 
                          data_metadata: List[DataMetadata],
                          dependency_results: Dict[str, BioToolResult]) -> BioToolResult:
        """Execute a single step with proper resource management and monitoring"""
        
        start_time = time.time()
        
        try:
            # Prepare parameters with dependency outputs
            enhanced_params = await self._enhance_parameters(
                step.parameters, dependency_results
            )
            
            # Execute the tool
            result = await step.tool.execute(enhanced_params, data_metadata)
            
            # Add execution metadata
            execution_time = time.time() - start_time
            if not result.metadata:
                result.metadata = {}
            result.metadata['execution_time'] = execution_time
            result.metadata['step_id'] = step.step_id
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return BioToolResult(
                success=False,
                error=f"Step execution failed after {execution_time:.2f}s: {str(e)}",
                metadata={
                    'execution_time': execution_time,
                    'step_id': step.step_id
                }
            )
    
    async def _enhance_parameters(self, base_params: Dict[str, Any],
                                dependency_results: Dict[str, BioToolResult]) -> Dict[str, Any]:
        """Enhance step parameters with outputs from dependencies"""
        
        enhanced_params = base_params.copy()
        
        # Simple parameter substitution
        for key, value in enhanced_params.items():
            if isinstance(value, str) and value.startswith("$output:"):
                # Extract output reference: $output:step_id:output_name
                parts = value.split(":")
                if len(parts) >= 2:
                    dep_step_id = parts[1]
                    if dep_step_id in dependency_results:
                        dep_result = dependency_results[dep_step_id]
                        if hasattr(dep_result, 'output') and dep_result.output:
                            enhanced_params[key] = dep_result.output
        
        return enhanced_params


# =================== Pipeline Templates Library ===================

class PipelineTemplateLibrary:
    """Library of common bioinformatics pipeline templates"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize common pipeline templates"""
        
        # RNA-seq analysis template
        self.templates['rnaseq_de'] = self._create_rnaseq_template()
        
        # Variant calling template
        self.templates['variant_calling'] = self._create_variant_calling_template()
        
        # Protein analysis template
        self.templates['protein_analysis'] = self._create_protein_analysis_template()
        
        # ChIP-seq analysis template
        self.templates['chipseq'] = self._create_chipseq_template()
    
    def _create_rnaseq_template(self) -> PipelineTemplate:
        """Create RNA-seq differential expression analysis template"""
        
        template = PipelineTemplate(
            name="rnaseq_differential_expression",
            description="Complete RNA-seq differential expression analysis pipeline"
        )
        
        # Import tools (these would be actual tool implementations)
        from bioagent_tools import (
            SequenceStatsTool, RNASeqDifferentialExpressionTool, 
            BioinformaticsVisualizationTool
        )
        
        # Quality control step
        qc_step = PipelineStep(
            step_id="quality_control",
            tool=SequenceStatsTool(),
            parameters={
                "input_file": "${input_file}",
                "sequence_type": "rna",
                "output_format": "json"
            },
            outputs=["qc_report"],
            estimated_runtime=timedelta(minutes=30),
            resource_requirements={"cpu_cores": 2, "memory_gb": 8}
        )
        template.add_step(qc_step)
        
        # Differential expression analysis
        de_step = PipelineStep(
            step_id="differential_expression",
            tool=RNASeqDifferentialExpressionTool(),
            parameters={
                "count_matrix": "${count_matrix}",
                "sample_info": "${sample_info}",
                "condition_column": "${condition_column}",
                "control_condition": "${control_condition}",
                "treatment_condition": "${treatment_condition}"
            },
            dependencies=["quality_control"],
            outputs=["de_results"],
            estimated_runtime=timedelta(hours=1),
            resource_requirements={"cpu_cores": 4, "memory_gb": 16}
        )
        template.add_step(de_step)
        
        # Visualization
        viz_step = PipelineStep(
            step_id="visualization",
            tool=BioinformaticsVisualizationTool(),
            parameters={
                "data_file": "$output:differential_expression:de_results",
                "plot_type": "volcano",
                "output_format": "png"
            },
            dependencies=["differential_expression"],
            outputs=["plots"],
            estimated_runtime=timedelta(minutes=15),
            resource_requirements={"cpu_cores": 1, "memory_gb": 4}
        )
        template.add_step(viz_step)
        
        return template
    
    def _create_variant_calling_template(self) -> PipelineTemplate:
        """Create variant calling analysis template"""
        
        template = PipelineTemplate(
            name="variant_calling",
            description="Variant calling and annotation pipeline"
        )
        
        # This would include steps like:
        # 1. Read alignment
        # 2. Variant calling
        # 3. Variant annotation
        # 4. Quality filtering
        # 5. Report generation
        
        return template
    
    def _create_protein_analysis_template(self) -> PipelineTemplate:
        """Create protein analysis template"""
        
        template = PipelineTemplate(
            name="protein_analysis",
            description="Comprehensive protein sequence and structure analysis"
        )
        
        # This would include steps like:
        # 1. Sequence statistics
        # 2. Domain prediction
        # 3. Structure analysis
        # 4. Functional annotation
        # 5. Comparative analysis
        
        return template
    
    def _create_chipseq_template(self) -> PipelineTemplate:
        """Create ChIP-seq analysis template"""
        
        template = PipelineTemplate(
            name="chipseq_analysis",
            description="ChIP-seq peak calling and analysis pipeline"
        )
        
        # This would include steps like:
        # 1. Quality control
        # 2. Read alignment
        # 3. Peak calling
        # 4. Peak annotation
        # 5. Motif analysis
        # 6. Differential binding
        
        return template
    
    def get_template(self, template_name: str) -> Optional[PipelineTemplate]:
        """Get a template by name"""
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        """List available template names"""
        return list(self.templates.keys())


# =================== Pipeline Manager ===================

class PipelineManager:
    """High-level manager for bioinformatics pipelines"""
    
    def __init__(self):
        self.pipelines: Dict[str, BioinformaticsPipeline] = {}
        self.template_library = PipelineTemplateLibrary()
        self.execution_history: List[PipelineResult] = []
    
    async def create_pipeline_from_template(self, template_name: str, 
                                          parameters: Dict[str, Any]) -> BioinformaticsPipeline:
        """Create a pipeline from a template"""
        
        template = self.template_library.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        pipeline = template.instantiate(parameters)
        self.pipelines[pipeline.pipeline_id] = pipeline
        
        return pipeline
    
    async def execute_pipeline(self, pipeline_id: str, 
                             data_metadata: List[DataMetadata]) -> PipelineResult:
        """Execute a pipeline by ID"""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")
        
        pipeline = self.pipelines[pipeline_id]
        result = await pipeline.execute(data_metadata)
        
        # Store in history
        self.execution_history.append(result)
        
        return result
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get the current status of a pipeline"""
        
        if pipeline_id not in self.pipelines:
            return {"error": "Pipeline not found"}
        
        pipeline = self.pipelines[pipeline_id]
        
        status = {
            "pipeline_id": pipeline_id,
            "is_running": pipeline.is_running,
            "created_at": pipeline.created_at.isoformat(),
            "step_count": len(pipeline.steps),
            "execution_mode": pipeline.execution_mode.value
        }
        
        if pipeline.current_result:
            status.update({
                "current_status": pipeline.current_result.status,
                "completed_steps": len(pipeline.current_result.step_results),
                "success_rate": pipeline.current_result.success_rate
            })
        
        return status
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines"""
        return [
            {
                "pipeline_id": pid,
                "status": self.get_pipeline_status(pid)
            }
            for pid in self.pipelines.keys()
        ]


# =================== Example Usage ===================

async def example_pipeline_usage():
    """Example of using the pipeline orchestration system"""
    
    # Initialize pipeline manager
    manager = PipelineManager()
    
    # Create pipeline from template
    pipeline = await manager.create_pipeline_from_template(
        template_name="rnaseq_de",
        parameters={
            "input_file": "/data/rnaseq_reads.fastq",
            "count_matrix": "/data/count_matrix.csv",
            "sample_info": "/data/sample_info.csv",
            "condition_column": "treatment",
            "control_condition": "control",
            "treatment_condition": "drug_A"
        }
    )
    
    print(f"Created pipeline: {pipeline.pipeline_id}")
    
    # Validate pipeline
    is_valid, errors = pipeline.validate_pipeline()
    print(f"Pipeline valid: {is_valid}")
    if errors:
        print(f"Validation errors: {errors}")
    
    # Create sample data metadata
    from bioagent_architecture import DataMetadata, DataType
    
    data_metadata = [
        DataMetadata(
            data_type=DataType.EXPRESSION_MATRIX,
            file_path="/data/count_matrix.csv",
            organism="Homo sapiens",
            tissue_type="liver",
            experimental_condition="drug_treatment"
        )
    ]
    
    # Execute pipeline
    try:
        result = await manager.execute_pipeline(pipeline.pipeline_id, data_metadata)
        print(f"Pipeline completed with status: {result.status}")
        print(f"Success rate: {result.success_rate:.2%}")
        if result.total_runtime:
            print(f"Total runtime: {result.total_runtime}")
    
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
    
    # Check status
    status = manager.get_pipeline_status(pipeline.pipeline_id)
    print(f"Pipeline status: {status}")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(example_pipeline_usage())