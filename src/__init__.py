"""
BioinformaticsAgent: An Advanced AI System for Computational Biology

This package provides a sophisticated AI-powered system designed specifically 
for bioinformatics and computational biology analysis.
"""

__version__ = "1.0.0"
__author__ = "Ashwin Gopinath"
__email__ = "ashwin@example.com"

# Import main components for easy access
from .bioagent_architecture import (
    BioinformaticsAgent,
    DataMetadata,
    DataType,
    AnalysisTask,
    ReasoningType
)

from .bioagent_tools import (
    get_all_bioinformatics_tools,
    SequenceStatsTool,
    RNASeqDifferentialExpressionTool,
    VariantAnnotationTool,
    ProteinStructureAnalysisTool,
    BioinformaticsVisualizationTool
)

from .bioagent_pipeline import (
    BioinformaticsPipeline,
    PipelineManager,
    PipelineStep,
    ExecutionMode
)

__all__ = [
    "BioinformaticsAgent",
    "DataMetadata", 
    "DataType",
    "AnalysisTask",
    "ReasoningType",
    "get_all_bioinformatics_tools",
    "SequenceStatsTool",
    "RNASeqDifferentialExpressionTool", 
    "VariantAnnotationTool",
    "ProteinStructureAnalysisTool",
    "BioinformaticsVisualizationTool",
    "BioinformaticsPipeline",
    "PipelineManager",
    "PipelineStep",
    "ExecutionMode"
]