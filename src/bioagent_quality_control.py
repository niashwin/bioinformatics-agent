#!/usr/bin/env python3
"""
BioinformaticsAgent Quality Control Module: Comprehensive QC analysis

This module provides extensive quality control capabilities:
- FastQC integration and parsing
- MultiQC report generation
- Custom QC metrics calculation
- Quality filtering and preprocessing
- Batch effect detection
- Contamination screening
- Library complexity assessment
"""

import asyncio
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile
import shutil
import os
import logging
import re
from collections import defaultdict

# Quality control libraries
try:
    import multiqc
    MULTIQC_AVAILABLE = True
except ImportError:
    MULTIQC_AVAILABLE = False

# Bioinformatics libraries
from Bio import SeqIO
try:
    from Bio.SeqUtils import gc_fraction as GC  # Updated BioPython import
except ImportError:
    try:
        from Bio.SeqUtils import GC  # Older BioPython versions
    except ImportError:
        def GC(seq):
            """Fallback GC content calculation"""
            seq_str = str(seq).upper()
            gc_count = seq_str.count('G') + seq_str.count('C')
            return (gc_count / len(seq_str)) * 100 if len(seq_str) > 0 else 0

try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

from bioagent_architecture import BioinformaticsTool, BioToolResult, DataType, DataMetadata
from bioagent_io import SequenceFileHandler, ExpressionDataHandler
from bioagent_external_tools import ExternalBioinformaticsTool, ToolConfiguration


# =================== QC Data Structures ===================

class QCMetricType(Enum):
    """Types of QC metrics"""
    SEQUENCE_QUALITY = "sequence_quality"
    LIBRARY_COMPLEXITY = "library_complexity"
    CONTAMINATION = "contamination"
    BATCH_EFFECTS = "batch_effects"
    ALIGNMENT_QUALITY = "alignment_quality"
    EXPRESSION_QUALITY = "expression_quality"
    VARIANT_QUALITY = "variant_quality"


@dataclass
class QCMetric:
    """Individual QC metric"""
    name: str
    value: Union[float, int, str]
    metric_type: QCMetricType
    pass_fail: Optional[bool] = None
    threshold: Optional[float] = None
    description: Optional[str] = None
    category: Optional[str] = None


@dataclass
class QCReport:
    """Comprehensive QC report"""
    sample_id: str
    data_type: DataType
    metrics: List[QCMetric] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    plots: Dict[str, str] = field(default_factory=dict)  # plot_name -> file_path
    overall_pass: bool = True
    
    def add_metric(self, metric: QCMetric):
        """Add a QC metric to the report"""
        self.metrics.append(metric)
        
        # Update overall pass status
        if metric.pass_fail is False:
            self.overall_pass = False
    
    def get_metrics_by_type(self, metric_type: QCMetricType) -> List[QCMetric]:
        """Get metrics of a specific type"""
        return [m for m in self.metrics if m.metric_type == metric_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "sample_id": self.sample_id,
            "data_type": self.data_type.value,
            "overall_pass": self.overall_pass,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "type": m.metric_type.value,
                    "pass_fail": m.pass_fail,
                    "threshold": m.threshold,
                    "description": m.description,
                    "category": m.category
                }
                for m in self.metrics
            ],
            "warnings": self.warnings,
            "errors": self.errors,
            "plots": self.plots
        }


# =================== Enhanced FastQC Tool ===================

class EnhancedFastQCTool(ExternalBioinformaticsTool):
    """Enhanced FastQC with detailed parsing and custom metrics"""
    
    def __init__(self, fastqc_path: str = "fastqc"):
        config = ToolConfiguration(
            executable_path=fastqc_path,
            default_parameters={"threads": 4}
        )
        
        super().__init__(
            name="enhanced_fastqc",
            description="Enhanced FastQC with comprehensive result parsing",
            supported_data_types=[DataType.FASTQ_READS],
            tool_type=ExternalToolType.QUALITY_CONTROL,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_files": {"type": "array", "items": {"type": "string"}},
                "output_dir": {"type": "string"},
                "threads": {"type": "integer", "default": 4},
                "extract": {"type": "boolean", "default": True},
                "custom_thresholds": {
                    "type": "object",
                    "properties": {
                        "min_quality": {"type": "number", "default": 20},
                        "min_gc": {"type": "number", "default": 30},
                        "max_gc": {"type": "number", "default": 70},
                        "max_duplication": {"type": "number", "default": 50}
                    }
                }
            },
            "required": ["input_files", "output_dir"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute enhanced FastQC analysis"""
        
        try:
            # Run standard FastQC
            fastqc_result = await self._run_fastqc(params)
            if not fastqc_result.success:
                return fastqc_result
            
            # Parse FastQC results in detail
            qc_reports = await self._parse_fastqc_results(
                params["output_dir"], 
                params["input_files"],
                params.get("custom_thresholds", {})
            )
            
            # Generate summary statistics
            summary_stats = self._generate_summary_stats(qc_reports)
            
            # Create visualizations
            plot_files = await self._create_qc_plots(qc_reports, params["output_dir"])
            
            return BioToolResult(
                success=True,
                output="Enhanced FastQC analysis completed",
                metadata={
                    "qc_reports": [report.to_dict() for report in qc_reports],
                    "summary_stats": summary_stats,
                    "plot_files": plot_files,
                    "output_directory": params["output_dir"]
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Enhanced FastQC failed: {str(e)}"
            )
    
    async def _run_fastqc(self, params: Dict[str, Any]) -> BioToolResult:
        """Run FastQC command"""
        
        output_dir = params["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        command = [
            self.config.executable_path,
            "--outdir", output_dir,
            "--threads", str(params.get("threads", 4))
        ]
        
        if params.get("extract", True):
            command.append("--extract")
        
        command.extend(params["input_files"])
        
        result = await self._run_command(command, timeout=1800)
        
        if result.success:
            return BioToolResult(success=True, output="FastQC completed")
        else:
            return BioToolResult(
                success=False,
                error=f"FastQC failed: {result.stderr}"
            )
    
    async def _parse_fastqc_results(self, output_dir: str, input_files: List[str],
                                  thresholds: Dict[str, float]) -> List[QCReport]:
        """Parse FastQC results in detail"""
        
        reports = []
        
        for input_file in input_files:
            sample_name = Path(input_file).stem
            if sample_name.endswith('.fastq'):
                sample_name = sample_name[:-6]
            elif sample_name.endswith('.fq'):
                sample_name = sample_name[:-3]
            
            # Create QC report for this sample
            qc_report = QCReport(
                sample_id=sample_name,
                data_type=DataType.FASTQ_READS
            )
            
            # Parse FastQC data file
            data_file = Path(output_dir) / f"{sample_name}_fastqc" / "fastqc_data.txt"
            summary_file = Path(output_dir) / f"{sample_name}_fastqc" / "summary.txt"
            
            if data_file.exists():
                await self._parse_fastqc_data_file(data_file, qc_report, thresholds)
            
            if summary_file.exists():
                await self._parse_fastqc_summary_file(summary_file, qc_report)
            
            reports.append(qc_report)
        
        return reports
    
    async def _parse_fastqc_data_file(self, data_file: Path, qc_report: QCReport,
                                    thresholds: Dict[str, float]):
        """Parse FastQC data file for detailed metrics"""
        
        with open(data_file, 'r') as f:
            content = f.read()
        
        # Parse different sections
        sections = content.split(">>END_MODULE\n")
        
        for section in sections:
            if section.startswith(">>Basic Statistics"):
                await self._parse_basic_statistics(section, qc_report, thresholds)
            elif section.startswith(">>Per base sequence quality"):
                await self._parse_per_base_quality(section, qc_report, thresholds)
            elif section.startswith(">>Per sequence quality scores"):
                await self._parse_per_sequence_quality(section, qc_report, thresholds)
            elif section.startswith(">>Per base sequence content"):
                await self._parse_sequence_content(section, qc_report, thresholds)
            elif section.startswith(">>Per sequence GC content"):
                await self._parse_gc_content(section, qc_report, thresholds)
            elif section.startswith(">>Sequence Duplication Levels"):
                await self._parse_duplication_levels(section, qc_report, thresholds)
            elif section.startswith(">>Adapter Content"):
                await self._parse_adapter_content(section, qc_report, thresholds)
    
    async def _parse_basic_statistics(self, section: str, qc_report: QCReport,
                                    thresholds: Dict[str, float]):
        """Parse basic statistics section"""
        
        lines = section.split('\n')
        for line in lines:
            if line.startswith('Total Sequences'):
                total_seqs = int(line.split('\t')[1])
                qc_report.add_metric(QCMetric(
                    name="total_sequences",
                    value=total_seqs,
                    metric_type=QCMetricType.SEQUENCE_QUALITY,
                    description="Total number of sequences"
                ))
            
            elif line.startswith('Sequence length'):
                seq_length = line.split('\t')[1]
                qc_report.add_metric(QCMetric(
                    name="sequence_length",
                    value=seq_length,
                    metric_type=QCMetricType.SEQUENCE_QUALITY,
                    description="Sequence length range"
                ))
            
            elif line.startswith('%GC'):
                gc_content = float(line.split('\t')[1])
                min_gc = thresholds.get("min_gc", 30)
                max_gc = thresholds.get("max_gc", 70)
                
                qc_report.add_metric(QCMetric(
                    name="gc_content",
                    value=gc_content,
                    metric_type=QCMetricType.SEQUENCE_QUALITY,
                    pass_fail=min_gc <= gc_content <= max_gc,
                    threshold=f"{min_gc}-{max_gc}%",
                    description="Overall GC content"
                ))
    
    async def _parse_per_base_quality(self, section: str, qc_report: QCReport,
                                    thresholds: Dict[str, float]):
        """Parse per base sequence quality"""
        
        lines = section.split('\n')
        quality_scores = []
        
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    mean_quality = float(parts[1])
                    quality_scores.append(mean_quality)
                except ValueError:
                    continue
        
        if quality_scores:
            min_quality = min(quality_scores)
            mean_quality = np.mean(quality_scores)
            min_threshold = thresholds.get("min_quality", 20)
            
            qc_report.add_metric(QCMetric(
                name="min_per_base_quality",
                value=min_quality,
                metric_type=QCMetricType.SEQUENCE_QUALITY,
                pass_fail=min_quality >= min_threshold,
                threshold=min_threshold,
                description="Minimum per-base quality score"
            ))
            
            qc_report.add_metric(QCMetric(
                name="mean_per_base_quality",
                value=mean_quality,
                metric_type=QCMetricType.SEQUENCE_QUALITY,
                description="Mean per-base quality score"
            ))
    
    async def _parse_duplication_levels(self, section: str, qc_report: QCReport,
                                      thresholds: Dict[str, float]):
        """Parse sequence duplication levels"""
        
        lines = section.split('\n')
        for line in lines:
            if line.startswith('#Total Duplicate Percentage'):
                dup_percentage = float(line.split('\t')[1])
                max_dup = thresholds.get("max_duplication", 50)
                
                qc_report.add_metric(QCMetric(
                    name="duplication_percentage",
                    value=dup_percentage,
                    metric_type=QCMetricType.LIBRARY_COMPLEXITY,
                    pass_fail=dup_percentage <= max_dup,
                    threshold=f"<={max_dup}%",
                    description="Percentage of duplicate sequences"
                ))
                break
    
    async def _parse_fastqc_summary_file(self, summary_file: Path, qc_report: QCReport):
        """Parse FastQC summary file for pass/fail status"""
        
        with open(summary_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    status, module, filename = parts[0], parts[1], parts[2]
                    
                    if status == "FAIL":
                        qc_report.errors.append(f"{module}: FAIL")
                        qc_report.overall_pass = False
                    elif status == "WARN":
                        qc_report.warnings.append(f"{module}: WARNING")
    
    def _generate_summary_stats(self, qc_reports: List[QCReport]) -> Dict[str, Any]:
        """Generate summary statistics across all samples"""
        
        summary = {
            "total_samples": len(qc_reports),
            "passing_samples": sum(1 for r in qc_reports if r.overall_pass),
            "failing_samples": sum(1 for r in qc_reports if not r.overall_pass),
            "average_metrics": {}
        }
        
        # Calculate average metrics
        metric_values = defaultdict(list)
        
        for report in qc_reports:
            for metric in report.metrics:
                if isinstance(metric.value, (int, float)):
                    metric_values[metric.name].append(metric.value)
        
        for metric_name, values in metric_values.items():
            if values:
                summary["average_metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return summary
    
    async def _create_qc_plots(self, qc_reports: List[QCReport], 
                             output_dir: str) -> List[str]:
        """Create QC visualization plots"""
        
        plot_files = []
        
        # Quality score distribution plot
        plt.figure(figsize=(12, 8))
        
        # Extract quality metrics
        quality_scores = []
        gc_contents = []
        dup_percentages = []
        sample_names = []
        
        for report in qc_reports:
            sample_names.append(report.sample_id)
            
            for metric in report.metrics:
                if metric.name == "mean_per_base_quality":
                    quality_scores.append(metric.value)
                elif metric.name == "gc_content":
                    gc_contents.append(metric.value)
                elif metric.name == "duplication_percentage":
                    dup_percentages.append(metric.value)
        
        # Plot 1: Quality score distribution
        if quality_scores:
            plt.subplot(2, 2, 1)
            plt.hist(quality_scores, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Mean Quality Score')
            plt.ylabel('Number of Samples')
            plt.title('Distribution of Mean Quality Scores')
            plt.axvline(x=20, color='r', linestyle='--', label='Threshold (Q20)')
            plt.legend()
        
        # Plot 2: GC content distribution
        if gc_contents:
            plt.subplot(2, 2, 2)
            plt.hist(gc_contents, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('GC Content (%)')
            plt.ylabel('Number of Samples')
            plt.title('Distribution of GC Content')
            plt.axvline(x=30, color='r', linestyle='--', label='Min Threshold')
            plt.axvline(x=70, color='r', linestyle='--', label='Max Threshold')
            plt.legend()
        
        # Plot 3: Duplication levels
        if dup_percentages:
            plt.subplot(2, 2, 3)
            plt.hist(dup_percentages, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Duplication Percentage (%)')
            plt.ylabel('Number of Samples')
            plt.title('Distribution of Duplication Levels')
            plt.axvline(x=50, color='r', linestyle='--', label='Threshold (50%)')
            plt.legend()
        
        # Plot 4: Sample pass/fail status
        plt.subplot(2, 2, 4)
        pass_counts = [1 if r.overall_pass else 0 for r in qc_reports]
        pass_rate = np.mean(pass_counts) * 100
        
        plt.pie([pass_rate, 100 - pass_rate], 
               labels=['Pass', 'Fail'],
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%')
        plt.title('Sample QC Pass/Fail Rate')
        
        plt.tight_layout()
        
        plot_file = Path(output_dir) / "qc_summary_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_files.append(str(plot_file))
        
        return plot_files


# =================== MultiQC Integration ===================

class MultiQCTool(BioinformaticsTool):
    """MultiQC tool for aggregating QC reports"""
    
    def __init__(self):
        super().__init__(
            name="multiqc",
            description="Aggregate QC reports from multiple tools",
            supported_data_types=[DataType.FASTQ_READS, DataType.ALIGNMENT_SAM]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_dirs": {"type": "array", "items": {"type": "string"}},
                "output_dir": {"type": "string"},
                "report_name": {"type": "string", "default": "multiqc_report"},
                "force": {"type": "boolean", "default": True}
            },
            "required": ["input_dirs", "output_dir"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute MultiQC aggregation"""
        
        try:
            if not MULTIQC_AVAILABLE:
                return BioToolResult(
                    success=False,
                    error="MultiQC package not available. Please install with: pip install multiqc"
                )
            
            # Prepare MultiQC command
            command = [
                "multiqc",
                "--outdir", params["output_dir"],
                "--filename", params.get("report_name", "multiqc_report")
            ]
            
            if params.get("force", True):
                command.append("--force")
            
            # Add input directories
            command.extend(params["input_dirs"])
            
            # Run MultiQC
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Parse MultiQC output
                report_file = Path(params["output_dir"]) / f"{params.get('report_name', 'multiqc_report')}.html"
                data_file = Path(params["output_dir"]) / "multiqc_data" / "multiqc_general_stats.txt"
                
                # Extract summary statistics
                summary_stats = {}
                if data_file.exists():
                    summary_stats = await self._parse_multiqc_data(data_file)
                
                return BioToolResult(
                    success=True,
                    output=f"MultiQC report generated: {report_file}",
                    metadata={
                        "report_file": str(report_file),
                        "data_directory": str(Path(params["output_dir"]) / "multiqc_data"),
                        "summary_stats": summary_stats
                    }
                )
            else:
                return BioToolResult(
                    success=False,
                    error=f"MultiQC failed: {stderr.decode()}"
                )
                
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"MultiQC execution failed: {str(e)}"
            )
    
    async def _parse_multiqc_data(self, data_file: Path) -> Dict[str, Any]:
        """Parse MultiQC general statistics"""
        
        try:
            df = pd.read_csv(data_file, sep='\t')
            
            summary = {
                "total_samples": len(df),
                "sample_names": df.iloc[:, 0].tolist() if len(df.columns) > 0 else [],
                "metrics_available": df.columns.tolist()[1:] if len(df.columns) > 1 else []
            }
            
            # Calculate summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                summary[f"{col}_mean"] = df[col].mean()
                summary[f"{col}_std"] = df[col].std()
            
            return summary
            
        except Exception as e:
            logging.warning(f"Failed to parse MultiQC data: {e}")
            return {}


# =================== Custom QC Metrics ===================

class CustomQCAnalyzer(BioinformaticsTool):
    """Custom quality control analysis beyond standard tools"""
    
    def __init__(self):
        super().__init__(
            name="custom_qc",
            description="Custom quality control metrics and analysis",
            supported_data_types=[DataType.FASTQ_READS, DataType.EXPRESSION_MATRIX, DataType.ALIGNMENT_SAM]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_files": {"type": "array", "items": {"type": "string"}},
                "analysis_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["contamination", "batch_effects", "complexity"]}
                },
                "output_dir": {"type": "string"},
                "reference_files": {"type": "object"}
            },
            "required": ["input_files", "analysis_types", "output_dir"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute custom QC analysis"""
        
        try:
            output_dir = Path(params["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_types = params["analysis_types"]
            results = {}
            
            for analysis_type in analysis_types:
                if analysis_type == "contamination":
                    results["contamination"] = await self._contamination_analysis(
                        params["input_files"], output_dir
                    )
                elif analysis_type == "batch_effects":
                    results["batch_effects"] = await self._batch_effect_analysis(
                        params["input_files"], output_dir
                    )
                elif analysis_type == "complexity":
                    results["complexity"] = await self._complexity_analysis(
                        params["input_files"], output_dir
                    )
            
            return BioToolResult(
                success=True,
                output="Custom QC analysis completed",
                metadata={
                    "results": results,
                    "output_directory": str(output_dir)
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Custom QC analysis failed: {str(e)}"
            )
    
    async def _contamination_analysis(self, input_files: List[str], 
                                    output_dir: Path) -> Dict[str, Any]:
        """Analyze potential contamination in sequencing data"""
        
        contamination_results = {}
        
        for input_file in input_files:
            sample_name = Path(input_file).stem
            
            # Basic contamination indicators
            handler = SequenceFileHandler(input_file)
            stats = handler.get_sequence_stats()
            
            # Check for unusual GC content distribution
            gc_contents = []
            for record in handler.read_sequences(max_records=10000):
                gc_content = GC(record.seq)
                gc_contents.append(gc_content)
            
            if gc_contents:
                gc_mean = np.mean(gc_contents)
                gc_std = np.std(gc_contents)
                
                # Flag samples with very high GC variation (potential contamination)
                contamination_risk = "high" if gc_std > 15 else "low"
                
                contamination_results[sample_name] = {
                    "gc_mean": gc_mean,
                    "gc_std": gc_std,
                    "contamination_risk": contamination_risk,
                    "total_sequences": stats["total_sequences"]
                }
        
        return contamination_results
    
    async def _batch_effect_analysis(self, input_files: List[str], 
                                   output_dir: Path) -> Dict[str, Any]:
        """Analyze potential batch effects in data"""
        
        # This is a simplified batch effect analysis
        # In practice, would use more sophisticated methods
        
        batch_results = {}
        
        if len(input_files) > 1:
            # Collect GC content distributions
            all_gc_contents = {}
            
            for input_file in input_files:
                sample_name = Path(input_file).stem
                handler = SequenceFileHandler(input_file)
                
                gc_contents = []
                for record in handler.read_sequences(max_records=1000):
                    gc_content = GC(record.seq)
                    gc_contents.append(gc_content)
                
                all_gc_contents[sample_name] = gc_contents
            
            # Calculate pairwise correlations
            sample_names = list(all_gc_contents.keys())
            correlations = {}
            
            for i, sample1 in enumerate(sample_names):
                for j, sample2 in enumerate(sample_names[i+1:], i+1):
                    if len(all_gc_contents[sample1]) > 0 and len(all_gc_contents[sample2]) > 0:
                        # Simple correlation of GC distributions
                        hist1, _ = np.histogram(all_gc_contents[sample1], bins=20, range=(0, 100))
                        hist2, _ = np.histogram(all_gc_contents[sample2], bins=20, range=(0, 100))
                        
                        correlation = np.corrcoef(hist1, hist2)[0, 1]
                        correlations[f"{sample1}_vs_{sample2}"] = correlation
            
            batch_results = {
                "sample_correlations": correlations,
                "potential_batches": "detected" if min(correlations.values()) < 0.8 else "none"
            }
        
        return batch_results
    
    async def _complexity_analysis(self, input_files: List[str], 
                                 output_dir: Path) -> Dict[str, Any]:
        """Analyze library complexity"""
        
        complexity_results = {}
        
        for input_file in input_files:
            sample_name = Path(input_file).stem
            
            # Calculate library complexity metrics
            handler = SequenceFileHandler(input_file)
            
            sequences = set()
            total_reads = 0
            
            # Sample sequences to estimate complexity
            for record in handler.read_sequences(max_records=100000):
                sequences.add(str(record.seq))
                total_reads += 1
            
            unique_sequences = len(sequences)
            complexity_ratio = unique_sequences / total_reads if total_reads > 0 else 0
            
            # Estimate library complexity using capture-recapture
            if total_reads > 1000:
                # Simplified Good-Turing estimate
                estimated_complexity = unique_sequences / (1 - (unique_sequences / total_reads))
            else:
                estimated_complexity = unique_sequences
            
            complexity_results[sample_name] = {
                "total_reads_sampled": total_reads,
                "unique_sequences": unique_sequences,
                "complexity_ratio": complexity_ratio,
                "estimated_total_complexity": estimated_complexity,
                "complexity_class": "high" if complexity_ratio > 0.8 else "medium" if complexity_ratio > 0.5 else "low"
            }
        
        return complexity_results


# =================== QC Pipeline ===================

class QualityControlPipeline:
    """Comprehensive quality control pipeline"""
    
    def __init__(self):
        self.tools = {
            "fastqc": EnhancedFastQCTool(),
            "multiqc": MultiQCTool(),
            "custom_qc": CustomQCAnalyzer()
        }
        self.results = {}
    
    async def run_full_qc_pipeline(self, input_files: List[str], 
                                 output_dir: str,
                                 data_type: DataType = DataType.FASTQ_READS) -> Dict[str, Any]:
        """Run complete QC pipeline"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pipeline_results = {
            "input_files": input_files,
            "output_directory": output_dir,
            "data_type": data_type.value,
            "tools_run": [],
            "overall_summary": {}
        }
        
        try:
            # Step 1: Run FastQC
            if data_type == DataType.FASTQ_READS:
                print("Running enhanced FastQC analysis...")
                fastqc_params = {
                    "input_files": input_files,
                    "output_dir": str(output_path / "fastqc"),
                    "threads": 4,
                    "extract": True
                }
                
                fastqc_result = await self.tools["fastqc"].execute(fastqc_params, [])
                pipeline_results["fastqc_result"] = fastqc_result.metadata if fastqc_result.success else {"error": fastqc_result.error}
                pipeline_results["tools_run"].append("fastqc")
                
                # Step 2: Run MultiQC
                print("Running MultiQC aggregation...")
                multiqc_params = {
                    "input_dirs": [str(output_path / "fastqc")],
                    "output_dir": str(output_path / "multiqc"),
                    "report_name": "qc_report"
                }
                
                multiqc_result = await self.tools["multiqc"].execute(multiqc_params, [])
                pipeline_results["multiqc_result"] = multiqc_result.metadata if multiqc_result.success else {"error": multiqc_result.error}
                pipeline_results["tools_run"].append("multiqc")
            
            # Step 3: Run custom QC analysis
            print("Running custom QC analysis...")
            custom_qc_params = {
                "input_files": input_files,
                "analysis_types": ["contamination", "batch_effects", "complexity"],
                "output_dir": str(output_path / "custom_qc")
            }
            
            custom_result = await self.tools["custom_qc"].execute(custom_qc_params, [])
            pipeline_results["custom_qc_result"] = custom_result.metadata if custom_result.success else {"error": custom_result.error}
            pipeline_results["tools_run"].append("custom_qc")
            
            # Generate overall summary
            pipeline_results["overall_summary"] = self._generate_pipeline_summary(pipeline_results)
            
            # Save results
            results_file = output_path / "qc_pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results["error"] = str(e)
            return pipeline_results
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of QC pipeline"""
        
        summary = {
            "total_samples": len(results["input_files"]),
            "tools_succeeded": [],
            "tools_failed": [],
            "overall_pass_rate": 0,
            "key_findings": []
        }
        
        # Check tool success rates
        for tool in results["tools_run"]:
            result_key = f"{tool}_result"
            if result_key in results and "error" not in results[result_key]:
                summary["tools_succeeded"].append(tool)
            else:
                summary["tools_failed"].append(tool)
        
        # Extract key findings from FastQC if available
        if "fastqc_result" in results and "qc_reports" in results["fastqc_result"]:
            qc_reports = results["fastqc_result"]["qc_reports"]
            passing_samples = sum(1 for report in qc_reports if report["overall_pass"])
            summary["overall_pass_rate"] = passing_samples / len(qc_reports) if qc_reports else 0
            
            # Identify common issues
            common_issues = {}
            for report in qc_reports:
                for error in report.get("errors", []):
                    common_issues[error] = common_issues.get(error, 0) + 1
            
            if common_issues:
                most_common = max(common_issues, key=common_issues.get)
                summary["key_findings"].append(f"Most common issue: {most_common} ({common_issues[most_common]} samples)")
        
        return summary


# =================== Example Usage ===================

async def example_quality_control():
    """Example of comprehensive quality control analysis"""
    
    print("Quality Control Analysis Example")
    print("=" * 50)
    
    # Initialize QC pipeline
    qc_pipeline = QualityControlPipeline()
    
    # Example input files (would be real FASTQ files in practice)
    example_files = [
        "sample1.fastq.gz",
        "sample2.fastq.gz",
        "sample3.fastq.gz"
    ]
    
    output_directory = "qc_analysis_output"
    
    print(f"Running QC pipeline on {len(example_files)} samples...")
    print(f"Output directory: {output_directory}")
    
    # This would work with real files
    # results = await qc_pipeline.run_full_qc_pipeline(
    #     example_files, 
    #     output_directory,
    #     DataType.FASTQ_READS
    # )
    
    # For demonstration, show what the pipeline would do
    print("\nQC Pipeline Steps:")
    print("1. Enhanced FastQC analysis with custom thresholds")
    print("2. MultiQC aggregation and reporting")
    print("3. Custom contamination screening")
    print("4. Batch effect detection")
    print("5. Library complexity assessment")
    print("6. Comprehensive visualization")
    
    print("\nExample completed (requires real FASTQ files to run)")


if __name__ == "__main__":
    asyncio.run(example_quality_control())