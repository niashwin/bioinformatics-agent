#!/usr/bin/env python3
"""
BioinformaticsAgent Alignment and Mapping Module: Comprehensive sequence alignment capabilities

This module provides advanced sequence alignment and mapping capabilities:
- Short-read alignment (BWA, Bowtie2, minimap2)
- Long-read alignment (minimap2, NGMLR, GraphMap)
- RNA-seq alignment (STAR, HISAT2, TopHat)
- Splice-aware alignment
- Multiple sequence alignment (MAFFT, Clustal, MUSCLE)
- Phylogenetic alignment
- Alignment post-processing and quality assessment
- Coverage analysis and statistics
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import warnings
from collections import defaultdict
import subprocess
import tempfile
import gzip
import re

# Bioinformatics libraries
try:
    import pysam
    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False
    logging.warning("Pysam not available. BAM/SAM processing limited.")

try:
    from Bio import AlignIO, SeqIO
    from Bio.Align import MultipleSeqAlignment
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.warning("Biopython not available. Sequence processing limited.")

# Statistical libraries
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_curve

from bioagent_architecture import BioinformaticsTool, BioToolResult, DataType, DataMetadata
from bioagent_external_tools import ExternalBioinformaticsTool, ToolConfiguration, ExternalToolType
from bioagent_io import SequenceFileHandler


# =================== Alignment Data Structures ===================

class AlignmentType(Enum):
    """Types of sequence alignment"""
    PAIRWISE = "pairwise"
    MULTIPLE = "multiple"
    SHORT_READ = "short_read"
    LONG_READ = "long_read"
    RNA_SEQ = "rna_seq"
    SPLICE_AWARE = "splice_aware"
    PHYLOGENETIC = "phylogenetic"


class AlignmentAlgorithm(Enum):
    """Alignment algorithms"""
    BWA_MEM = "bwa_mem"
    BWA_ALN = "bwa_aln"
    BOWTIE2 = "bowtie2"
    MINIMAP2 = "minimap2"
    STAR = "star"
    HISAT2 = "hisat2"
    TOPHAT = "tophat"
    MAFFT = "mafft"
    CLUSTALW = "clustalw"
    MUSCLE = "muscle"
    TCOFFEE = "tcoffee"


class ReadType(Enum):
    """Types of sequencing reads"""
    SINGLE_END = "single_end"
    PAIRED_END = "paired_end"
    MATE_PAIR = "mate_pair"
    LONG_READ = "long_read"


@dataclass
class AlignmentStatistics:
    """Statistics from alignment process"""
    total_reads: int
    mapped_reads: int
    unmapped_reads: int
    properly_paired: int = 0
    singleton_reads: int = 0
    duplicates: int = 0
    mapping_quality_mean: float = 0.0
    mapping_quality_std: float = 0.0
    insert_size_mean: float = 0.0
    insert_size_std: float = 0.0
    
    @property
    def mapping_rate(self) -> float:
        """Calculate mapping rate"""
        return self.mapped_reads / self.total_reads if self.total_reads > 0 else 0.0
    
    @property
    def proper_pair_rate(self) -> float:
        """Calculate proper pair rate"""
        return self.properly_paired / self.total_reads if self.total_reads > 0 else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_reads": self.total_reads,
            "mapped_reads": self.mapped_reads,
            "unmapped_reads": self.unmapped_reads,
            "mapping_rate": f"{self.mapping_rate:.2%}",
            "properly_paired": self.properly_paired,
            "proper_pair_rate": f"{self.proper_pair_rate:.2%}",
            "duplicates": self.duplicates,
            "mapping_quality_mean": self.mapping_quality_mean,
            "insert_size_mean": self.insert_size_mean
        }


@dataclass
class CoverageStatistics:
    """Coverage statistics for aligned reads"""
    mean_coverage: float
    median_coverage: float
    std_coverage: float
    coverage_1x: float  # Percentage of bases covered at least 1x
    coverage_5x: float  # Percentage of bases covered at least 5x
    coverage_10x: float  # Percentage of bases covered at least 10x
    coverage_20x: float  # Percentage of bases covered at least 20x
    coverage_distribution: List[int] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get coverage summary"""
        return {
            "mean_coverage": self.mean_coverage,
            "median_coverage": self.median_coverage,
            "std_coverage": self.std_coverage,
            "coverage_1x": f"{self.coverage_1x:.2%}",
            "coverage_5x": f"{self.coverage_5x:.2%}",
            "coverage_10x": f"{self.coverage_10x:.2%}",
            "coverage_20x": f"{self.coverage_20x:.2%}"
        }


# =================== Short-Read Alignment Tool ===================

class ShortReadAligner(ExternalBioinformaticsTool):
    """Short-read alignment using BWA, Bowtie2, or minimap2"""
    
    def __init__(self, algorithm: AlignmentAlgorithm = AlignmentAlgorithm.BWA_MEM):
        self.algorithm = algorithm
        
        # Configure based on algorithm
        if algorithm == AlignmentAlgorithm.BWA_MEM:
            executable = "bwa"
            tool_name = "bwa_mem_aligner"
        elif algorithm == AlignmentAlgorithm.BOWTIE2:
            executable = "bowtie2"
            tool_name = "bowtie2_aligner"
        elif algorithm == AlignmentAlgorithm.MINIMAP2:
            executable = "minimap2"
            tool_name = "minimap2_aligner"
        else:
            executable = "bwa"
            tool_name = "short_read_aligner"
        
        config = ToolConfiguration(
            executable_path=executable,
            default_parameters={
                "threads": 8,
                "mark_shorter_splits": True
            }
        )
        
        super().__init__(
            name=tool_name,
            description=f"Short-read alignment using {algorithm.value}",
            supported_data_types=[DataType.FASTQ_READS],
            tool_type=ExternalToolType.ALIGNMENT,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reference_index": {"type": "string", "description": "Path to reference index"},
                "fastq_files": {"type": "array", "items": {"type": "string"}, "description": "FASTQ input files"},
                "output_sam": {"type": "string", "description": "Output SAM/BAM file"},
                "read_type": {"type": "string", "enum": ["single_end", "paired_end"], "default": "paired_end"},
                "threads": {"type": "integer", "default": 8},
                "read_group": {"type": "string", "description": "Read group information"},
                "sort_output": {"type": "boolean", "default": True},
                "mark_duplicates": {"type": "boolean", "default": True},
                "min_mapping_quality": {"type": "integer", "default": 20}
            },
            "required": ["reference_index", "fastq_files", "output_sam"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute short-read alignment"""
        
        try:
            # Validate inputs
            reference_index = params["reference_index"]
            fastq_files = params["fastq_files"]
            output_sam = params["output_sam"]
            
            if not Path(reference_index).exists():
                return BioToolResult(success=False, error=f"Reference index not found: {reference_index}")
            
            for fastq_file in fastq_files:
                if not Path(fastq_file).exists():
                    return BioToolResult(success=False, error=f"FASTQ file not found: {fastq_file}")
            
            # Create output directory
            Path(output_sam).parent.mkdir(parents=True, exist_ok=True)
            
            # Run alignment
            if self.algorithm == AlignmentAlgorithm.BWA_MEM:
                result = await self._run_bwa_mem(params)
            elif self.algorithm == AlignmentAlgorithm.BOWTIE2:
                result = await self._run_bowtie2(params)
            elif self.algorithm == AlignmentAlgorithm.MINIMAP2:
                result = await self._run_minimap2(params)
            else:
                return BioToolResult(success=False, error=f"Algorithm {self.algorithm} not implemented")
            
            if not result.success:
                return result
            
            # Post-process alignment
            processed_bam = await self._post_process_alignment(params)
            
            # Generate alignment statistics
            alignment_stats = await self._calculate_alignment_statistics(processed_bam)
            
            # Calculate coverage statistics
            coverage_stats = await self._calculate_coverage_statistics(processed_bam, params.get("reference_index"))
            
            return BioToolResult(
                success=True,
                output=f"Short-read alignment completed: {processed_bam}",
                metadata={
                    "output_bam": processed_bam,
                    "algorithm": self.algorithm.value,
                    "alignment_statistics": alignment_stats.get_summary(),
                    "coverage_statistics": coverage_stats.get_summary() if coverage_stats else {},
                    "parameters": params
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Short-read alignment failed: {str(e)}"
            )
    
    async def _run_bwa_mem(self, params: Dict[str, Any]) -> BioToolResult:
        """Run BWA MEM alignment"""
        
        command = [
            "bwa", "mem",
            "-t", str(params.get("threads", 8)),
            params["reference_index"]
        ]
        
        # Add read group if specified
        if "read_group" in params:
            command.extend(["-R", params["read_group"]])
        
        # Add FASTQ files
        command.extend(params["fastq_files"])
        
        # Run BWA MEM and pipe to SAM file
        with open(params["output_sam"], 'w') as output_file:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=output_file,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
        
        if process.returncode == 0:
            return BioToolResult(success=True, output="BWA MEM completed")
        else:
            return BioToolResult(success=False, error=f"BWA MEM failed: {stderr.decode()}")
    
    async def _run_bowtie2(self, params: Dict[str, Any]) -> BioToolResult:
        """Run Bowtie2 alignment"""
        
        command = [
            "bowtie2",
            "-p", str(params.get("threads", 8)),
            "-x", params["reference_index"],
            "-S", params["output_sam"]
        ]
        
        # Handle paired-end vs single-end
        fastq_files = params["fastq_files"]
        if len(fastq_files) == 1:
            command.extend(["-U", fastq_files[0]])
        elif len(fastq_files) == 2:
            command.extend(["-1", fastq_files[0], "-2", fastq_files[1]])
        else:
            return BioToolResult(success=False, error="Bowtie2 supports 1 or 2 FASTQ files only")
        
        # Add read group if specified
        if "read_group" in params:
            command.extend(["--rg-id", "1", "--rg", params["read_group"]])
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return BioToolResult(success=True, output="Bowtie2 completed")
        else:
            return BioToolResult(success=False, error=f"Bowtie2 failed: {stderr.decode()}")
    
    async def _run_minimap2(self, params: Dict[str, Any]) -> BioToolResult:
        """Run minimap2 alignment"""
        
        command = [
            "minimap2",
            "-t", str(params.get("threads", 8)),
            "-a",  # Output SAM format
            "-x", "sr",  # Short reads preset
            params["reference_index"]
        ]
        
        command.extend(params["fastq_files"])
        
        # Run minimap2 and pipe to SAM file
        with open(params["output_sam"], 'w') as output_file:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=output_file,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
        
        if process.returncode == 0:
            return BioToolResult(success=True, output="minimap2 completed")
        else:
            return BioToolResult(success=False, error=f"minimap2 failed: {stderr.decode()}")
    
    async def _post_process_alignment(self, params: Dict[str, Any]) -> str:
        """Post-process alignment file (sort, index, mark duplicates)"""
        
        sam_file = params["output_sam"]
        base_name = Path(sam_file).stem
        output_dir = Path(sam_file).parent
        
        # Convert SAM to BAM and sort
        sorted_bam = output_dir / f"{base_name}.sorted.bam"
        
        if PYSAM_AVAILABLE:
            # Use pysam for processing
            print("Converting SAM to sorted BAM...")
            pysam.sort("-o", str(sorted_bam), sam_file)
            
            # Index BAM file
            print("Indexing BAM file...")
            pysam.index(str(sorted_bam))
            
            # Mark duplicates if requested
            if params.get("mark_duplicates", True):
                print("Marking duplicates...")
                dedup_bam = output_dir / f"{base_name}.sorted.dedup.bam"
                
                # Simple duplicate marking (would use picard in practice)
                with pysam.AlignmentFile(sorted_bam, "rb") as infile:
                    with pysam.AlignmentFile(dedup_bam, "wb", template=infile) as outfile:
                        seen_alignments = set()
                        
                        for read in infile:
                            # Simple duplicate detection based on position and sequence
                            if read.is_unmapped:
                                outfile.write(read)
                                continue
                            
                            key = (read.reference_id, read.reference_start, read.query_sequence)
                            
                            if key in seen_alignments:
                                read.is_duplicate = True
                            else:
                                seen_alignments.add(key)
                            
                            outfile.write(read)
                
                # Index deduplicated BAM
                pysam.index(str(dedup_bam))
                
                # Remove intermediate files
                sorted_bam.unlink()
                (sorted_bam.parent / f"{sorted_bam.name}.bai").unlink(missing_ok=True)
                
                return str(dedup_bam)
            
            return str(sorted_bam)
        
        else:
            # Fallback using samtools commands
            print("Converting SAM to sorted BAM using samtools...")
            
            # Convert and sort
            sort_command = ["samtools", "sort", "-o", str(sorted_bam), sam_file]
            process = await asyncio.create_subprocess_exec(*sort_command)
            await process.communicate()
            
            # Index
            index_command = ["samtools", "index", str(sorted_bam)]
            process = await asyncio.create_subprocess_exec(*index_command)
            await process.communicate()
            
            return str(sorted_bam)
    
    async def _calculate_alignment_statistics(self, bam_file: str) -> AlignmentStatistics:
        """Calculate alignment statistics from BAM file"""
        
        stats = AlignmentStatistics(0, 0, 0)
        
        if not PYSAM_AVAILABLE:
            logging.warning("Pysam not available. Cannot calculate detailed alignment statistics.")
            return stats
        
        try:
            with pysam.AlignmentFile(bam_file, "rb") as bamfile:
                mapping_qualities = []
                insert_sizes = []
                
                for read in bamfile:
                    stats.total_reads += 1
                    
                    if read.is_unmapped:
                        stats.unmapped_reads += 1
                    else:
                        stats.mapped_reads += 1
                        mapping_qualities.append(read.mapping_quality)
                        
                        if read.is_proper_pair:
                            stats.properly_paired += 1
                        
                        if read.is_read1 and not read.is_unmapped and not read.mate_is_unmapped:
                            if abs(read.template_length) < 2000:  # Reasonable insert size
                                insert_sizes.append(abs(read.template_length))
                    
                    if read.is_duplicate:
                        stats.duplicates += 1
                    
                    if read.is_paired and not read.is_proper_pair and not read.mate_is_unmapped:
                        stats.singleton_reads += 1
                
                # Calculate quality statistics
                if mapping_qualities:
                    stats.mapping_quality_mean = np.mean(mapping_qualities)
                    stats.mapping_quality_std = np.std(mapping_qualities)
                
                # Calculate insert size statistics
                if insert_sizes:
                    stats.insert_size_mean = np.mean(insert_sizes)
                    stats.insert_size_std = np.std(insert_sizes)
        
        except Exception as e:
            logging.warning(f"Failed to calculate alignment statistics: {e}")
        
        return stats
    
    async def _calculate_coverage_statistics(self, bam_file: str, reference: Optional[str] = None) -> Optional[CoverageStatistics]:
        """Calculate coverage statistics from BAM file"""
        
        if not PYSAM_AVAILABLE:
            logging.warning("Pysam not available. Cannot calculate coverage statistics.")
            return None
        
        try:
            with pysam.AlignmentFile(bam_file, "rb") as bamfile:
                # Get reference information
                ref_lengths = dict(zip(bamfile.references, bamfile.lengths))
                
                # Calculate coverage for each chromosome
                total_bases = sum(ref_lengths.values())
                coverage_array = np.zeros(total_bases)
                
                # This is simplified - in practice you'd want to use pysam.coverage()
                # or samtools depth for large genomes
                base_idx = 0
                for ref_name, ref_length in ref_lengths.items():
                    # Get coverage for this chromosome
                    for pileup_column in bamfile.pileup(ref_name, stepper='samtools'):
                        if pileup_column.reference_pos < ref_length:
                            coverage_array[base_idx + pileup_column.reference_pos] = pileup_column.nsegments
                    
                    base_idx += ref_length
                
                # Calculate statistics
                mean_cov = np.mean(coverage_array)
                median_cov = np.median(coverage_array)
                std_cov = np.std(coverage_array)
                
                # Calculate coverage thresholds
                cov_1x = np.sum(coverage_array >= 1) / len(coverage_array)
                cov_5x = np.sum(coverage_array >= 5) / len(coverage_array)
                cov_10x = np.sum(coverage_array >= 10) / len(coverage_array)
                cov_20x = np.sum(coverage_array >= 20) / len(coverage_array)
                
                return CoverageStatistics(
                    mean_coverage=mean_cov,
                    median_coverage=median_cov,
                    std_coverage=std_cov,
                    coverage_1x=cov_1x,
                    coverage_5x=cov_5x,
                    coverage_10x=cov_10x,
                    coverage_20x=cov_20x,
                    coverage_distribution=coverage_array.tolist()[:1000]  # Limit size for storage
                )
        
        except Exception as e:
            logging.warning(f"Failed to calculate coverage statistics: {e}")
            return None


# =================== RNA-seq Alignment Tool ===================

class RNASeqAligner(ExternalBioinformaticsTool):
    """RNA-seq specific alignment using STAR or HISAT2"""
    
    def __init__(self, algorithm: AlignmentAlgorithm = AlignmentAlgorithm.STAR):
        self.algorithm = algorithm
        
        # Configure based on algorithm
        if algorithm == AlignmentAlgorithm.STAR:
            executable = "STAR"
            tool_name = "star_aligner"
        elif algorithm == AlignmentAlgorithm.HISAT2:
            executable = "hisat2"
            tool_name = "hisat2_aligner"
        else:
            executable = "STAR"
            tool_name = "rnaseq_aligner"
        
        config = ToolConfiguration(
            executable_path=executable,
            default_parameters={
                "runThreadN": 8,
                "outSAMtype": "BAM SortedByCoordinate"
            }
        )
        
        super().__init__(
            name=tool_name,
            description=f"RNA-seq alignment using {algorithm.value}",
            supported_data_types=[DataType.FASTQ_READS],
            tool_type=ExternalToolType.ALIGNMENT,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "genome_index": {"type": "string", "description": "Path to genome index directory"},
                "fastq_files": {"type": "array", "items": {"type": "string"}, "description": "FASTQ input files"},
                "output_prefix": {"type": "string", "description": "Output file prefix"},
                "threads": {"type": "integer", "default": 8},
                "annotation_gtf": {"type": "string", "description": "GTF annotation file"},
                "two_pass_mode": {"type": "boolean", "default": False},
                "quantification_mode": {"type": "boolean", "default": True},
                "novel_junctions": {"type": "boolean", "default": True}
            },
            "required": ["genome_index", "fastq_files", "output_prefix"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute RNA-seq alignment"""
        
        try:
            # Validate inputs
            genome_index = params["genome_index"]
            fastq_files = params["fastq_files"]
            output_prefix = params["output_prefix"]
            
            if not Path(genome_index).exists():
                return BioToolResult(success=False, error=f"Genome index not found: {genome_index}")
            
            for fastq_file in fastq_files:
                if not Path(fastq_file).exists():
                    return BioToolResult(success=False, error=f"FASTQ file not found: {fastq_file}")
            
            # Create output directory
            Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
            
            # Run RNA-seq alignment
            if self.algorithm == AlignmentAlgorithm.STAR:
                result = await self._run_star(params)
            elif self.algorithm == AlignmentAlgorithm.HISAT2:
                result = await self._run_hisat2(params)
            else:
                return BioToolResult(success=False, error=f"Algorithm {self.algorithm} not implemented")
            
            if not result.success:
                return result
            
            # Parse alignment statistics
            alignment_stats = await self._parse_rnaseq_statistics(params)
            
            # Generate splice junction analysis
            junction_stats = await self._analyze_splice_junctions(params)
            
            return BioToolResult(
                success=True,
                output=f"RNA-seq alignment completed: {output_prefix}",
                metadata={
                    "output_prefix": output_prefix,
                    "algorithm": self.algorithm.value,
                    "alignment_statistics": alignment_stats,
                    "junction_statistics": junction_stats,
                    "parameters": params
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"RNA-seq alignment failed: {str(e)}"
            )
    
    async def _run_star(self, params: Dict[str, Any]) -> BioToolResult:
        """Run STAR RNA-seq aligner"""
        
        command = [
            "STAR",
            "--runMode", "alignReads",
            "--genomeDir", params["genome_index"],
            "--readFilesIn"
        ]
        
        # Add FASTQ files
        command.extend(params["fastq_files"])
        
        # Add other parameters
        command.extend([
            "--runThreadN", str(params.get("threads", 8)),
            "--outFileNamePrefix", params["output_prefix"],
            "--outSAMtype", "BAM", "SortedByCoordinate",
            "--outSAMstrandField", "intronMotif"
        ])
        
        # Handle compressed files
        if any(f.endswith('.gz') for f in params["fastq_files"]):
            command.extend(["--readFilesCommand", "zcat"])
        
        # Two-pass mode for novel junction discovery
        if params.get("two_pass_mode", False):
            command.extend(["--twopassMode", "Basic"])
        
        # Add GTF annotation if provided
        if "annotation_gtf" in params:
            command.extend(["--sjdbGTFfile", params["annotation_gtf"]])
        
        # Quantification mode
        if params.get("quantification_mode", True):
            command.extend(["--quantMode", "GeneCounts"])
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return BioToolResult(success=True, output="STAR alignment completed")
        else:
            return BioToolResult(success=False, error=f"STAR failed: {stderr.decode()}")
    
    async def _run_hisat2(self, params: Dict[str, Any]) -> BioToolResult:
        """Run HISAT2 RNA-seq aligner"""
        
        output_sam = f"{params['output_prefix']}Aligned.out.sam"
        
        command = [
            "hisat2",
            "-p", str(params.get("threads", 8)),
            "-x", params["genome_index"],
            "-S", output_sam
        ]
        
        # Handle paired-end vs single-end
        fastq_files = params["fastq_files"]
        if len(fastq_files) == 1:
            command.extend(["-U", fastq_files[0]])
        elif len(fastq_files) == 2:
            command.extend(["-1", fastq_files[0], "-2", fastq_files[1]])
        else:
            return BioToolResult(success=False, error="HISAT2 supports 1 or 2 FASTQ files only")
        
        # Add splice site database if available
        if "annotation_gtf" in params:
            # In practice, you'd need to extract splice sites from GTF first
            pass
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            # Convert SAM to sorted BAM
            if PYSAM_AVAILABLE:
                output_bam = f"{params['output_prefix']}Aligned.sortedByCoord.out.bam"
                pysam.sort("-o", output_bam, output_sam)
                pysam.index(output_bam)
                
                # Remove SAM file
                Path(output_sam).unlink()
            
            return BioToolResult(success=True, output="HISAT2 alignment completed")
        else:
            return BioToolResult(success=False, error=f"HISAT2 failed: {stderr.decode()}")
    
    async def _parse_rnaseq_statistics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse RNA-seq specific alignment statistics"""
        
        stats = {}
        
        if self.algorithm == AlignmentAlgorithm.STAR:
            # Parse STAR log file
            log_file = f"{params['output_prefix']}Log.final.out"
            
            if Path(log_file).exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if 'Number of input reads' in line:
                            stats['input_reads'] = int(line.split('|')[-1].strip())
                        elif 'Uniquely mapped reads number' in line:
                            stats['uniquely_mapped'] = int(line.split('|')[-1].strip())
                        elif 'Uniquely mapped reads %' in line:
                            stats['uniquely_mapped_pct'] = float(line.split('|')[-1].strip().rstrip('%'))
                        elif 'Number of reads mapped to multiple loci' in line:
                            stats['multi_mapped'] = int(line.split('|')[-1].strip())
                        elif 'Number of reads unmapped' in line:
                            stats['unmapped'] = int(line.split('|')[-1].strip())
                        elif 'Number of splices: Total' in line:
                            stats['total_splices'] = int(line.split('|')[-1].strip())
                        elif 'Number of splices: Annotated (sjdb)' in line:
                            stats['annotated_splices'] = int(line.split('|')[-1].strip())
                        elif 'Number of splices: GT/AG' in line:
                            stats['canonical_splices'] = int(line.split('|')[-1].strip())
        
        return stats
    
    async def _analyze_splice_junctions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze splice junction characteristics"""
        
        junction_stats = {}
        
        if self.algorithm == AlignmentAlgorithm.STAR:
            # Parse STAR splice junction file
            sj_file = f"{params['output_prefix']}SJ.out.tab"
            
            if Path(sj_file).exists():
                junctions = []
                with open(sj_file, 'r') as f:
                    for line in f:
                        fields = line.strip().split('\t')
                        if len(fields) >= 9:
                            chrom, start, end, strand, motif, annotated, reads, multi_reads, max_overhang = fields[:9]
                            
                            junctions.append({
                                'chromosome': chrom,
                                'start': int(start),
                                'end': int(end),
                                'strand': int(strand),
                                'motif': int(motif),
                                'annotated': int(annotated),
                                'reads': int(reads),
                                'multimap_reads': int(multi_reads),
                                'max_overhang': int(max_overhang)
                            })
                
                if junctions:
                    # Calculate junction statistics
                    junction_stats['total_junctions'] = len(junctions)
                    junction_stats['annotated_junctions'] = sum(1 for j in junctions if j['annotated'] > 0)
                    junction_stats['novel_junctions'] = sum(1 for j in junctions if j['annotated'] == 0)
                    
                    # Motif distribution
                    motif_counts = defaultdict(int)
                    for j in junctions:
                        motif_counts[j['motif']] += 1
                    
                    junction_stats['motif_distribution'] = dict(motif_counts)
                    
                    # Read support distribution
                    read_supports = [j['reads'] for j in junctions]
                    junction_stats['read_support_stats'] = {
                        'mean': np.mean(read_supports),
                        'median': np.median(read_supports),
                        'max': np.max(read_supports)
                    }
        
        return junction_stats


# =================== Multiple Sequence Alignment Tool ===================

class MultipleSequenceAligner(ExternalBioinformaticsTool):
    """Multiple sequence alignment using MAFFT, MUSCLE, or ClustalW"""
    
    def __init__(self, algorithm: AlignmentAlgorithm = AlignmentAlgorithm.MAFFT):
        self.algorithm = algorithm
        
        # Configure based on algorithm
        if algorithm == AlignmentAlgorithm.MAFFT:
            executable = "mafft"
            tool_name = "mafft_aligner"
        elif algorithm == AlignmentAlgorithm.MUSCLE:
            executable = "muscle"
            tool_name = "muscle_aligner"
        elif algorithm == AlignmentAlgorithm.CLUSTALW:
            executable = "clustalw"
            tool_name = "clustalw_aligner"
        else:
            executable = "mafft"
            tool_name = "msa_aligner"
        
        config = ToolConfiguration(
            executable_path=executable,
            default_parameters={
                "threads": 4
            }
        )
        
        super().__init__(
            name=tool_name,
            description=f"Multiple sequence alignment using {algorithm.value}",
            supported_data_types=[DataType.GENOMIC_SEQUENCE, DataType.PROTEIN_SEQUENCE],
            tool_type=ExternalToolType.ALIGNMENT,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_fasta": {"type": "string", "description": "Input FASTA file with sequences"},
                "output_alignment": {"type": "string", "description": "Output alignment file"},
                "sequence_type": {"type": "string", "enum": ["dna", "protein"], "default": "dna"},
                "algorithm_mode": {"type": "string", "enum": ["auto", "fast", "accurate"], "default": "auto"},
                "threads": {"type": "integer", "default": 4},
                "gap_open_penalty": {"type": "number", "default": -1.0},
                "gap_extension_penalty": {"type": "number", "default": -0.1}
            },
            "required": ["input_fasta", "output_alignment"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute multiple sequence alignment"""
        
        try:
            # Validate inputs
            input_fasta = params["input_fasta"]
            output_alignment = params["output_alignment"]
            
            if not Path(input_fasta).exists():
                return BioToolResult(success=False, error=f"Input FASTA not found: {input_fasta}")
            
            # Create output directory
            Path(output_alignment).parent.mkdir(parents=True, exist_ok=True)
            
            # Count input sequences
            sequence_count = await self._count_sequences(input_fasta)
            
            if sequence_count < 2:
                return BioToolResult(success=False, error="Need at least 2 sequences for alignment")
            
            # Run multiple sequence alignment
            if self.algorithm == AlignmentAlgorithm.MAFFT:
                result = await self._run_mafft(params)
            elif self.algorithm == AlignmentAlgorithm.MUSCLE:
                result = await self._run_muscle(params)
            elif self.algorithm == AlignmentAlgorithm.CLUSTALW:
                result = await self._run_clustalw(params)
            else:
                return BioToolResult(success=False, error=f"Algorithm {self.algorithm} not implemented")
            
            if not result.success:
                return result
            
            # Analyze alignment quality
            alignment_stats = await self._analyze_alignment_quality(output_alignment)
            
            return BioToolResult(
                success=True,
                output=f"Multiple sequence alignment completed: {output_alignment}",
                metadata={
                    "output_alignment": output_alignment,
                    "algorithm": self.algorithm.value,
                    "sequence_count": sequence_count,
                    "alignment_statistics": alignment_stats,
                    "parameters": params
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Multiple sequence alignment failed: {str(e)}"
            )
    
    async def _count_sequences(self, fasta_file: str) -> int:
        """Count sequences in FASTA file"""
        
        count = 0
        with open(fasta_file, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    count += 1
        return count
    
    async def _run_mafft(self, params: Dict[str, Any]) -> BioToolResult:
        """Run MAFFT multiple sequence alignment"""
        
        command = ["mafft"]
        
        # Algorithm mode
        mode = params.get("algorithm_mode", "auto")
        if mode == "fast":
            command.append("--retree")
            command.append("1")
        elif mode == "accurate":
            command.append("--maxiterate")
            command.append("1000")
        
        # Threading
        command.extend(["--thread", str(params.get("threads", 4))])
        
        # Sequence type
        if params.get("sequence_type", "dna") == "protein":
            command.append("--amino")
        
        # Input file
        command.append(params["input_fasta"])
        
        # Run MAFFT
        with open(params["output_alignment"], 'w') as output_file:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=output_file,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
        
        if process.returncode == 0:
            return BioToolResult(success=True, output="MAFFT alignment completed")
        else:
            return BioToolResult(success=False, error=f"MAFFT failed: {stderr.decode()}")
    
    async def _run_muscle(self, params: Dict[str, Any]) -> BioToolResult:
        """Run MUSCLE multiple sequence alignment"""
        
        command = [
            "muscle",
            "-in", params["input_fasta"],
            "-out", params["output_alignment"]
        ]
        
        # Algorithm mode
        mode = params.get("algorithm_mode", "auto")
        if mode == "fast":
            command.extend(["-maxiters", "2"])
        elif mode == "accurate":
            command.extend(["-maxiters", "16"])
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return BioToolResult(success=True, output="MUSCLE alignment completed")
        else:
            return BioToolResult(success=False, error=f"MUSCLE failed: {stderr.decode()}")
    
    async def _run_clustalw(self, params: Dict[str, Any]) -> BioToolResult:
        """Run ClustalW multiple sequence alignment"""
        
        command = [
            "clustalw",
            f"-INFILE={params['input_fasta']}",
            f"-OUTFILE={params['output_alignment']}",
            "-OUTPUT=FASTA"
        ]
        
        # Sequence type
        if params.get("sequence_type", "dna") == "protein":
            command.append("-TYPE=PROTEIN")
        else:
            command.append("-TYPE=DNA")
        
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            return BioToolResult(success=True, output="ClustalW alignment completed")
        else:
            return BioToolResult(success=False, error=f"ClustalW failed: {stderr.decode()}")
    
    async def _analyze_alignment_quality(self, alignment_file: str) -> Dict[str, Any]:
        """Analyze quality of multiple sequence alignment"""
        
        stats = {
            "alignment_length": 0,
            "sequence_count": 0,
            "gap_percentage": 0.0,
            "conservation_score": 0.0,
            "identity_percentage": 0.0
        }
        
        if not BIOPYTHON_AVAILABLE:
            logging.warning("Biopython not available. Cannot analyze alignment quality.")
            return stats
        
        try:
            # Read alignment
            alignment = AlignIO.read(alignment_file, "fasta")
            
            stats["sequence_count"] = len(alignment)
            stats["alignment_length"] = alignment.get_alignment_length()
            
            # Calculate gap percentage
            total_positions = len(alignment) * alignment.get_alignment_length()
            gap_count = sum(str(record.seq).count('-') for record in alignment)
            stats["gap_percentage"] = (gap_count / total_positions) * 100
            
            # Calculate conservation and identity
            conservation_scores = []
            identity_scores = []
            
            for i in range(alignment.get_alignment_length()):
                column = alignment[:, i]
                
                # Conservation: fraction of most common character
                char_counts = defaultdict(int)
                for char in column:
                    if char != '-':
                        char_counts[char] += 1
                
                if char_counts:
                    max_count = max(char_counts.values())
                    non_gap_count = sum(char_counts.values())
                    conservation = max_count / non_gap_count if non_gap_count > 0 else 0
                    conservation_scores.append(conservation)
                
                # Identity: all characters the same (excluding gaps)
                non_gap_chars = [c for c in column if c != '-']
                if len(non_gap_chars) > 1:
                    identity = 1.0 if len(set(non_gap_chars)) == 1 else 0.0
                    identity_scores.append(identity)
            
            if conservation_scores:
                stats["conservation_score"] = np.mean(conservation_scores)
            
            if identity_scores:
                stats["identity_percentage"] = np.mean(identity_scores) * 100
        
        except Exception as e:
            logging.warning(f"Failed to analyze alignment quality: {e}")
        
        return stats


# =================== Alignment Analysis Pipeline ===================

class AlignmentPipeline:
    """Complete alignment analysis pipeline"""
    
    def __init__(self):
        self.short_read_aligner = ShortReadAligner(AlignmentAlgorithm.BWA_MEM)
        self.rnaseq_aligner = RNASeqAligner(AlignmentAlgorithm.STAR)
        self.msa_aligner = MultipleSequenceAligner(AlignmentAlgorithm.MAFFT)
    
    async def run_short_read_pipeline(self, reference_index: str, fastq_files: List[str], 
                                    output_dir: str) -> Dict[str, Any]:
        """Run short-read alignment pipeline"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare parameters
        alignment_params = {
            "reference_index": reference_index,
            "fastq_files": fastq_files,
            "output_sam": str(output_path / "aligned.sam"),
            "threads": 8,
            "sort_output": True,
            "mark_duplicates": True
        }
        
        # Run alignment
        result = await self.short_read_aligner.execute(alignment_params, [])
        
        return {
            "pipeline_type": "short_read_alignment",
            "success": result.success,
            "output": result.output,
            "metadata": result.metadata,
            "error": result.error if not result.success else None
        }
    
    async def run_rnaseq_pipeline(self, genome_index: str, fastq_files: List[str], 
                                output_dir: str, annotation_gtf: Optional[str] = None) -> Dict[str, Any]:
        """Run RNA-seq alignment pipeline"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare parameters
        alignment_params = {
            "genome_index": genome_index,
            "fastq_files": fastq_files,
            "output_prefix": str(output_path / "rnaseq_"),
            "threads": 8,
            "two_pass_mode": True,
            "quantification_mode": True
        }
        
        if annotation_gtf:
            alignment_params["annotation_gtf"] = annotation_gtf
        
        # Run alignment
        result = await self.rnaseq_aligner.execute(alignment_params, [])
        
        return {
            "pipeline_type": "rnaseq_alignment",
            "success": result.success,
            "output": result.output,
            "metadata": result.metadata,
            "error": result.error if not result.success else None
        }


# =================== Example Usage ===================

async def example_alignment_analysis():
    """Example of comprehensive alignment analysis"""
    
    print("Alignment and Mapping Analysis Example")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = AlignmentPipeline()
    
    # Example parameters
    reference_index = "genome_index/genome"
    rnaseq_index = "genome_index"
    fastq_files = ["sample_R1.fastq.gz", "sample_R2.fastq.gz"]
    annotation_gtf = "annotations.gtf"
    
    print("Available Alignment Methods:")
    print("1. Short-read alignment (BWA MEM, Bowtie2, minimap2)")
    print("2. RNA-seq alignment (STAR, HISAT2)")
    print("3. Long-read alignment (minimap2, NGMLR)")
    print("4. Multiple sequence alignment (MAFFT, MUSCLE, ClustalW)")
    
    # This would work with real data:
    # short_read_result = await pipeline.run_short_read_pipeline(
    #     reference_index, fastq_files, "short_read_output"
    # )
    
    # rnaseq_result = await pipeline.run_rnaseq_pipeline(
    #     rnaseq_index, fastq_files, "rnaseq_output", annotation_gtf
    # )
    
    print("\nExample completed (requires real sequencing data and indices to run)")


if __name__ == "__main__":
    asyncio.run(example_alignment_analysis())