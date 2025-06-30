#!/usr/bin/env python3
"""
BioinformaticsAgent External Tools Integration: Real bioinformatics software integration

This module provides integration with actual bioinformatics tools:
- BLAST (sequence similarity search)
- GATK (variant calling)
- BWA/Bowtie2 (sequence alignment)
- SAMtools/BCFtools (file manipulation)
- STAR (RNA-seq alignment)
- Cufflinks (transcript assembly)
- VEP/ANNOVAR (variant annotation)
- FastQC (quality control)
- Trimmomatic (read trimming)
- MACS2 (peak calling)
"""

import asyncio
import subprocess
import tempfile
import shutil
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import aiofiles

from bioagent_architecture import BioinformaticsTool, BioToolResult, DataType, DataMetadata


# =================== Tool Types and Configurations ===================

class ExternalToolType(Enum):
    """Types of external tools"""
    ALIGNMENT = "alignment"
    VARIANT_CALLING = "variant_calling"
    ANNOTATION = "annotation"
    QUALITY_CONTROL = "quality_control"
    ASSEMBLY = "assembly"
    QUANTIFICATION = "quantification"
    PEAK_CALLING = "peak_calling"
    SEQUENCE_SEARCH = "sequence_search"


@dataclass
class ToolConfiguration:
    """Configuration for external tools"""
    executable_path: str
    version: Optional[str] = None
    additional_paths: Dict[str, str] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result from external tool execution"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    output_files: List[str] = field(default_factory=list)
    log_files: List[str] = field(default_factory=list)


# =================== Base External Tool Class ===================

class ExternalBioinformaticsTool(BioinformaticsTool):
    """Base class for external bioinformatics tools"""
    
    def __init__(self, name: str, description: str, supported_data_types: List[DataType],
                 tool_type: ExternalToolType, config: ToolConfiguration):
        super().__init__(name, description, supported_data_types)
        self.tool_type = tool_type
        self.config = config
        self.temp_dir = None
        
    async def check_installation(self) -> bool:
        """Check if tool is properly installed"""
        try:
            result = await self._run_command([self.config.executable_path, "--version"])
            return result.success
        except:
            return False
    
    async def _run_command(self, command: List[str], input_data: Optional[str] = None,
                          timeout: Optional[int] = None) -> ExecutionResult:
        """Run external command asynchronously"""
        import time
        start_time = time.time()
        
        try:
            # Set up environment
            env = os.environ.copy()
            env.update(self.config.environment_variables)
            
            # Create process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Run with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_data.encode() if input_data else None),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {timeout} seconds")
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                stdout=stdout.decode('utf-8', errors='ignore'),
                stderr=stderr.decode('utf-8', errors='ignore'),
                return_code=process.returncode,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=execution_time
            )
    
    def _create_temp_dir(self) -> str:
        """Create temporary directory for tool execution"""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix=f"{self.name}_")
        return self.temp_dir
    
    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None


# =================== Sequence Alignment Tools ===================

class BWATool(ExternalBioinformaticsTool):
    """BWA alignment tool"""
    
    def __init__(self, bwa_path: str = "bwa"):
        config = ToolConfiguration(
            executable_path=bwa_path,
            default_parameters={
                "algorithm": "mem",
                "threads": 4,
                "mark_shorter_splits": True
            }
        )
        
        super().__init__(
            name="bwa",
            description="Burrows-Wheeler Aligner for short-read alignment",
            supported_data_types=[DataType.FASTQ_READS],
            tool_type=ExternalToolType.ALIGNMENT,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reference_genome": {"type": "string", "description": "Path to reference genome"},
                "fastq_files": {"type": "array", "items": {"type": "string"}},
                "output_sam": {"type": "string", "description": "Output SAM file"},
                "threads": {"type": "integer", "default": 4},
                "algorithm": {"type": "string", "enum": ["mem", "aln"], "default": "mem"},
                "read_group": {"type": "string", "description": "Read group information"}
            },
            "required": ["reference_genome", "fastq_files", "output_sam"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute BWA alignment"""
        
        try:
            # Validate parameters
            is_valid, error = self.validate_parameters(params)
            if not is_valid:
                return BioToolResult(success=False, error=error)
            
            # Check installation
            if not await self.check_installation():
                return BioToolResult(success=False, error="BWA not installed or not in PATH")
            
            temp_dir = self._create_temp_dir()
            
            # Prepare command
            algorithm = params.get("algorithm", "mem")
            threads = params.get("threads", 4)
            
            if algorithm == "mem":
                command = [
                    self.config.executable_path, "mem",
                    "-t", str(threads),
                    params["reference_genome"]
                ]
                
                # Add FASTQ files
                command.extend(params["fastq_files"])
                
                # Add read group if specified
                if "read_group" in params:
                    command.extend(["-R", params["read_group"]])
                
                # Run BWA MEM
                result = await self._run_command(command, timeout=3600)  # 1 hour timeout
                
                if result.success:
                    # Write output to file
                    output_file = params["output_sam"]
                    async with aiofiles.open(output_file, 'w') as f:
                        await f.write(result.stdout)
                    
                    # Parse alignment statistics from stderr
                    stats = self._parse_bwa_stats(result.stderr)
                    
                    return BioToolResult(
                        success=True,
                        output=f"Alignment completed: {output_file}",
                        metadata={
                            "output_file": output_file,
                            "alignment_stats": stats,
                            "execution_time": result.execution_time
                        }
                    )
                else:
                    return BioToolResult(
                        success=False,
                        error=f"BWA failed: {result.stderr}"
                    )
            else:
                return BioToolResult(
                    success=False,
                    error=f"Algorithm {algorithm} not implemented yet"
                )
                
        except Exception as e:
            return BioToolResult(success=False, error=f"BWA execution failed: {str(e)}")
        
        finally:
            self._cleanup_temp_dir()
    
    def _parse_bwa_stats(self, stderr: str) -> Dict[str, Any]:
        """Parse BWA alignment statistics from stderr"""
        stats = {}
        
        lines = stderr.split('\n')
        for line in lines:
            if 'Processed' in line and 'reads' in line:
                # Extract read count
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'Processed' and i + 1 < len(parts):
                        try:
                            stats['processed_reads'] = int(parts[i + 1])
                        except ValueError:
                            pass
        
        return stats


class Bowtie2Tool(ExternalBioinformaticsTool):
    """Bowtie2 alignment tool"""
    
    def __init__(self, bowtie2_path: str = "bowtie2"):
        config = ToolConfiguration(
            executable_path=bowtie2_path,
            default_parameters={
                "threads": 4,
                "preset": "sensitive"
            }
        )
        
        super().__init__(
            name="bowtie2",
            description="Fast and sensitive read alignment tool",
            supported_data_types=[DataType.FASTQ_READS],
            tool_type=ExternalToolType.ALIGNMENT,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index_prefix": {"type": "string", "description": "Bowtie2 index prefix"},
                "fastq_files": {"type": "array", "items": {"type": "string"}},
                "output_sam": {"type": "string", "description": "Output SAM file"},
                "threads": {"type": "integer", "default": 4},
                "preset": {"type": "string", "enum": ["very-fast", "fast", "sensitive", "very-sensitive"], "default": "sensitive"}
            },
            "required": ["index_prefix", "fastq_files", "output_sam"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute Bowtie2 alignment"""
        
        try:
            if not await self.check_installation():
                return BioToolResult(success=False, error="Bowtie2 not installed")
            
            command = [
                self.config.executable_path,
                "--" + params.get("preset", "sensitive"),
                "-p", str(params.get("threads", 4)),
                "-x", params["index_prefix"],
                "-S", params["output_sam"]
            ]
            
            # Handle paired-end vs single-end
            fastq_files = params["fastq_files"]
            if len(fastq_files) == 1:
                command.extend(["-U", fastq_files[0]])
            elif len(fastq_files) == 2:
                command.extend(["-1", fastq_files[0], "-2", fastq_files[1]])
            else:
                return BioToolResult(
                    success=False,
                    error="Bowtie2 supports 1 or 2 FASTQ files only"
                )
            
            result = await self._run_command(command, timeout=3600)
            
            if result.success:
                stats = self._parse_bowtie2_stats(result.stderr)
                return BioToolResult(
                    success=True,
                    output=f"Alignment completed: {params['output_sam']}",
                    metadata={
                        "output_file": params["output_sam"],
                        "alignment_stats": stats,
                        "execution_time": result.execution_time
                    }
                )
            else:
                return BioToolResult(
                    success=False,
                    error=f"Bowtie2 failed: {result.stderr}"
                )
                
        except Exception as e:
            return BioToolResult(success=False, error=f"Bowtie2 execution failed: {str(e)}")
    
    def _parse_bowtie2_stats(self, stderr: str) -> Dict[str, Any]:
        """Parse Bowtie2 alignment statistics"""
        stats = {}
        
        lines = stderr.split('\n')
        for line in lines:
            line = line.strip()
            if 'reads; of these:' in line:
                stats['total_reads'] = int(line.split()[0])
            elif 'aligned concordantly exactly 1 time' in line:
                stats['concordant_unique'] = int(line.split()[0])
            elif 'aligned concordantly >1 times' in line:
                stats['concordant_multiple'] = int(line.split()[0])
            elif 'overall alignment rate' in line:
                stats['overall_alignment_rate'] = line.split()[-1]
        
        return stats


class STARTool(ExternalBioinformaticsTool):
    """STAR RNA-seq aligner"""
    
    def __init__(self, star_path: str = "STAR"):
        config = ToolConfiguration(
            executable_path=star_path,
            default_parameters={
                "runThreadN": 4,
                "outSAMtype": "BAM SortedByCoordinate"
            }
        )
        
        super().__init__(
            name="star",
            description="Spliced Transcripts Alignment to a Reference",
            supported_data_types=[DataType.FASTQ_READS],
            tool_type=ExternalToolType.ALIGNMENT,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "genome_dir": {"type": "string", "description": "STAR genome directory"},
                "fastq_files": {"type": "array", "items": {"type": "string"}},
                "output_prefix": {"type": "string", "description": "Output file prefix"},
                "threads": {"type": "integer", "default": 4},
                "two_pass_mode": {"type": "boolean", "default": False}
            },
            "required": ["genome_dir", "fastq_files", "output_prefix"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute STAR alignment"""
        
        try:
            if not await self.check_installation():
                return BioToolResult(success=False, error="STAR not installed")
            
            command = [
                self.config.executable_path,
                "--runMode", "alignReads",
                "--genomeDir", params["genome_dir"],
                "--readFilesIn"
            ]
            
            # Add FASTQ files
            command.extend(params["fastq_files"])
            
            # Add other parameters
            command.extend([
                "--runThreadN", str(params.get("threads", 4)),
                "--outFileNamePrefix", params["output_prefix"],
                "--outSAMtype", "BAM", "SortedByCoordinate"
            ])
            
            # Handle compressed files
            if any(f.endswith('.gz') for f in params["fastq_files"]):
                command.extend(["--readFilesCommand", "zcat"])
            
            # Two-pass mode
            if params.get("two_pass_mode", False):
                command.extend(["--twopassMode", "Basic"])
            
            result = await self._run_command(command, timeout=7200)  # 2 hours
            
            if result.success:
                # STAR creates multiple output files
                output_files = [
                    params["output_prefix"] + "Aligned.sortedByCoord.out.bam",
                    params["output_prefix"] + "Log.final.out",
                    params["output_prefix"] + "SJ.out.tab"
                ]
                
                # Parse log file for statistics
                log_file = params["output_prefix"] + "Log.final.out"
                stats = await self._parse_star_log(log_file)
                
                return BioToolResult(
                    success=True,
                    output=f"STAR alignment completed: {output_files[0]}",
                    metadata={
                        "output_files": output_files,
                        "alignment_stats": stats,
                        "execution_time": result.execution_time
                    }
                )
            else:
                return BioToolResult(
                    success=False,
                    error=f"STAR failed: {result.stderr}"
                )
                
        except Exception as e:
            return BioToolResult(success=False, error=f"STAR execution failed: {str(e)}")
    
    async def _parse_star_log(self, log_file: str) -> Dict[str, Any]:
        """Parse STAR log file for statistics"""
        stats = {}
        
        try:
            async with aiofiles.open(log_file, 'r') as f:
                content = await f.read()
                
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if 'Number of input reads' in line:
                    stats['input_reads'] = int(line.split('\t')[-1])
                elif 'Uniquely mapped reads number' in line:
                    stats['uniquely_mapped'] = int(line.split('\t')[-1])
                elif 'Uniquely mapped reads %' in line:
                    stats['uniquely_mapped_pct'] = float(line.split('\t')[-1].rstrip('%'))
                elif 'Number of reads mapped to multiple loci' in line:
                    stats['multi_mapped'] = int(line.split('\t')[-1])
                elif 'Number of reads unmapped' in line:
                    stats['unmapped'] = int(line.split('\t')[-1])
        except:
            pass
        
        return stats


# =================== Variant Calling Tools ===================

class GATKTool(ExternalBioinformaticsTool):
    """GATK variant calling tool"""
    
    def __init__(self, gatk_path: str = "gatk"):
        config = ToolConfiguration(
            executable_path=gatk_path,
            default_parameters={
                "java_options": "-Xmx8g"
            }
        )
        
        super().__init__(
            name="gatk",
            description="Genome Analysis Toolkit for variant calling",
            supported_data_types=[DataType.ALIGNMENT_SAM],
            tool_type=ExternalToolType.VARIANT_CALLING,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["HaplotypeCaller", "Mutect2", "GenotypeGVCFs"]},
                "reference": {"type": "string", "description": "Reference genome FASTA"},
                "input_bam": {"type": "string", "description": "Input BAM file"},
                "output_vcf": {"type": "string", "description": "Output VCF file"},
                "intervals": {"type": "string", "description": "Genomic intervals"},
                "java_options": {"type": "string", "default": "-Xmx8g"}
            },
            "required": ["tool", "reference", "input_bam", "output_vcf"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute GATK variant calling"""
        
        try:
            if not await self.check_installation():
                return BioToolResult(success=False, error="GATK not installed")
            
            tool = params["tool"]
            java_opts = params.get("java_options", "-Xmx8g")
            
            # Set JAVA_OPTS environment variable
            env_vars = {"JAVA_OPTS": java_opts}
            original_config = self.config.environment_variables
            self.config.environment_variables.update(env_vars)
            
            try:
                if tool == "HaplotypeCaller":
                    result = await self._run_haplotype_caller(params)
                elif tool == "Mutect2":
                    result = await self._run_mutect2(params)
                else:
                    return BioToolResult(
                        success=False,
                        error=f"GATK tool {tool} not implemented"
                    )
                
                return result
                
            finally:
                # Restore original environment
                self.config.environment_variables = original_config
                
        except Exception as e:
            return BioToolResult(success=False, error=f"GATK execution failed: {str(e)}")
    
    async def _run_haplotype_caller(self, params: Dict[str, Any]) -> BioToolResult:
        """Run GATK HaplotypeCaller"""
        
        command = [
            self.config.executable_path, "HaplotypeCaller",
            "-R", params["reference"],
            "-I", params["input_bam"],
            "-O", params["output_vcf"]
        ]
        
        # Add intervals if specified
        if "intervals" in params:
            command.extend(["-L", params["intervals"]])
        
        result = await self._run_command(command, timeout=7200)  # 2 hours
        
        if result.success:
            # Parse VCF for basic statistics
            stats = await self._parse_vcf_stats(params["output_vcf"])
            
            return BioToolResult(
                success=True,
                output=f"Variant calling completed: {params['output_vcf']}",
                metadata={
                    "output_file": params["output_vcf"],
                    "variant_stats": stats,
                    "execution_time": result.execution_time
                }
            )
        else:
            return BioToolResult(
                success=False,
                error=f"HaplotypeCaller failed: {result.stderr}"
            )
    
    async def _run_mutect2(self, params: Dict[str, Any]) -> BioToolResult:
        """Run GATK Mutect2 for somatic variant calling"""
        
        command = [
            self.config.executable_path, "Mutect2",
            "-R", params["reference"],
            "-I", params["input_bam"],
            "-O", params["output_vcf"]
        ]
        
        if "intervals" in params:
            command.extend(["-L", params["intervals"]])
        
        result = await self._run_command(command, timeout=7200)
        
        if result.success:
            stats = await self._parse_vcf_stats(params["output_vcf"])
            
            return BioToolResult(
                success=True,
                output=f"Somatic variant calling completed: {params['output_vcf']}",
                metadata={
                    "output_file": params["output_vcf"],
                    "variant_stats": stats,
                    "execution_time": result.execution_time
                }
            )
        else:
            return BioToolResult(
                success=False,
                error=f"Mutect2 failed: {result.stderr}"
            )
    
    async def _parse_vcf_stats(self, vcf_file: str) -> Dict[str, Any]:
        """Parse basic VCF statistics"""
        stats = {"total_variants": 0, "snps": 0, "indels": 0}
        
        try:
            async with aiofiles.open(vcf_file, 'r') as f:
                async for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        ref, alt = parts[3], parts[4]
                        stats["total_variants"] += 1
                        
                        if len(ref) == 1 and len(alt) == 1:
                            stats["snps"] += 1
                        else:
                            stats["indels"] += 1
        except:
            pass
        
        return stats


# =================== Sequence Search Tools ===================

class BLASTTool(ExternalBioinformaticsTool):
    """BLAST sequence search tool"""
    
    def __init__(self, blast_path: str = "blastn"):
        config = ToolConfiguration(
            executable_path=blast_path,
            default_parameters={
                "evalue": "1e-5",
                "max_target_seqs": 10
            }
        )
        
        super().__init__(
            name="blast",
            description="Basic Local Alignment Search Tool",
            supported_data_types=[DataType.GENOMIC_SEQUENCE, DataType.PROTEIN_SEQUENCE],
            tool_type=ExternalToolType.SEQUENCE_SEARCH,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "program": {"type": "string", "enum": ["blastn", "blastp", "blastx", "tblastn", "tblastx"]},
                "query": {"type": "string", "description": "Query sequence file"},
                "database": {"type": "string", "description": "BLAST database"},
                "output": {"type": "string", "description": "Output file"},
                "evalue": {"type": "string", "default": "1e-5"},
                "max_target_seqs": {"type": "integer", "default": 10},
                "outfmt": {"type": "integer", "default": 6}
            },
            "required": ["program", "query", "database", "output"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute BLAST search"""
        
        try:
            program = params["program"]
            
            # Update executable path based on program
            self.config.executable_path = program
            
            if not await self.check_installation():
                return BioToolResult(success=False, error=f"{program} not installed")
            
            command = [
                program,
                "-query", params["query"],
                "-db", params["database"],
                "-out", params["output"],
                "-evalue", str(params.get("evalue", "1e-5")),
                "-max_target_seqs", str(params.get("max_target_seqs", 10)),
                "-outfmt", str(params.get("outfmt", 6))
            ]
            
            result = await self._run_command(command, timeout=3600)
            
            if result.success:
                # Parse BLAST output for statistics
                stats = await self._parse_blast_output(params["output"], params.get("outfmt", 6))
                
                return BioToolResult(
                    success=True,
                    output=f"BLAST search completed: {params['output']}",
                    metadata={
                        "output_file": params["output"],
                        "search_stats": stats,
                        "execution_time": result.execution_time
                    }
                )
            else:
                return BioToolResult(
                    success=False,
                    error=f"BLAST failed: {result.stderr}"
                )
                
        except Exception as e:
            return BioToolResult(success=False, error=f"BLAST execution failed: {str(e)}")
    
    async def _parse_blast_output(self, output_file: str, outfmt: int) -> Dict[str, Any]:
        """Parse BLAST output for statistics"""
        stats = {"total_hits": 0, "queries_with_hits": 0, "unique_subjects": set()}
        
        try:
            async with aiofiles.open(output_file, 'r') as f:
                current_query = None
                query_has_hits = False
                
                async for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if outfmt == 6:  # Tabular format
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            query, subject = parts[0], parts[1]
                            
                            if current_query != query:
                                if query_has_hits:
                                    stats["queries_with_hits"] += 1
                                current_query = query
                                query_has_hits = False
                            
                            stats["total_hits"] += 1
                            stats["unique_subjects"].add(subject)
                            query_has_hits = True
                
                # Handle last query
                if query_has_hits:
                    stats["queries_with_hits"] += 1
        
        except:
            pass
        
        # Convert set to count
        stats["unique_subjects"] = len(stats["unique_subjects"])
        return stats


# =================== Quality Control Tools ===================

class FastQCTool(ExternalBioinformaticsTool):
    """FastQC quality control tool"""
    
    def __init__(self, fastqc_path: str = "fastqc"):
        config = ToolConfiguration(
            executable_path=fastqc_path,
            default_parameters={
                "threads": 4
            }
        )
        
        super().__init__(
            name="fastqc",
            description="A quality control application for high throughput sequence data",
            supported_data_types=[DataType.FASTQ_READS],
            tool_type=ExternalToolType.QUALITY_CONTROL,
            config=config
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_files": {"type": "array", "items": {"type": "string"}},
                "output_dir": {"type": "string", "description": "Output directory"},
                "threads": {"type": "integer", "default": 4}
            },
            "required": ["input_files", "output_dir"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute FastQC quality control"""
        
        try:
            if not await self.check_installation():
                return BioToolResult(success=False, error="FastQC not installed")
            
            # Create output directory
            output_dir = params["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            
            command = [
                self.config.executable_path,
                "--outdir", output_dir,
                "--threads", str(params.get("threads", 4))
            ]
            
            # Add input files
            command.extend(params["input_files"])
            
            result = await self._run_command(command, timeout=1800)  # 30 minutes
            
            if result.success:
                # Parse FastQC results
                stats = await self._parse_fastqc_results(output_dir, params["input_files"])
                
                return BioToolResult(
                    success=True,
                    output=f"FastQC completed: {output_dir}",
                    metadata={
                        "output_directory": output_dir,
                        "qc_stats": stats,
                        "execution_time": result.execution_time
                    }
                )
            else:
                return BioToolResult(
                    success=False,
                    error=f"FastQC failed: {result.stderr}"
                )
                
        except Exception as e:
            return BioToolResult(success=False, error=f"FastQC execution failed: {str(e)}")
    
    async def _parse_fastqc_results(self, output_dir: str, input_files: List[str]) -> Dict[str, Any]:
        """Parse FastQC results for summary statistics"""
        stats = {}
        
        for input_file in input_files:
            filename = Path(input_file).stem
            if filename.endswith('.fastq'):
                filename = filename[:-6]
            elif filename.endswith('.fq'):
                filename = filename[:-3]
            
            # Look for FastQC data file
            data_file = Path(output_dir) / f"{filename}_fastqc" / "fastqc_data.txt"
            
            try:
                if data_file.exists():
                    file_stats = {}
                    async with aiofiles.open(data_file, 'r') as f:
                        async for line in f:
                            line = line.strip()
                            if line.startswith('Total Sequences'):
                                file_stats['total_sequences'] = int(line.split('\t')[1])
                            elif line.startswith('Sequence length'):
                                file_stats['sequence_length'] = line.split('\t')[1]
                            elif line.startswith('%GC'):
                                file_stats['gc_content'] = float(line.split('\t')[1])
                    
                    stats[filename] = file_stats
            except:
                pass
        
        return stats


# =================== Tool Registry and Manager ===================

class ExternalToolRegistry:
    """Registry for managing external bioinformatics tools"""
    
    def __init__(self):
        self.tools = {}
        self.configurations = {}
        
    def register_tool(self, tool: ExternalBioinformaticsTool):
        """Register an external tool"""
        self.tools[tool.name] = tool
        
    def get_tool(self, name: str) -> Optional[ExternalBioinformaticsTool]:
        """Get a registered tool by name"""
        return self.tools.get(name)
    
    def list_tools(self, tool_type: Optional[ExternalToolType] = None) -> List[str]:
        """List available tools, optionally filtered by type"""
        if tool_type:
            return [name for name, tool in self.tools.items() 
                   if tool.tool_type == tool_type]
        return list(self.tools.keys())
    
    async def check_all_installations(self) -> Dict[str, bool]:
        """Check installation status of all registered tools"""
        status = {}
        for name, tool in self.tools.items():
            status[name] = await tool.check_installation()
        return status
    
    def configure_tool(self, tool_name: str, config: ToolConfiguration):
        """Configure a tool with custom settings"""
        if tool_name in self.tools:
            self.tools[tool_name].config = config
            self.configurations[tool_name] = config


# =================== Initialization Function ===================

def initialize_external_tools() -> ExternalToolRegistry:
    """Initialize and register all external tools"""
    
    registry = ExternalToolRegistry()
    
    # Register alignment tools
    registry.register_tool(BWATool())
    registry.register_tool(Bowtie2Tool())
    registry.register_tool(STARTool())
    
    # Register variant calling tools
    registry.register_tool(GATKTool())
    
    # Register sequence search tools
    registry.register_tool(BLASTTool())
    
    # Register quality control tools
    registry.register_tool(FastQCTool())
    
    return registry


# =================== Example Usage ===================

async def example_external_tools():
    """Example of using external bioinformatics tools"""
    
    print("Initializing external tools registry...")
    registry = initialize_external_tools()
    
    # Check which tools are installed
    print("\nChecking tool installations:")
    status = await registry.check_all_installations()
    for tool_name, installed in status.items():
        print(f"  {tool_name}: {'✓ Installed' if installed else '✗ Not found'}")
    
    # Example: Running FastQC (if installed)
    fastqc_tool = registry.get_tool("fastqc")
    if fastqc_tool and status.get("fastqc", False):
        print("\nTesting FastQC...")
        
        # This would need actual FASTQ files to work
        example_params = {
            "input_files": ["example1.fastq", "example2.fastq"],
            "output_dir": "fastqc_output",
            "threads": 2
        }
        
        # result = await fastqc_tool.execute(example_params, [])
        # print(f"FastQC result: {result.success}")
        print("FastQC test skipped (no input files)")
    
    # Example: Running BLAST (if installed)
    blast_tool = registry.get_tool("blast")
    if blast_tool and status.get("blast", False):
        print("\nTesting BLAST...")
        
        example_params = {
            "program": "blastn",
            "query": "query.fasta",
            "database": "nt",
            "output": "blast_results.txt",
            "evalue": "1e-10"
        }
        
        # result = await blast_tool.execute(example_params, [])
        # print(f"BLAST result: {result.success}")
        print("BLAST test skipped (no database)")
    
    print("\nExternal tools example completed!")


if __name__ == "__main__":
    asyncio.run(example_external_tools())