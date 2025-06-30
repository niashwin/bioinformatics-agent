#!/usr/bin/env python3
"""
BioinformaticsAgent Tools: Comprehensive collection of bioinformatics tools
for the specialized agent system.

This module provides implementations of common bioinformatics tools including:
- Sequence analysis tools
- Genomics and transcriptomics tools  
- Proteomics tools
- Structural biology tools
- Statistical analysis tools
- Visualization tools
"""

import asyncio
import json
import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import the base classes from the main module
from bioagent_architecture import (
    BioinformaticsTool, BioToolResult, DataType, DataMetadata
)


# =================== Sequence Analysis Tools ===================

class SequenceStatsTool(BioinformaticsTool):
    """Calculate basic statistics for DNA/RNA/protein sequences"""
    
    def __init__(self):
        super().__init__(
            name="sequence_stats",
            description="Calculate sequence composition, length, and basic statistics",
            supported_data_types=[
                DataType.GENOMIC_SEQUENCE, 
                DataType.PROTEIN_SEQUENCE, 
                DataType.RNA_SEQUENCE
            ]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string", 
                    "description": "Path to FASTA file containing sequences"
                },
                "sequence_type": {
                    "type": "string",
                    "enum": ["dna", "rna", "protein"],
                    "description": "Type of sequences in the file"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["json", "csv", "table"],
                    "default": "json"
                }
            },
            "required": ["input_file", "sequence_type"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        try:
            from Bio import SeqIO
            from Bio.SeqUtils import GC, molecular_weight
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
            
            input_file = params["input_file"]
            seq_type = params["sequence_type"]
            
            results = []
            
            # Parse sequences
            for record in SeqIO.parse(input_file, "fasta"):
                seq_str = str(record.seq)
                stats = {
                    "sequence_id": record.id,
                    "length": len(seq_str),
                    "description": record.description
                }
                
                if seq_type in ["dna", "rna"]:
                    # Nucleotide sequence statistics
                    stats.update({
                        "gc_content": GC(record.seq),
                        "a_count": seq_str.count('A'),
                        "t_count": seq_str.count('T') if seq_type == "dna" else seq_str.count('U'),
                        "g_count": seq_str.count('G'),
                        "c_count": seq_str.count('C'),
                        "n_count": seq_str.count('N'),
                        "molecular_weight": molecular_weight(record.seq, seq_type)
                    })
                    
                elif seq_type == "protein":
                    # Protein sequence statistics
                    try:
                        protein_analysis = ProteinAnalysis(seq_str)
                        stats.update({
                            "molecular_weight": protein_analysis.molecular_weight(),
                            "isoelectric_point": protein_analysis.isoelectric_point(),
                            "instability_index": protein_analysis.instability_index(),
                            "gravy": protein_analysis.gravy(),
                            "aromaticity": protein_analysis.aromaticity(),
                            "amino_acid_composition": protein_analysis.get_amino_acids_percent()
                        })
                    except Exception as e:
                        stats["analysis_error"] = str(e)
                
                results.append(stats)
            
            # Format output
            df = pd.DataFrame(results)
            output_format = params.get("output_format", "json")
            
            if output_format == "csv":
                output_str = df.to_csv(index=False)
            elif output_format == "table":
                output_str = df.to_string(index=False)
            else:
                output_str = df.to_json(orient="records", indent=2)
            
            return BioToolResult(
                success=True,
                output=output_str,
                metadata={
                    "total_sequences": len(results),
                    "sequence_type": seq_type,
                    "average_length": df["length"].mean() if len(results) > 0 else 0
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Sequence statistics calculation failed: {str(e)}"
            )


class MultipleSequenceAlignmentTool(BioinformaticsTool):
    """Perform multiple sequence alignment using various algorithms"""
    
    def __init__(self):
        super().__init__(
            name="msa",
            description="Perform multiple sequence alignment using MUSCLE, ClustalW, or MAFFT",
            supported_data_types=[
                DataType.GENOMIC_SEQUENCE,
                DataType.PROTEIN_SEQUENCE,
                DataType.RNA_SEQUENCE
            ]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_file": {
                    "type": "string",
                    "description": "Path to FASTA file with sequences to align"
                },
                "algorithm": {
                    "type": "string",
                    "enum": ["muscle", "clustalw", "mafft"],
                    "default": "muscle"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["fasta", "clustal", "phylip"],
                    "default": "fasta"
                },
                "gap_open_penalty": {"type": "number", "default": -10},
                "gap_extend_penalty": {"type": "number", "default": -0.5}
            },
            "required": ["input_file"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        try:
            from Bio.Align.Applications import MuscleCommandline, ClustalwCommandline
            from Bio import AlignIO
            
            input_file = params["input_file"]
            algorithm = params.get("algorithm", "muscle")
            output_format = params.get("output_format", "fasta")
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.aln', delete=False) as tmp_out:
                output_file = tmp_out.name
            
            # Run alignment based on selected algorithm
            if algorithm == "muscle":
                muscle_cline = MuscleCommandline(
                    input=input_file,
                    out=output_file,
                    gapopen=params.get("gap_open_penalty", -10)
                )
                stdout, stderr = muscle_cline()
                
            elif algorithm == "clustalw":
                clustalw_cline = ClustalwCommandline(
                    "clustalw2",
                    infile=input_file,
                    outfile=output_file,
                    outorder="input"
                )
                stdout, stderr = clustalw_cline()
            
            else:  # mafft
                cmd = f"mafft --auto {input_file} > {output_file}"
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
            
            # Read and analyze alignment
            alignment = AlignIO.read(output_file, output_format)
            
            # Calculate alignment statistics
            alignment_length = alignment.get_alignment_length()
            num_sequences = len(alignment)
            
            # Calculate conservation scores (simplified)
            conservation_scores = []
            for i in range(alignment_length):
                column = alignment[:, i]
                unique_chars = len(set(column))
                conservation = 1 - (unique_chars - 1) / num_sequences
                conservation_scores.append(conservation)
            
            avg_conservation = np.mean(conservation_scores)
            
            # Read alignment content
            with open(output_file, 'r') as f:
                alignment_content = f.read()
            
            # Cleanup
            os.unlink(output_file)
            
            return BioToolResult(
                success=True,
                output=alignment_content,
                metadata={
                    "algorithm_used": algorithm,
                    "alignment_length": alignment_length,
                    "num_sequences": num_sequences,
                    "average_conservation": avg_conservation,
                    "output_format": output_format
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Multiple sequence alignment failed: {str(e)}"
            )


# =================== Genomics Tools ===================

class RNASeqDifferentialExpressionTool(BioinformaticsTool):
    """Perform differential expression analysis on RNA-seq count data"""
    
    def __init__(self):
        super().__init__(
            name="rnaseq_de",
            description="Differential expression analysis using DESeq2-like methodology",
            supported_data_types=[DataType.EXPRESSION_MATRIX]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "count_matrix": {
                    "type": "string",
                    "description": "Path to count matrix file (genes x samples)"
                },
                "sample_info": {
                    "type": "string",
                    "description": "Path to sample information file with conditions"
                },
                "condition_column": {
                    "type": "string",
                    "description": "Column name in sample_info containing conditions"
                },
                "control_condition": {
                    "type": "string",
                    "description": "Name of control condition"
                },
                "treatment_condition": {
                    "type": "string",
                    "description": "Name of treatment condition"
                },
                "padj_threshold": {"type": "number", "default": 0.05},
                "fc_threshold": {"type": "number", "default": 1.5},
                "min_counts": {"type": "integer", "default": 10}
            },
            "required": ["count_matrix", "sample_info", "condition_column", 
                        "control_condition", "treatment_condition"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        try:
            from scipy import stats
            from sklearn.preprocessing import StandardScaler
            
            # Load data
            counts = pd.read_csv(params["count_matrix"], index_col=0)
            sample_info = pd.read_csv(params["sample_info"], index_col=0)
            
            condition_col = params["condition_column"]
            control_cond = params["control_condition"]
            treatment_cond = params["treatment_condition"]
            
            # Filter samples
            control_samples = sample_info[sample_info[condition_col] == control_cond].index
            treatment_samples = sample_info[sample_info[condition_col] == treatment_cond].index
            
            # Filter counts
            min_counts = params.get("min_counts", 10)
            expressed_genes = counts.sum(axis=1) >= min_counts
            counts_filtered = counts[expressed_genes]
            
            # Normalize (simplified approach - in practice use DESeq2 normalization)
            # Log2 transform with pseudocount
            log_counts = np.log2(counts_filtered + 1)
            
            results = []
            
            for gene in log_counts.index:
                control_expr = log_counts.loc[gene, control_samples]
                treatment_expr = log_counts.loc[gene, treatment_samples]
                
                # Calculate statistics
                control_mean = control_expr.mean()
                treatment_mean = treatment_expr.mean()
                fold_change = treatment_mean - control_mean  # log2 fold change
                
                # T-test
                if len(control_expr) > 1 and len(treatment_expr) > 1:
                    t_stat, p_value = stats.ttest_ind(treatment_expr, control_expr)
                else:
                    t_stat, p_value = 0, 1
                
                results.append({
                    'gene_id': gene,
                    'baseMean': (control_mean + treatment_mean) / 2,
                    'log2FoldChange': fold_change,
                    'pvalue': p_value,
                    'control_mean': control_mean,
                    'treatment_mean': treatment_mean
                })
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Multiple testing correction (Benjamini-Hochberg)
            from statsmodels.stats.multitest import multipletests
            
            if len(results_df) > 0:
                _, padj, _, _ = multipletests(
                    results_df['pvalue'], 
                    method='fdr_bh'
                )
                results_df['padj'] = padj
            else:
                results_df['padj'] = []
            
            # Filter significant genes
            padj_thresh = params.get("padj_threshold", 0.05)
            fc_thresh = np.log2(params.get("fc_threshold", 1.5))
            
            significant_genes = results_df[
                (results_df['padj'] < padj_thresh) & 
                (abs(results_df['log2FoldChange']) > fc_thresh)
            ].copy()
            
            # Sort by significance
            significant_genes = significant_genes.sort_values('padj')
            
            # Generate summary statistics
            summary = {
                "total_genes_tested": len(results_df),
                "significant_genes": len(significant_genes),
                "upregulated_genes": len(significant_genes[significant_genes['log2FoldChange'] > 0]),
                "downregulated_genes": len(significant_genes[significant_genes['log2FoldChange'] < 0]),
                "padj_threshold": padj_thresh,
                "fc_threshold": params.get("fc_threshold", 1.5)
            }
            
            return BioToolResult(
                success=True,
                output={
                    "all_results": results_df.to_dict('records'),
                    "significant_genes": significant_genes.to_dict('records'),
                    "summary": summary
                },
                metadata=summary
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Differential expression analysis failed: {str(e)}"
            )


class VariantAnnotationTool(BioinformaticsTool):
    """Annotate genetic variants with functional consequences"""
    
    def __init__(self):
        super().__init__(
            name="variant_annotation",
            description="Annotate genetic variants with gene, transcript, and functional information",
            supported_data_types=[DataType.VARIANT_DATA]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "vcf_file": {
                    "type": "string",
                    "description": "Path to VCF file containing variants"
                },
                "genome_build": {
                    "type": "string",
                    "enum": ["hg19", "hg38", "mm10", "mm39"],
                    "default": "hg38"
                },
                "annotation_database": {
                    "type": "string",
                    "enum": ["refseq", "ensembl", "ucsc"],
                    "default": "refseq"
                },
                "include_population_freq": {"type": "boolean", "default": True},
                "include_pathogenicity": {"type": "boolean", "default": True}
            },
            "required": ["vcf_file"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        try:
            # This is a simplified implementation
            # In practice, would use tools like VEP, ANNOVAR, or SnpEff
            
            vcf_file = params["vcf_file"]
            genome_build = params.get("genome_build", "hg38")
            
            # Parse VCF file (simplified)
            variants = []
            with open(vcf_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        chrom, pos, var_id, ref, alt = parts[:5]
                        
                        # Simulate annotation
                        annotation = {
                            "chromosome": chrom,
                            "position": int(pos),
                            "variant_id": var_id if var_id != '.' else f"{chrom}:{pos}:{ref}:{alt}",
                            "reference": ref,
                            "alternate": alt,
                            "variant_type": self._classify_variant(ref, alt),
                            "gene_symbol": f"GENE_{hash(f'{chrom}:{pos}') % 1000}",
                            "transcript_consequence": self._simulate_consequence(),
                            "amino_acid_change": self._simulate_aa_change(ref, alt),
                            "population_frequency": np.random.random() * 0.5,
                            "pathogenicity_score": np.random.random()
                        }
                        
                        variants.append(annotation)
            
            # Filter and prioritize variants
            high_impact_variants = [
                v for v in variants 
                if v["transcript_consequence"] in ["missense", "nonsense", "frameshift"]
            ]
            
            summary = {
                "total_variants": len(variants),
                "high_impact_variants": len(high_impact_variants),
                "variant_types": self._count_variant_types(variants),
                "genome_build": genome_build
            }
            
            return BioToolResult(
                success=True,
                output={
                    "annotated_variants": variants,
                    "high_impact_variants": high_impact_variants,
                    "summary": summary
                },
                metadata=summary
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Variant annotation failed: {str(e)}"
            )
    
    def _classify_variant(self, ref: str, alt: str) -> str:
        """Classify variant type based on ref and alt alleles"""
        if len(ref) == 1 and len(alt) == 1:
            return "SNV"
        elif len(ref) > len(alt):
            return "deletion"
        elif len(ref) < len(alt):
            return "insertion"
        else:
            return "complex"
    
    def _simulate_consequence(self) -> str:
        """Simulate transcript consequence"""
        consequences = ["synonymous", "missense", "nonsense", "splice_site", "frameshift", "intronic"]
        return np.random.choice(consequences)
    
    def _simulate_aa_change(self, ref: str, alt: str) -> str:
        """Simulate amino acid change"""
        if len(ref) == 1 and len(alt) == 1:
            aa_codes = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", 
                       "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
            ref_aa = np.random.choice(aa_codes)
            alt_aa = np.random.choice(aa_codes)
            pos = np.random.randint(1, 500)
            return f"{ref_aa}{pos}{alt_aa}"
        return ""
    
    def _count_variant_types(self, variants: List[Dict]) -> Dict[str, int]:
        """Count variants by type"""
        type_counts = {}
        for variant in variants:
            vtype = variant["variant_type"]
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        return type_counts


# =================== Proteomics Tools ===================

class ProteinStructureAnalysisTool(BioinformaticsTool):
    """Analyze protein structure from PDB files"""
    
    def __init__(self):
        super().__init__(
            name="protein_structure",
            description="Analyze protein structure properties from PDB files",
            supported_data_types=[DataType.STRUCTURE_PDB]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pdb_file": {
                    "type": "string",
                    "description": "Path to PDB structure file"
                },
                "chain_id": {
                    "type": "string",
                    "description": "Specific chain to analyze (optional)"
                },
                "calculate_surface_area": {"type": "boolean", "default": True},
                "find_binding_sites": {"type": "boolean", "default": True},
                "secondary_structure": {"type": "boolean", "default": True}
            },
            "required": ["pdb_file"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        try:
            from Bio.PDB import PDBParser, DSSP, is_aa
            
            pdb_file = params["pdb_file"]
            chain_id = params.get("chain_id")
            
            # Parse PDB structure
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_file)
            
            results = {}
            
            # Iterate through models and chains
            for model in structure:
                for chain in model:
                    if chain_id and chain.id != chain_id:
                        continue
                    
                    chain_analysis = {
                        "chain_id": chain.id,
                        "num_residues": len([r for r in chain if is_aa(r)]),
                        "residue_composition": {},
                        "coordinates": []
                    }
                    
                    # Analyze residues
                    aa_counts = {}
                    for residue in chain:
                        if is_aa(residue):
                            aa_name = residue.get_resname()
                            aa_counts[aa_name] = aa_counts.get(aa_name, 0) + 1
                            
                            # Get CA coordinates
                            if 'CA' in residue:
                                ca_atom = residue['CA']
                                chain_analysis["coordinates"].append({
                                    "residue": f"{aa_name}{residue.id[1]}",
                                    "x": ca_atom.coord[0],
                                    "y": ca_atom.coord[1],
                                    "z": ca_atom.coord[2]
                                })
                    
                    chain_analysis["residue_composition"] = aa_counts
                    
                    # Calculate basic geometric properties
                    if chain_analysis["coordinates"]:
                        coords = np.array([[c["x"], c["y"], c["z"]] 
                                         for c in chain_analysis["coordinates"]])
                        
                        # Center of mass
                        center_of_mass = np.mean(coords, axis=0)
                        
                        # Radius of gyration
                        distances = np.linalg.norm(coords - center_of_mass, axis=1)
                        radius_of_gyration = np.sqrt(np.mean(distances**2))
                        
                        chain_analysis.update({
                            "center_of_mass": center_of_mass.tolist(),
                            "radius_of_gyration": radius_of_gyration,
                            "max_distance": np.max(distances),
                            "min_distance": np.min(distances)
                        })
                    
                    results[chain.id] = chain_analysis
            
            # Secondary structure analysis (if DSSP available)
            if params.get("secondary_structure", True):
                try:
                    dssp = DSSP(model, pdb_file)
                    ss_summary = {"helix": 0, "sheet": 0, "coil": 0}
                    
                    for key in dssp.keys():
                        ss = dssp[key][2]
                        if ss in ['H', 'G', 'I']:
                            ss_summary["helix"] += 1
                        elif ss in ['E', 'B']:
                            ss_summary["sheet"] += 1
                        else:
                            ss_summary["coil"] += 1
                    
                    results["secondary_structure"] = ss_summary
                    
                except Exception:
                    results["secondary_structure"] = "DSSP analysis failed"
            
            summary = {
                "total_chains": len(results) - (1 if "secondary_structure" in results else 0),
                "pdb_file": pdb_file,
                "analysis_completed": True
            }
            
            return BioToolResult(
                success=True,
                output=results,
                metadata=summary
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Protein structure analysis failed: {str(e)}"
            )


class MassSpecProteomicsTool(BioinformaticsTool):
    """Analyze mass spectrometry proteomics data"""
    
    def __init__(self):
        super().__init__(
            name="ms_proteomics",
            description="Analyze mass spectrometry proteomics data for protein identification and quantification",
            supported_data_types=[DataType.PROTEOMICS_DATA]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "intensity_matrix": {
                    "type": "string",
                    "description": "Path to protein intensity matrix"
                },
                "sample_info": {
                    "type": "string",
                    "description": "Path to sample information file"
                },
                "protein_annotations": {
                    "type": "string",
                    "description": "Path to protein annotation file"
                },
                "normalization_method": {
                    "type": "string",
                    "enum": ["median", "quantile", "vsn", "none"],
                    "default": "median"
                },
                "missing_value_threshold": {"type": "number", "default": 0.5},
                "fold_change_threshold": {"type": "number", "default": 1.5}
            },
            "required": ["intensity_matrix", "sample_info"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        try:
            # Load data
            intensities = pd.read_csv(params["intensity_matrix"], index_col=0)
            sample_info = pd.read_csv(params["sample_info"], index_col=0)
            
            # Handle missing values
            missing_threshold = params.get("missing_value_threshold", 0.5)
            valid_proteins = intensities.dropna(thresh=int(missing_threshold * intensities.shape[1]))
            
            # Log2 transform
            log_intensities = np.log2(valid_proteins + 1)
            
            # Normalization
            norm_method = params.get("normalization_method", "median")
            if norm_method == "median":
                normalized = log_intensities - log_intensities.median()
            elif norm_method == "quantile":
                from sklearn.preprocessing import QuantileTransformer
                qt = QuantileTransformer(output_distribution='normal')
                normalized = pd.DataFrame(
                    qt.fit_transform(log_intensities.T).T,
                    index=log_intensities.index,
                    columns=log_intensities.columns
                )
            else:
                normalized = log_intensities
            
            # Statistical analysis
            results = []
            conditions = sample_info['condition'].unique()
            
            if len(conditions) == 2:
                # Two-group comparison
                cond1, cond2 = conditions
                group1_samples = sample_info[sample_info['condition'] == cond1].index
                group2_samples = sample_info[sample_info['condition'] == cond2].index
                
                for protein in normalized.index:
                    group1_values = normalized.loc[protein, group1_samples].dropna()
                    group2_values = normalized.loc[protein, group2_samples].dropna()
                    
                    if len(group1_values) > 0 and len(group2_values) > 0:
                        fold_change = group2_values.mean() - group1_values.mean()
                        
                        # T-test
                        from scipy import stats
                        if len(group1_values) > 1 and len(group2_values) > 1:
                            t_stat, p_value = stats.ttest_ind(group2_values, group1_values)
                        else:
                            t_stat, p_value = 0, 1
                        
                        results.append({
                            'protein_id': protein,
                            'log2_fold_change': fold_change,
                            'pvalue': p_value,
                            'group1_mean': group1_values.mean(),
                            'group2_mean': group2_values.mean(),
                            'group1_std': group1_values.std(),
                            'group2_std': group2_values.std()
                        })
            
            results_df = pd.DataFrame(results)
            
            # Multiple testing correction
            if len(results_df) > 0:
                from statsmodels.stats.multitest import multipletests
                _, padj, _, _ = multipletests(results_df['pvalue'], method='fdr_bh')
                results_df['padj'] = padj
            
            # Filter significant proteins
            fc_threshold = np.log2(params.get("fold_change_threshold", 1.5))
            significant_proteins = results_df[
                (results_df['padj'] < 0.05) & 
                (abs(results_df['log2_fold_change']) > fc_threshold)
            ]
            
            summary = {
                "total_proteins": len(intensities),
                "valid_proteins_after_filtering": len(valid_proteins),
                "significant_proteins": len(significant_proteins),
                "upregulated_proteins": len(significant_proteins[significant_proteins['log2_fold_change'] > 0]),
                "downregulated_proteins": len(significant_proteins[significant_proteins['log2_fold_change'] < 0]),
                "normalization_method": norm_method
            }
            
            return BioToolResult(
                success=True,
                output={
                    "normalized_intensities": normalized.to_dict(),
                    "statistical_results": results_df.to_dict('records'),
                    "significant_proteins": significant_proteins.to_dict('records'),
                    "summary": summary
                },
                metadata=summary
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Mass spectrometry proteomics analysis failed: {str(e)}"
            )


# =================== Phylogenetic Analysis Tools ===================

class PhylogeneticAnalysisTool(BioinformaticsTool):
    """Construct and analyze phylogenetic trees"""
    
    def __init__(self):
        super().__init__(
            name="phylogenetic_analysis",
            description="Construct phylogenetic trees and perform evolutionary analysis",
            supported_data_types=[
                DataType.GENOMIC_SEQUENCE,
                DataType.PROTEIN_SEQUENCE,
                DataType.PHYLOGENETIC_TREE
            ]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "alignment_file": {
                    "type": "string",
                    "description": "Path to multiple sequence alignment file"
                },
                "method": {
                    "type": "string",
                    "enum": ["neighbor_joining", "maximum_likelihood", "parsimony"],
                    "default": "neighbor_joining"
                },
                "bootstrap_replicates": {"type": "integer", "default": 100},
                "substitution_model": {
                    "type": "string",
                    "enum": ["JC69", "K2P", "GTR", "WAG", "LG"],
                    "default": "K2P"
                },
                "outgroup": {"type": "string", "description": "Outgroup sequence ID"}
            },
            "required": ["alignment_file"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        try:
            from Bio import AlignIO, Phylo
            from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
            
            alignment_file = params["alignment_file"]
            method = params.get("method", "neighbor_joining")
            
            # Load alignment
            alignment = AlignIO.read(alignment_file, "fasta")
            
            # Calculate distance matrix
            calculator = DistanceCalculator('identity')
            distance_matrix = calculator.get_distance(alignment)
            
            # Construct tree
            constructor = DistanceTreeConstructor()
            
            if method == "neighbor_joining":
                tree = constructor.nj(distance_matrix)
            else:  # UPGMA as fallback
                tree = constructor.upgma(distance_matrix)
            
            # Root tree if outgroup specified
            outgroup = params.get("outgroup")
            if outgroup:
                try:
                    tree.root_with_outgroup(outgroup)
                except:
                    pass  # Continue without rooting if outgroup not found
            
            # Calculate tree statistics
            tree_stats = {
                "total_length": tree.total_branch_length(),
                "num_terminals": tree.count_terminals(),
                "num_internal_nodes": len(tree.get_nonterminals()),
                "max_distance": max([tree.distance(tree.root, terminal) 
                                   for terminal in tree.get_terminals()]),
                "method_used": method
            }
            
            # Get tree in different formats
            import io
            
            # Newick format
            newick_output = io.StringIO()
            Phylo.write(tree, newick_output, "newick")
            newick_string = newick_output.getvalue()
            
            # Nexus format
            nexus_output = io.StringIO()
            Phylo.write(tree, nexus_output, "nexus")
            nexus_string = nexus_output.getvalue()
            
            # Extract terminal distances for analysis
            terminals = tree.get_terminals()
            pairwise_distances = []
            
            for i, term1 in enumerate(terminals):
                for j, term2 in enumerate(terminals[i+1:], i+1):
                    distance = tree.distance(term1, term2)
                    pairwise_distances.append({
                        "sequence1": term1.name,
                        "sequence2": term2.name,
                        "distance": distance
                    })
            
            return BioToolResult(
                success=True,
                output={
                    "tree_newick": newick_string,
                    "tree_nexus": nexus_string,
                    "tree_statistics": tree_stats,
                    "pairwise_distances": pairwise_distances,
                    "distance_matrix": distance_matrix._matrix
                },
                metadata=tree_stats
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Phylogenetic analysis failed: {str(e)}"
            )


# =================== Visualization Tools ===================

class BioinformaticsVisualizationTool(BioinformaticsTool):
    """Create various bioinformatics visualizations"""
    
    def __init__(self):
        super().__init__(
            name="bio_visualization",
            description="Create publication-quality visualizations for biological data",
            supported_data_types=[
                DataType.EXPRESSION_MATRIX,
                DataType.VARIANT_DATA,
                DataType.PROTEOMICS_DATA,
                DataType.PHYLOGENETIC_TREE
            ]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "data_file": {
                    "type": "string",
                    "description": "Path to data file"
                },
                "plot_type": {
                    "type": "string",
                    "enum": ["heatmap", "volcano", "pca", "boxplot", "scatter", "histogram"],
                    "description": "Type of plot to create"
                },
                "color_scheme": {
                    "type": "string",
                    "enum": ["viridis", "plasma", "Blues", "Reds", "RdBu"],
                    "default": "viridis"
                },
                "figure_size": {
                    "type": "array",
                    "items": {"type": "number"},
                    "default": [10, 8]
                },
                "output_format": {
                    "type": "string",
                    "enum": ["png", "pdf", "svg"],
                    "default": "png"
                },
                "dpi": {"type": "integer", "default": 300}
            },
            "required": ["data_file", "plot_type"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            data_file = params["data_file"]
            plot_type = params["plot_type"]
            color_scheme = params.get("color_scheme", "viridis")
            fig_size = params.get("figure_size", [10, 8])
            
            # Load data
            data = pd.read_csv(data_file, index_col=0)
            
            # Set up plot
            plt.figure(figsize=fig_size)
            
            if plot_type == "heatmap":
                # Correlation heatmap
                corr_matrix = data.corr()
                sns.heatmap(corr_matrix, annot=True, cmap=color_scheme, 
                           center=0, square=True, cbar_kws={"shrink": .8})
                plt.title("Correlation Heatmap")
                
            elif plot_type == "volcano":
                # Volcano plot (assumes log2FC and pvalue columns)
                if 'log2FoldChange' in data.columns and 'pvalue' in data.columns:
                    x = data['log2FoldChange']
                    y = -np.log10(data['pvalue'] + 1e-300)  # Avoid log(0)
                    
                    plt.scatter(x, y, alpha=0.6, c=y, cmap=color_scheme)
                    plt.xlabel('Log2 Fold Change')
                    plt.ylabel('-Log10 P-value')
                    plt.title('Volcano Plot')
                    plt.colorbar(label='-Log10 P-value')
                else:
                    raise ValueError("Volcano plot requires 'log2FoldChange' and 'pvalue' columns")
                    
            elif plot_type == "pca":
                # PCA plot
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data.T)
                
                pca = PCA(n_components=2)
                pca_coords = pca.fit_transform(scaled_data)
                
                plt.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                           c=range(len(pca_coords)), cmap=color_scheme)
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.title('PCA Plot')
                plt.colorbar(label='Sample Index')
                
            elif plot_type == "boxplot":
                # Box plot of data distribution
                data_melted = data.reset_index().melt(id_vars='index', var_name='Sample', value_name='Value')
                sns.boxplot(data=data_melted, x='Sample', y='Value')
                plt.xticks(rotation=45)
                plt.title('Data Distribution by Sample')
                
            elif plot_type == "scatter":
                # Scatter plot of first two numeric columns
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[:2]
                    plt.scatter(data[x_col], data[y_col], alpha=0.6, c=range(len(data)), cmap=color_scheme)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'{x_col} vs {y_col}')
                    plt.colorbar(label='Data Point Index')
                else:
                    raise ValueError("Scatter plot requires at least 2 numeric columns")
                    
            elif plot_type == "histogram":
                # Histogram of first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    plt.hist(data[col], bins=30, alpha=0.7, color=plt.cm.get_cmap(color_scheme)(0.5))
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of {col}')
                else:
                    raise ValueError("Histogram requires at least 1 numeric column")
            
            plt.tight_layout()
            
            # Save plot
            output_format = params.get("output_format", "png")
            dpi = params.get("dpi", 300)
            
            with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as tmp_file:
                output_path = tmp_file.name
                plt.savefig(output_path, format=output_format, dpi=dpi, bbox_inches='tight')
            
            plt.close()
            
            # Read plot data for return
            with open(output_path, 'rb') as f:
                plot_data = f.read()
            
            os.unlink(output_path)
            
            return BioToolResult(
                success=True,
                output=f"{plot_type} plot created successfully",
                metadata={
                    "plot_type": plot_type,
                    "data_shape": data.shape,
                    "output_format": output_format,
                    "file_size_bytes": len(plot_data)
                },
                visualization_data=plot_data
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Visualization creation failed: {str(e)}"
            )


# =================== Tool Registry ===================

def get_all_bioinformatics_tools() -> List[BioinformaticsTool]:
    """Return a list of all available bioinformatics tools"""
    return [
        SequenceStatsTool(),
        MultipleSequenceAlignmentTool(),
        RNASeqDifferentialExpressionTool(),
        VariantAnnotationTool(),
        ProteinStructureAnalysisTool(),
        MassSpecProteomicsTool(),
        PhylogeneticAnalysisTool(),
        BioinformaticsVisualizationTool()
    ]


# =================== Example Usage ===================

async def test_tools():
    """Test the bioinformatics tools"""
    
    # Create sample data
    sample_metadata = DataMetadata(
        data_type=DataType.GENOMIC_SEQUENCE,
        file_path="sample_sequences.fasta",
        organism="Homo sapiens"
    )
    
    # Test sequence stats tool
    seq_tool = SequenceStatsTool()
    
    # Create a sample FASTA file for testing
    sample_fasta = """
>seq1
ATGCGATCGATCGATCGATCG
>seq2
CGATCGATCGATCGATCGATC
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(sample_fasta)
        fasta_path = f.name
    
    # Test the tool
    result = await seq_tool.execute(
        {"input_file": fasta_path, "sequence_type": "dna"},
        [sample_metadata]
    )
    
    print(f"Sequence stats result: {result.success}")
    if result.success:
        print(f"Output: {result.output[:200]}...")
    else:
        print(f"Error: {result.error}")
    
    # Cleanup
    os.unlink(fasta_path)


if __name__ == "__main__":
    asyncio.run(test_tools())