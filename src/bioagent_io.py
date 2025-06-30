#!/usr/bin/env python3
"""
BioinformaticsAgent I/O Module: Comprehensive file handling and data management

This module provides robust file I/O capabilities for bioinformatics data formats:
- FASTA/FASTQ files
- VCF/BCF variant files  
- SAM/BAM alignment files
- Expression matrices (CSV, TSV, HDF5, Excel)
- Annotation files (GFF, GTF, BED)
- Single-cell data (H5, H5AD, Loom, MEX)
- Database connectivity
- Cloud storage integration
"""

import asyncio
import gzip
import bz2
import lzma
# Core dependencies
import sqlite3
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Iterator, AsyncIterator, BinaryIO, TextIO
from dataclasses import dataclass, field
from enum import Enum
import logging

import pandas as pd
import numpy as np
from Bio import SeqIO, Align
from Bio.SeqRecord import SeqRecord

# Optional dependencies with fallbacks
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False

try:
    import cyvcf2
    HAS_CYVCF2 = True
except ImportError:
    HAS_CYVCF2 = False

try:
    import pybedtools
    HAS_PYBEDTOOLS = True
except ImportError:
    HAS_PYBEDTOOLS = False

try:
    import loompy
    HAS_LOOMPY = True
except ImportError:
    HAS_LOOMPY = False

try:
    import tables
    HAS_TABLES = True
except ImportError:
    HAS_TABLES = False

# Cloud storage (optional)
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    from azure.storage.blob import BlobServiceClient
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

# Async file operations
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False
import aiohttp


# =================== Data Format Detection ===================

class FileFormat(Enum):
    """Supported file formats"""
    FASTA = "fasta"
    FASTQ = "fastq"
    VCF = "vcf"
    BCF = "bcf"
    SAM = "sam"
    BAM = "bam"
    CRAM = "cram"
    GFF = "gff"
    GTF = "gtf"
    BED = "bed"
    BEDGRAPH = "bedgraph"
    WIG = "wig"
    BIGWIG = "bigwig"
    CSV = "csv"
    TSV = "tsv"
    EXCEL = "excel"
    HDF5 = "h5"
    H5AD = "h5ad"
    LOOM = "loom"
    MEX = "mex"
    ZARR = "zarr"
    JSON = "json"
    YAML = "yaml"
    PICKLE = "pickle"


class CompressionType(Enum):
    """Supported compression types"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


@dataclass
class FileMetadata:
    """Metadata for bioinformatics files"""
    file_path: str
    file_format: FileFormat
    compression: CompressionType
    file_size: int
    line_count: Optional[int] = None
    record_count: Optional[int] = None
    genome_build: Optional[str] = None
    sample_names: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    checksum: Optional[str] = None
    created_date: Optional[str] = None
    last_modified: Optional[str] = None


# =================== File Format Detection ===================

class FormatDetector:
    """Automatically detect file formats and compression"""
    
    @staticmethod
    def detect_format(file_path: str) -> tuple[FileFormat, CompressionType]:
        """Detect file format and compression from path and content"""
        path = Path(file_path)
        
        # Check compression first
        compression = CompressionType.NONE
        name = path.name.lower()
        
        if name.endswith('.gz'):
            compression = CompressionType.GZIP
            name = name[:-3]  # Remove .gz
        elif name.endswith('.bz2'):
            compression = CompressionType.BZIP2
            name = name[:-4]  # Remove .bz2
        elif name.endswith('.xz') or name.endswith('.lzma'):
            compression = CompressionType.LZMA
            name = name[:-3] if name.endswith('.xz') else name[:-5]
        
        # Detect format by extension
        format_map = {
            '.fasta': FileFormat.FASTA,
            '.fa': FileFormat.FASTA,
            '.fas': FileFormat.FASTA,
            '.fastq': FileFormat.FASTQ,
            '.fq': FileFormat.FASTQ,
            '.vcf': FileFormat.VCF,
            '.bcf': FileFormat.BCF,
            '.sam': FileFormat.SAM,
            '.bam': FileFormat.BAM,
            '.cram': FileFormat.CRAM,
            '.gff': FileFormat.GFF,
            '.gff3': FileFormat.GFF,
            '.gtf': FileFormat.GTF,
            '.bed': FileFormat.BED,
            '.bedgraph': FileFormat.BEDGRAPH,
            '.wig': FileFormat.WIG,
            '.bigwig': FileFormat.BIGWIG,
            '.bw': FileFormat.BIGWIG,
            '.csv': FileFormat.CSV,
            '.tsv': FileFormat.TSV,
            '.txt': FileFormat.TSV,  # Assume tab-separated
            '.xlsx': FileFormat.EXCEL,
            '.xls': FileFormat.EXCEL,
            '.h5': FileFormat.HDF5,
            '.hdf5': FileFormat.HDF5,
            '.h5ad': FileFormat.H5AD,
            '.loom': FileFormat.LOOM,
            '.zarr': FileFormat.ZARR,
            '.json': FileFormat.JSON,
            '.yaml': FileFormat.YAML,
            '.yml': FileFormat.YAML,
            '.pkl': FileFormat.PICKLE,
            '.pickle': FileFormat.PICKLE
        }
        
        # Find matching extension
        for ext, fmt in format_map.items():
            if name.endswith(ext):
                return fmt, compression
        
        # If no extension match, try content-based detection
        try:
            file_format = FormatDetector._detect_by_content(file_path, compression)
            return file_format, compression
        except:
            raise ValueError(f"Could not detect format for {file_path}")
    
    @staticmethod
    def _detect_by_content(file_path: str, compression: CompressionType) -> FileFormat:
        """Detect format by examining file content"""
        
        def open_file(path: str, comp: CompressionType):
            if comp == CompressionType.GZIP:
                return gzip.open(path, 'rt')
            elif comp == CompressionType.BZIP2:
                return bz2.open(path, 'rt')
            elif comp == CompressionType.LZMA:
                return lzma.open(path, 'rt')
            else:
                return open(path, 'r')
        
        try:
            with open_file(file_path, compression) as f:
                first_line = f.readline().strip()
                
                if first_line.startswith('>'):
                    return FileFormat.FASTA
                elif first_line.startswith('@'):
                    # Could be FASTQ or SAM
                    second_line = f.readline().strip()
                    third_line = f.readline().strip()
                    if third_line.startswith('+'):
                        return FileFormat.FASTQ
                    else:
                        return FileFormat.SAM
                elif first_line.startswith('##fileformat=VCF'):
                    return FileFormat.VCF
                elif first_line.startswith('#') and 'gff-version' in first_line.lower():
                    return FileFormat.GFF
                elif '\t' in first_line:
                    return FileFormat.TSV
                elif ',' in first_line:
                    return FileFormat.CSV
                
        except Exception:
            pass
        
        raise ValueError("Could not detect file format from content")


# =================== Base File Reader/Writer ===================

class BaseFileHandler:
    """Base class for file handlers"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.format, self.compression = FormatDetector.detect_format(str(self.file_path))
        
    def _open_file(self, mode: str = 'r'):
        """Open file with appropriate compression"""
        if self.compression == CompressionType.GZIP:
            return gzip.open(self.file_path, mode + 't' if 't' not in mode else mode)
        elif self.compression == CompressionType.BZIP2:
            return bz2.open(self.file_path, mode + 't' if 't' not in mode else mode)
        elif self.compression == CompressionType.LZMA:
            return lzma.open(self.file_path, mode + 't' if 't' not in mode else mode)
        else:
            return open(self.file_path, mode)
    
    async def _open_file_async(self, mode: str = 'r'):
        """Open file asynchronously"""
        if self.compression == CompressionType.NONE:
            return await aiofiles.open(self.file_path, mode)
        else:
            # For compressed files, we need to handle them synchronously
            # In a real implementation, you might want to use async compression libraries
            return self._open_file(mode)


# =================== Sequence File Handlers ===================

class SequenceFileHandler(BaseFileHandler):
    """Handler for FASTA and FASTQ files"""
    
    def read_sequences(self, max_records: Optional[int] = None) -> Iterator[SeqRecord]:
        """Read sequences from FASTA/FASTQ files"""
        format_str = "fasta" if self.format == FileFormat.FASTA else "fastq"
        
        with self._open_file() as handle:
            count = 0
            for record in SeqIO.parse(handle, format_str):
                if max_records and count >= max_records:
                    break
                yield record
                count += 1
    
    async def read_sequences_async(self, max_records: Optional[int] = None) -> AsyncIterator[SeqRecord]:
        """Asynchronously read sequences"""
        format_str = "fasta" if self.format == FileFormat.FASTA else "fastq"
        
        # For demonstration - in practice, you'd want a truly async sequence parser
        for record in self.read_sequences(max_records):
            yield record
            await asyncio.sleep(0)  # Allow other tasks to run
    
    def write_sequences(self, sequences: List[SeqRecord], append: bool = False):
        """Write sequences to file"""
        format_str = "fasta" if self.format == FileFormat.FASTA else "fastq"
        mode = 'a' if append else 'w'
        
        with self._open_file(mode) as handle:
            SeqIO.write(sequences, handle, format_str)
    
    def get_sequence_stats(self) -> Dict[str, Any]:
        """Get basic statistics about sequences"""
        stats = {
            'total_sequences': 0,
            'total_bases': 0,
            'min_length': float('inf'),
            'max_length': 0,
            'avg_length': 0,
            'gc_content': 0
        }
        
        total_gc = 0
        lengths = []
        
        for record in self.read_sequences():
            seq_len = len(record.seq)
            stats['total_sequences'] += 1
            stats['total_bases'] += seq_len
            stats['min_length'] = min(stats['min_length'], seq_len)
            stats['max_length'] = max(stats['max_length'], seq_len)
            lengths.append(seq_len)
            
            # GC content
            if self.format == FileFormat.FASTA:
                gc_count = record.seq.count('G') + record.seq.count('C')
                total_gc += gc_count
        
        if stats['total_sequences'] > 0:
            stats['avg_length'] = stats['total_bases'] / stats['total_sequences']
            if stats['total_bases'] > 0:
                stats['gc_content'] = total_gc / stats['total_bases']
        
        if stats['min_length'] == float('inf'):
            stats['min_length'] = 0
        
        return stats


# =================== Alignment File Handlers ===================

class AlignmentFileHandler(BaseFileHandler):
    """Handler for SAM/BAM/CRAM files"""
    
    def __init__(self, file_path: str, reference_fasta: Optional[str] = None):
        super().__init__(file_path)
        self.reference_fasta = reference_fasta
    
    def read_alignments(self, region: Optional[str] = None, 
                       max_reads: Optional[int] = None) -> Iterator[Any]:
        """Read alignments from SAM/BAM/CRAM files"""
        mode = 'r'
        if self.format == FileFormat.BAM:
            mode = 'rb'
        elif self.format == FileFormat.CRAM:
            mode = 'rc'
        
        if not HAS_PYSAM:
            raise ImportError("pysam is required for alignment file handling. Install with: pip install pysam")
        
        with pysam.AlignmentFile(str(self.file_path), mode, 
                               reference_filename=self.reference_fasta) as samfile:
            
            iterator = samfile.fetch(region=region) if region else samfile.fetch()
            
            count = 0
            for read in iterator:
                if max_reads and count >= max_reads:
                    break
                yield read
                count += 1
    
    def get_alignment_stats(self) -> Dict[str, Any]:
        """Get alignment statistics"""
        stats = {
            'total_reads': 0,
            'mapped_reads': 0,
            'unmapped_reads': 0,
            'properly_paired': 0,
            'duplicate_reads': 0,
            'avg_mapping_quality': 0,
            'chromosomes': set()
        }
        
        total_mapq = 0
        
        for read in self.read_alignments():
            stats['total_reads'] += 1
            
            if read.is_unmapped:
                stats['unmapped_reads'] += 1
            else:
                stats['mapped_reads'] += 1
                total_mapq += read.mapping_quality
                stats['chromosomes'].add(read.reference_name)
            
            if read.is_proper_pair:
                stats['properly_paired'] += 1
            
            if read.is_duplicate:
                stats['duplicate_reads'] += 1
        
        if stats['mapped_reads'] > 0:
            stats['avg_mapping_quality'] = total_mapq / stats['mapped_reads']
        
        stats['chromosomes'] = list(stats['chromosomes'])
        return stats
    
    def create_index(self):
        """Create index for BAM/CRAM files"""
        pysam.index(str(self.file_path))


# =================== Variant File Handlers ===================

class VariantFileHandler(BaseFileHandler):
    """Handler for VCF/BCF files"""
    
    def read_variants(self, region: Optional[str] = None,
                     max_variants: Optional[int] = None) -> Iterator[Any]:
        """Read variants from VCF/BCF files"""
        if not HAS_CYVCF2:
            raise ImportError("cyvcf2 is required for variant file handling. Install with: pip install cyvcf2")
        
        vcf = cyvcf2.VCF(str(self.file_path))
        
        iterator = vcf(region) if region else vcf
        
        count = 0
        for variant in iterator:
            if max_variants and count >= max_variants:
                break
            yield variant
            count += 1
        
        vcf.close()
    
    def get_variant_stats(self) -> Dict[str, Any]:
        """Get variant statistics"""
        stats = {
            'total_variants': 0,
            'snvs': 0,
            'indels': 0,
            'complex_variants': 0,
            'chromosomes': set(),
            'samples': [],
            'avg_quality': 0
        }
        
        vcf = cyvcf2.VCF(str(self.file_path))
        stats['samples'] = vcf.samples
        
        total_qual = 0
        qual_count = 0
        
        for variant in vcf:
            stats['total_variants'] += 1
            stats['chromosomes'].add(variant.CHROM)
            
            # Classify variant type
            if variant.is_snp:
                stats['snvs'] += 1
            elif variant.is_indel:
                stats['indels'] += 1
            else:
                stats['complex_variants'] += 1
            
            # Quality score
            if variant.QUAL is not None:
                total_qual += variant.QUAL
                qual_count += 1
        
        if qual_count > 0:
            stats['avg_quality'] = total_qual / qual_count
        
        stats['chromosomes'] = list(stats['chromosomes'])
        vcf.close()
        return stats
    
    def filter_variants(self, output_file: str, filters: Dict[str, Any]):
        """Filter variants based on criteria"""
        vcf_in = cyvcf2.VCF(str(self.file_path))
        vcf_out = cyvcf2.Writer(output_file, vcf_in)
        
        for variant in vcf_in:
            keep = True
            
            # Apply filters
            if 'min_qual' in filters and variant.QUAL < filters['min_qual']:
                keep = False
            
            if 'max_missing' in filters:
                missing_rate = sum(1 for gt in variant.genotypes if gt == [None, None, False]) / len(variant.genotypes)
                if missing_rate > filters['max_missing']:
                    keep = False
            
            if keep:
                vcf_out.write_record(variant)
        
        vcf_out.close()
        vcf_in.close()


# =================== Expression Data Handlers ===================

class ExpressionDataHandler(BaseFileHandler):
    """Handler for expression matrices and count data"""
    
    def read_expression_matrix(self, **kwargs) -> pd.DataFrame:
        """Read expression matrix from various formats"""
        if self.format == FileFormat.CSV:
            return pd.read_csv(self.file_path, index_col=0, **kwargs)
        elif self.format == FileFormat.TSV:
            return pd.read_csv(self.file_path, sep='\t', index_col=0, **kwargs)
        elif self.format == FileFormat.EXCEL:
            return pd.read_excel(self.file_path, index_col=0, **kwargs)
        elif self.format == FileFormat.HDF5:
            return pd.read_hdf(self.file_path, **kwargs)
        elif self.format == FileFormat.H5AD:
            adata = ad.read_h5ad(self.file_path)
            return pd.DataFrame(adata.X.T, 
                              columns=adata.obs.index, 
                              index=adata.var.index)
        else:
            raise ValueError(f"Unsupported format for expression data: {self.format}")
    
    def write_expression_matrix(self, data: pd.DataFrame, **kwargs):
        """Write expression matrix to file"""
        if self.format == FileFormat.CSV:
            data.to_csv(self.file_path, **kwargs)
        elif self.format == FileFormat.TSV:
            data.to_csv(self.file_path, sep='\t', **kwargs)
        elif self.format == FileFormat.EXCEL:
            data.to_excel(self.file_path, **kwargs)
        elif self.format == FileFormat.HDF5:
            data.to_hdf(self.file_path, key='data', **kwargs)
        else:
            raise ValueError(f"Unsupported format for writing: {self.format}")
    
    def get_expression_stats(self) -> Dict[str, Any]:
        """Get basic statistics about expression data"""
        data = self.read_expression_matrix()
        
        stats = {
            'n_genes': data.shape[0],
            'n_samples': data.shape[1],
            'total_counts': data.sum().sum(),
            'median_counts_per_gene': data.sum(axis=1).median(),
            'median_counts_per_sample': data.sum(axis=0).median(),
            'zero_rate': (data == 0).sum().sum() / (data.shape[0] * data.shape[1]),
            'gene_names': data.index.tolist()[:100],  # First 100 genes
            'sample_names': data.columns.tolist()
        }
        
        return stats


# =================== Single-cell Data Handlers ===================

class SingleCellDataHandler(BaseFileHandler):
    """Handler for single-cell data formats"""
    
    def read_single_cell_data(self, **kwargs) -> Any:
        """Read single-cell data from various formats"""
        if self.format == FileFormat.H5AD:
            if not HAS_ANNDATA:
                raise ImportError("anndata is required for H5AD files. Install with: pip install anndata")
            return ad.read_h5ad(self.file_path, **kwargs)
        elif self.format == FileFormat.LOOM:
            return ad.read_loom(self.file_path, **kwargs)
        elif self.format == FileFormat.HDF5:
            return ad.read_h5(self.file_path, **kwargs)
        elif self.format == FileFormat.CSV or self.format == FileFormat.TSV:
            # Read as expression matrix and convert to AnnData
            sep = ',' if self.format == FileFormat.CSV else '\t'
            df = pd.read_csv(self.file_path, sep=sep, index_col=0)
            return ad.AnnData(df.T)  # Transpose so cells are observations
        else:
            raise ValueError(f"Unsupported format for single-cell data: {self.format}")
    
    def write_single_cell_data(self, adata: Any, **kwargs):
        """Write single-cell data to file"""
        if self.format == FileFormat.H5AD:
            adata.write_h5ad(self.file_path, **kwargs)
        elif self.format == FileFormat.LOOM:
            adata.write_loom(self.file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format for writing single-cell data: {self.format}")
    
    def get_single_cell_stats(self) -> Dict[str, Any]:
        """Get basic statistics about single-cell data"""
        adata = self.read_single_cell_data()
        
        stats = {
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'total_counts': adata.X.sum(),
            'median_genes_per_cell': np.median(np.array((adata.X > 0).sum(axis=1)).flatten()),
            'median_counts_per_cell': np.median(np.array(adata.X.sum(axis=1)).flatten()),
            'cell_names': adata.obs.index.tolist()[:100],  # First 100 cells
            'gene_names': adata.var.index.tolist()[:100]   # First 100 genes
        }
        
        return stats


# =================== Annotation File Handlers ===================

class AnnotationFileHandler(BaseFileHandler):
    """Handler for annotation files (GFF, GTF, BED)"""
    
    def read_annotations(self, feature_types: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Read annotations from GFF/GTF/BED files"""
        if self.format in [FileFormat.GFF, FileFormat.GTF]:
            return self._read_gff_gtf(feature_types)
        elif self.format == FileFormat.BED:
            return self._read_bed()
        else:
            raise ValueError(f"Unsupported annotation format: {self.format}")
    
    def _read_gff_gtf(self, feature_types: Optional[List[str]] = None) -> Iterator[Dict[str, Any]]:
        """Read GFF/GTF files"""
        with self._open_file() as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 9:
                    continue
                
                feature = {
                    'seqname': parts[0],
                    'source': parts[1],
                    'feature': parts[2],
                    'start': int(parts[3]),
                    'end': int(parts[4]),
                    'score': parts[5] if parts[5] != '.' else None,
                    'strand': parts[6],
                    'frame': parts[7] if parts[7] != '.' else None,
                    'attributes': self._parse_attributes(parts[8])
                }
                
                if feature_types is None or feature['feature'] in feature_types:
                    yield feature
    
    def _read_bed(self) -> Iterator[Dict[str, Any]]:
        """Read BED files"""
        with self._open_file() as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split('\t')
                feature = {
                    'chrom': parts[0],
                    'start': int(parts[1]),
                    'end': int(parts[2])
                }
                
                # Optional BED fields
                if len(parts) > 3:
                    feature['name'] = parts[3]
                if len(parts) > 4:
                    feature['score'] = int(parts[4])
                if len(parts) > 5:
                    feature['strand'] = parts[5]
                
                yield feature
    
    def _parse_attributes(self, attr_string: str) -> Dict[str, str]:
        """Parse GFF/GTF attributes"""
        attributes = {}
        
        if self.format == FileFormat.GTF:
            # GTF format: gene_id "value"; transcript_id "value";
            for item in attr_string.split(';'):
                item = item.strip()
                if not item:
                    continue
                parts = item.split(' ', 1)
                if len(parts) == 2:
                    key = parts[0]
                    value = parts[1].strip('"')
                    attributes[key] = value
        else:
            # GFF format: ID=value;Parent=value
            for item in attr_string.split(';'):
                item = item.strip()
                if '=' in item:
                    key, value = item.split('=', 1)
                    attributes[key] = value
        
        return attributes


# =================== Database Connectivity ===================

class DatabaseHandler:
    """Handler for biological databases"""
    
    def __init__(self, db_type: str, connection_params: Dict[str, Any]):
        self.db_type = db_type
        self.connection_params = connection_params
        self.connection = None
    
    async def connect(self):
        """Connect to database"""
        if self.db_type == 'sqlite':
            self.connection = sqlite3.connect(self.connection_params['database'])
        elif self.db_type == 'postgresql':
            import psycopg2
            self.connection = psycopg2.connect(**self.connection_params)
        elif self.db_type == 'mongodb':
            import pymongo
            client = pymongo.MongoClient(**self.connection_params)
            self.connection = client[self.connection_params['database']]
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    async def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute database query"""
        if not self.connection:
            await self.connect()
        
        if self.db_type in ['sqlite', 'postgresql']:
            cursor = self.connection.cursor()
            cursor.execute(query, params or ())
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch results
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            return results
        
        elif self.db_type == 'mongodb':
            # MongoDB query would be different
            # This is just a placeholder
            collection = self.connection[self.connection_params.get('collection', 'default')]
            return list(collection.find({}))
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()


# =================== Cloud Storage Integration ===================

class CloudStorageHandler:
    """Handler for cloud storage operations"""
    
    def __init__(self, provider: str, credentials: Dict[str, Any]):
        self.provider = provider
        self.credentials = credentials
        self.client = None
    
    async def connect(self):
        """Connect to cloud storage"""
        if self.provider == 'aws':
            self.client = boto3.client('s3', **self.credentials)
        elif self.provider == 'gcp':
            self.client = gcs.Client(**self.credentials)
        elif self.provider == 'azure':
            self.client = BlobServiceClient(**self.credentials)
        else:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")
    
    async def download_file(self, remote_path: str, local_path: str):
        """Download file from cloud storage"""
        if not self.client:
            await self.connect()
        
        if self.provider == 'aws':
            bucket, key = remote_path.split('/', 1)
            self.client.download_file(bucket, key, local_path)
        elif self.provider == 'gcp':
            bucket_name, blob_name = remote_path.split('/', 1)
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
        elif self.provider == 'azure':
            container, blob = remote_path.split('/', 1)
            blob_client = self.client.get_blob_client(container=container, blob=blob)
            with open(local_path, 'wb') as f:
                f.write(blob_client.download_blob().readall())
    
    async def upload_file(self, local_path: str, remote_path: str):
        """Upload file to cloud storage"""
        if not self.client:
            await self.connect()
        
        if self.provider == 'aws':
            bucket, key = remote_path.split('/', 1)
            self.client.upload_file(local_path, bucket, key)
        elif self.provider == 'gcp':
            bucket_name, blob_name = remote_path.split('/', 1)
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
        elif self.provider == 'azure':
            container, blob = remote_path.split('/', 1)
            blob_client = self.client.get_blob_client(container=container, blob=blob)
            with open(local_path, 'rb') as f:
                blob_client.upload_blob(f, overwrite=True)


# =================== Universal File Manager ===================

class UniversalFileManager:
    """Universal file manager that handles all bioinformatics file types"""
    
    def __init__(self):
        self.handlers = {}
        self.cloud_handlers = {}
        self.db_handlers = {}
    
    def get_handler(self, file_path: str) -> BaseFileHandler:
        """Get appropriate handler for file"""
        if file_path not in self.handlers:
            file_format, _ = FormatDetector.detect_format(file_path)
            
            if file_format in [FileFormat.FASTA, FileFormat.FASTQ]:
                self.handlers[file_path] = SequenceFileHandler(file_path)
            elif file_format in [FileFormat.SAM, FileFormat.BAM, FileFormat.CRAM]:
                self.handlers[file_path] = AlignmentFileHandler(file_path)
            elif file_format in [FileFormat.VCF, FileFormat.BCF]:
                self.handlers[file_path] = VariantFileHandler(file_path)
            elif file_format in [FileFormat.CSV, FileFormat.TSV, FileFormat.EXCEL, FileFormat.HDF5]:
                self.handlers[file_path] = ExpressionDataHandler(file_path)
            elif file_format in [FileFormat.H5AD, FileFormat.LOOM]:
                self.handlers[file_path] = SingleCellDataHandler(file_path)
            elif file_format in [FileFormat.GFF, FileFormat.GTF, FileFormat.BED]:
                self.handlers[file_path] = AnnotationFileHandler(file_path)
            else:
                self.handlers[file_path] = BaseFileHandler(file_path)
        
        return self.handlers[file_path]
    
    async def read_file(self, file_path: str, **kwargs) -> Any:
        """Universal file reader"""
        handler = self.get_handler(file_path)
        
        if isinstance(handler, SequenceFileHandler):
            return list(handler.read_sequences(**kwargs))
        elif isinstance(handler, ExpressionDataHandler):
            return handler.read_expression_matrix(**kwargs)
        elif isinstance(handler, SingleCellDataHandler):
            return handler.read_single_cell_data(**kwargs)
        elif isinstance(handler, VariantFileHandler):
            return list(handler.read_variants(**kwargs))
        elif isinstance(handler, AlignmentFileHandler):
            return list(handler.read_alignments(**kwargs))
        elif isinstance(handler, AnnotationFileHandler):
            return list(handler.read_annotations(**kwargs))
        else:
            raise ValueError(f"No reader available for {file_path}")
    
    async def get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Get statistics for any file type"""
        handler = self.get_handler(file_path)
        
        if isinstance(handler, SequenceFileHandler):
            return handler.get_sequence_stats()
        elif isinstance(handler, ExpressionDataHandler):
            return handler.get_expression_stats()
        elif isinstance(handler, SingleCellDataHandler):
            return handler.get_single_cell_stats()
        elif isinstance(handler, VariantFileHandler):
            return handler.get_variant_stats()
        elif isinstance(handler, AlignmentFileHandler):
            return handler.get_alignment_stats()
        else:
            return {"file_path": file_path, "format": handler.format.value}


# =================== Example Usage ===================

async def example_file_operations():
    """Example of using the file I/O system"""
    
    # Initialize universal file manager
    file_manager = UniversalFileManager()
    
    # Example files (these would be real files in practice)
    example_files = [
        "example.fasta",
        "example.fastq.gz",
        "example.vcf",
        "example.bam",
        "expression_matrix.csv",
        "single_cell_data.h5ad",
        "annotations.gtf"
    ]
    
    for file_path in example_files:
        try:
            print(f"\nProcessing {file_path}:")
            
            # Get file statistics
            stats = await file_manager.get_file_stats(file_path)
            print(f"Stats: {stats}")
            
            # Read file content (limited for demo)
            # data = await file_manager.read_file(file_path, max_records=10)
            # print(f"Data preview: {type(data)}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Cloud storage example
    print("\nCloud storage operations:")
    
    # AWS S3 example
    aws_handler = CloudStorageHandler('aws', {
        'aws_access_key_id': 'your_key',
        'aws_secret_access_key': 'your_secret',
        'region_name': 'us-west-2'
    })
    
    # Download file from S3 (commented out for demo)
    # await aws_handler.download_file('my-bucket/data.fastq.gz', 'local_data.fastq.gz')
    
    print("File I/O system demonstration complete!")


if __name__ == "__main__":
    asyncio.run(example_file_operations())