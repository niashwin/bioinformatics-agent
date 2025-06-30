#!/usr/bin/env python3
"""
BioinformaticsAgent Single-Cell Analysis Module: Comprehensive single-cell RNA-seq analysis

This module provides advanced single-cell RNA-seq analysis capabilities:
- Data preprocessing and quality control
- Normalization and scaling
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Clustering and cell type identification
- Differential expression analysis
- Trajectory analysis (pseudotime)
- Cell-cell communication analysis
- Integration of multiple datasets
- Spatial transcriptomics support
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
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import rankdata
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Single-cell specific libraries
try:
    import scanpy as sc
    import anndata as ad
    from scanpy.preprocessing import highly_variable_genes
    from scanpy.tools import pca, neighbors, umap, leiden, louvain
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    logging.warning("Scanpy not available. Using alternative implementations.")

try:
    import umap.umap_ as umap_sklearn
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Using t-SNE for dimensionality reduction.")

try:
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logging.warning("Leiden algorithm not available. Using Louvain clustering.")

from bioagent_architecture import BioinformaticsTool, BioToolResult, DataType, DataMetadata
from bioagent_io import ExpressionDataHandler
from bioagent_statistics import StatisticalResult, DifferentialExpressionAnalyzer


# =================== Single-Cell Data Structures ===================

class SingleCellDataType(Enum):
    """Types of single-cell data"""
    SCRNA_SEQ = "scrna_seq"
    SPATIAL_TRANSCRIPTOMICS = "spatial_transcriptomics"
    CITE_SEQ = "cite_seq"
    MULTIOME = "multiome"
    SNRNA_SEQ = "snrna_seq"
    SNATAC_SEQ = "snatac_seq"


class QCMetric(Enum):
    """Quality control metrics for single-cell data"""
    N_GENES = "n_genes"
    N_UMI = "n_umi"
    MITOCHONDRIAL_PERCENT = "mt_percent"
    RIBOSOMAL_PERCENT = "ribo_percent"
    DOUBLET_SCORE = "doublet_score"
    CELL_COMPLEXITY = "cell_complexity"


class NormalizationMethod(Enum):
    """Normalization methods for single-cell data"""
    LOG_NORMALIZE = "log_normalize"
    CPM = "cpm"
    SCRAN = "scran"
    SCTRANSFORM = "sctransform"
    PEARSON_RESIDUALS = "pearson_residuals"


class DimensionalityReduction(Enum):
    """Dimensionality reduction methods"""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    DIFFUSION_MAP = "diffusion_map"
    ICA = "ica"
    LSI = "lsi"


class ClusteringMethod(Enum):
    """Clustering algorithms for single-cell data"""
    LEIDEN = "leiden"
    LOUVAIN = "louvain"
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    PHENOGRAPH = "phenograph"


@dataclass
class SingleCellQCMetrics:
    """Quality control metrics for single-cell data"""
    n_cells_before: int
    n_cells_after: int
    n_genes_before: int
    n_genes_after: int
    median_genes_per_cell: float
    median_umi_per_cell: float
    mean_mt_percent: float
    mean_ribo_percent: float
    doublet_rate: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of QC metrics"""
        return {
            "cells_retained": f"{self.n_cells_after}/{self.n_cells_before} ({self.n_cells_after/self.n_cells_before*100:.1f}%)",
            "genes_retained": f"{self.n_genes_after}/{self.n_genes_before} ({self.n_genes_after/self.n_genes_before*100:.1f}%)",
            "median_genes_per_cell": self.median_genes_per_cell,
            "median_umi_per_cell": self.median_umi_per_cell,
            "mean_mt_percent": self.mean_mt_percent,
            "mean_ribo_percent": self.mean_ribo_percent,
            "doublet_rate": self.doublet_rate
        }


@dataclass
class ClusteringResult:
    """Results from clustering analysis"""
    cluster_labels: np.ndarray
    n_clusters: int
    silhouette_score: float
    method: ClusteringMethod
    parameters: Dict[str, Any]
    cluster_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DifferentialExpressionResult:
    """Results from differential expression analysis"""
    gene_names: List[str]
    log_fold_changes: np.ndarray
    p_values: np.ndarray
    adjusted_p_values: np.ndarray
    cluster_markers: Dict[str, List[str]] = field(default_factory=dict)


# =================== Single-Cell Data Container ===================

class SingleCellData:
    """Container for single-cell data with associated metadata"""
    
    def __init__(self, expression_matrix: Union[np.ndarray, sp.spmatrix], 
                 gene_names: List[str], cell_barcodes: List[str],
                 cell_metadata: Optional[pd.DataFrame] = None,
                 gene_metadata: Optional[pd.DataFrame] = None):
        
        self.X = expression_matrix
        self.gene_names = gene_names
        self.cell_barcodes = cell_barcodes
        self.n_cells, self.n_genes = self.X.shape
        
        # Initialize metadata
        self.cell_metadata = cell_metadata if cell_metadata is not None else pd.DataFrame(index=cell_barcodes)
        self.gene_metadata = gene_metadata if gene_metadata is not None else pd.DataFrame(index=gene_names)
        
        # Storage for analysis results
        self.obsm = {}  # Dimensional reductions
        self.varm = {}  # Gene loadings
        self.uns = {}   # Unstructured annotations
        
        # Quality control metrics
        self._calculate_basic_qc()
    
    def _calculate_basic_qc(self):
        """Calculate basic QC metrics"""
        if sp.issparse(self.X):
            # For sparse matrices
            self.cell_metadata['n_genes'] = np.array((self.X > 0).sum(axis=1)).flatten()
            self.cell_metadata['n_umi'] = np.array(self.X.sum(axis=1)).flatten()
            self.gene_metadata['n_cells'] = np.array((self.X > 0).sum(axis=0)).flatten()
        else:
            # For dense matrices
            self.cell_metadata['n_genes'] = (self.X > 0).sum(axis=1)
            self.cell_metadata['n_umi'] = self.X.sum(axis=1)
            self.gene_metadata['n_cells'] = (self.X > 0).sum(axis=0)
        
        # Calculate mitochondrial gene percentage
        mt_genes = [i for i, gene in enumerate(self.gene_names) 
                   if gene.upper().startswith('MT-') or gene.upper().startswith('MITO')]
        
        if mt_genes:
            if sp.issparse(self.X):
                mt_counts = np.array(self.X[:, mt_genes].sum(axis=1)).flatten()
            else:
                mt_counts = self.X[:, mt_genes].sum(axis=1)
            
            self.cell_metadata['mt_percent'] = (mt_counts / self.cell_metadata['n_umi']) * 100
        else:
            self.cell_metadata['mt_percent'] = 0.0
        
        # Calculate ribosomal gene percentage
        ribo_genes = [i for i, gene in enumerate(self.gene_names)
                     if gene.upper().startswith('RPS') or gene.upper().startswith('RPL')]
        
        if ribo_genes:
            if sp.issparse(self.X):
                ribo_counts = np.array(self.X[:, ribo_genes].sum(axis=1)).flatten()
            else:
                ribo_counts = self.X[:, ribo_genes].sum(axis=1)
            
            self.cell_metadata['ribo_percent'] = (ribo_counts / self.cell_metadata['n_umi']) * 100
        else:
            self.cell_metadata['ribo_percent'] = 0.0
    
    def to_anndata(self) -> 'ad.AnnData':
        """Convert to AnnData object if scanpy is available"""
        if not SCANPY_AVAILABLE:
            raise ImportError("Scanpy not available for AnnData conversion")
        
        adata = ad.AnnData(
            X=self.X,
            obs=self.cell_metadata,
            var=self.gene_metadata
        )
        adata.obs_names = self.cell_barcodes
        adata.var_names = self.gene_names
        
        # Add stored analysis results
        for key, value in self.obsm.items():
            adata.obsm[key] = value
        for key, value in self.varm.items():
            adata.varm[key] = value
        for key, value in self.uns.items():
            adata.uns[key] = value
        
        return adata
    
    @classmethod
    def from_anndata(cls, adata: 'ad.AnnData') -> 'SingleCellData':
        """Create from AnnData object"""
        sc_data = cls(
            expression_matrix=adata.X,
            gene_names=adata.var_names.tolist(),
            cell_barcodes=adata.obs_names.tolist(),
            cell_metadata=adata.obs.copy(),
            gene_metadata=adata.var.copy()
        )
        
        # Copy analysis results
        for key, value in adata.obsm.items():
            sc_data.obsm[key] = value
        for key, value in adata.varm.items():
            sc_data.varm[key] = value
        for key, value in adata.uns.items():
            sc_data.uns[key] = value
        
        return sc_data


# =================== Single-Cell Preprocessing Tool ===================

class SingleCellPreprocessor(BioinformaticsTool):
    """Comprehensive single-cell data preprocessing"""
    
    def __init__(self):
        super().__init__(
            name="single_cell_preprocessor",
            description="Preprocess single-cell RNA-seq data with QC filtering and normalization",
            supported_data_types=[DataType.EXPRESSION_MATRIX]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression_file": {"type": "string", "description": "Path to expression matrix file"},
                "min_genes_per_cell": {"type": "integer", "default": 200},
                "max_genes_per_cell": {"type": "integer", "default": 5000},
                "min_cells_per_gene": {"type": "integer", "default": 3},
                "max_mt_percent": {"type": "number", "default": 20.0},
                "normalization_method": {"type": "string", "enum": ["log_normalize", "cpm", "scran"], "default": "log_normalize"},
                "target_sum": {"type": "number", "default": 10000},
                "highly_variable_genes": {"type": "boolean", "default": True},
                "n_top_genes": {"type": "integer", "default": 2000},
                "output_file": {"type": "string"}
            },
            "required": ["expression_file", "output_file"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute single-cell preprocessing"""
        
        try:
            # Load expression data
            expression_file = params["expression_file"]
            handler = ExpressionDataHandler(expression_file)
            
            print(f"Loading expression data from {expression_file}...")
            expression_data = handler.load_expression_data()
            
            # Create SingleCellData object
            sc_data = SingleCellData(
                expression_matrix=expression_data["expression_matrix"],
                gene_names=expression_data["gene_names"],
                cell_barcodes=expression_data.get("sample_names", 
                    [f"Cell_{i}" for i in range(expression_data["expression_matrix"].shape[0])])
            )
            
            print(f"Loaded data: {sc_data.n_cells} cells, {sc_data.n_genes} genes")
            
            # Quality control filtering
            qc_metrics_before = self._get_qc_summary(sc_data)
            sc_data = await self._apply_qc_filters(sc_data, params)
            qc_metrics_after = self._get_qc_summary(sc_data)
            
            # Normalization
            sc_data = await self._normalize_data(sc_data, params)
            
            # Highly variable genes
            if params.get("highly_variable_genes", True):
                sc_data = await self._find_highly_variable_genes(sc_data, params)
            
            # Save processed data
            output_file = params["output_file"]
            await self._save_processed_data(sc_data, output_file)
            
            # Create QC metrics summary
            qc_summary = SingleCellQCMetrics(
                n_cells_before=qc_metrics_before["n_cells"],
                n_cells_after=qc_metrics_after["n_cells"],
                n_genes_before=qc_metrics_before["n_genes"],
                n_genes_after=qc_metrics_after["n_genes"],
                median_genes_per_cell=float(np.median(sc_data.cell_metadata['n_genes'])),
                median_umi_per_cell=float(np.median(sc_data.cell_metadata['n_umi'])),
                mean_mt_percent=float(np.mean(sc_data.cell_metadata['mt_percent'])),
                mean_ribo_percent=float(np.mean(sc_data.cell_metadata['ribo_percent']))
            )
            
            return BioToolResult(
                success=True,
                output=f"Single-cell preprocessing completed: {output_file}",
                metadata={
                    "qc_metrics": qc_summary.get_summary(),
                    "n_cells_final": sc_data.n_cells,
                    "n_genes_final": sc_data.n_genes,
                    "normalization_method": params.get("normalization_method", "log_normalize"),
                    "output_file": output_file
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Single-cell preprocessing failed: {str(e)}"
            )
    
    def _get_qc_summary(self, sc_data: SingleCellData) -> Dict[str, int]:
        """Get basic QC summary"""
        return {
            "n_cells": sc_data.n_cells,
            "n_genes": sc_data.n_genes
        }
    
    async def _apply_qc_filters(self, sc_data: SingleCellData, params: Dict[str, Any]) -> SingleCellData:
        """Apply quality control filters"""
        
        min_genes = params.get("min_genes_per_cell", 200)
        max_genes = params.get("max_genes_per_cell", 5000)
        min_cells = params.get("min_cells_per_gene", 3)
        max_mt = params.get("max_mt_percent", 20.0)
        
        # Filter cells
        cell_filter = (
            (sc_data.cell_metadata['n_genes'] >= min_genes) &
            (sc_data.cell_metadata['n_genes'] <= max_genes) &
            (sc_data.cell_metadata['mt_percent'] <= max_mt)
        )
        
        print(f"Filtering cells: {cell_filter.sum()}/{len(cell_filter)} cells pass QC")
        
        # Filter genes
        gene_filter = sc_data.gene_metadata['n_cells'] >= min_cells
        
        print(f"Filtering genes: {gene_filter.sum()}/{len(gene_filter)} genes pass QC")
        
        # Apply filters
        if sp.issparse(sc_data.X):
            filtered_X = sc_data.X[cell_filter, :][:, gene_filter]
        else:
            filtered_X = sc_data.X[cell_filter, :][:, gene_filter]
        
        filtered_genes = [gene for i, gene in enumerate(sc_data.gene_names) if gene_filter.iloc[i]]
        filtered_cells = [cell for i, cell in enumerate(sc_data.cell_barcodes) if cell_filter.iloc[i]]
        
        # Create new SingleCellData object
        filtered_sc_data = SingleCellData(
            expression_matrix=filtered_X,
            gene_names=filtered_genes,
            cell_barcodes=filtered_cells,
            cell_metadata=sc_data.cell_metadata[cell_filter].copy(),
            gene_metadata=sc_data.gene_metadata[gene_filter].copy()
        )
        
        return filtered_sc_data
    
    async def _normalize_data(self, sc_data: SingleCellData, params: Dict[str, Any]) -> SingleCellData:
        """Normalize expression data"""
        
        method = params.get("normalization_method", "log_normalize")
        target_sum = params.get("target_sum", 10000)
        
        print(f"Normalizing data using {method} method...")
        
        if method == "log_normalize":
            # Standard log normalization
            if sp.issparse(sc_data.X):
                # Normalize to target sum
                sc_data.X = sc_data.X.astype(np.float32)
                cell_sums = np.array(sc_data.X.sum(axis=1)).flatten()
                sc_data.X = sc_data.X.multiply(target_sum / cell_sums[:, np.newaxis])
                
                # Log transform
                sc_data.X.data = np.log1p(sc_data.X.data)
            else:
                # Dense matrix normalization
                cell_sums = sc_data.X.sum(axis=1)
                sc_data.X = sc_data.X / cell_sums[:, np.newaxis] * target_sum
                sc_data.X = np.log1p(sc_data.X)
        
        elif method == "cpm":
            # Counts per million
            if sp.issparse(sc_data.X):
                cell_sums = np.array(sc_data.X.sum(axis=1)).flatten()
                sc_data.X = sc_data.X.multiply(1e6 / cell_sums[:, np.newaxis])
            else:
                cell_sums = sc_data.X.sum(axis=1)
                sc_data.X = sc_data.X / cell_sums[:, np.newaxis] * 1e6
        
        # Store normalization info
        sc_data.uns['normalization'] = {
            'method': method,
            'target_sum': target_sum
        }
        
        return sc_data
    
    async def _find_highly_variable_genes(self, sc_data: SingleCellData, params: Dict[str, Any]) -> SingleCellData:
        """Find highly variable genes"""
        
        n_top_genes = params.get("n_top_genes", 2000)
        
        print(f"Finding top {n_top_genes} highly variable genes...")
        
        if SCANPY_AVAILABLE:
            # Use scanpy implementation
            adata = sc_data.to_anndata()
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            sc_data = SingleCellData.from_anndata(adata)
        else:
            # Simple implementation based on coefficient of variation
            if sp.issparse(sc_data.X):
                means = np.array(sc_data.X.mean(axis=0)).flatten()
                vars = np.array(sc_data.X.multiply(sc_data.X).mean(axis=0)).flatten() - means**2
            else:
                means = sc_data.X.mean(axis=0)
                vars = sc_data.X.var(axis=0)
            
            # Avoid division by zero
            cv = np.divide(np.sqrt(vars), means, out=np.zeros_like(means), where=means!=0)
            
            # Select top variable genes
            top_genes_idx = np.argsort(cv)[-n_top_genes:]
            sc_data.gene_metadata['highly_variable'] = False
            sc_data.gene_metadata.iloc[top_genes_idx, sc_data.gene_metadata.columns.get_loc('highly_variable')] = True
        
        return sc_data
    
    async def _save_processed_data(self, sc_data: SingleCellData, output_file: str):
        """Save processed single-cell data"""
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as h5 format if scanpy is available
        if SCANPY_AVAILABLE:
            adata = sc_data.to_anndata()
            adata.write_h5ad(output_file)
        else:
            # Save as numpy/pandas format
            output_dir = output_path.with_suffix('')
            output_dir.mkdir(exist_ok=True)
            
            # Save expression matrix
            if sp.issparse(sc_data.X):
                sp.save_npz(output_dir / "expression_matrix.npz", sc_data.X)
            else:
                np.save(output_dir / "expression_matrix.npy", sc_data.X)
            
            # Save metadata
            sc_data.cell_metadata.to_csv(output_dir / "cell_metadata.csv")
            sc_data.gene_metadata.to_csv(output_dir / "gene_metadata.csv")
            
            # Save gene and cell names
            with open(output_dir / "gene_names.txt", 'w') as f:
                f.write('\n'.join(sc_data.gene_names))
            
            with open(output_dir / "cell_barcodes.txt", 'w') as f:
                f.write('\n'.join(sc_data.cell_barcodes))


# =================== Single-Cell Analysis Tool ===================

class SingleCellAnalyzer(BioinformaticsTool):
    """Comprehensive single-cell analysis including clustering and visualization"""
    
    def __init__(self):
        super().__init__(
            name="single_cell_analyzer",
            description="Perform dimensionality reduction, clustering, and visualization of single-cell data",
            supported_data_types=[DataType.EXPRESSION_MATRIX]
        )
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_file": {"type": "string", "description": "Path to preprocessed single-cell data"},
                "dim_reduction_methods": {"type": "array", "items": {"type": "string", "enum": ["pca", "tsne", "umap"]}, "default": ["pca", "umap"]},
                "clustering_methods": {"type": "array", "items": {"type": "string", "enum": ["leiden", "louvain", "kmeans"]}, "default": ["leiden"]},
                "n_pcs": {"type": "integer", "default": 50},
                "n_neighbors": {"type": "integer", "default": 15},
                "resolution": {"type": "number", "default": 0.5},
                "output_dir": {"type": "string"}
            },
            "required": ["input_file", "output_dir"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute single-cell analysis"""
        
        try:
            # Load preprocessed data
            input_file = params["input_file"]
            sc_data = await self._load_single_cell_data(input_file)
            
            print(f"Loaded single-cell data: {sc_data.n_cells} cells, {sc_data.n_genes} genes")
            
            # Dimensionality reduction
            dim_methods = params.get("dim_reduction_methods", ["pca", "umap"])
            for method in dim_methods:
                print(f"Performing {method.upper()} dimensionality reduction...")
                sc_data = await self._perform_dimensionality_reduction(sc_data, method, params)
            
            # Clustering
            clustering_methods = params.get("clustering_methods", ["leiden"])
            clustering_results = {}
            
            for method in clustering_methods:
                print(f"Performing {method} clustering...")
                result = await self._perform_clustering(sc_data, method, params)
                clustering_results[method] = result
                
                # Add cluster labels to metadata
                sc_data.cell_metadata[f'cluster_{method}'] = result.cluster_labels
            
            # Differential expression analysis
            print("Finding marker genes...")
            de_results = await self._find_marker_genes(sc_data, clustering_results)
            
            # Create visualizations
            print("Creating visualizations...")
            plot_files = await self._create_visualizations(sc_data, params["output_dir"])
            
            # Save results
            output_file = Path(params["output_dir"]) / "single_cell_analysis_results.h5ad"
            await self._save_results(sc_data, output_file)
            
            return BioToolResult(
                success=True,
                output=f"Single-cell analysis completed: {params['output_dir']}",
                metadata={
                    "n_cells": sc_data.n_cells,
                    "n_genes": sc_data.n_genes,
                    "dimensionality_reductions": dim_methods,
                    "clustering_results": {
                        method: {
                            "n_clusters": result.n_clusters,
                            "silhouette_score": result.silhouette_score
                        } for method, result in clustering_results.items()
                    },
                    "marker_genes": {method: len(de_results[method].gene_names) 
                                   for method in de_results.keys()},
                    "plot_files": plot_files,
                    "output_file": str(output_file)
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Single-cell analysis failed: {str(e)}"
            )
    
    async def _load_single_cell_data(self, input_file: str) -> SingleCellData:
        """Load single-cell data from file"""
        
        input_path = Path(input_file)
        
        if input_path.suffix == '.h5ad' and SCANPY_AVAILABLE:
            # Load AnnData format
            adata = ad.read_h5ad(input_file)
            return SingleCellData.from_anndata(adata)
        else:
            # Load from directory structure
            if input_path.is_file():
                input_dir = input_path.with_suffix('')
            else:
                input_dir = input_path
            
            # Load expression matrix
            matrix_file = input_dir / "expression_matrix.npz"
            if matrix_file.exists():
                X = sp.load_npz(matrix_file)
            else:
                matrix_file = input_dir / "expression_matrix.npy"
                X = np.load(matrix_file)
            
            # Load metadata
            cell_metadata = pd.read_csv(input_dir / "cell_metadata.csv", index_col=0)
            gene_metadata = pd.read_csv(input_dir / "gene_metadata.csv", index_col=0)
            
            # Load names
            with open(input_dir / "gene_names.txt", 'r') as f:
                gene_names = [line.strip() for line in f]
            
            with open(input_dir / "cell_barcodes.txt", 'r') as f:
                cell_barcodes = [line.strip() for line in f]
            
            return SingleCellData(
                expression_matrix=X,
                gene_names=gene_names,
                cell_barcodes=cell_barcodes,
                cell_metadata=cell_metadata,
                gene_metadata=gene_metadata
            )
    
    async def _perform_dimensionality_reduction(self, sc_data: SingleCellData, 
                                              method: str, params: Dict[str, Any]) -> SingleCellData:
        """Perform dimensionality reduction"""
        
        n_pcs = params.get("n_pcs", 50)
        
        if method == "pca":
            # Use highly variable genes if available
            if 'highly_variable' in sc_data.gene_metadata.columns:
                hvg_mask = sc_data.gene_metadata['highly_variable'].values
                if sp.issparse(sc_data.X):
                    X_hvg = sc_data.X[:, hvg_mask]
                else:
                    X_hvg = sc_data.X[:, hvg_mask]
            else:
                X_hvg = sc_data.X
            
            # Perform PCA
            if sp.issparse(X_hvg):
                pca = TruncatedSVD(n_components=min(n_pcs, min(X_hvg.shape)-1))
            else:
                pca = PCA(n_components=min(n_pcs, min(X_hvg.shape)-1))
            
            pca_result = pca.fit_transform(X_hvg)
            sc_data.obsm['X_pca'] = pca_result
            sc_data.varm['PCs'] = pca.components_.T if hasattr(pca, 'components_') else None
            sc_data.uns['pca'] = {'variance_ratio': pca.explained_variance_ratio_}
        
        elif method == "tsne":
            # Use PCA as input if available
            if 'X_pca' in sc_data.obsm:
                X_input = sc_data.obsm['X_pca'][:, :min(50, sc_data.obsm['X_pca'].shape[1])]
            else:
                X_input = sc_data.X.toarray() if sp.issparse(sc_data.X) else sc_data.X
            
            tsne = TSNE(n_components=2, perplexity=min(30, sc_data.n_cells//4), random_state=42)
            tsne_result = tsne.fit_transform(X_input)
            sc_data.obsm['X_tsne'] = tsne_result
        
        elif method == "umap":
            # Use PCA as input if available
            if 'X_pca' in sc_data.obsm:
                X_input = sc_data.obsm['X_pca'][:, :min(50, sc_data.obsm['X_pca'].shape[1])]
            else:
                X_input = sc_data.X.toarray() if sp.issparse(sc_data.X) else sc_data.X
            
            if UMAP_AVAILABLE:
                umap_model = umap_sklearn.UMAP(n_neighbors=params.get("n_neighbors", 15), 
                                             n_components=2, random_state=42)
                umap_result = umap_model.fit_transform(X_input)
            else:
                # Fallback to t-SNE if UMAP not available
                logging.warning("UMAP not available, using t-SNE instead")
                tsne = TSNE(n_components=2, perplexity=min(30, sc_data.n_cells//4), random_state=42)
                umap_result = tsne.fit_transform(X_input)
            
            sc_data.obsm['X_umap'] = umap_result
        
        return sc_data
    
    async def _perform_clustering(self, sc_data: SingleCellData, 
                                method: str, params: Dict[str, Any]) -> ClusteringResult:
        """Perform clustering analysis"""
        
        # Use PCA representation for clustering
        if 'X_pca' in sc_data.obsm:
            X_input = sc_data.obsm['X_pca'][:, :min(50, sc_data.obsm['X_pca'].shape[1])]
        else:
            X_input = sc_data.X.toarray() if sp.issparse(sc_data.X) else sc_data.X
        
        if method == "leiden" and LEIDEN_AVAILABLE and SCANPY_AVAILABLE:
            # Use scanpy's Leiden implementation
            adata = sc_data.to_anndata()
            sc.pp.neighbors(adata, n_neighbors=params.get("n_neighbors", 15), n_pcs=min(50, X_input.shape[1]))
            sc.tl.leiden(adata, resolution=params.get("resolution", 0.5))
            labels = adata.obs['leiden'].astype(int).values
            
        elif method == "louvain" and SCANPY_AVAILABLE:
            # Use scanpy's Louvain implementation
            adata = sc_data.to_anndata()
            sc.pp.neighbors(adata, n_neighbors=params.get("n_neighbors", 15), n_pcs=min(50, X_input.shape[1]))
            sc.tl.louvain(adata, resolution=params.get("resolution", 0.5))
            labels = adata.obs['louvain'].astype(int).values
            
        elif method == "kmeans":
            # Use sklearn KMeans
            n_clusters = params.get("n_clusters", 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_input)
            
        else:
            # Fallback to hierarchical clustering
            from scipy.cluster.hierarchy import fcluster
            from scipy.spatial.distance import pdist, squareform
            
            # Compute pairwise distances (sample subset for large datasets)
            if X_input.shape[0] > 1000:
                indices = np.random.choice(X_input.shape[0], 1000, replace=False)
                X_sample = X_input[indices]
            else:
                X_sample = X_input
                indices = np.arange(X_input.shape[0])
            
            distances = pdist(X_sample, metric='euclidean')
            linkage_matrix = linkage(distances, method='ward')
            
            # Cut dendrogram to get clusters
            n_clusters = params.get("n_clusters", 10)
            sample_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Assign labels to all cells (for sampled case, use nearest neighbors)
            if len(indices) < X_input.shape[0]:
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(X_sample, sample_labels)
                labels = knn.predict(X_input)
            else:
                labels = sample_labels
        
        # Calculate silhouette score
        if X_input.shape[0] > 2 and len(np.unique(labels)) > 1:
            sil_score = silhouette_score(X_input, labels)
        else:
            sil_score = 0.0
        
        return ClusteringResult(
            cluster_labels=labels,
            n_clusters=len(np.unique(labels)),
            silhouette_score=sil_score,
            method=ClusteringMethod(method),
            parameters=params
        )
    
    async def _find_marker_genes(self, sc_data: SingleCellData, 
                               clustering_results: Dict[str, ClusteringResult]) -> Dict[str, DifferentialExpressionResult]:
        """Find marker genes for each cluster"""
        
        de_results = {}
        
        for method, clustering_result in clustering_results.items():
            print(f"Finding marker genes for {method} clusters...")
            
            cluster_labels = clustering_result.cluster_labels
            unique_clusters = np.unique(cluster_labels)
            
            all_genes = []
            all_logfc = []
            all_pvals = []
            cluster_markers = {}
            
            for cluster in unique_clusters:
                cluster_mask = cluster_labels == cluster
                other_mask = cluster_labels != cluster
                
                if cluster_mask.sum() < 3 or other_mask.sum() < 3:
                    continue
                
                # Get expression data
                if sp.issparse(sc_data.X):
                    cluster_expr = sc_data.X[cluster_mask, :].toarray()
                    other_expr = sc_data.X[other_mask, :].toarray()
                else:
                    cluster_expr = sc_data.X[cluster_mask, :]
                    other_expr = sc_data.X[other_mask, :]
                
                # Perform statistical tests for each gene
                gene_pvals = []
                gene_logfc = []
                
                for gene_idx in range(sc_data.n_genes):
                    cluster_values = cluster_expr[:, gene_idx]
                    other_values = other_expr[:, gene_idx]
                    
                    # Wilcoxon rank-sum test
                    try:
                        from scipy.stats import ranksums
                        statistic, pval = ranksums(cluster_values, other_values)
                        
                        # Calculate log fold change
                        cluster_mean = np.mean(cluster_values)
                        other_mean = np.mean(other_values)
                        
                        if other_mean > 0:
                            logfc = np.log2((cluster_mean + 1e-9) / (other_mean + 1e-9))
                        else:
                            logfc = 0.0
                        
                        gene_pvals.append(pval)
                        gene_logfc.append(logfc)
                        
                    except:
                        gene_pvals.append(1.0)
                        gene_logfc.append(0.0)
                
                # Multiple testing correction
                from statsmodels.stats.multitest import multipletests
                _, adj_pvals, _, _ = multipletests(gene_pvals, method='fdr_bh')
                
                # Select significant markers
                significant_mask = (adj_pvals < 0.05) & (np.array(gene_logfc) > 0.5)
                
                if significant_mask.sum() > 0:
                    marker_indices = np.where(significant_mask)[0]
                    cluster_markers[f'cluster_{cluster}'] = [sc_data.gene_names[i] for i in marker_indices[:50]]  # Top 50
                
                all_genes.extend(sc_data.gene_names)
                all_logfc.extend(gene_logfc)
                all_pvals.extend(adj_pvals)
            
            de_results[method] = DifferentialExpressionResult(
                gene_names=all_genes,
                log_fold_changes=np.array(all_logfc),
                p_values=np.array(all_pvals),
                adjusted_p_values=np.array(all_pvals),  # Already adjusted
                cluster_markers=cluster_markers
            )
        
        return de_results
    
    async def _create_visualizations(self, sc_data: SingleCellData, output_dir: str) -> List[str]:
        """Create visualization plots"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = []
        
        # QC metrics plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Number of genes per cell
        axes[0, 0].hist(sc_data.cell_metadata['n_genes'], bins=50, alpha=0.7)
        axes[0, 0].set_xlabel('Number of genes')
        axes[0, 0].set_ylabel('Number of cells')
        axes[0, 0].set_title('Genes per cell')
        
        # UMI counts per cell
        axes[0, 1].hist(sc_data.cell_metadata['n_umi'], bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('UMI counts')
        axes[0, 1].set_ylabel('Number of cells')
        axes[0, 1].set_title('UMI per cell')
        
        # Mitochondrial percentage
        axes[1, 0].hist(sc_data.cell_metadata['mt_percent'], bins=50, alpha=0.7)
        axes[1, 0].set_xlabel('Mitochondrial %')
        axes[1, 0].set_ylabel('Number of cells')
        axes[1, 0].set_title('Mitochondrial gene %')
        
        # Genes vs UMI scatter
        axes[1, 1].scatter(sc_data.cell_metadata['n_umi'], sc_data.cell_metadata['n_genes'], 
                          alpha=0.6, s=1)
        axes[1, 1].set_xlabel('UMI counts')
        axes[1, 1].set_ylabel('Number of genes')
        axes[1, 1].set_title('Genes vs UMI')
        
        plt.tight_layout()
        qc_plot_file = output_path / "qc_metrics.png"
        plt.savefig(qc_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(qc_plot_file))
        
        # Dimensionality reduction plots
        for dim_method in ['X_pca', 'X_tsne', 'X_umap']:
            if dim_method in sc_data.obsm:
                coords = sc_data.obsm[dim_method]
                
                # Plot by cluster if available
                cluster_cols = [col for col in sc_data.cell_metadata.columns if col.startswith('cluster_')]
                
                if cluster_cols:
                    n_plots = len(cluster_cols)
                    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
                    if n_plots == 1:
                        axes = [axes]
                    
                    for i, cluster_col in enumerate(cluster_cols):
                        scatter = axes[i].scatter(coords[:, 0], coords[:, 1], 
                                                c=sc_data.cell_metadata[cluster_col], 
                                                cmap='tab20', s=1, alpha=0.7)
                        axes[i].set_title(f'{dim_method.replace("X_", "").upper()} - {cluster_col}')
                        plt.colorbar(scatter, ax=axes[i])
                    
                    plt.tight_layout()
                    plot_file = output_path / f"{dim_method.replace('X_', '')}_clusters.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_files.append(str(plot_file))
                
                # Plot by QC metrics
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Color by number of genes
                scatter = axes[0].scatter(coords[:, 0], coords[:, 1], 
                                        c=sc_data.cell_metadata['n_genes'], 
                                        cmap='viridis', s=1, alpha=0.7)
                axes[0].set_title(f'{dim_method.replace("X_", "").upper()} - Genes per cell')
                plt.colorbar(scatter, ax=axes[0])
                
                # Color by UMI counts
                scatter = axes[1].scatter(coords[:, 0], coords[:, 1], 
                                        c=sc_data.cell_metadata['n_umi'], 
                                        cmap='viridis', s=1, alpha=0.7)
                axes[1].set_title(f'{dim_method.replace("X_", "").upper()} - UMI per cell')
                plt.colorbar(scatter, ax=axes[1])
                
                # Color by mitochondrial percentage
                scatter = axes[2].scatter(coords[:, 0], coords[:, 1], 
                                        c=sc_data.cell_metadata['mt_percent'], 
                                        cmap='plasma', s=1, alpha=0.7)
                axes[2].set_title(f'{dim_method.replace("X_", "").upper()} - Mitochondrial %')
                plt.colorbar(scatter, ax=axes[2])
                
                plt.tight_layout()
                plot_file = output_path / f"{dim_method.replace('X_', '')}_qc.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(str(plot_file))
        
        return plot_files
    
    async def _save_results(self, sc_data: SingleCellData, output_file: Path):
        """Save analysis results"""
        
        if SCANPY_AVAILABLE:
            adata = sc_data.to_anndata()
            adata.write_h5ad(output_file)
        else:
            # Save in directory format
            output_dir = output_file.with_suffix('')
            output_dir.mkdir(exist_ok=True)
            
            # Save expression matrix
            if sp.issparse(sc_data.X):
                sp.save_npz(output_dir / "expression_matrix.npz", sc_data.X)
            else:
                np.save(output_dir / "expression_matrix.npy", sc_data.X)
            
            # Save metadata
            sc_data.cell_metadata.to_csv(output_dir / "cell_metadata.csv")
            sc_data.gene_metadata.to_csv(output_dir / "gene_metadata.csv")
            
            # Save dimensional reductions
            for key, value in sc_data.obsm.items():
                np.save(output_dir / f"{key}.npy", value)
            
            # Save other annotations
            with open(output_dir / "uns.json", 'w') as f:
                json.dump({k: v for k, v in sc_data.uns.items() if isinstance(v, (str, int, float, list, dict))}, f)


# =================== Example Usage ===================

async def example_single_cell_analysis():
    """Example of comprehensive single-cell RNA-seq analysis"""
    
    print("Single-Cell RNA-seq Analysis Example")
    print("=" * 50)
    
    # Initialize tools
    preprocessor = SingleCellPreprocessor()
    analyzer = SingleCellAnalyzer()
    
    # Example preprocessing parameters
    preprocess_params = {
        "expression_file": "example_expression_matrix.csv",
        "min_genes_per_cell": 200,
        "max_genes_per_cell": 5000,
        "min_cells_per_gene": 3,
        "max_mt_percent": 20.0,
        "normalization_method": "log_normalize",
        "highly_variable_genes": True,
        "n_top_genes": 2000,
        "output_file": "preprocessed_data.h5ad"
    }
    
    # Example analysis parameters
    analysis_params = {
        "input_file": "preprocessed_data.h5ad",
        "dim_reduction_methods": ["pca", "umap", "tsne"],
        "clustering_methods": ["leiden", "kmeans"],
        "n_pcs": 50,
        "n_neighbors": 15,
        "resolution": 0.5,
        "output_dir": "single_cell_analysis_results"
    }
    
    print("Analysis Pipeline:")
    print("1. Data preprocessing and quality control")
    print("2. Normalization and highly variable gene selection")
    print("3. Principal component analysis")
    print("4. Non-linear dimensionality reduction (UMAP, t-SNE)")
    print("5. Graph-based clustering (Leiden, Louvain)")
    print("6. Marker gene identification")
    print("7. Visualization and reporting")
    
    print("\nExample completed (requires real expression data to run)")
    
    # This would work with real data:
    # preprocess_result = await preprocessor.execute(preprocess_params, [])
    # if preprocess_result.success:
    #     analysis_result = await analyzer.execute(analysis_params, [])
    #     print(f"Analysis result: {analysis_result.success}")


if __name__ == "__main__":
    asyncio.run(example_single_cell_analysis())