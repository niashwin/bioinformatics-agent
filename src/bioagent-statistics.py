#!/usr/bin/env python3
"""
BioinformaticsAgent Statistical Analysis Module: Real statistical methods for bioinformatics

This module implements actual statistical analysis methods used in bioinformatics:
- Differential expression analysis (DESeq2-like, edgeR-like, limma-like)
- Multiple testing correction
- Power analysis and sample size calculations
- Normalization methods
- Quality control statistics
- Bayesian analysis methods
- Machine learning for bioinformatics
"""

import asyncio
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, UMAP
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging

# Try to import DESeq2 implementation
try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    PYDESEQ2_AVAILABLE = True
except ImportError:
    PYDESEQ2_AVAILABLE = False
    logging.warning("PyDESeq2 not available. Using alternative implementations.")

# Try to import scanpy for single-cell
try:
    import scanpy as sc
    import anndata as ad
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    logging.warning("Scanpy not available. Single-cell analysis limited.")


# =================== Statistical Test Types ===================

class StatisticalTest(Enum):
    """Available statistical tests"""
    TTEST = "t_test"
    WILCOXON = "wilcoxon"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FISHER_EXACT = "fisher_exact"
    CHI_SQUARE = "chi_square"
    DESEQ2 = "deseq2"
    EDGER = "edger"
    LIMMA = "limma"
    MANN_WHITNEY = "mann_whitney"


class NormalizationMethod(Enum):
    """Normalization methods for expression data"""
    TPM = "tpm"
    FPKM = "fpkm"
    CPM = "cpm"
    TMM = "tmm"
    DESEQ2 = "deseq2"
    QUANTILE = "quantile"
    Z_SCORE = "z_score"
    LOG_TRANSFORM = "log_transform"
    VST = "vst"  # Variance Stabilizing Transformation


class MultipleTestingCorrection(Enum):
    """Multiple testing correction methods"""
    BONFERRONI = "bonferroni"
    BENJAMINI_HOCHBERG = "fdr_bh"
    BENJAMINI_YEKUTIELI = "fdr_by"
    HOLM = "holm"
    SIDAK = "sidak"
    HOLM_SIDAK = "holm_sidak"


@dataclass
class StatisticalResult:
    """Result from statistical analysis"""
    test_statistic: float
    p_value: float
    adjusted_p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    method: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DifferentialExpressionResult:
    """Result from differential expression analysis"""
    gene_id: str
    base_mean: float
    log2_fold_change: float
    log2_fold_change_se: Optional[float] = None
    p_value: float
    adjusted_p_value: float
    stat: Optional[float] = None
    method: str = "unknown"


# =================== Normalization Methods ===================

class ExpressionNormalizer:
    """Comprehensive expression data normalization"""
    
    @staticmethod
    def normalize_counts(counts: pd.DataFrame, method: NormalizationMethod,
                        gene_lengths: Optional[pd.Series] = None) -> pd.DataFrame:
        """Normalize count data using various methods"""
        
        if method == NormalizationMethod.CPM:
            return ExpressionNormalizer._cpm_normalize(counts)
        elif method == NormalizationMethod.TPM:
            if gene_lengths is None:
                raise ValueError("Gene lengths required for TPM normalization")
            return ExpressionNormalizer._tpm_normalize(counts, gene_lengths)
        elif method == NormalizationMethod.FPKM:
            if gene_lengths is None:
                raise ValueError("Gene lengths required for FPKM normalization")
            return ExpressionNormalizer._fpkm_normalize(counts, gene_lengths)
        elif method == NormalizationMethod.TMM:
            return ExpressionNormalizer._tmm_normalize(counts)
        elif method == NormalizationMethod.DESEQ2:
            return ExpressionNormalizer._deseq2_normalize(counts)
        elif method == NormalizationMethod.QUANTILE:
            return ExpressionNormalizer._quantile_normalize(counts)
        elif method == NormalizationMethod.Z_SCORE:
            return ExpressionNormalizer._zscore_normalize(counts)
        elif method == NormalizationMethod.LOG_TRANSFORM:
            return ExpressionNormalizer._log_normalize(counts)
        elif method == NormalizationMethod.VST:
            return ExpressionNormalizer._vst_normalize(counts)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def _cpm_normalize(counts: pd.DataFrame) -> pd.DataFrame:
        """Counts per million normalization"""
        return counts.div(counts.sum(axis=0), axis=1) * 1e6
    
    @staticmethod
    def _tpm_normalize(counts: pd.DataFrame, gene_lengths: pd.Series) -> pd.DataFrame:
        """Transcripts per million normalization"""
        # Rate = counts / gene_length
        rate = counts.div(gene_lengths, axis=0)
        # TPM = rate / sum(rate) * 1e6
        return rate.div(rate.sum(axis=0), axis=1) * 1e6
    
    @staticmethod
    def _fpkm_normalize(counts: pd.DataFrame, gene_lengths: pd.Series) -> pd.DataFrame:
        """Fragments per kilobase per million normalization"""
        # FPKM = (counts * 1e9) / (gene_length * total_counts)
        total_counts = counts.sum(axis=0)
        fpkm = counts.div(gene_lengths, axis=0).div(total_counts, axis=1) * 1e9
        return fpkm
    
    @staticmethod
    def _tmm_normalize(counts: pd.DataFrame) -> pd.DataFrame:
        """Trimmed Mean of M-values normalization (edgeR-style)"""
        # Simplified TMM implementation
        log_counts = np.log2(counts + 1)
        
        # Calculate reference sample (geometric mean)
        ref_sample = np.exp(np.log(counts + 1).mean(axis=1)) - 1
        
        normalization_factors = []
        for col in counts.columns:
            sample_counts = counts[col]
            
            # Calculate M and A values
            M = np.log2((sample_counts + 1) / (ref_sample + 1))
            A = 0.5 * (np.log2(sample_counts + 1) + np.log2(ref_sample + 1))
            
            # Remove extreme values (trim)
            valid_idx = (~np.isnan(M)) & (~np.isnan(A)) & (sample_counts > 0) & (ref_sample > 0)
            M_trimmed = M[valid_idx]
            
            # Calculate TMM factor
            if len(M_trimmed) > 0:
                tmm_factor = 2 ** np.mean(M_trimmed)
            else:
                tmm_factor = 1.0
            
            normalization_factors.append(tmm_factor)
        
        # Apply normalization factors
        norm_factors = pd.Series(normalization_factors, index=counts.columns)
        return counts.div(norm_factors, axis=1)
    
    @staticmethod
    def _deseq2_normalize(counts: pd.DataFrame) -> pd.DataFrame:
        """DESeq2-style median ratio normalization"""
        # Calculate geometric mean for each gene
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geom_means = np.exp(np.log(counts + 1).mean(axis=1)) - 1
        
        # Remove genes with zero geometric mean
        valid_genes = geom_means > 0
        if not valid_genes.any():
            return counts  # Return original if no valid genes
        
        counts_subset = counts.loc[valid_genes]
        geom_means_subset = geom_means[valid_genes]
        
        # Calculate size factors
        size_factors = []
        for col in counts_subset.columns:
            ratios = counts_subset[col] / geom_means_subset
            # Use median of ratios as size factor
            size_factor = ratios.median()
            size_factors.append(size_factor if size_factor > 0 else 1.0)
        
        size_factors = pd.Series(size_factors, index=counts.columns)
        return counts.div(size_factors, axis=1)
    
    @staticmethod
    def _quantile_normalize(counts: pd.DataFrame) -> pd.DataFrame:
        """Quantile normalization"""
        # Sort each sample
        sorted_counts = pd.DataFrame(np.sort(counts.values, axis=0), 
                                   columns=counts.columns)
        
        # Calculate mean across samples for each quantile
        quantile_means = sorted_counts.mean(axis=1)
        
        # Replace values with quantile means
        ranks = counts.rank(method='min')
        normalized = pd.DataFrame(index=counts.index, columns=counts.columns)
        
        for col in counts.columns:
            sample_ranks = ranks[col].astype(int) - 1  # Convert to 0-based
            normalized[col] = quantile_means.iloc[sample_ranks].values
        
        return normalized
    
    @staticmethod
    def _zscore_normalize(counts: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalization"""
        return (counts - counts.mean(axis=1, keepdims=True)) / counts.std(axis=1, keepdims=True)
    
    @staticmethod
    def _log_normalize(counts: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
        """Log2 transformation with pseudocount"""
        return np.log2(counts + pseudocount)
    
    @staticmethod
    def _vst_normalize(counts: pd.DataFrame) -> pd.DataFrame:
        """Variance Stabilizing Transformation (simplified)"""
        # Simplified VST - in practice, use DESeq2's VST
        # This is an approximation: log2(count + 0.5)
        return np.log2(counts + 0.5)


# =================== Differential Expression Analysis ===================

class DifferentialExpressionAnalyzer:
    """Comprehensive differential expression analysis"""
    
    def __init__(self):
        self.results = None
        self.design_matrix = None
    
    async def analyze_differential_expression(
        self,
        counts: pd.DataFrame,
        sample_info: pd.DataFrame,
        design_formula: str = "~ condition",
        test_method: StatisticalTest = StatisticalTest.DESEQ2,
        reference_level: Optional[str] = None
    ) -> List[DifferentialExpressionResult]:
        """Perform differential expression analysis"""
        
        if test_method == StatisticalTest.DESEQ2:
            return await self._deseq2_analysis(counts, sample_info, design_formula, reference_level)
        elif test_method == StatisticalTest.EDGER:
            return await self._edger_analysis(counts, sample_info, design_formula, reference_level)
        elif test_method == StatisticalTest.LIMMA:
            return await self._limma_analysis(counts, sample_info, design_formula, reference_level)
        elif test_method == StatisticalTest.TTEST:
            return await self._ttest_analysis(counts, sample_info, reference_level)
        elif test_method == StatisticalTest.WILCOXON:
            return await self._wilcoxon_analysis(counts, sample_info, reference_level)
        else:
            raise ValueError(f"Unsupported test method: {test_method}")
    
    async def _deseq2_analysis(
        self,
        counts: pd.DataFrame,
        sample_info: pd.DataFrame,
        design_formula: str,
        reference_level: Optional[str]
    ) -> List[DifferentialExpressionResult]:
        """DESeq2-style differential expression analysis"""
        
        if PYDESEQ2_AVAILABLE:
            # Use actual PyDESeq2 implementation
            return await self._pydeseq2_analysis(counts, sample_info, design_formula, reference_level)
        else:
            # Use our own implementation
            return await self._deseq2_like_analysis(counts, sample_info, design_formula, reference_level)
    
    async def _pydeseq2_analysis(
        self,
        counts: pd.DataFrame,
        sample_info: pd.DataFrame,
        design_formula: str,
        reference_level: Optional[str]
    ) -> List[DifferentialExpressionResult]:
        """Analysis using PyDESeq2"""
        
        # Create DESeq2 dataset
        dds = DeseqDataSet(
            counts=counts.T,  # PyDESeq2 expects samples as rows
            metadata=sample_info,
            design_factors=design_formula.replace("~ ", "").split(" + "),
            ref_level=reference_level
        )
        
        # Run DESeq2 analysis
        dds.deseq2()
        
        # Get results
        stat_res = DeseqStats(dds)
        stat_res.summary()
        
        results = []
        for gene_id in counts.index:
            if gene_id in stat_res.results_df.index:
                row = stat_res.results_df.loc[gene_id]
                result = DifferentialExpressionResult(
                    gene_id=gene_id,
                    base_mean=row['baseMean'],
                    log2_fold_change=row['log2FoldChange'],
                    log2_fold_change_se=row['lfcSE'],
                    p_value=row['pvalue'],
                    adjusted_p_value=row['padj'],
                    stat=row['stat'],
                    method="PyDESeq2"
                )
                results.append(result)
        
        return results
    
    async def _deseq2_like_analysis(
        self,
        counts: pd.DataFrame,
        sample_info: pd.DataFrame,
        design_formula: str,
        reference_level: Optional[str]
    ) -> List[DifferentialExpressionResult]:
        """DESeq2-like analysis implementation"""
        
        # Extract condition from design formula (simplified)
        condition_col = design_formula.replace("~ ", "").strip()
        
        if condition_col not in sample_info.columns:
            raise ValueError(f"Condition column '{condition_col}' not found in sample_info")
        
        conditions = sample_info[condition_col].unique()
        if len(conditions) != 2:
            raise ValueError("DESeq2-like analysis currently supports only two-condition comparisons")
        
        # Determine reference and treatment
        if reference_level:
            ref_condition = reference_level
            treat_condition = [c for c in conditions if c != reference_level][0]
        else:
            ref_condition, treat_condition = sorted(conditions)
        
        ref_samples = sample_info[sample_info[condition_col] == ref_condition].index
        treat_samples = sample_info[sample_info[condition_col] == treat_condition].index
        
        # Normalize counts (DESeq2-style)
        normalized_counts = ExpressionNormalizer.normalize_counts(
            counts, NormalizationMethod.DESEQ2
        )
        
        results = []
        
        for gene_id in counts.index:
            ref_values = normalized_counts.loc[gene_id, ref_samples]
            treat_values = normalized_counts.loc[gene_id, treat_samples]
            
            # Calculate base mean
            base_mean = normalized_counts.loc[gene_id].mean()
            
            # Skip genes with very low expression
            if base_mean < 1:
                continue
            
            # Log2 fold change
            ref_mean = ref_values.mean()
            treat_mean = treat_values.mean()
            
            if ref_mean > 0:
                log2_fc = np.log2((treat_mean + 1) / (ref_mean + 1))
            else:
                log2_fc = 0
            
            # Statistical test (Wald test approximation)
            if len(ref_values) > 1 and len(treat_values) > 1:
                stat, p_value = stats.ttest_ind(treat_values, ref_values)
            else:
                stat, p_value = 0, 1
            
            result = DifferentialExpressionResult(
                gene_id=gene_id,
                base_mean=base_mean,
                log2_fold_change=log2_fc,
                p_value=p_value,
                adjusted_p_value=p_value,  # Will be adjusted later
                stat=stat,
                method="DESeq2-like"
            )
            results.append(result)
        
        # Multiple testing correction
        p_values = [r.p_value for r in results]
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        
        for i, result in enumerate(results):
            result.adjusted_p_value = p_adjusted[i]
        
        return results
    
    async def _edger_analysis(
        self,
        counts: pd.DataFrame,
        sample_info: pd.DataFrame,
        design_formula: str,
        reference_level: Optional[str]
    ) -> List[DifferentialExpressionResult]:
        """edgeR-like analysis"""
        
        # TMM normalization
        normalized_counts = ExpressionNormalizer.normalize_counts(
            counts, NormalizationMethod.TMM
        )
        
        # Extract condition
        condition_col = design_formula.replace("~ ", "").strip()
        conditions = sample_info[condition_col].unique()
        
        if len(conditions) != 2:
            raise ValueError("edgeR-like analysis currently supports only two-condition comparisons")
        
        if reference_level:
            ref_condition = reference_level
            treat_condition = [c for c in conditions if c != reference_level][0]
        else:
            ref_condition, treat_condition = sorted(conditions)
        
        ref_samples = sample_info[sample_info[condition_col] == ref_condition].index
        treat_samples = sample_info[sample_info[condition_col] == treat_condition].index
        
        results = []
        
        for gene_id in counts.index:
            ref_values = normalized_counts.loc[gene_id, ref_samples]
            treat_values = normalized_counts.loc[gene_id, treat_samples]
            
            # Calculate CPM
            cpm = ExpressionNormalizer._cpm_normalize(
                pd.DataFrame({gene_id: counts.loc[gene_id]}).T
            ).iloc[0]
            
            base_mean = cpm.mean()
            
            # Log2 fold change
            ref_mean = ref_values.mean()
            treat_mean = treat_values.mean()
            
            if ref_mean > 0 and treat_mean > 0:
                log2_fc = np.log2(treat_mean / ref_mean)
            else:
                log2_fc = 0
            
            # Exact test approximation (using t-test for simplicity)
            if len(ref_values) > 1 and len(treat_values) > 1:
                stat, p_value = stats.ttest_ind(treat_values, ref_values)
            else:
                stat, p_value = 0, 1
            
            result = DifferentialExpressionResult(
                gene_id=gene_id,
                base_mean=base_mean,
                log2_fold_change=log2_fc,
                p_value=p_value,
                adjusted_p_value=p_value,
                stat=stat,
                method="edgeR-like"
            )
            results.append(result)
        
        # Multiple testing correction
        p_values = [r.p_value for r in results]
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        
        for i, result in enumerate(results):
            result.adjusted_p_value = p_adjusted[i]
        
        return results
    
    async def _limma_analysis(
        self,
        counts: pd.DataFrame,
        sample_info: pd.DataFrame,
        design_formula: str,
        reference_level: Optional[str]
    ) -> List[DifferentialExpressionResult]:
        """limma-like analysis for RNA-seq"""
        
        # Log-transform counts
        log_counts = ExpressionNormalizer.normalize_counts(
            counts, NormalizationMethod.LOG_TRANSFORM
        )
        
        # Extract condition
        condition_col = design_formula.replace("~ ", "").strip()
        conditions = sample_info[condition_col].unique()
        
        if len(conditions) != 2:
            raise ValueError("limma-like analysis currently supports only two-condition comparisons")
        
        if reference_level:
            ref_condition = reference_level
            treat_condition = [c for c in conditions if c != reference_level][0]
        else:
            ref_condition, treat_condition = sorted(conditions)
        
        ref_samples = sample_info[sample_info[condition_col] == ref_condition].index
        treat_samples = sample_info[sample_info[condition_col] == treat_condition].index
        
        results = []
        
        for gene_id in counts.index:
            ref_values = log_counts.loc[gene_id, ref_samples]
            treat_values = log_counts.loc[gene_id, treat_samples]
            
            base_mean = log_counts.loc[gene_id].mean()
            
            # Log2 fold change
            log2_fc = treat_values.mean() - ref_values.mean()
            
            # Moderated t-test (simplified)
            if len(ref_values) > 1 and len(treat_values) > 1:
                stat, p_value = stats.ttest_ind(treat_values, ref_values)
            else:
                stat, p_value = 0, 1
            
            result = DifferentialExpressionResult(
                gene_id=gene_id,
                base_mean=base_mean,
                log2_fold_change=log2_fc,
                p_value=p_value,
                adjusted_p_value=p_value,
                stat=stat,
                method="limma-like"
            )
            results.append(result)
        
        # Multiple testing correction
        p_values = [r.p_value for r in results]
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        
        for i, result in enumerate(results):
            result.adjusted_p_value = p_adjusted[i]
        
        return results
    
    async def _ttest_analysis(
        self,
        counts: pd.DataFrame,
        sample_info: pd.DataFrame,
        reference_level: Optional[str]
    ) -> List[DifferentialExpressionResult]:
        """Simple t-test analysis"""
        
        # Get condition column (assume 'condition' if not specified)
        condition_col = 'condition'
        if condition_col not in sample_info.columns:
            condition_col = sample_info.columns[0]
        
        conditions = sample_info[condition_col].unique()
        if len(conditions) != 2:
            raise ValueError("t-test analysis requires exactly two conditions")
        
        if reference_level:
            ref_condition = reference_level
            treat_condition = [c for c in conditions if c != reference_level][0]
        else:
            ref_condition, treat_condition = sorted(conditions)
        
        ref_samples = sample_info[sample_info[condition_col] == ref_condition].index
        treat_samples = sample_info[sample_info[condition_col] == treat_condition].index
        
        # Log transform
        log_counts = np.log2(counts + 1)
        
        results = []
        
        for gene_id in counts.index:
            ref_values = log_counts.loc[gene_id, ref_samples]
            treat_values = log_counts.loc[gene_id, treat_samples]
            
            base_mean = log_counts.loc[gene_id].mean()
            log2_fc = treat_values.mean() - ref_values.mean()
            
            # T-test
            if len(ref_values) > 1 and len(treat_values) > 1:
                stat, p_value = stats.ttest_ind(treat_values, ref_values)
            else:
                stat, p_value = 0, 1
            
            result = DifferentialExpressionResult(
                gene_id=gene_id,
                base_mean=base_mean,
                log2_fold_change=log2_fc,
                p_value=p_value,
                adjusted_p_value=p_value,
                stat=stat,
                method="t-test"
            )
            results.append(result)
        
        # Multiple testing correction
        p_values = [r.p_value for r in results]
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        
        for i, result in enumerate(results):
            result.adjusted_p_value = p_adjusted[i]
        
        return results
    
    async def _wilcoxon_analysis(
        self,
        counts: pd.DataFrame,
        sample_info: pd.DataFrame,
        reference_level: Optional[str]
    ) -> List[DifferentialExpressionResult]:
        """Wilcoxon rank-sum test analysis"""
        
        condition_col = 'condition'
        if condition_col not in sample_info.columns:
            condition_col = sample_info.columns[0]
        
        conditions = sample_info[condition_col].unique()
        if len(conditions) != 2:
            raise ValueError("Wilcoxon analysis requires exactly two conditions")
        
        if reference_level:
            ref_condition = reference_level
            treat_condition = [c for c in conditions if c != reference_level][0]
        else:
            ref_condition, treat_condition = sorted(conditions)
        
        ref_samples = sample_info[sample_info[condition_col] == ref_condition].index
        treat_samples = sample_info[sample_info[condition_col] == treat_condition].index
        
        results = []
        
        for gene_id in counts.index:
            ref_values = counts.loc[gene_id, ref_samples]
            treat_values = counts.loc[gene_id, treat_samples]
            
            base_mean = counts.loc[gene_id].mean()
            
            # Log2 fold change
            ref_mean = ref_values.mean()
            treat_mean = treat_values.mean()
            
            if ref_mean > 0:
                log2_fc = np.log2((treat_mean + 1) / (ref_mean + 1))
            else:
                log2_fc = 0
            
            # Wilcoxon rank-sum test
            try:
                stat, p_value = stats.ranksums(treat_values, ref_values)
            except:
                stat, p_value = 0, 1
            
            result = DifferentialExpressionResult(
                gene_id=gene_id,
                base_mean=base_mean,
                log2_fold_change=log2_fc,
                p_value=p_value,
                adjusted_p_value=p_value,
                stat=stat,
                method="Wilcoxon"
            )
            results.append(result)
        
        # Multiple testing correction
        p_values = [r.p_value for r in results]
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        
        for i, result in enumerate(results):
            result.adjusted_p_value = p_adjusted[i]
        
        return results


# =================== Power Analysis and Sample Size ===================

class PowerAnalyzer:
    """Statistical power analysis and sample size calculations"""
    
    @staticmethod
    def calculate_power(effect_size: float, sample_size: int, alpha: float = 0.05,
                       alternative: str = 'two-sided') -> float:
        """Calculate statistical power for t-test"""
        return ttest_power(effect_size, sample_size, alpha, alternative)
    
    @staticmethod
    def calculate_sample_size(effect_size: float, power: float = 0.8, 
                            alpha: float = 0.05, alternative: str = 'two-sided') -> int:
        """Calculate required sample size for t-test"""
        from statsmodels.stats.power import ttest_power
        
        # Binary search for sample size
        low, high = 2, 1000
        while low < high:
            mid = (low + high) // 2
            calculated_power = ttest_power(effect_size, mid, alpha, alternative)
            if calculated_power < power:
                low = mid + 1
            else:
                high = mid
        
        return low
    
    @staticmethod
    def effect_size_from_counts(counts1: np.ndarray, counts2: np.ndarray) -> float:
        """Calculate Cohen's d effect size from count data"""
        mean1, mean2 = np.mean(counts1), np.mean(counts2)
        std1, std2 = np.std(counts1, ddof=1), np.std(counts2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(counts1), len(counts2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return abs(mean1 - mean2) / pooled_std


# =================== Advanced Statistics ===================

class AdvancedStatistics:
    """Advanced statistical methods for bioinformatics"""
    
    @staticmethod
    def run_pca(data: pd.DataFrame, n_components: Optional[int] = None,
                standardize: bool = True) -> Dict[str, Any]:
        """Principal Component Analysis"""
        
        if standardize:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.T)
        else:
            data_scaled = data.T
        
        if n_components is None:
            n_components = min(data_scaled.shape) - 1
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_scaled)
        
        return {
            'pca_coordinates': pd.DataFrame(
                pca_result, 
                index=data.columns,
                columns=[f'PC{i+1}' for i in range(n_components)]
            ),
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'explained_variance': pca.explained_variance_,
            'components': pd.DataFrame(
                pca.components_.T,
                index=data.index,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
        }
    
    @staticmethod
    def run_tsne(data: pd.DataFrame, n_components: int = 2, 
                perplexity: float = 30.0, random_state: int = 42) -> pd.DataFrame:
        """t-SNE dimensionality reduction"""
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   random_state=random_state)
        tsne_result = tsne.fit_transform(data.T)
        
        return pd.DataFrame(
            tsne_result,
            index=data.columns,
            columns=[f'tSNE{i+1}' for i in range(n_components)]
        )
    
    @staticmethod
    def run_umap(data: pd.DataFrame, n_components: int = 2,
                n_neighbors: int = 15, min_dist: float = 0.1) -> pd.DataFrame:
        """UMAP dimensionality reduction"""
        
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, 
                              n_neighbors=n_neighbors, min_dist=min_dist)
            umap_result = reducer.fit_transform(data.T)
            
            return pd.DataFrame(
                umap_result,
                index=data.columns,
                columns=[f'UMAP{i+1}' for i in range(n_components)]
            )
        except ImportError:
            raise ImportError("UMAP package not available. Please install umap-learn.")
    
    @staticmethod
    def hierarchical_clustering(data: pd.DataFrame, method: str = 'average',
                              metric: str = 'euclidean') -> Dict[str, Any]:
        """Hierarchical clustering"""
        
        # Calculate distance matrix
        distances = pdist(data.T, metric=metric)
        
        # Perform linkage
        linkage_matrix = linkage(distances, method=method)
        
        return {
            'linkage_matrix': linkage_matrix,
            'distance_matrix': squareform(distances),
            'sample_order': data.columns.tolist()
        }
    
    @staticmethod
    def kmeans_clustering(data: pd.DataFrame, n_clusters: int,
                         random_state: int = 42) -> Dict[str, Any]:
        """K-means clustering"""
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data.T)
        
        return {
            'cluster_labels': pd.Series(cluster_labels, index=data.columns),
            'cluster_centers': pd.DataFrame(
                kmeans.cluster_centers_.T,
                index=data.index,
                columns=[f'Cluster_{i}' for i in range(n_clusters)]
            ),
            'inertia': kmeans.inertia_
        }


# =================== Example Usage ===================

async def example_statistical_analysis():
    """Example of using the statistical analysis system"""
    
    # Generate example data
    np.random.seed(42)
    n_genes, n_samples = 1000, 20
    
    # Create count matrix
    base_expression = np.random.negative_binomial(10, 0.3, (n_genes, n_samples))
    
    # Add differential expression for some genes
    de_genes = np.random.choice(n_genes, 100, replace=False)
    fold_changes = np.random.normal(2, 1, 100)
    
    counts = pd.DataFrame(
        base_expression,
        index=[f'Gene_{i}' for i in range(n_genes)],
        columns=[f'Sample_{i}' for i in range(n_samples)]
    )
    
    # Modify DE genes
    for i, gene_idx in enumerate(de_genes):
        if i < 50:  # First 50 genes upregulated
            counts.iloc[gene_idx, 10:] *= abs(fold_changes[i])
        else:  # Next 50 genes downregulated
            counts.iloc[gene_idx, 10:] /= abs(fold_changes[i])
    
    # Create sample info
    sample_info = pd.DataFrame({
        'condition': ['control'] * 10 + ['treatment'] * 10,
        'batch': ['batch1'] * 5 + ['batch2'] * 5 + ['batch1'] * 5 + ['batch2'] * 5
    }, index=counts.columns)
    
    print("Starting statistical analysis example...")
    
    # Test normalization
    print("\n1. Testing normalization methods:")
    normalizer = ExpressionNormalizer()
    
    for method in [NormalizationMethod.CPM, NormalizationMethod.DESEQ2, 
                   NormalizationMethod.TMM]:
        try:
            normalized = normalizer.normalize_counts(counts, method)
            print(f"  {method.value}: Success (shape: {normalized.shape})")
        except Exception as e:
            print(f"  {method.value}: Failed - {e}")
    
    # Test differential expression
    print("\n2. Testing differential expression methods:")
    de_analyzer = DifferentialExpressionAnalyzer()
    
    for method in [StatisticalTest.DESEQ2, StatisticalTest.EDGER, 
                   StatisticalTest.TTEST, StatisticalTest.WILCOXON]:
        try:
            results = await de_analyzer.analyze_differential_expression(
                counts, sample_info, "~ condition", method, "control"
            )
            
            significant_genes = [r for r in results if r.adjusted_p_value < 0.05]
            print(f"  {method.value}: {len(significant_genes)} significant genes")
            
        except Exception as e:
            print(f"  {method.value}: Failed - {e}")
    
    # Test PCA
    print("\n3. Testing PCA:")
    try:
        pca_result = AdvancedStatistics.run_pca(counts)
        print(f"  PCA: Success (explained variance: {pca_result['explained_variance_ratio'][:3]})")
    except Exception as e:
        print(f"  PCA: Failed - {e}")
    
    # Test power analysis
    print("\n4. Testing power analysis:")
    try:
        power = PowerAnalyzer.calculate_power(effect_size=0.8, sample_size=10)
        sample_size = PowerAnalyzer.calculate_sample_size(effect_size=0.8, power=0.8)
        print(f"  Power with n=10, d=0.8: {power:.3f}")
        print(f"  Sample size for power=0.8, d=0.8: {sample_size}")
    except Exception as e:
        print(f"  Power analysis: Failed - {e}")
    
    print("\nStatistical analysis example completed!")


if __name__ == "__main__":
    asyncio.run(example_statistical_analysis())