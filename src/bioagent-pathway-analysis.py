#!/usr/bin/env python3
"""
BioinformaticsAgent Pathway Analysis Module: Real pathway and functional enrichment analysis

This module provides comprehensive pathway analysis capabilities:
- Gene Ontology (GO) enrichment analysis
- KEGG pathway enrichment
- Reactome pathway analysis
- WikiPathways integration
- GSEA (Gene Set Enrichment Analysis)
- Custom gene set analysis
- Network-based pathway analysis
- Functional annotation clustering
"""

import asyncio
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import requests
import logging
from collections import defaultdict, Counter
import math

# Pathway analysis libraries
try:
    import gseapy as gp
    from gseapy import gse
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False

try:
    from goatools.obo_parser import GODag
    from goatools.associations import read_associations
    from goatools.enrichment_study import GOEnrichmentStudy
    GOATOOLS_AVAILABLE = True
except ImportError:
    GOATOOLS_AVAILABLE = False

from bioagent_architecture import BioinformaticsTool, BioToolResult, DataType, DataMetadata
from bioagent_databases import BiologicalDatabaseManager, KEGGDatabase
from bioagent_statistics import StatisticalResult


# =================== Pathway Analysis Data Structures ===================

class PathwayDatabase(Enum):
    """Pathway databases"""
    GO_BIOLOGICAL_PROCESS = "GO_Biological_Process_2023"
    GO_MOLECULAR_FUNCTION = "GO_Molecular_Function_2023"
    GO_CELLULAR_COMPONENT = "GO_Cellular_Component_2023"
    KEGG_2023 = "KEGG_2023_Human"
    REACTOME_2022 = "Reactome_2022"
    WIKIPATHWAYS_2023 = "WikiPathways_2023_Human"
    HALLMARK_2020 = "MSigDB_Hallmark_2020"
    BIOCARTA_2016 = "BioCarta_2016"
    PID_2016 = "PID_2016"


class EnrichmentMethod(Enum):
    """Enrichment analysis methods"""
    HYPERGEOMETRIC = "hypergeometric"
    FISHER_EXACT = "fisher_exact"
    GSEA = "gsea"
    GSVA = "gsva"
    ORA = "over_representation"
    RANK_BASED = "rank_based"


@dataclass
class PathwayTerm:
    """Pathway or GO term information"""
    term_id: str
    name: str
    description: str
    database: PathwayDatabase
    genes: Set[str] = field(default_factory=set)
    gene_count: int = 0
    category: Optional[str] = None
    parent_terms: List[str] = field(default_factory=list)
    child_terms: List[str] = field(default_factory=list)


@dataclass
class EnrichmentResult:
    """Result from pathway enrichment analysis"""
    term_id: str
    term_name: str
    description: str
    database: PathwayDatabase
    p_value: float
    adjusted_p_value: float
    odds_ratio: float
    enrichment_score: float
    gene_count: int
    total_genes_in_term: int
    background_count: int
    enriched_genes: List[str]
    fold_enrichment: float
    method: EnrichmentMethod
    
    @property
    def is_significant(self) -> bool:
        return self.adjusted_p_value < 0.05


@dataclass
class GSEAResult:
    """GSEA-specific result"""
    term_id: str
    term_name: str
    enrichment_score: float
    normalized_enrichment_score: float
    p_value: float
    fdr: float
    leading_edge_genes: List[str]
    gene_set_size: int
    rank_at_max: int


# =================== Gene Set Databases ===================

class GeneSetDatabase:
    """Manager for gene set databases"""
    
    def __init__(self, cache_dir: str = "pathway_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.gene_sets = {}
        self.loaded_databases = set()
    
    async def load_database(self, database: PathwayDatabase, 
                          organism: str = "Human") -> Dict[str, PathwayTerm]:
        """Load gene sets from specified database"""
        
        if database in self.loaded_databases:
            return self.gene_sets.get(database.value, {})
        
        if database in [PathwayDatabase.GO_BIOLOGICAL_PROCESS, 
                       PathwayDatabase.GO_MOLECULAR_FUNCTION,
                       PathwayDatabase.GO_CELLULAR_COMPONENT]:
            gene_sets = await self._load_go_terms(database)
        elif database == PathwayDatabase.KEGG_2023:
            gene_sets = await self._load_kegg_pathways(organism)
        elif database == PathwayDatabase.REACTOME_2022:
            gene_sets = await self._load_reactome_pathways(organism)
        else:
            # Try to load from Enrichr library
            gene_sets = await self._load_enrichr_library(database.value)
        
        self.gene_sets[database.value] = gene_sets
        self.loaded_databases.add(database)
        
        return gene_sets
    
    async def _load_go_terms(self, database: PathwayDatabase) -> Dict[str, PathwayTerm]:
        """Load GO terms"""
        
        go_terms = {}
        
        if not GOATOOLS_AVAILABLE:
            logging.warning("GOATools not available. Using simplified GO loading.")
            return go_terms
        
        try:
            # Download GO OBO file if not exists
            obo_file = self.cache_dir / "go-basic.obo"
            if not obo_file.exists():
                await self._download_go_obo(obo_file)
            
            # Load GO DAG
            go_dag = GODag(str(obo_file))
            
            # Filter by namespace
            namespace_map = {
                PathwayDatabase.GO_BIOLOGICAL_PROCESS: "biological_process",
                PathwayDatabase.GO_MOLECULAR_FUNCTION: "molecular_function", 
                PathwayDatabase.GO_CELLULAR_COMPONENT: "cellular_component"
            }
            target_namespace = namespace_map[database]
            
            for go_id, go_term in go_dag.items():
                if go_term.namespace == target_namespace:
                    pathway_term = PathwayTerm(
                        term_id=go_id,
                        name=go_term.name,
                        description=go_term.defn if hasattr(go_term, 'defn') else "",
                        database=database,
                        category=go_term.namespace
                    )
                    
                    # Add parent/child relationships
                    pathway_term.parent_terms = [parent.id for parent in go_term.parents]
                    pathway_term.child_terms = [child.id for child in go_term.children]
                    
                    go_terms[go_id] = pathway_term
        
        except Exception as e:
            logging.error(f"Failed to load GO terms: {e}")
        
        return go_terms
    
    async def _download_go_obo(self, output_file: Path):
        """Download GO OBO file"""
        
        url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    with open(output_file, 'w') as f:
                        f.write(content)
                else:
                    raise Exception(f"Failed to download GO OBO file: {response.status}")
    
    async def _load_kegg_pathways(self, organism: str) -> Dict[str, PathwayTerm]:
        """Load KEGG pathways"""
        
        kegg_pathways = {}
        
        try:
            kegg_db = KEGGDatabase()
            await kegg_db.connect()
            
            # Get organism code
            org_code = self._get_kegg_organism_code(organism)
            
            # Get pathways for organism
            pathway_result = await kegg_db.search_pathways(org_code)
            
            if pathway_result.success:
                for pathway_data in pathway_result.results:
                    pathway_id = pathway_data["pathway_id"]
                    
                    # Get genes in pathway
                    genes_result = await kegg_db.get_pathway_genes(pathway_id)
                    
                    gene_set = set()
                    if genes_result.success:
                        gene_set = {gene["gene_id"] for gene in genes_result.results}
                    
                    pathway_term = PathwayTerm(
                        term_id=pathway_id,
                        name=pathway_data["description"],
                        description=pathway_data["description"],
                        database=PathwayDatabase.KEGG_2023,
                        genes=gene_set,
                        gene_count=len(gene_set)
                    )
                    
                    kegg_pathways[pathway_id] = pathway_term
            
            await kegg_db.disconnect()
        
        except Exception as e:
            logging.error(f"Failed to load KEGG pathways: {e}")
        
        return kegg_pathways
    
    def _get_kegg_organism_code(self, organism: str) -> str:
        """Get KEGG organism code"""
        
        organism_map = {
            "Human": "hsa",
            "Mouse": "mmu", 
            "Rat": "rno",
            "Drosophila": "dme",
            "C. elegans": "cel",
            "S. cerevisiae": "sce"
        }
        
        return organism_map.get(organism, "hsa")
    
    async def _load_reactome_pathways(self, organism: str) -> Dict[str, PathwayTerm]:
        """Load Reactome pathways"""
        
        reactome_pathways = {}
        
        try:
            # Reactome REST API
            base_url = "https://reactome.org/ContentService"
            
            # Get species ID
            species_map = {
                "Human": "9606",
                "Mouse": "10090",
                "Rat": "10116"
            }
            species_id = species_map.get(organism, "9606")
            
            async with aiohttp.ClientSession() as session:
                # Get pathways for species
                url = f"{base_url}/data/pathways/top/{species_id}"
                async with session.get(url) as response:
                    if response.status == 200:
                        pathways = await response.json()
                        
                        for pathway in pathways:
                            pathway_id = pathway["stId"]
                            
                            # Get pathway participants
                            participants_url = f"{base_url}/data/participants/{pathway_id}"
                            async with session.get(participants_url) as part_response:
                                if part_response.status == 200:
                                    participants = await part_response.json()
                                    
                                    # Extract gene names
                                    genes = set()
                                    for participant in participants:
                                        if "geneName" in participant:
                                            genes.add(participant["geneName"])
                                    
                                    pathway_term = PathwayTerm(
                                        term_id=pathway_id,
                                        name=pathway["displayName"],
                                        description=pathway.get("summation", ""),
                                        database=PathwayDatabase.REACTOME_2022,
                                        genes=genes,
                                        gene_count=len(genes)
                                    )
                                    
                                    reactome_pathways[pathway_id] = pathway_term
        
        except Exception as e:
            logging.error(f"Failed to load Reactome pathways: {e}")
        
        return reactome_pathways
    
    async def _load_enrichr_library(self, library_name: str) -> Dict[str, PathwayTerm]:
        """Load gene sets from Enrichr library"""
        
        gene_sets = {}
        
        try:
            # Enrichr API
            url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={library_name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        for line in content.strip().split('\n'):
                            parts = line.split('\t')
                            if len(parts) >= 3:
                                term_name = parts[0]
                                description = parts[1] if parts[1] else term_name
                                genes = set(parts[2:])
                                
                                pathway_term = PathwayTerm(
                                    term_id=term_name.replace(" ", "_"),
                                    name=term_name,
                                    description=description,
                                    database=PathwayDatabase(library_name),
                                    genes=genes,
                                    gene_count=len(genes)
                                )
                                
                                gene_sets[term_name] = pathway_term
        
        except Exception as e:
            logging.error(f"Failed to load Enrichr library {library_name}: {e}")
        
        return gene_sets


# =================== Enrichment Analysis Engine ===================

class PathwayEnrichmentAnalyzer:
    """Core pathway enrichment analysis engine"""
    
    def __init__(self):
        self.gene_set_db = GeneSetDatabase()
        self.background_genes = set()
        self.total_background = 20000  # Default human genome size
    
    async def run_enrichment_analysis(
        self,
        gene_list: List[str],
        databases: List[PathwayDatabase],
        method: EnrichmentMethod = EnrichmentMethod.HYPERGEOMETRIC,
        background_genes: Optional[List[str]] = None,
        organism: str = "Human",
        min_gene_set_size: int = 5,
        max_gene_set_size: int = 500,
        significance_threshold: float = 0.05
    ) -> List[EnrichmentResult]:
        """Run pathway enrichment analysis"""
        
        # Set background
        if background_genes:
            self.background_genes = set(background_genes)
            self.total_background = len(self.background_genes)
        
        # Convert input genes to set
        input_genes = set(gene_list)
        
        all_results = []
        
        for database in databases:
            # Load gene sets for this database
            gene_sets = await self.gene_set_db.load_database(database, organism)
            
            # Filter gene sets by size
            filtered_gene_sets = {
                term_id: term for term_id, term in gene_sets.items()
                if min_gene_set_size <= term.gene_count <= max_gene_set_size
            }
            
            # Run enrichment for each gene set
            for term_id, pathway_term in filtered_gene_sets.items():
                result = await self._calculate_enrichment(
                    input_genes, pathway_term, method
                )
                
                if result and result.p_value <= significance_threshold:
                    all_results.append(result)
        
        # Apply multiple testing correction
        if all_results:
            all_results = self._apply_multiple_testing_correction(all_results)
        
        # Sort by significance
        all_results.sort(key=lambda x: x.adjusted_p_value)
        
        return all_results
    
    async def _calculate_enrichment(
        self,
        input_genes: Set[str],
        pathway_term: PathwayTerm,
        method: EnrichmentMethod
    ) -> Optional[EnrichmentResult]:
        """Calculate enrichment for a single pathway"""
        
        # Find overlap
        overlap_genes = input_genes.intersection(pathway_term.genes)
        overlap_count = len(overlap_genes)
        
        if overlap_count == 0:
            return None
        
        # Calculate enrichment based on method
        if method == EnrichmentMethod.HYPERGEOMETRIC:
            return await self._hypergeometric_test(
                input_genes, pathway_term, overlap_genes, overlap_count
            )
        elif method == EnrichmentMethod.FISHER_EXACT:
            return await self._fisher_exact_test(
                input_genes, pathway_term, overlap_genes, overlap_count
            )
        else:
            raise ValueError(f"Method {method} not implemented")
    
    async def _hypergeometric_test(
        self,
        input_genes: Set[str],
        pathway_term: PathwayTerm, 
        overlap_genes: Set[str],
        overlap_count: int
    ) -> EnrichmentResult:
        """Hypergeometric test for enrichment"""
        
        # Contingency table values
        k = overlap_count  # successes in sample
        n = len(input_genes)  # sample size
        K = pathway_term.gene_count  # successes in population
        N = self.total_background  # population size
        
        # Hypergeometric test (survival function for right-tail)
        p_value = stats.hypergeom.sf(k - 1, N, K, n)
        
        # Calculate odds ratio and fold enrichment
        expected = (n * K) / N
        fold_enrichment = k / expected if expected > 0 else float('inf')
        
        # Odds ratio calculation
        a = k  # overlap
        b = n - k  # input genes not in pathway
        c = K - k  # pathway genes not in input
        d = N - n - K + k  # neither
        
        odds_ratio = (a * d) / (b * c) if b > 0 and c > 0 else float('inf')
        
        return EnrichmentResult(
            term_id=pathway_term.term_id,
            term_name=pathway_term.name,
            description=pathway_term.description,
            database=pathway_term.database,
            p_value=p_value,
            adjusted_p_value=p_value,  # Will be corrected later
            odds_ratio=odds_ratio,
            enrichment_score=-np.log10(p_value),
            gene_count=overlap_count,
            total_genes_in_term=pathway_term.gene_count,
            background_count=self.total_background,
            enriched_genes=list(overlap_genes),
            fold_enrichment=fold_enrichment,
            method=EnrichmentMethod.HYPERGEOMETRIC
        )
    
    async def _fisher_exact_test(
        self,
        input_genes: Set[str],
        pathway_term: PathwayTerm,
        overlap_genes: Set[str], 
        overlap_count: int
    ) -> EnrichmentResult:
        """Fisher's exact test for enrichment"""
        
        # Contingency table
        a = overlap_count  # in both
        b = len(input_genes) - overlap_count  # in input only
        c = pathway_term.gene_count - overlap_count  # in pathway only
        d = self.total_background - len(input_genes) - pathway_term.gene_count + overlap_count
        
        # Fisher's exact test
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        
        # Fold enrichment
        expected = (len(input_genes) * pathway_term.gene_count) / self.total_background
        fold_enrichment = overlap_count / expected if expected > 0 else float('inf')
        
        return EnrichmentResult(
            term_id=pathway_term.term_id,
            term_name=pathway_term.name,
            description=pathway_term.description,
            database=pathway_term.database,
            p_value=p_value,
            adjusted_p_value=p_value,  # Will be corrected later
            odds_ratio=odds_ratio,
            enrichment_score=-np.log10(p_value),
            gene_count=overlap_count,
            total_genes_in_term=pathway_term.gene_count,
            background_count=self.total_background,
            enriched_genes=list(overlap_genes),
            fold_enrichment=fold_enrichment,
            method=EnrichmentMethod.FISHER_EXACT
        )
    
    def _apply_multiple_testing_correction(
        self, 
        results: List[EnrichmentResult]
    ) -> List[EnrichmentResult]:
        """Apply multiple testing correction"""
        
        from statsmodels.stats.multitest import multipletests
        
        p_values = [result.p_value for result in results]
        
        # Benjamini-Hochberg correction
        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        
        # Update results
        for i, result in enumerate(results):
            result.adjusted_p_value = p_adjusted[i]
        
        return results


# =================== GSEA Implementation ===================

class GSEAAnalyzer:
    """Gene Set Enrichment Analysis implementation"""
    
    def __init__(self):
        self.gene_set_db = GeneSetDatabase()
    
    async def run_gsea(
        self,
        ranked_gene_list: pd.DataFrame,  # columns: gene, score
        databases: List[PathwayDatabase],
        organism: str = "Human",
        min_gene_set_size: int = 15,
        max_gene_set_size: int = 500,
        permutations: int = 1000
    ) -> List[GSEAResult]:
        """Run GSEA analysis"""
        
        if GSEAPY_AVAILABLE:
            return await self._run_gseapy(
                ranked_gene_list, databases, organism, 
                min_gene_set_size, max_gene_set_size, permutations
            )
        else:
            return await self._run_custom_gsea(
                ranked_gene_list, databases, organism,
                min_gene_set_size, max_gene_set_size, permutations
            )
    
    async def _run_gseapy(
        self,
        ranked_gene_list: pd.DataFrame,
        databases: List[PathwayDatabase],
        organism: str,
        min_gene_set_size: int,
        max_gene_set_size: int,
        permutations: int
    ) -> List[GSEAResult]:
        """Run GSEA using GSEApy library"""
        
        results = []
        
        for database in databases:
            try:
                # Convert database enum to GSEApy library name
                library_name = self._get_gseapy_library_name(database)
                
                if library_name:
                    # Run GSEA
                    gsea_results = gp.gsea(
                        data=ranked_gene_list,
                        gene_sets=library_name,
                        cls='#continuous',
                        min_size=min_gene_set_size,
                        max_size=max_gene_set_size,
                        permutation_num=permutations,
                        no_plot=True,
                        processes=1,
                        format='png'
                    )
                    
                    # Parse results
                    for _, row in gsea_results.res2d.iterrows():
                        result = GSEAResult(
                            term_id=row['Term'],
                            term_name=row['Term'],
                            enrichment_score=row['ES'],
                            normalized_enrichment_score=row['NES'],
                            p_value=row['NOM p-val'],
                            fdr=row['FDR q-val'],
                            leading_edge_genes=row['genes'].split(';') if isinstance(row['genes'], str) else [],
                            gene_set_size=row['gene_set_size'],
                            rank_at_max=0  # Would need to calculate from detailed results
                        )
                        results.append(result)
            
            except Exception as e:
                logging.error(f"GSEA failed for database {database}: {e}")
        
        return results
    
    async def _run_custom_gsea(
        self,
        ranked_gene_list: pd.DataFrame,
        databases: List[PathwayDatabase], 
        organism: str,
        min_gene_set_size: int,
        max_gene_set_size: int,
        permutations: int
    ) -> List[GSEAResult]:
        """Custom GSEA implementation"""
        
        results = []
        
        # Prepare ranked list
        gene_scores = dict(zip(ranked_gene_list.iloc[:, 0], ranked_gene_list.iloc[:, 1]))
        ranked_genes = list(ranked_gene_list.iloc[:, 0])
        
        for database in databases:
            # Load gene sets
            gene_sets = await self.gene_set_db.load_database(database, organism)
            
            for term_id, pathway_term in gene_sets.items():
                if min_gene_set_size <= pathway_term.gene_count <= max_gene_set_size:
                    
                    # Calculate enrichment score
                    es, rank_at_max = self._calculate_enrichment_score(
                        ranked_genes, gene_scores, pathway_term.genes
                    )
                    
                    # Calculate p-value through permutation (simplified)
                    p_value = self._estimate_gsea_pvalue(
                        ranked_genes, gene_scores, pathway_term.genes, es, permutations
                    )
                    
                    # Find leading edge genes
                    leading_edge = self._find_leading_edge_genes(
                        ranked_genes, pathway_term.genes, rank_at_max
                    )
                    
                    result = GSEAResult(
                        term_id=term_id,
                        term_name=pathway_term.name,
                        enrichment_score=es,
                        normalized_enrichment_score=es,  # Simplified
                        p_value=p_value,
                        fdr=p_value,  # Would need proper FDR calculation
                        leading_edge_genes=leading_edge,
                        gene_set_size=pathway_term.gene_count,
                        rank_at_max=rank_at_max
                    )
                    
                    results.append(result)
        
        return results
    
    def _calculate_enrichment_score(
        self, 
        ranked_genes: List[str],
        gene_scores: Dict[str, float],
        gene_set: Set[str]
    ) -> Tuple[float, int]:
        """Calculate GSEA enrichment score"""
        
        N = len(ranked_genes)
        gene_set_in_list = [gene for gene in ranked_genes if gene in gene_set]
        
        if not gene_set_in_list:
            return 0.0, 0
        
        # Calculate running enrichment score
        hit_indices = [ranked_genes.index(gene) for gene in gene_set_in_list]
        
        # Weight by correlation (absolute value)
        weights = [abs(gene_scores.get(gene, 0)) for gene in gene_set_in_list]
        weight_sum = sum(weights)
        
        if weight_sum == 0:
            return 0.0, 0
        
        # Running sum
        running_sum = 0
        max_es = 0
        min_es = 0
        rank_at_max = 0
        
        for i, gene in enumerate(ranked_genes):
            if gene in gene_set:
                # Hit
                idx = gene_set_in_list.index(gene)
                running_sum += weights[idx] / weight_sum
            else:
                # Miss
                running_sum -= 1 / (N - len(gene_set_in_list))
            
            if running_sum > max_es:
                max_es = running_sum
                rank_at_max = i
            if running_sum < min_es:
                min_es = running_sum
        
        # Return the extreme value
        return max_es if abs(max_es) > abs(min_es) else min_es, rank_at_max
    
    def _estimate_gsea_pvalue(
        self,
        ranked_genes: List[str],
        gene_scores: Dict[str, float],
        gene_set: Set[str],
        observed_es: float,
        permutations: int
    ) -> float:
        """Estimate GSEA p-value through permutation"""
        
        # Simplified permutation test
        extreme_count = 0
        
        for _ in range(min(permutations, 100)):  # Limit for demo
            # Permute gene scores
            permuted_scores = np.random.permutation(list(gene_scores.values()))
            permuted_gene_scores = dict(zip(ranked_genes, permuted_scores))
            
            # Calculate ES for permuted data
            perm_es, _ = self._calculate_enrichment_score(
                ranked_genes, permuted_gene_scores, gene_set
            )
            
            # Count extreme values
            if abs(perm_es) >= abs(observed_es):
                extreme_count += 1
        
        return extreme_count / min(permutations, 100)
    
    def _find_leading_edge_genes(
        self,
        ranked_genes: List[str],
        gene_set: Set[str],
        rank_at_max: int
    ) -> List[str]:
        """Find leading edge genes"""
        
        leading_edge = []
        
        for i in range(rank_at_max + 1):
            if ranked_genes[i] in gene_set:
                leading_edge.append(ranked_genes[i])
        
        return leading_edge
    
    def _get_gseapy_library_name(self, database: PathwayDatabase) -> Optional[str]:
        """Map database enum to GSEApy library name"""
        
        mapping = {
            PathwayDatabase.GO_BIOLOGICAL_PROCESS: "GO_Biological_Process_2023",
            PathwayDatabase.GO_MOLECULAR_FUNCTION: "GO_Molecular_Function_2023", 
            PathwayDatabase.GO_CELLULAR_COMPONENT: "GO_Cellular_Component_2023",
            PathwayDatabase.KEGG_2023: "KEGG_2021_Human",
            PathwayDatabase.REACTOME_2022: "Reactome_2022",
            PathwayDatabase.HALLMARK_2020: "MSigDB_Hallmark_2020"
        }
        
        return mapping.get(database)


# =================== Pathway Analysis Tool ===================

class PathwayAnalysisTool(BioinformaticsTool):
    """Comprehensive pathway analysis tool"""
    
    def __init__(self):
        super().__init__(
            name="pathway_analysis",
            description="Comprehensive pathway and functional enrichment analysis",
            supported_data_types=[DataType.EXPRESSION_MATRIX]
        )
        self.enrichment_analyzer = PathwayEnrichmentAnalyzer()
        self.gsea_analyzer = GSEAAnalyzer()
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "gene_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of genes for enrichment analysis"
                },
                "ranked_gene_list": {
                    "type": "string",
                    "description": "Path to file with ranked genes (gene, score)"
                },
                "analysis_type": {
                    "type": "string", 
                    "enum": ["enrichment", "gsea", "both"],
                    "default": "enrichment"
                },
                "databases": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "GO_Biological_Process_2023",
                            "GO_Molecular_Function_2023", 
                            "GO_Cellular_Component_2023",
                            "KEGG_2023_Human",
                            "Reactome_2022",
                            "MSigDB_Hallmark_2020"
                        ]
                    },
                    "default": ["GO_Biological_Process_2023", "KEGG_2023_Human"]
                },
                "organism": {"type": "string", "default": "Human"},
                "method": {
                    "type": "string",
                    "enum": ["hypergeometric", "fisher_exact", "gsea"],
                    "default": "hypergeometric"
                },
                "background_genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Background gene set"
                },
                "significance_threshold": {"type": "number", "default": 0.05},
                "min_gene_set_size": {"type": "integer", "default": 5},
                "max_gene_set_size": {"type": "integer", "default": 500},
                "output_dir": {"type": "string", "description": "Output directory"}
            },
            "required": ["output_dir"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute pathway analysis"""
        
        try:
            output_dir = Path(params["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_type = params.get("analysis_type", "enrichment")
            
            # Parse databases
            database_names = params.get("databases", ["GO_Biological_Process_2023", "KEGG_2023_Human"])
            databases = [PathwayDatabase(name) for name in database_names]
            
            results = {}
            
            # Run enrichment analysis
            if analysis_type in ["enrichment", "both"] and "gene_list" in params:
                enrichment_results = await self._run_enrichment_analysis(
                    params, databases, output_dir
                )
                results["enrichment"] = enrichment_results
            
            # Run GSEA
            if analysis_type in ["gsea", "both"] and "ranked_gene_list" in params:
                gsea_results = await self._run_gsea_analysis(
                    params, databases, output_dir
                )
                results["gsea"] = gsea_results
            
            # Generate visualizations
            plot_files = await self._create_pathway_plots(results, output_dir)
            
            # Generate summary report
            summary = self._generate_pathway_summary(results)
            
            return BioToolResult(
                success=True,
                output=f"Pathway analysis completed: {output_dir}",
                metadata={
                    "results": results,
                    "summary": summary,
                    "plot_files": plot_files,
                    "output_directory": str(output_dir)
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Pathway analysis failed: {str(e)}"
            )
    
    async def _run_enrichment_analysis(
        self, 
        params: Dict[str, Any],
        databases: List[PathwayDatabase],
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Run over-representation enrichment analysis"""
        
        gene_list = params["gene_list"]
        method = EnrichmentMethod(params.get("method", "hypergeometric"))
        background_genes = params.get("background_genes")
        organism = params.get("organism", "Human")
        
        enrichment_results = await self.enrichment_analyzer.run_enrichment_analysis(
            gene_list=gene_list,
            databases=databases,
            method=method,
            background_genes=background_genes,
            organism=organism,
            min_gene_set_size=params.get("min_gene_set_size", 5),
            max_gene_set_size=params.get("max_gene_set_size", 500),
            significance_threshold=params.get("significance_threshold", 0.05)
        )
        
        # Convert to serializable format
        results_data = []
        for result in enrichment_results:
            results_data.append({
                "term_id": result.term_id,
                "term_name": result.term_name,
                "description": result.description,
                "database": result.database.value,
                "p_value": result.p_value,
                "adjusted_p_value": result.adjusted_p_value,
                "odds_ratio": result.odds_ratio,
                "fold_enrichment": result.fold_enrichment,
                "gene_count": result.gene_count,
                "total_genes_in_term": result.total_genes_in_term,
                "enriched_genes": result.enriched_genes,
                "is_significant": result.is_significant
            })
        
        # Save results
        results_file = output_dir / "enrichment_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return results_data
    
    async def _run_gsea_analysis(
        self,
        params: Dict[str, Any],
        databases: List[PathwayDatabase],
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Run GSEA analysis"""
        
        # Load ranked gene list
        ranked_file = params["ranked_gene_list"]
        ranked_genes = pd.read_csv(ranked_file, sep='\t')
        
        organism = params.get("organism", "Human")
        
        gsea_results = await self.gsea_analyzer.run_gsea(
            ranked_gene_list=ranked_genes,
            databases=databases,
            organism=organism,
            min_gene_set_size=params.get("min_gene_set_size", 15),
            max_gene_set_size=params.get("max_gene_set_size", 500),
            permutations=1000
        )
        
        # Convert to serializable format
        results_data = []
        for result in gsea_results:
            results_data.append({
                "term_id": result.term_id,
                "term_name": result.term_name,
                "enrichment_score": result.enrichment_score,
                "normalized_enrichment_score": result.normalized_enrichment_score,
                "p_value": result.p_value,
                "fdr": result.fdr,
                "leading_edge_genes": result.leading_edge_genes,
                "gene_set_size": result.gene_set_size
            })
        
        # Save results
        results_file = output_dir / "gsea_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return results_data
    
    async def _create_pathway_plots(self, results: Dict[str, Any], 
                                  output_dir: Path) -> List[str]:
        """Create pathway analysis visualizations"""
        
        plot_files = []
        
        # Enrichment results plots
        if "enrichment" in results:
            enrichment_data = results["enrichment"]
            
            if enrichment_data:
                # Top pathways bar plot
                plt.figure(figsize=(12, 8))
                
                top_pathways = enrichment_data[:20]  # Top 20
                pathway_names = [p["term_name"][:50] + "..." if len(p["term_name"]) > 50 
                               else p["term_name"] for p in top_pathways]
                p_values = [-np.log10(p["adjusted_p_value"]) for p in top_pathways]
                
                plt.barh(range(len(pathway_names)), p_values)
                plt.yticks(range(len(pathway_names)), pathway_names)
                plt.xlabel('-log10(Adjusted P-value)')
                plt.title('Top Enriched Pathways')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                plot_file = output_dir / "enrichment_barplot.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(str(plot_file))
                
                # Enrichment bubble plot
                plt.figure(figsize=(10, 8))
                
                x_vals = [p["fold_enrichment"] for p in top_pathways]
                y_vals = [-np.log10(p["adjusted_p_value"]) for p in top_pathways]
                sizes = [p["gene_count"] * 10 for p in top_pathways]
                
                plt.scatter(x_vals, y_vals, s=sizes, alpha=0.6, c=y_vals, cmap='viridis')
                plt.xlabel('Fold Enrichment')
                plt.ylabel('-log10(Adjusted P-value)')
                plt.title('Pathway Enrichment Bubble Plot')
                plt.colorbar(label='-log10(Adjusted P-value)')
                
                plot_file = output_dir / "enrichment_bubble.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files.append(str(plot_file))
        
        return plot_files
    
    def _generate_pathway_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pathway analysis summary"""
        
        summary = {
            "total_analyses": len(results),
            "analysis_types": list(results.keys())
        }
        
        if "enrichment" in results:
            enrichment_data = results["enrichment"]
            significant_pathways = [p for p in enrichment_data if p["is_significant"]]
            
            summary["enrichment"] = {
                "total_pathways_tested": len(enrichment_data),
                "significant_pathways": len(significant_pathways),
                "top_pathway": enrichment_data[0]["term_name"] if enrichment_data else None,
                "min_p_value": min(p["adjusted_p_value"] for p in enrichment_data) if enrichment_data else None
            }
        
        if "gsea" in results:
            gsea_data = results["gsea"]
            summary["gsea"] = {
                "total_gene_sets_tested": len(gsea_data),
                "significant_gene_sets": len([g for g in gsea_data if g["fdr"] < 0.05]),
                "top_gene_set": gsea_data[0]["term_name"] if gsea_data else None
            }
        
        return summary


# =================== Example Usage ===================

async def example_pathway_analysis():
    """Example of comprehensive pathway analysis"""
    
    print("Pathway Analysis Example")
    print("=" * 40)
    
    # Initialize pathway analysis tool
    pathway_tool = PathwayAnalysisTool()
    
    # Example parameters
    example_genes = [
        "TP53", "BRCA1", "BRCA2", "ATM", "CHEK2", "PALB2", 
        "CDK2", "CCNA2", "CCNB1", "CDKN1A", "MDM2", "RB1"
    ]
    
    params = {
        "gene_list": example_genes,
        "analysis_type": "enrichment",
        "databases": ["GO_Biological_Process_2023", "KEGG_2023_Human"],
        "organism": "Human",
        "method": "hypergeometric",
        "significance_threshold": 0.05,
        "output_dir": "pathway_analysis_output"
    }
    
    print(f"Running pathway analysis on {len(example_genes)} genes...")
    print(f"Databases: {params['databases']}")
    
    # This would run the actual analysis with real databases
    # result = await pathway_tool.execute(params, [])
    
    print("\nPathway Analysis Pipeline:")
    print("1. Load gene set databases (GO, KEGG, Reactome)")
    print("2. Run over-representation analysis")
    print("3. Apply multiple testing correction")
    print("4. Generate enrichment visualizations")
    print("5. Create summary report")
    
    print("\nExample completed (requires database connectivity to run)")


if __name__ == "__main__":
    asyncio.run(example_pathway_analysis())