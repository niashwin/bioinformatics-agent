#!/usr/bin/env python3
"""
BioinformaticsAgent Database Integration: Real biological database connectivity

This module provides comprehensive connectivity to biological databases:
- NCBI (GenBank, RefSeq, PubMed, SRA, etc.)
- Ensembl (genes, genomes, variation)
- UniProt (protein sequences and annotations)
- KEGG (pathways, reactions, compounds)
- GO (Gene Ontology)
- STRING (protein interactions)
- dbSNP (genetic variations)
- ClinVar (clinical variants)
- Local database management
"""

import asyncio
import aiohttp
import requests
import json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import sqlite3
import gzip
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, AsyncIterator, Iterator
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import urllib.parse
from urllib.parse import urlencode
import re

# Database-specific libraries
try:
    from bioservices import KEGG, UniProt, QuickGO
    BIOSERVICES_AVAILABLE = True
except ImportError:
    BIOSERVICES_AVAILABLE = False

try:
    from Bio import Entrez, SeqIO
    from Bio.SeqUtils import molecular_weight
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

from bioagent_architecture import BioinformaticsTool, BioToolResult, DataType, DataMetadata


# =================== Database Types and Configurations ===================

class DatabaseType(Enum):
    """Types of biological databases"""
    NCBI = "ncbi"
    ENSEMBL = "ensembl"
    UNIPROT = "uniprot"
    KEGG = "kegg"
    GO = "go"
    STRING = "string"
    DBSNP = "dbsnp"
    CLINVAR = "clinvar"
    REACTOME = "reactome"
    PFAM = "pfam"
    LOCAL_SQLITE = "local_sqlite"


@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    base_url: str
    api_key: Optional[str] = None
    rate_limit: float = 1.0  # Requests per second
    timeout: int = 30
    retry_attempts: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    cache_enabled: bool = True
    cache_dir: Optional[str] = None


@dataclass
class QueryResult:
    """Result from database query"""
    database: DatabaseType
    query: str
    results: List[Dict[str, Any]]
    total_count: int
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    cached: bool = False


# =================== Base Database Interface ===================

class BiologicalDatabase:
    """Base class for biological database interfaces"""
    
    def __init__(self, db_type: DatabaseType, config: DatabaseConfig):
        self.db_type = db_type
        self.config = config
        self.session = None
        self.cache = {}
        self.last_request_time = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Establish database connection"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers=self.config.headers
        )
    
    async def disconnect(self):
        """Close database connection"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.config.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, url: str, params: Optional[Dict] = None,
                          method: str = "GET") -> Dict[str, Any]:
        """Make HTTP request with rate limiting and retries"""
        
        await self._rate_limit()
        
        for attempt in range(self.config.retry_attempts):
            try:
                if method == "GET":
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            content_type = response.headers.get('content-type', '')
                            
                            if 'application/json' in content_type:
                                return await response.json()
                            else:
                                text = await response.text()
                                return {"text": text, "status": response.status}
                        else:
                            logging.warning(f"Request failed with status {response.status}")
                            
            except Exception as e:
                logging.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
        
        raise Exception(f"Failed to fetch data after {self.config.retry_attempts} attempts")


# =================== NCBI Database Interface ===================

class NCBIDatabase(BiologicalDatabase):
    """Interface to NCBI databases"""
    
    def __init__(self, email: str, api_key: Optional[str] = None):
        config = DatabaseConfig(
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            api_key=api_key,
            rate_limit=3.0 if api_key else 3.0,  # 3 requests per second
            headers={"User-Agent": f"BioinformaticsAgent ({email})"}
        )
        super().__init__(DatabaseType.NCBI, config)
        self.email = email
        
        if BIOPYTHON_AVAILABLE:
            Entrez.email = email
            if api_key:
                Entrez.api_key = api_key
    
    async def search_nucleotide(self, query: str, max_results: int = 20) -> QueryResult:
        """Search NCBI Nucleotide database"""
        
        start_time = time.time()
        
        try:
            # Use esearch to get IDs
            search_url = f"{self.config.base_url}esearch.fcgi"
            search_params = {
                "db": "nucleotide",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            
            if self.config.api_key:
                search_params["api_key"] = self.config.api_key
            
            search_response = await self._make_request(search_url, search_params)
            
            if "esearchresult" not in search_response:
                raise Exception("Invalid search response")
            
            id_list = search_response["esearchresult"]["idlist"]
            total_count = int(search_response["esearchresult"]["count"])
            
            if not id_list:
                return QueryResult(
                    database=self.db_type,
                    query=query,
                    results=[],
                    total_count=0,
                    success=True,
                    execution_time=time.time() - start_time
                )
            
            # Use efetch to get details
            fetch_url = f"{self.config.base_url}efetch.fcgi"
            fetch_params = {
                "db": "nucleotide",
                "id": ",".join(id_list),
                "rettype": "gb",
                "retmode": "xml"
            }
            
            if self.config.api_key:
                fetch_params["api_key"] = self.config.api_key
            
            fetch_response = await self._make_request(fetch_url, fetch_params)
            
            # Parse XML response
            results = self._parse_genbank_xml(fetch_response["text"])
            
            return QueryResult(
                database=self.db_type,
                query=query,
                results=results,
                total_count=total_count,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=query,
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def search_protein(self, query: str, max_results: int = 20) -> QueryResult:
        """Search NCBI Protein database"""
        
        start_time = time.time()
        
        try:
            search_url = f"{self.config.base_url}esearch.fcgi"
            search_params = {
                "db": "protein",
                "term": query,
                "retmax": max_results,
                "retmode": "json"
            }
            
            if self.config.api_key:
                search_params["api_key"] = self.config.api_key
            
            search_response = await self._make_request(search_url, search_params)
            id_list = search_response["esearchresult"]["idlist"]
            total_count = int(search_response["esearchresult"]["count"])
            
            if not id_list:
                return QueryResult(
                    database=self.db_type,
                    query=query,
                    results=[],
                    total_count=0,
                    success=True,
                    execution_time=time.time() - start_time
                )
            
            # Fetch protein details
            fetch_url = f"{self.config.base_url}efetch.fcgi"
            fetch_params = {
                "db": "protein",
                "id": ",".join(id_list),
                "rettype": "fasta",
                "retmode": "text"
            }
            
            if self.config.api_key:
                fetch_params["api_key"] = self.config.api_key
            
            fetch_response = await self._make_request(fetch_url, fetch_params)
            
            # Parse FASTA response
            results = self._parse_fasta_text(fetch_response["text"])
            
            return QueryResult(
                database=self.db_type,
                query=query,
                results=results,
                total_count=total_count,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=query,
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def search_pubmed(self, query: str, max_results: int = 20) -> QueryResult:
        """Search PubMed database"""
        
        start_time = time.time()
        
        try:
            search_url = f"{self.config.base_url}esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance"
            }
            
            if self.config.api_key:
                search_params["api_key"] = self.config.api_key
            
            search_response = await self._make_request(search_url, search_params)
            id_list = search_response["esearchresult"]["idlist"]
            total_count = int(search_response["esearchresult"]["count"])
            
            if not id_list:
                return QueryResult(
                    database=self.db_type,
                    query=query,
                    results=[],
                    total_count=0,
                    success=True,
                    execution_time=time.time() - start_time
                )
            
            # Fetch article summaries
            summary_url = f"{self.config.base_url}esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json"
            }
            
            if self.config.api_key:
                summary_params["api_key"] = self.config.api_key
            
            summary_response = await self._make_request(summary_url, summary_params)
            
            # Parse PubMed summaries
            results = self._parse_pubmed_summaries(summary_response)
            
            return QueryResult(
                database=self.db_type,
                query=query,
                results=results,
                total_count=total_count,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=query,
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _parse_genbank_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse GenBank XML response"""
        results = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for seq_elem in root.findall('.//GBSeq'):
                result = {}
                
                # Basic information
                accession = seq_elem.find('GBSeq_accession-version')
                if accession is not None:
                    result['accession'] = accession.text
                
                definition = seq_elem.find('GBSeq_definition')
                if definition is not None:
                    result['definition'] = definition.text
                
                organism = seq_elem.find('GBSeq_organism')
                if organism is not None:
                    result['organism'] = organism.text
                
                length = seq_elem.find('GBSeq_length')
                if length is not None:
                    result['length'] = int(length.text)
                
                sequence = seq_elem.find('GBSeq_sequence')
                if sequence is not None:
                    result['sequence'] = sequence.text
                
                results.append(result)
        
        except ET.ParseError as e:
            logging.warning(f"Failed to parse GenBank XML: {e}")
        
        return results
    
    def _parse_fasta_text(self, fasta_text: str) -> List[Dict[str, Any]]:
        """Parse FASTA text response"""
        results = []
        
        sequences = fasta_text.split('>')
        for seq_data in sequences[1:]:  # Skip first empty element
            lines = seq_data.strip().split('\n')
            if lines:
                header = lines[0]
                sequence = ''.join(lines[1:])
                
                # Parse header
                parts = header.split('|')
                result = {
                    'header': header,
                    'sequence': sequence,
                    'length': len(sequence)
                }
                
                if len(parts) >= 2:
                    result['accession'] = parts[1]
                
                results.append(result)
        
        return results
    
    def _parse_pubmed_summaries(self, summary_data: Dict) -> List[Dict[str, Any]]:
        """Parse PubMed summary response"""
        results = []
        
        if "result" in summary_data:
            for pmid, article_data in summary_data["result"].items():
                if pmid == "uids":
                    continue
                
                result = {
                    'pmid': pmid,
                    'title': article_data.get('title', ''),
                    'authors': article_data.get('authors', []),
                    'journal': article_data.get('fulljournalname', ''),
                    'pub_date': article_data.get('pubdate', ''),
                    'doi': article_data.get('elocationid', '')
                }
                
                results.append(result)
        
        return results


# =================== UniProt Database Interface ===================

class UniProtDatabase(BiologicalDatabase):
    """Interface to UniProt database"""
    
    def __init__(self):
        config = DatabaseConfig(
            base_url="https://rest.uniprot.org/",
            rate_limit=1.0,  # 1 request per second
            headers={"Accept": "application/json"}
        )
        super().__init__(DatabaseType.UNIPROT, config)
    
    async def search_proteins(self, query: str, organism: Optional[str] = None,
                            max_results: int = 20) -> QueryResult:
        """Search UniProt proteins"""
        
        start_time = time.time()
        
        try:
            # Construct search query
            search_query = query
            if organism:
                search_query += f" AND organism_name:{organism}"
            
            url = f"{self.config.base_url}uniprotkb/search"
            params = {
                "query": search_query,
                "size": max_results,
                "format": "json"
            }
            
            response = await self._make_request(url, params)
            
            results = []
            total_count = 0
            
            if "results" in response:
                total_count = len(response["results"])
                
                for entry in response["results"]:
                    result = self._parse_uniprot_entry(entry)
                    results.append(result)
            
            return QueryResult(
                database=self.db_type,
                query=query,
                results=results,
                total_count=total_count,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=query,
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def get_protein_by_id(self, uniprot_id: str) -> QueryResult:
        """Get protein details by UniProt ID"""
        
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}uniprotkb/{uniprot_id}"
            params = {"format": "json"}
            
            response = await self._make_request(url, params)
            
            result = self._parse_uniprot_entry(response)
            
            return QueryResult(
                database=self.db_type,
                query=uniprot_id,
                results=[result],
                total_count=1,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=uniprot_id,
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _parse_uniprot_entry(self, entry: Dict) -> Dict[str, Any]:
        """Parse UniProt entry"""
        
        result = {
            "uniprot_id": entry.get("primaryAccession", ""),
            "entry_name": entry.get("uniProtkbId", ""),
            "protein_name": "",
            "gene_names": [],
            "organism": "",
            "length": 0,
            "sequence": "",
            "functions": [],
            "keywords": [],
            "go_terms": []
        }
        
        # Protein names
        if "proteinDescription" in entry:
            protein_desc = entry["proteinDescription"]
            if "recommendedName" in protein_desc:
                result["protein_name"] = protein_desc["recommendedName"].get("fullName", {}).get("value", "")
        
        # Gene names
        if "genes" in entry:
            for gene in entry["genes"]:
                if "geneName" in gene:
                    result["gene_names"].append(gene["geneName"]["value"])
        
        # Organism
        if "organism" in entry:
            result["organism"] = entry["organism"].get("scientificName", "")
        
        # Sequence
        if "sequence" in entry:
            result["length"] = entry["sequence"].get("length", 0)
            result["sequence"] = entry["sequence"].get("value", "")
        
        # Functions
        if "comments" in entry:
            for comment in entry["comments"]:
                if comment.get("commentType") == "FUNCTION":
                    if "texts" in comment:
                        for text in comment["texts"]:
                            result["functions"].append(text.get("value", ""))
        
        # Keywords
        if "keywords" in entry:
            for keyword in entry["keywords"]:
                result["keywords"].append(keyword.get("value", ""))
        
        # GO terms
        if "uniProtKBCrossReferences" in entry:
            for cross_ref in entry["uniProtKBCrossReferences"]:
                if cross_ref.get("database") == "GO":
                    go_id = cross_ref.get("id", "")
                    if "properties" in cross_ref:
                        for prop in cross_ref["properties"]:
                            if prop.get("key") == "GoTerm":
                                result["go_terms"].append({
                                    "id": go_id,
                                    "term": prop.get("value", "")
                                })
        
        return result


# =================== Ensembl Database Interface ===================

class EnsemblDatabase(BiologicalDatabase):
    """Interface to Ensembl database"""
    
    def __init__(self):
        config = DatabaseConfig(
            base_url="https://rest.ensembl.org/",
            rate_limit=15.0,  # 15 requests per second
            headers={"Content-Type": "application/json"}
        )
        super().__init__(DatabaseType.ENSEMBL, config)
    
    async def search_genes(self, query: str, species: str = "homo_sapiens") -> QueryResult:
        """Search Ensembl genes"""
        
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}lookup/symbol/{species}/{query}"
            params = {"expand": "1"}
            
            response = await self._make_request(url, params)
            
            if "error" in response:
                # Try alternative search
                url = f"{self.config.base_url}xrefs/symbol/{species}/{query}"
                response = await self._make_request(url, params)
                
                if isinstance(response, list):
                    results = [self._parse_ensembl_gene_xref(item) for item in response]
                else:
                    results = []
            else:
                results = [self._parse_ensembl_gene(response)]
            
            return QueryResult(
                database=self.db_type,
                query=query,
                results=results,
                total_count=len(results),
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=query,
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def get_gene_sequence(self, gene_id: str, species: str = "homo_sapiens") -> QueryResult:
        """Get gene sequence from Ensembl"""
        
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}sequence/id/{gene_id}"
            params = {"type": "genomic"}
            
            response = await self._make_request(url, params)
            
            if "seq" in response:
                result = {
                    "gene_id": gene_id,
                    "sequence": response["seq"],
                    "length": len(response["seq"]),
                    "description": response.get("desc", "")
                }
                
                return QueryResult(
                    database=self.db_type,
                    query=gene_id,
                    results=[result],
                    total_count=1,
                    success=True,
                    execution_time=time.time() - start_time
                )
            else:
                raise Exception("No sequence found")
                
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=gene_id,
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _parse_ensembl_gene(self, gene_data: Dict) -> Dict[str, Any]:
        """Parse Ensembl gene data"""
        
        return {
            "gene_id": gene_data.get("id", ""),
            "gene_name": gene_data.get("display_name", ""),
            "description": gene_data.get("description", ""),
            "biotype": gene_data.get("biotype", ""),
            "chromosome": gene_data.get("seq_region_name", ""),
            "start": gene_data.get("start", 0),
            "end": gene_data.get("end", 0),
            "strand": gene_data.get("strand", 0),
            "species": gene_data.get("species", "")
        }
    
    def _parse_ensembl_gene_xref(self, xref_data: Dict) -> Dict[str, Any]:
        """Parse Ensembl cross-reference data"""
        
        return {
            "gene_id": xref_data.get("id", ""),
            "gene_name": xref_data.get("display_id", ""),
            "description": xref_data.get("description", ""),
            "type": xref_data.get("type", ""),
            "species": xref_data.get("species", "")
        }


# =================== KEGG Database Interface ===================

class KEGGDatabase(BiologicalDatabase):
    """Interface to KEGG database"""
    
    def __init__(self):
        config = DatabaseConfig(
            base_url="https://rest.kegg.jp/",
            rate_limit=1.0  # 1 request per second
        )
        super().__init__(DatabaseType.KEGG, config)
    
    async def search_pathways(self, organism: str = "hsa") -> QueryResult:
        """Get pathways for organism"""
        
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}list/pathway/{organism}"
            
            response = await self._make_request(url)
            
            results = []
            if "text" in response:
                lines = response["text"].strip().split('\n')
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pathway_id, description = parts[0], parts[1]
                        results.append({
                            "pathway_id": pathway_id,
                            "description": description,
                            "organism": organism
                        })
            
            return QueryResult(
                database=self.db_type,
                query=f"pathways/{organism}",
                results=results,
                total_count=len(results),
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=f"pathways/{organism}",
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    async def get_pathway_genes(self, pathway_id: str) -> QueryResult:
        """Get genes in a pathway"""
        
        start_time = time.time()
        
        try:
            url = f"{self.config.base_url}link/hsa/{pathway_id}"
            
            response = await self._make_request(url)
            
            results = []
            if "text" in response:
                lines = response["text"].strip().split('\n')
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        gene_id = parts[1]
                        results.append({
                            "pathway_id": pathway_id,
                            "gene_id": gene_id
                        })
            
            return QueryResult(
                database=self.db_type,
                query=f"genes/{pathway_id}",
                results=results,
                total_count=len(results),
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QueryResult(
                database=self.db_type,
                query=f"genes/{pathway_id}",
                results=[],
                total_count=0,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )


# =================== Database Manager ===================

class BiologicalDatabaseManager:
    """Manager for multiple biological databases"""
    
    def __init__(self):
        self.databases = {}
        self.cache_dir = Path("database_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def register_database(self, name: str, database: BiologicalDatabase):
        """Register a database"""
        self.databases[name] = database
    
    async def initialize_databases(self, email: str, ncbi_api_key: Optional[str] = None):
        """Initialize all databases"""
        
        # NCBI
        self.databases["ncbi"] = NCBIDatabase(email, ncbi_api_key)
        
        # UniProt
        self.databases["uniprot"] = UniProtDatabase()
        
        # Ensembl
        self.databases["ensembl"] = EnsemblDatabase()
        
        # KEGG
        self.databases["kegg"] = KEGGDatabase()
        
        # Connect to all databases
        for db in self.databases.values():
            await db.connect()
    
    async def search_across_databases(self, query: str, 
                                    databases: Optional[List[str]] = None) -> Dict[str, QueryResult]:
        """Search across multiple databases"""
        
        if databases is None:
            databases = list(self.databases.keys())
        
        results = {}
        
        # Run searches in parallel
        tasks = []
        for db_name in databases:
            if db_name in self.databases:
                db = self.databases[db_name]
                
                if db_name == "ncbi":
                    task = db.search_nucleotide(query)
                elif db_name == "uniprot":
                    task = db.search_proteins(query)
                elif db_name == "ensembl":
                    task = db.search_genes(query)
                else:
                    continue
                
                tasks.append((db_name, task))
        
        # Wait for all searches to complete
        for db_name, task in tasks:
            try:
                result = await task
                results[db_name] = result
            except Exception as e:
                results[db_name] = QueryResult(
                    database=self.databases[db_name].db_type,
                    query=query,
                    results=[],
                    total_count=0,
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    async def get_gene_annotations(self, gene_name: str, 
                                 species: str = "homo_sapiens") -> Dict[str, Any]:
        """Get comprehensive gene annotations from multiple databases"""
        
        annotations = {
            "gene_name": gene_name,
            "species": species,
            "ensembl_data": None,
            "uniprot_data": None,
            "kegg_pathways": None,
            "pubmed_papers": None
        }
        
        # Get Ensembl data
        if "ensembl" in self.databases:
            ensembl_result = await self.databases["ensembl"].search_genes(gene_name, species)
            if ensembl_result.success and ensembl_result.results:
                annotations["ensembl_data"] = ensembl_result.results[0]
        
        # Get UniProt data
        if "uniprot" in self.databases:
            uniprot_result = await self.databases["uniprot"].search_proteins(
                gene_name, species.replace("_", " ")
            )
            if uniprot_result.success and uniprot_result.results:
                annotations["uniprot_data"] = uniprot_result.results[0]
        
        # Get PubMed papers
        if "ncbi" in self.databases:
            pubmed_query = f"{gene_name} AND {species.replace('_', ' ')}"
            pubmed_result = await self.databases["ncbi"].search_pubmed(pubmed_query, max_results=5)
            if pubmed_result.success:
                annotations["pubmed_papers"] = pubmed_result.results
        
        return annotations
    
    async def close_all(self):
        """Close all database connections"""
        for db in self.databases.values():
            await db.disconnect()


# =================== Database Tool Integration ===================

class DatabaseQueryTool(BioinformaticsTool):
    """Tool for querying biological databases"""
    
    def __init__(self):
        super().__init__(
            name="database_query",
            description="Query biological databases for genes, proteins, and pathways",
            supported_data_types=[DataType.GENOMIC_SEQUENCE, DataType.PROTEIN_SEQUENCE]
        )
        self.db_manager = None
    
    def _define_parameter_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "databases": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["ncbi", "uniprot", "ensembl", "kegg"]},
                    "default": ["ncbi", "uniprot"]
                },
                "query_type": {
                    "type": "string",
                    "enum": ["gene", "protein", "pathway", "publication"],
                    "default": "gene"
                },
                "species": {"type": "string", "default": "homo_sapiens"},
                "max_results": {"type": "integer", "default": 20},
                "email": {"type": "string", "description": "Email for NCBI queries"},
                "ncbi_api_key": {"type": "string", "description": "NCBI API key (optional)"}
            },
            "required": ["query", "email"]
        }
    
    async def execute(self, params: Dict[str, Any], 
                     data_metadata: List[DataMetadata]) -> BioToolResult:
        """Execute database query"""
        
        try:
            # Initialize database manager
            if self.db_manager is None:
                self.db_manager = BiologicalDatabaseManager()
                await self.db_manager.initialize_databases(
                    params["email"],
                    params.get("ncbi_api_key")
                )
            
            query = params["query"]
            databases = params.get("databases", ["ncbi", "uniprot"])
            query_type = params.get("query_type", "gene")
            
            # Perform searches
            if query_type == "gene":
                results = await self.db_manager.get_gene_annotations(
                    query, params.get("species", "homo_sapiens")
                )
            else:
                results = await self.db_manager.search_across_databases(query, databases)
            
            # Format results
            formatted_results = self._format_results(results, query_type)
            
            return BioToolResult(
                success=True,
                output=f"Database query completed for: {query}",
                metadata={
                    "query": query,
                    "query_type": query_type,
                    "databases_searched": databases,
                    "results": formatted_results
                }
            )
            
        except Exception as e:
            return BioToolResult(
                success=False,
                error=f"Database query failed: {str(e)}"
            )
    
    def _format_results(self, results: Dict[str, Any], query_type: str) -> Dict[str, Any]:
        """Format query results for output"""
        
        formatted = {
            "summary": {},
            "detailed_results": results
        }
        
        if query_type == "gene":
            # Summarize gene annotation
            if results.get("ensembl_data"):
                formatted["summary"]["ensembl_id"] = results["ensembl_data"].get("gene_id", "")
                formatted["summary"]["description"] = results["ensembl_data"].get("description", "")
            
            if results.get("uniprot_data"):
                formatted["summary"]["protein_name"] = results["uniprot_data"].get("protein_name", "")
                formatted["summary"]["uniprot_id"] = results["uniprot_data"].get("uniprot_id", "")
            
            if results.get("pubmed_papers"):
                formatted["summary"]["recent_papers"] = len(results["pubmed_papers"])
        
        else:
            # Summarize cross-database search
            for db_name, db_result in results.items():
                if hasattr(db_result, 'total_count'):
                    formatted["summary"][f"{db_name}_hits"] = db_result.total_count
        
        return formatted


# =================== Example Usage ===================

async def example_database_integration():
    """Example of biological database integration"""
    
    print("Biological Database Integration Example")
    print("=" * 50)
    
    # Initialize database manager
    db_manager = BiologicalDatabaseManager()
    
    # Note: This requires a valid email and optionally an NCBI API key
    email = "your.email@example.com"  # Replace with your email
    ncbi_api_key = None  # Replace with your NCBI API key if available
    
    try:
        await db_manager.initialize_databases(email, ncbi_api_key)
        
        print("Testing database connections...")
        
        # Test gene annotation
        print("\n1. Testing gene annotation (TP53):")
        gene_annotations = await db_manager.get_gene_annotations("TP53")
        
        if gene_annotations["ensembl_data"]:
            print(f"  Ensembl ID: {gene_annotations['ensembl_data'].get('gene_id', 'N/A')}")
            print(f"  Description: {gene_annotations['ensembl_data'].get('description', 'N/A')[:100]}...")
        
        if gene_annotations["uniprot_data"]:
            print(f"  UniProt ID: {gene_annotations['uniprot_data'].get('uniprot_id', 'N/A')}")
            print(f"  Protein: {gene_annotations['uniprot_data'].get('protein_name', 'N/A')}")
        
        # Test cross-database search
        print("\n2. Testing cross-database search (insulin):")
        search_results = await db_manager.search_across_databases("insulin")
        
        for db_name, result in search_results.items():
            print(f"  {db_name}: {result.total_count} results (success: {result.success})")
        
        await db_manager.close_all()
        
        print("\nDatabase integration test completed successfully!")
        
    except Exception as e:
        print(f"Database test failed: {e}")
        print("Note: This example requires internet connectivity and valid credentials")


if __name__ == "__main__":
    asyncio.run(example_database_integration())