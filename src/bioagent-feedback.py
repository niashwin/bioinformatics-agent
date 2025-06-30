#!/usr/bin/env python3
"""
BioinformaticsAgent Feedback and Iterative Improvement System

This module implements sophisticated feedback mechanisms for continuous improvement:
- Multi-source feedback collection
- Automated quality assessment
- User feedback integration
- Literature-based validation
- Iterative refinement algorithms
- Learning from analysis history
"""

import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import pickle
from pathlib import Path

# Import base classes
from bioagent_architecture import (
    BioinformaticsTool, BioToolResult, DataType, DataMetadata, 
    AnalysisTask, ReflectionContext
)
from bioagent_reasoning import QualityAssessment


# =================== Feedback Types ===================

class FeedbackType(Enum):
    """Types of feedback that can be provided"""
    USER_CORRECTION = "user_correction"
    QUALITY_METRIC = "quality_metric"
    BIOLOGICAL_VALIDATION = "biological_validation"
    STATISTICAL_CHECK = "statistical_check"
    LITERATURE_COMPARISON = "literature_comparison"
    PEER_REVIEW = "peer_review"
    AUTOMATED_VALIDATION = "automated_validation"


class FeedbackSeverity(Enum):
    """Severity levels for feedback items"""
    CRITICAL = "critical"      # Must fix for valid results
    HIGH = "high"              # Should fix for quality
    MEDIUM = "medium"          # Recommended improvements
    LOW = "low"                # Nice to have enhancements
    INFO = "info"              # Informational only


@dataclass
class FeedbackItem:
    """Individual feedback item"""
    feedback_id: str
    feedback_type: FeedbackType
    severity: FeedbackSeverity
    category: str
    message: str
    suggested_action: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    validation_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class FeedbackReport:
    """Comprehensive feedback report for an analysis"""
    analysis_id: str
    feedback_items: List[FeedbackItem] = field(default_factory=list)
    overall_quality_score: float = 0.0
    improvement_priority_list: List[str] = field(default_factory=list)
    automated_fixes_available: List[str] = field(default_factory=list)
    estimated_improvement_impact: Dict[str, float] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def get_critical_items(self) -> List[FeedbackItem]:
        """Get all critical feedback items"""
        return [item for item in self.feedback_items if item.severity == FeedbackSeverity.CRITICAL]
    
    def get_unresolved_items(self) -> List[FeedbackItem]:
        """Get all unresolved feedback items"""
        return [item for item in self.feedback_items if not item.resolved]
    
    def add_feedback(self, item: FeedbackItem):
        """Add a feedback item to the report"""
        self.feedback_items.append(item)
        self._update_priority_list()
    
    def _update_priority_list(self):
        """Update the improvement priority list based on feedback"""
        # Sort by severity and timestamp
        severity_order = {
            FeedbackSeverity.CRITICAL: 0,
            FeedbackSeverity.HIGH: 1,
            FeedbackSeverity.MEDIUM: 2,
            FeedbackSeverity.LOW: 3,
            FeedbackSeverity.INFO: 4
        }
        
        sorted_items = sorted(
            self.get_unresolved_items(),
            key=lambda x: (severity_order[x.severity], x.timestamp)
        )
        
        self.improvement_priority_list = [
            f"{item.category}: {item.message}" for item in sorted_items
        ]


# =================== Feedback Collectors ===================

class FeedbackCollector(ABC):
    """Abstract base class for feedback collection"""
    
    @abstractmethod
    async def collect_feedback(self, analysis_result: Dict[str, Any], 
                             context: Dict[str, Any]) -> List[FeedbackItem]:
        """Collect feedback based on analysis results"""
        pass


class StatisticalFeedbackCollector(FeedbackCollector):
    """Collects feedback based on statistical validation"""
    
    async def collect_feedback(self, analysis_result: Dict[str, Any], 
                             context: Dict[str, Any]) -> List[FeedbackItem]:
        feedback_items = []
        
        # Check for p-value distribution issues
        if 'pvalues' in analysis_result:
            pvalues = np.array(analysis_result['pvalues'])
            
            # Check for p-value inflation
            if np.mean(pvalues < 0.05) > 0.5:
                feedback_items.append(FeedbackItem(
                    feedback_id="stat_001",
                    feedback_type=FeedbackType.STATISTICAL_CHECK,
                    severity=FeedbackSeverity.HIGH,
                    category="Statistical Validity",
                    message="Unusually high proportion of significant p-values detected",
                    suggested_action="Review multiple testing correction and check for p-hacking",
                    affected_components=["statistical_analysis"],
                    validation_data={"significant_proportion": float(np.mean(pvalues < 0.05))}
                ))
            
            # Check for lack of signal
            if np.mean(pvalues) > 0.9:
                feedback_items.append(FeedbackItem(
                    feedback_id="stat_002",
                    feedback_type=FeedbackType.STATISTICAL_CHECK,
                    severity=FeedbackSeverity.MEDIUM,
                    category="Statistical Power",
                    message="P-values suggest lack of statistical power",
                    suggested_action="Consider increasing sample size or effect size threshold",
                    affected_components=["statistical_analysis"],
                    validation_data={"mean_pvalue": float(np.mean(pvalues))}
                ))
        
        # Check for multiple testing correction
        if 'statistical_test' in analysis_result and 'correction' not in str(analysis_result):
            feedback_items.append(FeedbackItem(
                feedback_id="stat_003",
                feedback_type=FeedbackType.STATISTICAL_CHECK,
                severity=FeedbackSeverity.CRITICAL,
                category="Multiple Testing",
                message="No multiple testing correction detected",
                suggested_action="Apply FDR or Bonferroni correction",
                affected_components=["statistical_analysis"]
            ))
        
        # Check for effect size reporting
        if 'significant_results' in analysis_result and 'effect_size' not in str(analysis_result):
            feedback_items.append(FeedbackItem(
                feedback_id="stat_004",
                feedback_type=FeedbackType.STATISTICAL_CHECK,
                severity=FeedbackSeverity.HIGH,
                category="Effect Size",
                message="Effect sizes not reported alongside p-values",
                suggested_action="Calculate and report effect sizes (e.g., Cohen's d, fold change)",
                affected_components=["statistical_analysis", "reporting"]
            ))
        
        return feedback_items


class BiologicalValidationCollector(FeedbackCollector):
    """Collects feedback based on biological plausibility"""
    
    async def collect_feedback(self, analysis_result: Dict[str, Any], 
                             context: Dict[str, Any]) -> List[FeedbackItem]:
        feedback_items = []
        
        # Check for pathway analysis
        if 'gene_list' in analysis_result and 'pathway_enrichment' not in analysis_result:
            feedback_items.append(FeedbackItem(
                feedback_id="bio_001",
                feedback_type=FeedbackType.BIOLOGICAL_VALIDATION,
                severity=FeedbackSeverity.HIGH,
                category="Functional Analysis",
                message="Gene list identified but no pathway analysis performed",
                suggested_action="Perform pathway enrichment analysis (e.g., GO, KEGG)",
                affected_components=["biological_interpretation"]
            ))
        
        # Check for unrealistic fold changes
        if 'fold_changes' in analysis_result:
            fold_changes = np.array(analysis_result['fold_changes'])
            if np.any(np.abs(fold_changes) > 100):
                feedback_items.append(FeedbackItem(
                    feedback_id="bio_002",
                    feedback_type=FeedbackType.BIOLOGICAL_VALIDATION,
                    severity=FeedbackSeverity.HIGH,
                    category="Biological Plausibility",
                    message="Extremely high fold changes detected",
                    suggested_action="Verify data normalization and check for outliers",
                    affected_components=["data_processing", "normalization"],
                    validation_data={"max_fold_change": float(np.max(np.abs(fold_changes)))}
                ))
        
        # Check for literature validation
        if 'novel_findings' in analysis_result:
            feedback_items.append(FeedbackItem(
                feedback_id="bio_003",
                feedback_type=FeedbackType.BIOLOGICAL_VALIDATION,
                severity=FeedbackSeverity.MEDIUM,
                category="Literature Validation",
                message="Novel findings require literature validation",
                suggested_action="Compare findings with published studies in similar systems",
                affected_components=["result_validation"]
            ))
        
        return feedback_items


class QualityMetricCollector(FeedbackCollector):
    """Collects feedback based on data quality metrics"""
    
    async def collect_feedback(self, analysis_result: Dict[str, Any], 
                             context: Dict[str, Any]) -> List[FeedbackItem]:
        feedback_items = []
        
        # Check sequencing quality
        if 'quality_metrics' in context:
            quality = context['quality_metrics']
            
            if quality.get('average_quality', 30) < 20:
                feedback_items.append(FeedbackItem(
                    feedback_id="qual_001",
                    feedback_type=FeedbackType.QUALITY_METRIC,
                    severity=FeedbackSeverity.CRITICAL,
                    category="Data Quality",
                    message="Poor sequencing quality detected",
                    suggested_action="Filter low-quality reads or re-sequence samples",
                    affected_components=["raw_data"],
                    validation_data={"average_quality": quality.get('average_quality')}
                ))
            
            if quality.get('duplication_rate', 0) > 0.5:
                feedback_items.append(FeedbackItem(
                    feedback_id="qual_002",
                    feedback_type=FeedbackType.QUALITY_METRIC,
                    severity=FeedbackSeverity.HIGH,
                    category="Library Quality",
                    message="High duplication rate indicates potential PCR bias",
                    suggested_action="Remove PCR duplicates or adjust for bias",
                    affected_components=["preprocessing"],
                    validation_data={"duplication_rate": quality.get('duplication_rate')}
                ))
        
        # Check for batch effects
        if 'pca_results' in analysis_result and 'batch_info' in context:
            feedback_items.append(FeedbackItem(
                feedback_id="qual_003",
                feedback_type=FeedbackType.QUALITY_METRIC,
                severity=FeedbackSeverity.HIGH,
                category="Batch Effects",
                message="Potential batch effects detected in PCA",
                suggested_action="Apply batch correction (e.g., ComBat, limma removeBatchEffect)",
                affected_components=["data_normalization"]
            ))
        
        return feedback_items


class LiteratureComparisonCollector(FeedbackCollector):
    """Collects feedback by comparing with literature"""
    
    def __init__(self):
        # In practice, this would connect to literature databases
        self.known_associations = {
            "TP53": ["cancer", "cell_cycle", "apoptosis"],
            "BRCA1": ["breast_cancer", "DNA_repair"],
            "VEGFA": ["angiogenesis", "hypoxia"]
        }
    
    async def collect_feedback(self, analysis_result: Dict[str, Any], 
                             context: Dict[str, Any]) -> List[FeedbackItem]:
        feedback_items = []
        
        # Check gene-disease associations
        if 'significant_genes' in analysis_result:
            genes = analysis_result['significant_genes']
            
            for gene in genes:
                if gene in self.known_associations:
                    expected_terms = self.known_associations[gene]
                    if not any(term in str(analysis_result) for term in expected_terms):
                        feedback_items.append(FeedbackItem(
                            feedback_id=f"lit_{gene}",
                            feedback_type=FeedbackType.LITERATURE_COMPARISON,
                            severity=FeedbackSeverity.MEDIUM,
                            category="Literature Consistency",
                            message=f"{gene} is known to be associated with {', '.join(expected_terms)}",
                            suggested_action="Investigate expected biological connections",
                            affected_components=["biological_interpretation"],
                            validation_data={"gene": gene, "expected_associations": expected_terms}
                        ))
        
        return feedback_items


# =================== Feedback Integration Engine ===================

class FeedbackIntegrationEngine:
    """Integrates feedback from multiple sources and prioritizes improvements"""
    
    def __init__(self):
        self.collectors = [
            StatisticalFeedbackCollector(),
            BiologicalValidationCollector(),
            QualityMetricCollector(),
            LiteratureComparisonCollector()
        ]
        self.feedback_history = []
    
    async def collect_comprehensive_feedback(self, analysis_result: Dict[str, Any],
                                           context: Dict[str, Any]) -> FeedbackReport:
        """Collect feedback from all sources"""
        
        report = FeedbackReport(analysis_id=context.get('analysis_id', 'unknown'))
        
        # Collect from all sources
        for collector in self.collectors:
            try:
                feedback_items = await collector.collect_feedback(analysis_result, context)
                for item in feedback_items:
                    report.add_feedback(item)
            except Exception as e:
                logging.error(f"Error in {collector.__class__.__name__}: {e}")
        
        # Calculate overall quality score
        report.overall_quality_score = self._calculate_quality_score(report)
        
        # Identify automated fixes
        report.automated_fixes_available = self._identify_automated_fixes(report)
        
        # Estimate improvement impact
        report.estimated_improvement_impact = self._estimate_improvement_impact(report)
        
        # Store in history
        self.feedback_history.append(report)
        
        return report
    
    def _calculate_quality_score(self, report: FeedbackReport) -> float:
        """Calculate overall quality score based on feedback"""
        
        if not report.feedback_items:
            return 1.0  # Perfect score if no issues
        
        # Weight by severity
        severity_weights = {
            FeedbackSeverity.CRITICAL: 0.4,
            FeedbackSeverity.HIGH: 0.2,
            FeedbackSeverity.MEDIUM: 0.1,
            FeedbackSeverity.LOW: 0.05,
            FeedbackSeverity.INFO: 0.0
        }
        
        total_weight = sum(
            severity_weights[item.severity] 
            for item in report.feedback_items
        )
        
        # Convert to score (1.0 = perfect, 0.0 = many critical issues)
        quality_score = max(0.0, 1.0 - total_weight)
        
        return quality_score
    
    def _identify_automated_fixes(self, report: FeedbackReport) -> List[str]:
        """Identify which issues can be automatically fixed"""
        
        automated_fixes = []
        
        for item in report.feedback_items:
            # Define automatable fixes
            if item.feedback_id == "stat_003":  # Multiple testing
                automated_fixes.append("Apply FDR correction automatically")
            elif item.feedback_id == "qual_002":  # PCR duplicates
                automated_fixes.append("Remove PCR duplicates automatically")
            elif item.feedback_id == "bio_001":  # Pathway analysis
                automated_fixes.append("Run automated pathway enrichment")
        
        return automated_fixes
    
    def _estimate_improvement_impact(self, report: FeedbackReport) -> Dict[str, float]:
        """Estimate the impact of fixing each issue"""
        
        impact_estimates = {}
        
        for item in report.get_unresolved_items():
            # Estimate impact based on severity
            base_impact = {
                FeedbackSeverity.CRITICAL: 0.3,
                FeedbackSeverity.HIGH: 0.2,
                FeedbackSeverity.MEDIUM: 0.1,
                FeedbackSeverity.LOW: 0.05,
                FeedbackSeverity.INFO: 0.0
            }
            
            impact_estimates[item.feedback_id] = base_impact[item.severity]
        
        return impact_estimates


# =================== Iterative Improvement System ===================

class IterativeImprovementSystem:
    """Manages iterative improvement of analyses based on feedback"""
    
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.improvement_history = []
        self.feedback_engine = FeedbackIntegrationEngine()
    
    async def improve_analysis(self, initial_analysis: Dict[str, Any],
                             initial_code: str,
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Iteratively improve analysis based on feedback"""
        
        current_analysis = initial_analysis
        current_code = initial_code
        iteration = 0
        
        improvement_record = {
            "initial_analysis": initial_analysis,
            "iterations": [],
            "final_analysis": None,
            "total_improvements": 0
        }
        
        while iteration < self.max_iterations:
            # Collect feedback
            feedback_report = await self.feedback_engine.collect_comprehensive_feedback(
                current_analysis, context
            )
            
            # Check if improvements needed
            if feedback_report.overall_quality_score > 0.9:
                logging.info(f"Analysis quality sufficient ({feedback_report.overall_quality_score:.2f})")
                break
            
            critical_items = feedback_report.get_critical_items()
            if not critical_items and iteration > 0:
                logging.info("No critical issues remaining")
                break
            
            # Apply improvements
            improved_analysis, improved_code = await self._apply_improvements(
                current_analysis, current_code, feedback_report
            )
            
            # Record iteration
            improvement_record["iterations"].append({
                "iteration": iteration + 1,
                "feedback_report": feedback_report,
                "quality_score": feedback_report.overall_quality_score,
                "improvements_applied": len(feedback_report.automated_fixes_available)
            })
            
            # Update for next iteration
            current_analysis = improved_analysis
            current_code = improved_code
            iteration += 1
        
        improvement_record["final_analysis"] = current_analysis
        improvement_record["total_improvements"] = iteration
        
        self.improvement_history.append(improvement_record)
        
        return current_analysis
    
    async def _apply_improvements(self, analysis: Dict[str, Any], 
                                code: str,
                                feedback_report: FeedbackReport) -> Tuple[Dict, str]:
        """Apply improvements based on feedback"""
        
        improved_analysis = analysis.copy()
        improved_code = code
        
        # Apply automated fixes
        for fix in feedback_report.automated_fixes_available:
            if "FDR correction" in fix:
                improved_analysis = await self._apply_fdr_correction(improved_analysis)
                improved_code = self._update_code_fdr(improved_code)
            
            elif "PCR duplicates" in fix:
                improved_analysis = await self._remove_duplicates(improved_analysis)
                improved_code = self._update_code_duplicates(improved_code)
            
            elif "pathway enrichment" in fix:
                pathway_results = await self._run_pathway_analysis(improved_analysis)
                improved_analysis['pathway_enrichment'] = pathway_results
                improved_code = self._update_code_pathways(improved_code)
        
        # Mark fixed items as resolved
        for item in feedback_report.feedback_items:
            if item.suggested_action in feedback_report.automated_fixes_available:
                item.resolved = True
        
        return improved_analysis, improved_code
    
    async def _apply_fdr_correction(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply FDR correction to p-values"""
        
        if 'pvalues' in analysis:
            from statsmodels.stats.multitest import multipletests
            
            pvalues = np.array(analysis['pvalues'])
            _, padj, _, _ = multipletests(pvalues, method='fdr_bh')
            
            analysis['padj'] = padj.tolist()
            analysis['correction_method'] = 'FDR_BH'
        
        return analysis
    
    async def _remove_duplicates(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Remove PCR duplicates from analysis"""
        
        # This would implement actual duplicate removal
        analysis['duplicates_removed'] = True
        return analysis
    
    async def _run_pathway_analysis(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run pathway enrichment analysis"""
        
        # Simplified pathway analysis
        if 'significant_genes' in analysis:
            pathways = {
                "Cell cycle": {"pvalue": 0.001, "genes": ["CDK1", "CCNA2"]},
                "Apoptosis": {"pvalue": 0.01, "genes": ["TP53", "BAX"]},
                "DNA repair": {"pvalue": 0.05, "genes": ["BRCA1", "ATM"]}
            }
            return pathways
        
        return {}
    
    def _update_code_fdr(self, code: str) -> str:
        """Update code to include FDR correction"""
        
        if "multipletests" not in code:
            import_line = "from statsmodels.stats.multitest import multipletests\n"
            code = import_line + code
        
        if "padj" not in code:
            fdr_code = """
# Apply FDR correction
_, padj, _, _ = multipletests(results['pvalue'], method='fdr_bh')
results['padj'] = padj
"""
            code += fdr_code
        
        return code
    
    def _update_code_duplicates(self, code: str) -> str:
        """Update code to remove duplicates"""
        
        if "remove_duplicates" not in code:
            dup_code = """
# Remove PCR duplicates
data_dedup = remove_pcr_duplicates(data)
"""
            code += dup_code
        
        return code
    
    def _update_code_pathways(self, code: str) -> str:
        """Update code to include pathway analysis"""
        
        if "pathway" not in code.lower():
            pathway_code = """
# Pathway enrichment analysis
pathway_results = run_pathway_enrichment(
    gene_list=significant_genes,
    database='KEGG',
    correction='FDR'
)
"""
            code += pathway_code
        
        return code


# =================== User Feedback Integration ===================

class UserFeedbackHandler:
    """Handles user-provided feedback and integrates it into the system"""
    
    def __init__(self):
        self.feedback_database = []
        self.learning_patterns = {}
    
    async def process_user_feedback(self, analysis_id: str, 
                                  user_feedback: str,
                                  analysis_result: Dict[str, Any]) -> FeedbackReport:
        """Process user-provided feedback"""
        
        # Parse user feedback
        feedback_items = await self._parse_user_feedback(user_feedback, analysis_id)
        
        # Create feedback report
        report = FeedbackReport(analysis_id=analysis_id)
        for item in feedback_items:
            report.add_feedback(item)
        
        # Learn from feedback
        await self._learn_from_feedback(feedback_items, analysis_result)
        
        # Store feedback
        self.feedback_database.append({
            "analysis_id": analysis_id,
            "user_feedback": user_feedback,
            "feedback_items": feedback_items,
            "timestamp": datetime.now()
        })
        
        return report
    
    async def _parse_user_feedback(self, feedback: str, analysis_id: str) -> List[FeedbackItem]:
        """Parse natural language feedback into structured items"""
        
        feedback_items = []
        feedback_lower = feedback.lower()
        
        # Look for common patterns
        if "wrong" in feedback_lower or "incorrect" in feedback_lower:
            feedback_items.append(FeedbackItem(
                feedback_id=f"user_{analysis_id}_1",
                feedback_type=FeedbackType.USER_CORRECTION,
                severity=FeedbackSeverity.HIGH,
                category="Result Accuracy",
                message="User indicates results are incorrect",
                suggested_action="Review analysis methodology and assumptions"
            ))
        
        if "missing" in feedback_lower:
            feedback_items.append(FeedbackItem(
                feedback_id=f"user_{analysis_id}_2",
                feedback_type=FeedbackType.USER_CORRECTION,
                severity=FeedbackSeverity.HIGH,
                category="Completeness",
                message="User indicates missing analysis components",
                suggested_action="Identify and add missing analyses"
            ))
        
        if "pathway" in feedback_lower or "enrichment" in feedback_lower:
            feedback_items.append(FeedbackItem(
                feedback_id=f"user_{analysis_id}_3",
                feedback_type=FeedbackType.USER_CORRECTION,
                severity=FeedbackSeverity.MEDIUM,
                category="Functional Analysis",
                message="User requests pathway or enrichment analysis",
                suggested_action="Perform pathway enrichment analysis"
            ))
        
        if "quality" in feedback_lower or "qc" in feedback_lower:
            feedback_items.append(FeedbackItem(
                feedback_id=f"user_{analysis_id}_4",
                feedback_type=FeedbackType.USER_CORRECTION,
                severity=FeedbackSeverity.HIGH,
                category="Quality Control",
                message="User has concerns about data quality",
                suggested_action="Perform comprehensive quality control analysis"
            ))
        
        return feedback_items
    
    async def _learn_from_feedback(self, feedback_items: List[FeedbackItem],
                                 analysis_result: Dict[str, Any]):
        """Learn patterns from user feedback"""
        
        # Track common feedback patterns
        for item in feedback_items:
            pattern_key = f"{item.category}_{item.message}"
            
            if pattern_key not in self.learning_patterns:
                self.learning_patterns[pattern_key] = {
                    "count": 0,
                    "contexts": [],
                    "suggested_preventions": []
                }
            
            self.learning_patterns[pattern_key]["count"] += 1
            self.learning_patterns[pattern_key]["contexts"].append(
                str(analysis_result)[:200]  # Store context snippet
            )
    
    def get_common_issues(self) -> List[Dict[str, Any]]:
        """Get commonly reported issues"""
        
        sorted_patterns = sorted(
            self.learning_patterns.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        return [
            {
                "issue": pattern[0],
                "frequency": pattern[1]["count"],
                "suggested_prevention": pattern[1].get("suggested_preventions", [])
            }
            for pattern in sorted_patterns[:10]
        ]


# =================== Adaptive Learning System ===================

class AdaptiveLearningSystem:
    """Learns from feedback history to prevent future issues"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.issue_prevention_model = {}
        self.success_patterns = {}
        self.failure_patterns = {}
    
    async def learn_from_history(self, feedback_history: List[FeedbackReport],
                               analysis_history: List[Dict[str, Any]]):
        """Learn from historical feedback and analyses"""
        
        # Identify patterns in successful analyses
        for i, report in enumerate(feedback_history):
            if report.overall_quality_score > 0.9:
                self._extract_success_patterns(analysis_history[i])
            else:
                self._extract_failure_patterns(analysis_history[i], report)
        
        # Build prevention model
        await self._build_prevention_model()
    
    def _extract_success_patterns(self, analysis: Dict[str, Any]):
        """Extract patterns from successful analyses"""
        
        patterns = {
            "has_quality_control": "quality" in str(analysis).lower(),
            "has_multiple_testing": "fdr" in str(analysis).lower() or "bonferroni" in str(analysis).lower(),
            "has_pathway_analysis": "pathway" in str(analysis).lower(),
            "has_validation": "validation" in str(analysis).lower(),
            "has_effect_sizes": "fold_change" in str(analysis).lower() or "effect_size" in str(analysis).lower()
        }
        
        for pattern, present in patterns.items():
            if pattern not in self.success_patterns:
                self.success_patterns[pattern] = 0
            if present:
                self.success_patterns[pattern] += 1
    
    def _extract_failure_patterns(self, analysis: Dict[str, Any], report: FeedbackReport):
        """Extract patterns from analyses with issues"""
        
        for item in report.feedback_items:
            if item.severity in [FeedbackSeverity.CRITICAL, FeedbackSeverity.HIGH]:
                pattern_key = f"{item.category}_{item.feedback_type.value}"
                
                if pattern_key not in self.failure_patterns:
                    self.failure_patterns[pattern_key] = {
                        "count": 0,
                        "common_missing": []
                    }
                
                self.failure_patterns[pattern_key]["count"] += 1
                
                # Track what was missing
                if "pathway" not in str(analysis).lower():
                    self.failure_patterns[pattern_key]["common_missing"].append("pathway_analysis")
                if "correction" not in str(analysis).lower():
                    self.failure_patterns[pattern_key]["common_missing"].append("multiple_testing_correction")
    
    async def _build_prevention_model(self):
        """Build model to prevent common issues"""
        
        # Simple rule-based prevention model
        self.issue_prevention_model = {
            "always_include": [],
            "check_for": [],
            "validate": []
        }
        
        # Identify must-have components
        total_successes = sum(self.success_patterns.values()) / len(self.success_patterns) if self.success_patterns else 1
        
        for pattern, count in self.success_patterns.items():
            if count / total_successes > 0.8:  # Present in >80% of successful analyses
                self.issue_prevention_model["always_include"].append(pattern)
        
        # Identify common failure triggers
        for pattern, data in self.failure_patterns.items():
            if data["count"] > 3:  # Repeated issue
                self.issue_prevention_model["check_for"].extend(data["common_missing"])
    
    def get_prevention_recommendations(self, planned_analysis: Dict[str, Any]) -> List[str]:
        """Get recommendations to prevent issues"""
        
        recommendations = []
        
        # Check always-include items
        for required in self.issue_prevention_model.get("always_include", []):
            if required not in str(planned_analysis).lower():
                recommendations.append(f"Include {required.replace('_', ' ')} in analysis")
        
        # Check for common failure triggers
        for check in set(self.issue_prevention_model.get("check_for", [])):
            if check not in str(planned_analysis).lower():
                recommendations.append(f"Consider adding {check.replace('_', ' ')}")
        
        return recommendations


# =================== Example Usage ===================

async def example_feedback_system():
    """Example of using the feedback and improvement system"""
    
    # Sample analysis result
    analysis_result = {
        "analysis_id": "example_001",
        "pvalues": np.random.rand(1000).tolist(),
        "significant_genes": ["TP53", "BRCA1", "VEGFA"],
        "fold_changes": np.random.randn(1000).tolist(),
        "statistical_test": "t-test"
    }
    
    context = {
        "analysis_id": "example_001",
        "quality_metrics": {
            "average_quality": 28,
            "duplication_rate": 0.3
        }
    }
    
    # Initialize feedback system
    feedback_engine = FeedbackIntegrationEngine()
    improvement_system = IterativeImprovementSystem()
    
    # Collect initial feedback
    print("Collecting comprehensive feedback...")
    feedback_report = await feedback_engine.collect_comprehensive_feedback(
        analysis_result, context
    )
    
    print(f"Overall quality score: {feedback_report.overall_quality_score:.2f}")
    print(f"Critical issues: {len(feedback_report.get_critical_items())}")
    print(f"Total feedback items: {len(feedback_report.feedback_items)}")
    
    # Apply iterative improvements
    print("\nApplying iterative improvements...")
    initial_code = "# Initial analysis code\nimport pandas as pd\n# ... analysis ..."
    
    improved_analysis = await improvement_system.improve_analysis(
        analysis_result, initial_code, context
    )
    
    # Handle user feedback
    user_handler = UserFeedbackHandler()
    user_feedback = "The analysis looks good but please add pathway enrichment analysis"
    
    print("\nProcessing user feedback...")
    user_report = await user_handler.process_user_feedback(
        "example_001", user_feedback, improved_analysis
    )
    
    print(f"User feedback items: {len(user_report.feedback_items)}")
    
    # Adaptive learning
    learning_system = AdaptiveLearningSystem()
    await learning_system.learn_from_history(
        [feedback_report, user_report],
        [analysis_result, improved_analysis]
    )
    
    recommendations = learning_system.get_prevention_recommendations(
        {"planned_analysis": "differential expression"}
    )
    
    print("\nPrevention recommendations:")
    for rec in recommendations:
        print(f"- {rec}")


if __name__ == "__main__":
    asyncio.run(example_feedback_system())