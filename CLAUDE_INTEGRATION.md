# Claude-Powered BioinformaticsAgent

This enhanced version of BioinformaticsAgent integrates with Claude API to provide AI-powered bioinformatics analysis with conversational interfaces, dynamic tool creation, and intelligent reflection loops.

## 🚀 New Features

### 🤖 Claude API Integration
- **Conversational Planning**: Claude analyzes your requirements and creates detailed analysis plans
- **Intelligent Code Generation**: Production-ready Python code generated based on biological context
- **Natural Language Interface**: Chat directly with Claude about bioinformatics questions
- **Result Interpretation**: Claude provides biological insights and statistical interpretation

### 🛠️ Dynamic Tool System
- **Tool Modification**: Adapt existing tools to new requirements using AI
- **Tool Creation**: Generate completely new bioinformatics tools with Claude
- **Automatic Benchmarking**: AI-generated validation and testing code
- **Smart Tool Selection**: Claude selects optimal tools for your analysis

### 🔄 Reflection and Validation
- **Multi-Criteria Assessment**: Evaluate analyses across correctness, completeness, biological relevance
- **Iterative Improvement**: Automatic suggestions for enhancing analysis quality
- **Statistical Validation**: Built-in checks for statistical rigor and multiple testing
- **Confidence Scoring**: Quantified confidence levels for all results

### 🔒 Safe Code Execution
- **Sandboxed Environment**: Secure execution with resource limits and timeouts
- **Input Validation**: Comprehensive code validation before execution
- **Result Capture**: Automatic capture of outputs, plots, and generated files
- **Error Handling**: Graceful handling of execution failures with detailed diagnostics

## 📦 Installation

### 1. Install Dependencies
```bash
# Install the enhanced requirements
pip install -r requirements.txt

# Key new dependencies:
# - anthropic>=0.18.0 (Claude API client)
# - httpx>=0.24.0 (Async HTTP client)
```

### 2. Set Up Claude API
```bash
# Get your API key from https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-api-key-here"

# Or add to your .env file
echo "ANTHROPIC_API_KEY=your-api-key-here" >> .env
```

### 3. Test Installation
```bash
python test_claude_integration.py
```

## 🎯 Quick Start

### Interactive Mode
```bash
python interactive_agent.py
```

### Example Session
```
🧬 bioagent> load example_data/expression_matrix.csv
✅ Loaded expression_matrix (expression_matrix) from example_data/expression_matrix.csv

🧬 bioagent> analyze "Find differentially expressed genes between treatment and control groups, perform pathway enrichment, and create visualizations"

🤔 Analyzing: Find differentially expressed genes...
⚙️ Generating analysis plan...

✅ Analysis completed!

📋 Analysis Plan:
  1. Load and validate expression data
  2. Perform differential expression analysis with DESeq2
  3. Apply multiple testing correction (FDR)
  4. Filter significant genes (padj < 0.05, |log2FC| > 1)
  5. Perform pathway enrichment analysis
  6. Generate volcano plot and heatmap visualizations

💻 Generated Code:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# ... (complete analysis code)
```

🤖 Claude Analysis:
The analysis successfully identified 156 differentially expressed genes...

📈 Reflection Summary:
   Overall Score: 8.7/10
   Confidence: high
   Meets Requirements: ✅

💪 Strengths:
   • Strong statistical validity
   • Strong interpretability
   • Strong biological relevance
```

## 🔧 Advanced Features

### Chat with Claude
```bash
🧬 bioagent> chat "What are best practices for single-cell RNA-seq quality control?"
🤖 Claude: For single-cell RNA-seq QC, focus on these key metrics:
1. Mitochondrial gene percentage (<10-20%)
2. Total UMI counts (filter cells with <500 or >5000 UMIs)
3. Gene detection (filter cells expressing <200 genes)
...
```

### Modify Existing Tools
```bash
🧬 bioagent> modify sequence_stats "Add support for calculating Shannon entropy and complexity measures"
🔧 Modifying tool 'sequence_stats'...
✅ Tool modification completed!
```

### Create New Tools
```bash
🧬 bioagent> create "Codon usage analyzer" "Calculate codon usage bias and optimal codons for different organisms"
🛠️ Creating new tool...
✅ Tool creation completed!
```

## 📊 Analysis Pipeline

The enhanced agent follows this intelligent pipeline:

1. **Planning Phase**
   - Claude analyzes your request
   - Generates structured analysis plan
   - Selects appropriate tools and methods

2. **Code Generation**
   - Claude generates production-ready Python code
   - Includes proper error handling and validation
   - Follows bioinformatics best practices

3. **Safe Execution**
   - Code runs in sandboxed environment
   - Resource limits and timeouts enforced
   - Results and outputs captured

4. **Reflection & Validation**
   - Multi-criteria quality assessment
   - Statistical and biological validation
   - Improvement suggestions generated

5. **Iteration** (if needed)
   - Code modifications based on reflection
   - Tool adaptation or creation
   - Re-execution with improvements

## 🎛️ Configuration

### Environment Variables
```bash
# Required for Claude features
ANTHROPIC_API_KEY=your-api-key-here

# Optional configurations
BIOAGENT_WORK_DIR=/path/to/working/directory
BIOAGENT_OUTPUT_DIR=output
BIOAGENT_MAX_MEMORY_MB=2048
BIOAGENT_TIMEOUT_SECONDS=300
```

### Execution Configuration
```python
from bioagent_execution import ExecutionConfig

config = ExecutionConfig(
    timeout=300,           # 5 minutes
    max_memory_mb=2048,    # 2GB limit
    max_disk_mb=1024,      # 1GB disk usage
    output_dir="results"   # Output directory
)
```

## 🔬 Supported Analysis Types

### Genomics
- Sequence analysis and statistics
- Variant calling and annotation
- Comparative genomics
- Phylogenetic analysis

### Transcriptomics
- RNA-seq differential expression
- Single-cell RNA-seq analysis
- Pathway enrichment analysis
- Co-expression network analysis

### Proteomics
- Mass spectrometry data analysis
- Protein structure analysis
- Functional annotation
- Protein-protein interactions

### Multi-omics
- Data integration across platforms
- Biomarker discovery
- Systems biology analysis
- Pathway reconstruction

## 🛡️ Security Features

### Code Validation
- Syntax checking and parsing
- Dangerous function detection
- Import restriction enforcement
- Resource usage monitoring

### Execution Safety
- Sandboxed environment isolation
- Memory and disk limits
- Network access restrictions
- Timeout enforcement

### Data Protection
- Local execution only
- No data sent to external services
- Temporary file cleanup
- Access control validation

## 📈 Performance

### Resource Usage
- **Memory**: Configurable limits (default: 2GB)
- **CPU**: Multi-core support for parallel analyses
- **Disk**: Automatic cleanup of temporary files
- **Network**: Minimal usage (API calls only)

### Optimization Features
- **Caching**: Tool and prompt caching
- **Parallel Execution**: Multi-step pipeline parallelization
- **Lazy Loading**: On-demand tool initialization
- **Result Reuse**: Automatic result caching and reuse

## 🐛 Troubleshooting

### Common Issues

**API Key Issues**
```bash
# Check if API key is set
echo $ANTHROPIC_API_KEY

# Test API connection
python -c "import anthropic; print('API key valid' if anthropic.Anthropic().api_key else 'API key missing')"
```

**Memory Issues**
```bash
# Reduce memory limits
export BIOAGENT_MAX_MEMORY_MB=1024

# Monitor memory usage
python test_claude_integration.py
```

**Execution Timeouts**
```bash
# Increase timeout
export BIOAGENT_TIMEOUT_SECONDS=600

# Check system resources
htop
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
python interactive_agent.py
```

## 🤝 Contributing

### Adding New Tools
1. Create tool class inheriting from `BioinformaticsTool`
2. Implement parameter schema and execute method
3. Add to tool registry
4. Test with Claude integration

### Enhancing Prompts
1. Update `bioagent_prompts.py`
2. Add domain-specific templates
3. Test with various analysis types
4. Validate Claude responses

### Improving Reflection
1. Add new assessment criteria
2. Enhance scoring algorithms
3. Include domain-specific validations
4. Test with real datasets

## 📚 Examples

See the `examples/` directory for complete analysis examples:
- `rna_seq_analysis.py` - Complete RNA-seq pipeline
- `variant_analysis.py` - Genomic variant analysis
- `single_cell_analysis.py` - scRNA-seq workflow
- `multi_omics_integration.py` - Multi-platform analysis

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on Anthropic's Claude API
- Inspired by the Gemini CLI Agent Architecture
- Utilizes the comprehensive bioinformatics ecosystem
- Community contributions and feedback

---

**Ready to revolutionize your bioinformatics workflows with AI? 🚀**

Start with `python interactive_agent.py` and experience the future of computational biology!