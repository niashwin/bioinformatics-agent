# BioinformaticsAgent - Quick Start Guide

## 🚀 Installation & Setup

### Option 1: Minimal Installation (Recommended)
```bash
git clone https://github.com/niashwin/bioinformatics-agent.git
cd bioinformatics-agent
pip install -r requirements-minimal.txt
```

### Option 2: Full Installation
```bash
git clone https://github.com/niashwin/bioinformatics-agent.git
cd bioinformatics-agent
pip install setuptools>=65.0.0 wheel>=0.37.0
pip install -r requirements.txt
```

## 📖 Running the Demo

### Interactive Jupyter Notebook
```bash
jupyter notebook bioinformatics_agent_demo.ipynb
```

### Complete Test Suite
```bash
python tests/test_all_capabilities.py
```
This generates `bioinformatics_agent_test_report.html` with comprehensive results.

### Test Imports (Debugging)
```bash
python test_imports.py
```

## ✅ What's Fixed

- ✅ **Import errors resolved** - All modules now import correctly
- ✅ **Graceful dependency handling** - Missing packages won't crash the system
- ✅ **Clear error messages** - Helpful guidance for installing optional features
- ✅ **Works in any environment** - Core functionality available with minimal deps

## 🔧 Core vs Optional Features

### Core Features (Always Available)
- BioinformaticsAgent architecture
- Basic tool framework 
- Pipeline orchestration
- File I/O for common formats
- Statistical analysis

### Optional Features (Install as needed)
- Single-cell analysis: `pip install scanpy anndata`
- Advanced file formats: `pip install h5py zarr pysam cyvcf2`
- Cloud storage: `pip install boto3 google-cloud-storage`
- Specialized stats: `pip install pingouin pydeseq2`

## 🎯 Usage Example

```python
from bioagent_architecture import BioinformaticsAgent, DataMetadata, DataType
from bioagent_tools import get_all_bioinformatics_tools

# Initialize agent
agent = BioinformaticsAgent()

# Register tools
for tool in get_all_bioinformatics_tools():
    agent.register_tool(tool)

print(f"Agent ready with {len(agent.tools)} tools!")
```

## 🆘 Troubleshooting

**ImportError: No module named 'bioagent_architecture'**
- Make sure you're in the correct directory
- Run `python test_imports.py` to debug

**Missing optional dependencies**
- Check the error message for installation instructions
- Use `requirements-minimal.txt` for basic functionality

**Jupyter notebook issues**
- Ensure you've installed: `pip install jupyter ipywidgets`
- Restart kernel if imports fail

## 📊 System Status

Run `python test_imports.py` to see:
```
✅ bioagent_architecture: Core agent functionality
✅ bioagent_tools: Tool framework  
✅ bioagent_io: File I/O
✅ bioagent_statistics: Statistical analysis
✅ bioagent_pipeline: Pipeline orchestration
```

🎉 **BioinformaticsAgent is ready for production bioinformatics analysis!**