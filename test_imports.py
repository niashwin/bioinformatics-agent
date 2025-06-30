#!/usr/bin/env python3
"""
Test script to verify all BioinformaticsAgent imports work correctly.
Run this to debug import issues.
"""

import sys
import os
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

print("🔍 Testing BioinformaticsAgent imports...")
print(f"📁 Current directory: {current_dir}")
print(f"📁 Src directory: {src_dir}")
print(f"📁 Python path: {sys.path[:3]}...")

# Test individual module imports
modules_to_test = [
    'bioagent_architecture',
    'bioagent_tools', 
    'bioagent_io',
    'bioagent_statistics',
    'bioagent_pipeline'
]

successful_imports = []
failed_imports = []

for module_name in modules_to_test:
    try:
        module = __import__(module_name)
        successful_imports.append(module_name)
        print(f"✅ {module_name}")
    except ImportError as e:
        failed_imports.append((module_name, str(e)))
        print(f"❌ {module_name}: {e}")

print(f"\n📊 Import Results:")
print(f"✅ Successful: {len(successful_imports)}/{len(modules_to_test)}")
print(f"❌ Failed: {len(failed_imports)}/{len(modules_to_test)}")

if successful_imports:
    print(f"\n🎯 Testing core functionality...")
    
    try:
        from bioagent_architecture import BioinformaticsAgent, DataType
        agent = BioinformaticsAgent()
        print("✅ BioinformaticsAgent creation successful")
        
        # Test basic functionality
        print(f"✅ Available data types: {len(DataType)}")
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")

if failed_imports:
    print(f"\n🔧 Failed imports:")
    for module, error in failed_imports:
        print(f"  • {module}: {error}")

print(f"\n🎉 Import test complete!")