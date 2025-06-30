#!/usr/bin/env python3
"""
Setup script for BioinformaticsAgent
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
with open(readme_file, "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
with open(requirements_file, "r", encoding="utf-8") as f:
    requirements = [
        line.strip() 
        for line in f.readlines() 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="bioinformatics-agent",
    version="1.0.0",
    author="Ashwin Gopinath",
    author_email="ashwin@example.com",
    description="An advanced AI system for bioinformatics and computational biology analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/bioinformatics-agent",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "bio-tools": [
            "pysam>=0.19.0",
            "pyvcf>=0.6.8",
            "dendropy>=4.5.0",
            "ete3>=3.1.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "bioagent=bioagent_example:main",
            "bioagent-interactive=bioagent_example:interactive_mode",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="bioinformatics computational-biology ai machine-learning genomics",
    project_urls={
        "Bug Reports": "https://github.com/your-username/bioinformatics-agent/issues",
        "Source": "https://github.com/your-username/bioinformatics-agent",
        "Documentation": "https://github.com/your-username/bioinformatics-agent/wiki",
    },
)