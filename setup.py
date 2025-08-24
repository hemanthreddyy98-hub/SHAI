#!/usr/bin/env python3
"""
Setup script for SHAI - Super Human AI
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="shai-ai",
    version="1.0.0",
    author="SHAI Team",
    author_email="support@shai-ai.com",
    description="Super Human AI - Advanced AI Assistant with Multi-Model Intelligence",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/SHAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "web": [
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "streamlit>=1.28.0",
            "gradio>=4.0.0",
        ],
        "full": [
            "torch>=2.0.0",
            "tensorflow>=2.13.0",
            "transformers>=4.35.0",
            "spacy>=3.7.0",
            "chromadb>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "shai=shai_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "shai": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="ai artificial-intelligence chatbot nlp machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/SHAI/issues",
        "Source": "https://github.com/your-repo/SHAI",
        "Documentation": "https://github.com/your-repo/SHAI/docs",
    },
)
