"""
Setup script for wan-video-toolkit

This provides backward compatibility for older pip versions and
ensures proper package installation from Git repositories.
"""

from setuptools import setup, find_packages

# Read version from version.py
version = {}
with open("wan_video_toolkit/version.py") as fp:
    exec(fp.read(), version)

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wan-video-toolkit",
    version=version["__version__"],
    author="Wan Video Toolkit Contributors",
    description="A comprehensive toolkit for Wan video generation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kaid/wan_video_toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "diffusers>=0.24.0", 
        "transformers>=4.30.0",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "typing-extensions>=4.0.0",
        "huggingface-hub>=0.15.0",
    ],
    extras_require={
        "gradio": ["gradio>=4.0.0"],
        "dev": [
            "mypy>=1.0.0",
            "pyright>=1.1.300",
            "black>=22.0.0",
            "isort>=5.10.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=6.0.0",
        ],
        "all": ["wan-video-toolkit[gradio,dev]"],
    },
)