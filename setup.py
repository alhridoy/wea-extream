"""Setup file for wx-extreme package."""

from setuptools import setup, find_packages

setup(
    name="wx-extreme",
    version="0.1.0",
    description="Framework for detecting and analyzing extreme heat events",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "xarray>=2022.3.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.8",
)
