"""Setup file for wx-extreme package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wx-extreme",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced evaluation framework for extreme weather events in ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wx-extreme",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "xarray>=2022.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "cartopy>=0.20.0",  # For geographical plotting
        "eccodes>=1.4.0",  # For reading weather data formats
        "netcdf4>=1.5.7",  # For reading NetCDF files
        "dask>=2022.1.0",  # For parallel computing
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.900",
            "pylint>=2.8.0",
        ],
    },
)
