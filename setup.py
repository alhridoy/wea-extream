"""Setup file for wx-extreme package."""

from setuptools import setup, find_packages

setup(
    name="wx-extreme",
    version="1.0.0",
    description="Weather Extreme Event Detection and Validation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/wx-extreme",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "xarray>=2023.1.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "cartopy>=0.21.0",
        "scipy>=1.10.0",
        "dask>=2023.1.0",
        "distributed>=2023.1.0",
        "zarr>=2.14.0",
        "gcsfs>=2023.1.0",
        "netCDF4>=1.6.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)
