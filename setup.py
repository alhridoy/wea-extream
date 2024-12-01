"""Setup file for wx-extreme package."""

from setuptools import setup, find_packages

setup(
    name="wx-extreme",
    version="1.0.0",
    description="Weather Extreme Event Detection and Validation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Al-Ekram Elahee Hridoy",
    author_email="alekram.elahee@gmail.com",
    url="https://github.com/alhridoy/wx-extreme",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "xarray>=2022.3.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "cartopy>=0.20.0",
        "scipy>=1.8.0",
        "dask>=2022.1.0",
        "distributed>=2022.1.0",
        "zarr>=2.11.0",
        "gcsfs>=2022.1.0",
        "netCDF4>=1.5.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
    keywords="weather climate extreme-events machine-learning forecasting",
    project_urls={
        "Bug Reports": "https://github.com/alhridoy/wx-extreme/issues",
        "Source": "https://github.com/alhridoy/wx-extreme",
    },
)
