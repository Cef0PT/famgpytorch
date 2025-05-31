#!/usr/bin/env python3

from setuptools import find_packages, setup

VERSION = "0.1.0"

readme = open("README.md").read()

setup(
    name="famgpytorch",
    version=VERSION,
    description="An implementation of fast approximate Gaussian Processes utilizing GPyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Pascal Alexander Tölle",
    project_urls= {"Source": "https://github.com/Cef0PT/famgpytorch"},
    classifiers=["Development Status :: 3 - Alpha", "Programming Language :: Python :: 3"],
    packages=find_packages(),
    install_requires=["torch", "gpytorch", "linear_operator"],
    extras_require={"playground": ["ipython", "jupyter", "matplotlib", "numpy"]},
)