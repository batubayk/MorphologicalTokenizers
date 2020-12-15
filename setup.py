#!/usr/bin/env python
import setuptools

with open("requirements.txt", "r") as requirement_file:
    requirements = requirement_file.read()

setuptools.setup(
    name="Morphological Tokenizers",
    version="0.0.1",
    author="Batuhan Baykara",
    author_email="batuhanbaykara@yahoo.com",
    description="A package that contains tokenizers for morphologically rich languages such as Turkish and Hungarian.",
    packages=setuptools.find_packages(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent",],
    python_requires=">=3.5",
    install_requires=requirements,
)
