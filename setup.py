# adapted from github.com/vsbuffalo/cvtkpy

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparg",
    version="0.0.1",
    author="Matthew Osmond",
    author_email="mm.osmond@utoronto.ca",
    description="A python package to estimate dispersal rates and locate genetic ancestors from genome-wide genealogies (Osmond & Coop 2021)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmosmond/sparg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.9',
)

