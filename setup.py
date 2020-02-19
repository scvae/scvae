#!/usr/bin/env python3

import setuptools

import scvae

NAME = scvae.__name__
DESCRIPTION = scvae.__description__
URL = scvae.__url__
AUTHOR = scvae.__author__
EMAIL = scvae.__email__
VERSION = scvae.__version__
LICENSE = scvae.__license__
PYTHON_VERSION_REQUIREMENT = ">=3.6, <3.8"

PACKAGE_REQUIREMENTS = [
    "importlib_resources >= 1.0",
    "loompy >= 2.0",
    "numpy >= 1.16",
    "matplotlib >= 2.0",
    "pandas >= 0.24",
    "pillow >= 5.4",
    "scikit-learn >= 0.20",
    "scipy >= 1.2",
    "seaborn >= 0.9",
    "tables >= 3.5",
    "tensorflow >= 1.15.2, < 2",
    "tensorflow-probability == 0.7"
]

DOCUMENTATION_REQUIREMENTS = [
    "pygments >= 2.4",
    "sphinx >= 2.2"
]

EXTRAS_REQUIREMENTS = {
    "docs": DOCUMENTATION_REQUIREMENTS
}

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.md", "r") as changelog_file:
    changelog = changelog_file.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description="\n\n".join([readme, changelog]),
    long_description_content_type="text/markdown",
    url=URL,
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": ["scvae=scvae.__main__:main"],
    },
    python_requires=PYTHON_VERSION_REQUIREMENT,
    install_requires=PACKAGE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIREMENTS,
    license=LICENSE,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
