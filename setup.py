#!/usr/bin/env python3

import setuptools
from pathlib import Path

here = Path(__file__).resolve().parent

python_version_requirement = ">=3.6, <3.8"

package_requirements = [
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

documentation_requirements = [
    "pygments >= 2.4",
    "sphinx >= 2.2"
]

extras_requirements = {
    "docs": documentation_requirements
}

about = {}

with here.joinpath("scvae", "__init__.py").open(mode="r") as init_file:
    exec(init_file.read(), about)

with here.joinpath("README.md").open(mode="r") as readme_file:
    readme = readme_file.read()

with here.joinpath("CHANGELOG.md").open(mode="r") as changelog_file:
    changelog = changelog_file.read()

setuptools.setup(
    name=about["__name__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__email__"],
    description=about["__description__"],
    long_description="\n\n".join([readme, changelog]),
    long_description_content_type="text/markdown",
    url=about["__url__"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": ["scvae=scvae.__main__:main"],
    },
    python_requires=python_version_requirement,
    install_requires=package_requirements,
    extras_require=extras_requirements,
    license=about["__license__"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
