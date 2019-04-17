import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="scvae",
    version="2.0",
    author="Christopher Heje Grønbech",
    author_email="christopher.heje.groenbech@regionh.dk",
    description="Model single-cell transcript counts using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chgroenbech/scVAE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
