# Release History #

## 2.1.2 (2020-04-07) ##

* Export of decomposition of data sets and latent values as compressed TSV files.
* Export of predictions as compressed TSV files.
* Fix potential crash during *t*-SNE decomposition.

## 2.1.1 (2020-02-24) ##

* Requires TensorFlow 1.15.2 because of a security vulnerability.
* Export of latent values as compressed TSV files.
* Make folder names and filenames more safe on Windows.
* Regrouped analyses, so fewer analyses are performed by default. All available analyses can be performed using ``--included-analyses all``.
* Fix loading of KL divergences when evaluating VAE models.
* Fix crash during model analyses, if the model did not exist.

## 2.1.0 (2019-11-12) ##

* Requires Python 3.6 or 3.7 as well as TensorFlow 1.15.
* Documentation with user guide and tutorial.
* Support for sparse matrices in HDF5 format.
* Improved support for Loom files by following conventions.
* Scatter plots of classes against the primary latent feature as well as the two primary latent features against each other when evaluating a model.
* Fix crash related to `argparse` when using Python 3.6.

## 2.0.0 (2019-05-18) ##

* Complete refactor and clean-up including structuring as Python package.
* Easier loading of custom data sets.
* Batch correction included in models for data sets with batch indices.
* Learnable mixture coefficients for the GMVAE model.
* Full covariance matrix for the GMVAE model.
* Sampling from models.

## 1.0 (2018-04-25) ##

Initial release.
