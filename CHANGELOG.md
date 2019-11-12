# Release History

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
