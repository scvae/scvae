.. highlight:: terminal

User Guide
==========

scVAE model count data, primarily single-cell gene transcript counts, using variational auto-encoders (:ref:`Kingma and Welling, 2014 <kingma2014>`; :ref:`Rezende et al., 2014 <rezende2014>`).

Installing scVAE
----------------

scVAE requires Python 3.6--3.7, which can be installed in `several ways`_, for example, using Miniconda_.

.. _several ways: https://realpython.com/installing-python/
.. _Miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

With Python in place, scVAE can be installed using pip::

   $ python3 -m pip install scvae

Using scVAE
-----------

In general, scVAE is used in the following way::

   $ scvae $COMMAND $DATA_SET

where ``$COMMAND`` can be ``analyse`` (data analysis), ``train`` (model training), or ``evaluate`` (model evaluation and analysis). ``$DATA_SET`` is a path to a data set file or a short name for a data set.

By default, data are placed and cached in the subfolder ``data/``, models are saved in the subfolder ``models/``, and analyses are saved in the subfolder ``analyses/``.

In the following, the most relevant options are described. Use the help option to list all options for each command::

   $ scvae $COMMAND --help

Data sets
^^^^^^^^^

Several data sets are already included in scVAE:

.. Non-breaking space
.. |_| unicode:: 0xA0
   :trim:
   
* ``Macosko-MRC``: `GSE63472`_.
* ``10x-MBC``: `1.3 Million Brain Cells from E18 Mice`_ from `10x Genomics`_.
   * ``10x-MBC-20k``: 20 |_| 000 sampled cells.
* ``10x-PBMC-PP``: Nine data sets of `purified PBMC populations`_ from 10x Genomics as specified in :ref:`Grønbech et al. (2020) <groenbech2020>`.
* ``10x-PBMC-68k``: `Fresh 68k PBMCs (Donor A)`_ from 10x Genomics.
* ``TCGA-RSEM``: `"transcript expression RNAseq - TOIL RSEM expected_count"`_ data set from the `TCGA Pan-Cancer (PANCAN)`_ cohort.

.. _GSE63472: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63472
.. _1.3 Million Brain Cells from E18 Mice: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons
.. _10x Genomics: https://www.10xgenomics.com
.. _purified PBMC populations: https://support.10xgenomics.com/single-cell-gene-expression/datasets/
.. _Fresh 68k PBMCs (Donor A): https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a
.. _"transcript expression RNAseq - TOIL RSEM expected_count": https://xenabrowser.net/datapages/?dataset=tcga_expected_count&host=https%3A%2F%2Ftoil.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443
.. _TCGA Pan-Cancer (PANCAN): https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443

Data sets will be cached in the data directory, which defaults to ``data/``. This can be changed using the option ``--data-directory`` (or ``-D``).

Be aware that it might take some time to load and preprocess the data the first time for large data sets. Also note that to load and analyse the ``10x-MBC`` data set, 47 GB of memory is required (32 GB for the original data set in sparse representation and 15 GB for the reconstructed test set in dense representation).

The default model can be trained on, for example, the ``10x-PBMC-PP`` data set like this::

   $ scvae train 10x-PBMC-PP

Custom data sets
""""""""""""""""

scVAE can read `Loom`_ files, and it can read a dense data matrix from a TSV file or a sparse one from a HDF5 file without further configuration. As an example, a data set in Loom format can be imported and modelled in the following way::

   $ scvae train data_set.loom

.. _Loom: https://loompy.org

The TSV files can be compressed using gzip, but each row should represent a cell or sample and each column a gene (for the reverse case, see below). If a header row and/or a header column are provided, they are used as gene IDs/names and/or cell/sample names, respectively.

For Loom files, scVAE follows `Loompy's conventions`_: each column represent a cell or sample and each row a gene. Cell or sample names are specified using the column attribute ``CellID`` (or just ``Cell``), and the row attribute ``Gene`` is used for gene names.

.. _Loompy's conventions: http://linnarssonlab.org/loompy/conventions/index.html

HDF5 files should include a single directory containing arrays for the sparse matrix (with names as for SciPy's CSR/CSC sparse matrix format: ``data``, ``indices``, ``indptr``, ``shape``). Arrays for example/cell and feature/gene names are also supported with a varieity of naming conventions: for example, ``barcodes``, ``cells``, ``samples``, ``examples`` for the former; ``genes`` and ``features`` for the latter. If either name array or both are present, scVAE will try to orient the matrix to match their dimensions.

scVAE also supports the following formats (supplied using the ``--format`` option):

* ``10x``: Output format for 10x Genomics's Cell Ranger.
* ``gtex``: Format for data sets from `GTEx`_.
* ``matrix_ebf``: (gzip compressed) TSV file with cells/samples/examples as rows and gene/features as columns (examples-by-features).
* ``matrix_fbe``: (gzip compressed) TSV file with gene/features as rows and cells/samples/examples as columns (features-by-examples).

.. _GTEx: https://gtexportal.org/home/index.html

The last of these formats can be used to read a TSV file, which is in reverse order of the default case::

   $ scvae train data_set.tsv.gz --format matrix_fbe

Using the Loom format, included cell types and batch indices can also be imported without further configuration by using the column attributes "ClusterName" and "BatchID", respectively.

Cell types for other formats can be imported in TSV format by instead providing a `JSON`_ file with a ``values`` field with the filename for the read counts, a ``labels`` field with the filename the cell types, and ``format`` field with the format.

.. _JSON: https://en.wikipedia.org/wiki/JSON

A JSON file for a GTEx data set would look like this:

.. code-block:: json

   {
      "values": "GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.gct.gz",
      "labels": "GTEx_v7_Annotations_SampleAttributesDS.txt",
      "format": "gtex"
   }

Naming this file ``gtex.json``, the GTEx data set can then be imported and modelled::

   $ scvae train gtex.json

Withheld data
"""""""""""""

Any data set can be split into a training, a validation, and a test set using the ``--split-data-set`` option::

   $ scvae train $DATA_SET --split-data-set

Then, the training set is used to train the model, the validation set is used for early stopping as well as finding the best model parameters, and the test set is used when evaluating the model.

The data set can be split either randomly (``random``) or in the sequence in which it already is [#sequence]_ (``sequential``). This is done by specifying either value using the option ``--splitting-method``::

   $ scvae train $DATA_SET --split-data-set --splitting-method random

The fraction of the data set used for the training and validation sets is set using the option ``--splitting-fraction``::

   $ scvae train $DATA_SET --split-data-set --splitting-fraction 0.9

This option also determines the fraction of the training and validation sets used when training a model. The above command will then split the data sets into training, validation, and test sets using a :math:`81 \, \%`- :math:`9 \, \%`-:math:`10 \, \%` split.

Training a model
^^^^^^^^^^^^^^^^

The command ``train`` is used to train a model on a data set::

   $ scvae train $DATA_SET

By default, a VAE model with a Poisson likelihood function, two-dimensional latent variable, and one hidden layer of 100 units will be trained on the specified data set for 200 epochs with a learning rate of :math:`10^{-4}`.

The default model can be changed by using the following options:

* ``-m``: The model type, either ``VAE`` or ``GMVAE``.
* ``-r``: Likelihood function (or reconstruction distribution):
   * ``poisson``,
   * ``negative_binomial``,
   * ``zero_inflated_poisson``,
   * ``zero_inflated_negative_binomial``,
   * ``constrained_poisson``,
   * ``bernoulli``,
   * ``gaussian``, and
   * ``log_normal``.
* ``-k``: The threshold for modelling low counts using discrete probabilities and high counts using a shifted likelihood function (denoted by :math:`k_\text{max}` in :ref:`Grønbech et al., 2020 <groenbech2020>`). This turns the likelihood function into a corresponding piecewise categorical likehood function.
* ``-q``: The latent prior distribution. For the VAE model, this can only be a normal isotropic Gaussian distribution (``gaussian``) or one with unit variance (``unit_variance_gaussian``). For the GMVAE model, this can either be a Gaussian-mixture model with a diagonal covariance matrix (``gaussian_mixture``) or a full covariance matrix (``full_covariance_gaussian_mixture``). Note that a full covariance matrix should only be used for simpler GMVAE models.
* ``--prior-probabilites-method``: Method for how to set the mixture coefficients for the latent prior distribution of the GMVAE model. They can be fixed to either uniform values (``uniform``) or inferred values from labelled data (``infer``), or they can be learnt by the model (``learn``).
* ``-l``: The dimension of the latent variable.
* ``-H``: The number of hidden units in each layer separated by spaces. For example, ``-H 200 100`` will make both the inference (encoder) and the generative (decoder) networks two-layered with the first inference layer and the last generative layer consisting of 200 hidden units and the last inference layer and the first generative layer consisting of 100 hidden units.
* ``-K``: The number of components for the GMVAE (if possible, this is inferred from labelled data, but it can be overridden using this option).
* ``-w``: The number of epochs during the start of training with a linear weight on the KL divergence (the warm-up optimisation scheme described in :ref:`Grønbech et al., 2020 <groenbech2020>`). This weight is gradually increased linearly from 0 to 1 for this number of epochs.
* ``--batch-correction``: Perform batch correction if batch indices are available in data set (currently only possible with Loom data sets).

The training procedure can be changed using the following options (only applicable to the ``train`` command):

* ``-e``: The number of epochs to train the model.
* ``--learning-rate``: The learning rate of the model. The model is optimised using the Adam optimisation algorithm (:ref:`Kingma and Ba, 2015 <kingma2015>`).

A GMVAE model with a negative binomial likelihood function, a 100-dimensional latent variable, two hidden layers of each 100 units, and 200 epochs using the warm-up scheme is trained for 500 epochs on the ``10x-PBMC-PP`` data set like this::

   $ scvae train 10x-PBMC-PP -m GMVAE -l 100 -H 100 100 -w 200 -e 500

Trained models are saved to the subdirectory ``models/`` by default. This can be changed using the option ``--models-directory`` (or ``-M``).

Evaluating a model
^^^^^^^^^^^^^^^^^^

The command ``evaluate`` is used to evaluate a model on a data set::

   $ scvae evalaute $DATA_SET

Note the model has to have been trained already on the same data set.

The model is specified in the same way as when training the model, and the model will be evaluated at the last epoch to which it was trained. If withheld data were used, the model will also be evaluated at the early-stopping epoch and epoch with the most optimal marginal log-likelihood lower bound (if available). A number of analyses are conducted of the models and results, and these saved in the subdirectory ``analyses/``. This can be changed using the option ``--analyses-directory`` (or ``-A``). If you want the tool to perform all available analyses, you can use this option and argument: ``--included-analyses all``.

Cells can be clustered and cell types can be predicted using the option ``--prediction-method``. Currently only *k*-means clustering (``kmeans``) is supported. The GMVAE clusters cells and predict cell types using its built-in density-based clustering by default.

To visualise the data sets or latent spaces thereof, these are decomposed using a decomposition method. By default, this method is PCA. This can be changed using the option ``--decomposition-methods``, and as the name implies, multiple methods can be specified: PCA (``pca``), ICA (``ica``), SVD (``svd``), and *t*-SNE (``tsne``).

Decompositions of the data sets and of the latent values as well as predictions and the latent values themselves are also saved to compressed TSV files in the same directory.

The GMVAE model trained in the previous section is evaluated with PCA and *t*-SNE decomposition methods like this::

   $ scvae evaluate 10x-PBMC-PP -m GMVAE -l 100 -H 100 100 -w 200 --decomposition-methods pca tsne

Examples
^^^^^^^^

To reproduce the main results from :ref:`Grønbech et al. (2020) <groenbech2020>`, you can run the following commands:

* Combined PBMC data set from 10x Genomics::

      $ scvae train 10x-PBMC-PP --split-data-set -m GMVAE -r negative_binomial -l 100 -H 100 100 -w 200 -e 500
      $ scvae evaluate 10x-PBMC-PP --split-data-set -m GMVAE -r negative_binomial -l 100 -H 100 100 -w 200 --decomposition-methods pca tsne

* TCGA data set::

      $ scvae train TCGA-RSEM --map-features --feature-selection keep_highest_variances 5000 --split-data-set -m GMVAE -r negative_binomial -l 50 -H 1000 1000 -e 500
      $ scvae evaluate TCGA-RSEM --map-features --feature-selection keep_highest_variances 5000 --split-data-set -m GMVAE -r negative_binomial -l 50 -H 1000 1000 --decomposition-methods pca tsne

Tutorial
--------

Say you have a data set consisting of:

* single-cell transcript counts a file called ``"transcript_counts.tsv.gz"`` with genes as rows and cells as columns, and
* associated cell types in file called ``"cell_types.tsv"``.

To load these, you make a JSON file with the following contents:

.. code-block:: json

   {
      "values": "transcript_counts.tsv.gz",
      "labels": "cell_types.tsv",
      "format": "matrix_fbe"
   }

(See :ref:`Custom data sets` for more loading options.)

You then save the JSON file in the same directory as the TSV files with a memorable name like ``"data_set.json"``.

To load and split this data set with scVAE and train a GMVAE model with a Poisson distribution on the training set, you run the following command in the same directory::

   $ scvae train data_set.json --split-data-set -m GMVAE -r poisson

(See :ref:`Training a model` for more model options.)

You evaluate this model on the test set using the following command::

   $ scvae evaluate data_set.json --split-data-set -m GMVAE -r poisson

The resulting plots are saved in a subfolder called ``"analyses"``. If you want *t*-SNE plots, you use this command instead::

   $ scvae evaluate data_set.json --split-data-set -m GMVAE -r poisson --decomposition-methods tsne

----

.. [#sequence] With the first part becoming the training set, the second part the validation set, and the remaining part the test set.
