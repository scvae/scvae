# scVAE: Single-cell variational auto-encoders #

This software tool implements two variants of variational auto-encoders: one with a Gaussian prior and another with a Gaussian-mixture prior. In addition, several discrete probability functions and derivations are included to handle sparse count data like single-cell gene expression data. Easy access to recent single-cell and traditional gene expression data sets are also provided. Lastly, the tool can also produce relevant analytics of the data sets and the models.

The methods used by this tool is described and examined in the paper ["scVAE: Variational auto-encoders for single-cell gene expression data"][scVAE-paper] by [Christopher Heje Grønbech][Chris], [Maximillian Fornitz Vording][Max], [Pascal Nordgren Timshel][Pascal], [Casper Kaae Sønderby][Casper], [Tune Hannes Pers][Tune], and [Ole Winther][Ole].

The tool has been developed by Christopher and Maximillian at [Section for Cognitive Systems][CogSys] at [DTU Compute][] with help from Casper and [Lars Maaløe][Lars] supervised by Ole and in collaboration with [Pers Lab][]. It is being further developed by Christopher at [Centre for Genomic Medicine][GM] (only available in Danish) at [Rigshospitalet][RH].

[scVAE-paper]: https://www.biorxiv.org/content/10.1101/318295v2
[Chris]: https://github.com/chgroenbech
[Max]: https://github.com/maximillian91
[Pascal]: https://github.com/pascaltimshel
[Casper]: https://casperkaae.github.io
[Tune]: http://cbmr.ku.dk/research/section-for-metabolic-genetics/pers-group/
[Ole]: http://cogsys.imm.dtu.dk/staff/winther/

[Lars]: http://github.com/larsmaaloee

[CogSys]: https://github.com/DTUComputeCognitiveSystems
[DTU Compute]: http://compute.dtu.dk
[Pers Lab]: https://github.com/perslab
[GM]: https://www.rigshospitalet.dk/afdelinger-og-klinikker/diagnostisk/genomisk-medicin/Sider/default.aspx
[RH]: https://www.rigshospitalet.dk/english/Pages/default.aspx

## Setup ##

The tool is implemented in Python (versions 3.3--3.6) using [TensorFlow][]. [Pandas][], [PyTables][], [Beautiful Soup][], and the [stemming][] package are used for importing and preprocessing data. Analyses of the models and the results are performed using [NumPy][], [SciPy][], and [scikit-learn][], and figures are made using [matplotlib][], [Seaborn][], and [Pillow][].

[TensorFlow]: https://www.tensorflow.org
[Pandas]: http://pandas.pydata.org
[PyTables]: http://www.pytables.org
[Beautiful Soup]: https://www.crummy.com/software/BeautifulSoup/
[stemming]: https://bitbucket.org/mchaput/stemming
[NumPy]: http://www.numpy.org
[SciPy]: https://www.scipy.org
[scikit-learn]: http://scikit-learn.org
[matplotlib]: http://matplotlib.org
[Seaborn]: http://seaborn.pydata.org
[Pillow]: http://python-pillow.org

All included data sets are downloaded and processed automatically as needed.

## Installation ##

This tool is not available as a Python package yet. In the meantime you will first need to install its dependencies by yourself. This can be done by running

	$ pip install numpy scipy scikit-learn kneed tensorflow-gpu tensorflow-probability-gpu pandas tables beautifulsoup4 stemming matplotlib seaborn pillow

(If you do not have a GPU supported by TensorFlow, install the standard version by replacing `tensorflow-gpu` and `tensorflow-probability-gpu` with `tensorflow` and `tensorflow-probability`, respectively.)

After this, you can clone this tool to an appropriate folder:

	$ git clone https://github.com/chgroenbech/scVAE.git

## Running ##

The standard configuration of the model using the synthetic data set, can be run by just running the `main.py` script. Be aware that it might take some time to load and preprocess the data the first time for large data sets. Also note that to load and analyse the largest data set, which is made available by 10x Genomics and consists of 1.3 million mouse brain cells, 47 GB of memory is required (32 GB for the original data set in sparse representation and 15 GB for the reconstructed test set).

Per default, data is downloaded to the subfolder `data/`, models are saved in the subfolder `log/`, and results are saved in the subfolder `results/`.

To see how to change the standard configuration or use another data set, run the following command:

	$ ./main.py -h

### Examples ###

To reproduce the main results from our paper, you can run the following commands:

* Combined [PBMC data set][PBMC] (under "Single Cell 3′ Paper: Zheng et al. 2017") from 10x Genomics:

		$ ./main.py -i 10x-PBMC-PP -m GMVAE -r negative_binomial -l 100 -H 100 100 -w 200 -e 500 --decomposition-methods pca tsne

* [TCGA data set][TCGA]:

		$ ./main.py -i TCGA-RSEM --map-features --feature-selection keep_highest_variances 5000 -m GMVAE -r negative_binomial -l 50 -H 1000 1000 -e 500 --decomposition-methods pca tsne

* [MBC data set][MBC] from 10x Genomics:

		$ ./main.py -i 10x-MBC -m GMVAE -K 10 -r zero_inflated_negative_binomial -l 25 -H 250 250 -e 500 --decomposition-methods pca tsne

[PBMC]: https://support.10xgenomics.com/single-cell-gene-expression/datasets/
[TCGA]: https://xenabrowser.net/datapages/?dataset=tcga_gene_expected_count&host=https://toil.xenahubs.net
[MBC]: https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons

You can change the model setup using the following argument options:

* `-m`: The model type, either `VAE` or `GMVAE`.
* `-r`: Likelihood function (or reconstruction distribution). Choose from:
	* `poisson`,
	* `negative binomial`,
	* `zero_inflated_poisson`,
	* `zero_inflated_negative binomial`,
	* `constrained_poisson`,
	* `bernoulli`,
	* `gaussian`,
	* `log_normal`, and
	* `lomax`.
* `-k`: The threshold for the piecewise categorical distibution (denoted by *K* in the paper).
* `-l`: The dimension of the latent variable.
* `-H`: The number of hidden units in each layer separated by spaces. For example, `-H 200 100` will make both the inference (encoder) and the generative (decoder) networks two-layered with the first inference layer and the last generative layer consisting of 200 hidden units and the last inference layer and the first generative layer consisting of 100 hidden units.
* `-K`: The number of components for the GMVAE (if possible, this is inferred from labelled data, but it can be overridden using this option).
* `-e`: The number of epochs to train the model.
* `-w`: The number of epochs during training where the warm-up optimisation scheme is used.

By default, a number of analyses are conducted of the models and saved in the subdirectory `results/`. These can be skipped using the option `--skip-analyses`. Among the anlyses are visualisations of the latent representations and the reconstructions, and the method to visualise these can be customised using the option `--decomposition-methods`.

You can also model the MNIST data set. Three different versions are supported: [the original][MNIST-original], [the normalised][MNIST-normalised], and [the binarised][MNIST-binarised]. To run the GMVAE model for, e.g., the binarised version, issue the following command:

	$ ./main.py -i mnist_binarised -m GMVAE -r bernoulli -l 10 -H 200 200 -e 500 --decomposition-methods pca tsne

[MNIST-original]: http://yann.lecun.com/exdb/mnist/
[MNIST-normalised]: http://deeplearning.net/data/mnist/
[MNIST-binarised]: http://www.cs.toronto.edu/~larocheh/publications/icml-2008-discriminative-rbm.pdf

### Comparisons ###

The script `cross_analysis.py` is provided to compare different models. After running several different models with different network architectures and likelihood functions, this can be run to compare these models.

It requires the relative path to the results folder, so using the standard configuration, it is run using the following command:

	$ ./cross_analysis.py -R $RESULT_DIRECTORY

where  `$RESULT_DIRECTORY` should be replaced by the path to the results directory. By default, scVAE saves results in the subdirectory `results/`.

Logs can be saved by adding the `-s` argument, and these are saved together with produced figures in the results folder specified. Data sets, models, and prediction methods can also be included or excluded using specific arguments. For documentation on these, use the command `./cross_analysis.py -h`.
