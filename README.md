# scVAE: Single-cell variational auto-encoders #

This software tool implements two variants of variational auto-encoders: one with a Gaussian prior and another with a Gaussian-mixture prior. In addition, several discrete probability functions and derivations are included to handle sparse count data like single-cell gene expression data. Easy access to recent single-cell and traditional gene expression data sets are also provided. Lastly, the tool can also produce relevant analytics of the data sets and the models.

The methods used by this tool is described and examined in the paper ["scVAE: Variational auto-encoders for single-cell gene expression data"][scVAE-paper] by [Christopher Heje Grønbech][Chris], [Maximillian Fornitz Vording][Max], [Pascal Nordgren Timshel][Pascal], [Casper Kaae Sønderby][Casper], [Tune Hannes Pers][Tune], and [Ole Winther][Ole].

The tool has been developed by Christopher and Maximillian at [Section for Cognitive Systems][CogSys] at [DTU Compute][] with help from Casper and [Lars Maaløe][Lars] supervised by Ole and in collaboration with [Pers Lab][].

[scVAE-paper]: https://www.biorxiv.org/content/early/2018/05/16/318295
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

## Setup ##

The tool is implemented in Python using [TensorFlow][]. [Pandas][], [PyTables][],
[Beautiful Soup][], and the [stemming][] module are used for importing and preprocessing data. Analyses of the models and the results are performed using the modules [NumPy][], [SciPy][], and [scikit-learn][], and figures are made using the modules [matplotlib][], [Seaborn][], and [Pillow][].

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

This tool is not available as a Python module. So you will first need to install all other modules which it is dependent upon by yourself. This can be done by running

	$ pip install numpy scipy scikit-learn tensorflow-gpu pandas tables beautifulsoup4 stemming matplotlib seaborn pillow

(If you do not have a GPU to use with TensorFlow, install the standard version by replacing `tensorflow-gpu` with `tensorflow`.)

After this, you can clone this tool to an appropriate folder:

	$ git clone https://github.com/chgroenbech/scVAE.git

## Running ##

The standard configuration of the model using the synthetic data set, can be run by just running the `main.py` script. Be aware that it might take some time to load and preprocess the data the first time for large data sets. Also note to load and analyse the largest data set, which is made available by 10x Genomics and consists of 1.3 million mouse brain cells, 47 GB of memory is required (32 GB for the original data set in sparse representation and 15 GB for the reconstructed test set).

Per default, data is downloaded to the subfolder `data/`, models are saved in the subfolder `log/`, and results are saved in the subfolder `results/`.

To see how to change the standard configuration or use another data set, run the following command:

	$ ./main.py -h

### Examples ###

To reproduce the main results from our paper, you can run the following commands:

Purified immune cells data set from 10x Genomics:

	$ ./main.py -i 10x_pbmc -m GMVAE -r negative_binomial -l 100 -H 100 100 -e 500 --decomposition-methods pca tsne

Mouse brain cells data set from 10x Genomics:

	$ ./main.py -i 10x_mbc -m GMVAE -K 10 -r zero_inflated_negative_binomial -l 25 -H 250 250 -e 500 --decomposition-methods pca tsne

TCGA data set:

	$ ./main.py -i tcga_rsem -m GMVAE --map-features -r negative_binomial -l 50 -H 500 500 -e 500 --decomposition-methods pca tsne

### Comparisons ###

The script `cross_analysis.py` is provided to compare different models. After running several different models with different network architectures and likelihood functions, this can be run to compare these models.

It requires the relative path to the results folder, so using the standard configuration, it is run using the following command:

	$ ./cross_analysis.py -R results/

Logs can be saved by adding the `-s` argument, and these are saved together with produced figures in the results folder specified. Data sets, models, and prediction methods can also be included or excluded using specific arguments. For documentation on these, use the command `./cross_analysis.py -h`.
