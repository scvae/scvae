# scVAE: Variational auto-encoders for single-cell gene expression data #

This software tool implements two variants of variational auto-encoders: one with a Gaussian prior and another with a Gaussian-mixture prior. In addition, several discrete probability functions and derivations are included to handle sparse count data like single-cell gene expression data. Easy access to recent single-cell and traditional gene expression data sets are also provided. Lastly, the tool can also produce relevant analytics of the data sets and the models.

The methods used by this tool is described and examined in [Grønbech et al. (2018)][paper]. The tool has been created by [Christopher Heje Grønbech][Chris] and [Maximillian Fornitz Vording][Max] at [DTU Compute][] with help from [Casper Kaae Sønderby][Casper] supervised by [Ole Winther][Ole] and in collaboration with [Pers Lab][].

[paper]: TBA
[DTU Compute]: http://compute.dtu.dk
[Chris]: https://github.com/chgroenbech
[Max]: https://github.com/maximillian91
[Casper]: https://casperkaae.github.io
[Ole]: http://cogsys.imm.dtu.dk/staff/winther/
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
[Pillow]: https://python-pillow.org

All included data sets are downloaded and processed automatically as needed.

## Installation ##

This tool is not available as a real Python module yet. So you will need to install all other modules it is dependent upon first by yourself. This can be done by running

	$ pip install numpy scipy scikit-learn tensorflow-gpu pandas tables beautifulsoup4 stemming matplotlib seaborn pillow

After this, you can clone this tool to an appropriate folder:

	$ git clone https://github.com/chgroenbech/scVAE.git

## Running ##

The standard configuration of the model using the synthetic data set, can be run by just running the `main.py` script. Be aware that it might take some time to load and preprocess the data the first time for large data sets.

To change the standard configuration and to use another data set, run the following command:

	./main.py -h
