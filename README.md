# Deep learning for single-cell transcript counts #

Model implementation for master thesis by [Christopher Heje Grønbech][Chris] and [Maximillian Fornitz Vording][Max] at [DTU Compute][] with [Ole Winther][Ole] as supervisor and [Pers Lab][] as collaborators.

[DTU Compute]: http://compute.dtu.dk
[Chris]: https://github.com/chgroenbech
[Max]: https://github.com/maximillian91
[Ole]: http://cogsys.imm.dtu.dk/staff/winther/
[Pers Lab]: https://github.com/perslab

## Description ##

Biologist are today facing the challenge of making relevant biological inference from the massive amount of single-cell transcriptomic data generated. Typical approaches to model the data involve several empirically justified steps of dimensionality reduction, transformation, and normalisation, as seen in modern single analysis literature, e.g. [Macosko et al. (2015)][Macosko].

To model and understand the regulatory interdependencies between the vast amount of genes and define the type and state of it, we will need deeper models that combine all features into a lower-dimensional latent space representing the epigenetic landscape of the cells, that C.H. Waddington described in the "The Strategy of the Genes" (1957) over 60 years ago. The hypothesis is that the latent representation of cells of the same type will cluster and follow continuous trajectories over the state transitions and the original gene expression counts can be reconstructed by a count-based distribution conditioning on the latent representation.

In this project, we will propose variational deep learning models applied directly on the count-based gene expression levels from single cell transcriptomic experiments. Specifically, we want to use variational auto-encoders, as [Kingma and Welling (2013)][Kingma], modified to model both synthetic and real data. Here the optimal choice of a conditional reconstruction distribution for transcript counts will be investigated like [Salimans et al. (2017)][Salimans]. Additionally, we want to use this model to classify cell-types and cell-states in the latent representation.

This multidisciplinary project will be carried out in collaboration with Pers Lab at the Novo Nordisk Center for Basic Metabolic Research at University of Copenhagen and the Epidemiology Department at Statens Serum Institut.

[Macosko]: http://www.cell.com/abstract/S0092-8674(15)00549-8
[Kingma]: https://arxiv.org/abs/1312.6114
[Salimans]: https://arxiv.org/abs/1701.05517
