# ======================================================================== #
#
# Copyright (c) 2017 - 2020 scVAE authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ======================================================================== #

import re

import numpy
import pandas
import seaborn
from matplotlib import pyplot

from scvae.analyses.figures import saving, style
from scvae.analyses.figures.utilities import _axis_label_for_symbol
from scvae.utilities import normalise_string, capitalise_string


def plot_learning_curves(curves, model_type, epoch_offset=0, name=None):

    figure_name = "learning_curves"
    figure_name = saving.build_figure_name("learning_curves", name)

    x_label = "Epoch"
    y_label = "Nat"

    if model_type == "AE":
        figure = pyplot.figure()
        axis_1 = figure.add_subplot(1, 1, 1)
    elif model_type == "VAE":
        figure, (axis_1, axis_2) = pyplot.subplots(
            nrows=2, sharex=True, figsize=(6.4, 9.6))
        figure.subplots_adjust(hspace=0.1)
    elif model_type == "GMVAE":
        figure, (axis_1, axis_2, axis_3) = pyplot.subplots(
            nrows=3, sharex=True, figsize=(6.4, 14.4))
        figure.subplots_adjust(hspace=0.1)

    for curve_set_name, curve_set in sorted(curves.items()):

        if curve_set_name == "training":
            line_style = "solid"
            colour_index_offset = 0
        elif curve_set_name == "validation":
            line_style = "dashed"
            colour_index_offset = 1

        def curve_colour(i):
            return style.STANDARD_PALETTE[
                len(curves) * i + colour_index_offset]

        for curve_name, curve in sorted(curve_set.items()):
            if curve is None:
                continue
            elif curve_name == "lower_bound":
                curve_name = "$\\mathcal{L}$"
                colour = curve_colour(0)
                axis = axis_1
            elif curve_name == "reconstruction_error":
                curve_name = "$\\log p(x|z)$"
                colour = curve_colour(1)
                axis = axis_1
            elif "kl_divergence" in curve_name:
                if curve_name == "kl_divergence":
                    index = ""
                    colour = curve_colour(0)
                    axis = axis_2
                else:
                    latent_variable = curve_name.replace("kl_divergence_", "")
                    latent_variable = re.sub(
                        pattern=r"(\w)(\d)",
                        repl=r"\1_\2",
                        string=latent_variable
                    )
                    index = "$_{" + latent_variable + "}$"
                    if latent_variable in ["z", "z_2"]:
                        colour = curve_colour(0)
                        axis = axis_2
                    elif latent_variable == "z_1":
                        colour = curve_colour(1)
                        axis = axis_2
                    elif latent_variable == "y":
                        colour = curve_colour(0)
                        axis = axis_3
                curve_name = "KL" + index + "$(q||p)$"
            elif curve_name == "log_likelihood":
                curve_name = "$L$"
                axis = axis_1
            epochs = numpy.arange(len(curve)) + epoch_offset + 1
            label = "{} ({} set)".format(curve_name, curve_set_name)
            axis.plot(
                epochs,
                curve,
                color=colour,
                linestyle=line_style,
                label=label
            )

    handles, labels = axis_1.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    axis_1.legend(handles, labels, loc="best")

    if model_type == "AE":
        axis_1.set_xlabel(x_label)
        axis_1.set_ylabel(y_label)
    elif model_type == "VAE":
        handles, labels = axis_2.get_legend_handles_labels()
        labels, handles = zip(*sorted(
            zip(labels, handles), key=lambda t: t[0]))
        axis_2.legend(handles, labels, loc="best")
        axis_1.set_ylabel("")
        axis_2.set_ylabel("")

        if model_type == "GMVAE":
            axis_3.legend(loc="best")
            handles, labels = axis_3.get_legend_handles_labels()
            labels, handles = zip(*sorted(
                zip(labels, handles), key=lambda t: t[0]))
            axis_3.legend(handles, labels, loc="best")
            axis_3.set_xlabel(x_label)
            axis_3.set_ylabel("")
        else:
            axis_2.set_xlabel(x_label)
        figure.text(-0.01, 0.5, y_label, va="center", rotation="vertical")

    seaborn.despine()

    return figure, figure_name


def plot_separate_learning_curves(curves, loss, name=None):

    if not isinstance(loss, list):
        losses = [loss]
    else:
        losses = loss

    if not isinstance(name, list):
        names = [name]
    else:
        names = name

    names.extend(losses)
    figure_name = saving.build_figure_name("learning_curves", names)

    x_label = "Epoch"
    y_label = "Nat"

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    for curve_set_name, curve_set in sorted(curves.items()):

        if curve_set_name == "training":
            line_style = "solid"
            colour_index_offset = 0
        elif curve_set_name == "validation":
            line_style = "dashed"
            colour_index_offset = 1

        def curve_colour(i):
            return style.STANDARD_PALETTE[
                len(curves) * i + colour_index_offset]

        for curve_name, curve in sorted(curve_set.items()):
            if curve is None or curve_name not in losses:
                continue
            elif curve_name == "lower_bound":
                curve_name = "$\\mathcal{L}$"
                colour = curve_colour(0)
            elif curve_name == "reconstruction_error":
                curve_name = "$\\log p(x|z)$"
                colour = curve_colour(1)
            elif "kl_divergence" in curve_name:
                if curve_name == "kl_divergence":
                    index = ""
                    colour = curve_colour(0)
                else:
                    latent_variable = curve_name.replace("kl_divergence_", "")
                    latent_variable = re.sub(
                        pattern=r"(\w)(\d)",
                        repl=r"\1_\2",
                        string=latent_variable
                    )
                    index = "$_{" + latent_variable + "}$"
                    if latent_variable in ["z", "z_2"]:
                        colour = curve_colour(0)
                    elif latent_variable == "z_1":
                        colour = curve_colour(1)
                    elif latent_variable == "y":
                        colour = curve_colour(0)
                curve_name = "KL" + index + "$(q||p)$"
            elif curve_name == "log_likelihood":
                curve_name = "$L$"
            epochs = numpy.arange(len(curve)) + 1
            label = curve_name + " ({} set)".format(curve_set_name)
            axis.plot(
                epochs,
                curve,
                color=colour,
                linestyle=line_style,
                label=label
            )

    handles, labels = axis.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    axis.legend(handles, labels, loc="best")

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    return figure, figure_name


def plot_accuracy_evolution(accuracies, name=None):

    figure_name = saving.build_figure_name("accuracies", name)
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    for accuracies_kind, accuracies in sorted(accuracies.items()):
        if accuracies is None:
            continue
        elif accuracies_kind == "training":
            line_style = "solid"
            colour = style.STANDARD_PALETTE[0]
        elif accuracies_kind == "validation":
            line_style = "dashed"
            colour = style.STANDARD_PALETTE[1]

        label = "{} set".format(capitalise_string(accuracies_kind))
        epochs = numpy.arange(len(accuracies)) + 1
        axis.plot(
            epochs,
            100 * accuracies,
            color=colour,
            linestyle=line_style,
            label=label
        )

    handles, labels = axis.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    axis.legend(handles, labels, loc="best")

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracies")

    return figure, figure_name


def plot_kl_divergence_evolution(kl_neurons, scale="log", name=None):

    figure_name = saving.build_figure_name("kl_divergence_evolution", name)
    n_epochs, __ = kl_neurons.shape

    kl_neurons = numpy.sort(kl_neurons, axis=1)

    if scale == "log":
        kl_neurons = numpy.log(kl_neurons)
        scale_label = "$\\log$ "
    else:
        scale_label = ""

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)

    cbar_dict = {"label": scale_label + "KL$(p_i|q_i)$"}

    number_of_epoch_labels = 10
    if n_epochs > 2 * number_of_epoch_labels:
        epoch_label_frequency = int(numpy.floor(
            n_epochs / number_of_epoch_labels))
    else:
        epoch_label_frequency = True

    epochs = numpy.arange(n_epochs) + 1

    seaborn.heatmap(
        pandas.DataFrame(kl_neurons.T, columns=epochs),
        xticklabels=epoch_label_frequency,
        yticklabels=False,
        cbar=True, cbar_kws=cbar_dict, cmap=style.STANDARD_COLOUR_MAP,
        ax=axis
    )

    axis.set_xlabel("Epochs")
    axis.set_ylabel("$i$")

    seaborn.despine(ax=axis)

    return figure, figure_name


def plot_centroid_probabilities_evolution(probabilities, distribution,
                                          linestyle="solid", name=None):

    distribution = normalise_string(distribution)

    y_label = _axis_label_for_symbol(
        symbol="\\pi",
        distribution=distribution,
        suffix="^k"
    )

    figure_name = "centroids_evolution-{}-probabilities".format(distribution)
    figure_name = saving.build_figure_name(figure_name, name)

    n_epochs, n_centroids = probabilities.shape

    centroids_palette = style.darker_palette(n_centroids)
    epochs = numpy.arange(n_epochs) + 1

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    for k in range(n_centroids):
        axis.plot(
            epochs,
            probabilities[:, k],
            color=centroids_palette[k],
            linestyle=linestyle,
            label="$k = {}$".format(k)
        )

    axis.set_xlabel("Epochs")
    axis.set_ylabel(y_label)

    axis.legend(loc="best")

    return figure, figure_name


def plot_centroid_means_evolution(means, distribution, decomposed=False,
                                  name=None):

    symbol = "\\mu"
    if decomposed:
        decomposition_method = "PCA"
    else:
        decomposition_method = ""
    distribution = normalise_string(distribution)
    suffix = "(y = k)"

    x_label = _axis_label_for_symbol(
        symbol=symbol,
        coordinate=1,
        decomposition_method=decomposition_method,
        distribution=distribution,
        suffix=suffix
    )
    y_label = _axis_label_for_symbol(
        symbol=symbol,
        coordinate=2,
        decomposition_method=decomposition_method,
        distribution=distribution,
        suffix=suffix
    )

    figure_name = "centroids_evolution-{}-means".format(distribution)
    figure_name = saving.build_figure_name(figure_name, name)

    n_epochs, n_centroids, latent_size = means.shape

    if latent_size > 2:
        raise ValueError("Dimensions of means should be 2.")

    centroids_palette = style.darker_palette(n_centroids)
    epochs = numpy.arange(n_epochs) + 1

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    colour_bar_scatter_plot = axis.scatter(
        means[:, 0, 0], means[:, 0, 1], c=epochs,
        cmap=seaborn.dark_palette(style.NEUTRAL_COLOUR, as_cmap=True),
        zorder=0
    )

    for k in range(n_centroids):
        colour = centroids_palette[k]
        colour_map = seaborn.dark_palette(colour, as_cmap=True)
        axis.plot(
            means[:, k, 0],
            means[:, k, 1],
            color=colour,
            label="$k = {}$".format(k),
            zorder=k + 1
        )
        axis.scatter(
            means[:, k, 0],
            means[:, k, 1],
            c=epochs,
            cmap=colour_map,
            zorder=n_centroids + k + 1
        )

    axis.legend(loc="best")

    colour_bar = figure.colorbar(colour_bar_scatter_plot)
    colour_bar.outline.set_linewidth(0)
    colour_bar.set_label("Epochs")

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    return figure, figure_name


def plot_centroid_covariance_matrices_evolution(covariance_matrices,
                                                distribution, name=None):

    distribution = normalise_string(distribution)
    figure_name = "centroids_evolution-{}-covariance_matrices".format(
        distribution)
    figure_name = saving.build_figure_name(figure_name, name)

    y_label = _axis_label_for_symbol(
        symbol="\\Sigma",
        distribution=distribution,
        prefix="|",
        suffix="(y = k)|"
    )

    n_epochs, n_centroids, __, __ = covariance_matrices.shape
    determinants = numpy.empty([n_epochs, n_centroids])

    for e in range(n_epochs):
        for k in range(n_centroids):
            determinants[e, k] = numpy.prod(numpy.diag(
                covariance_matrices[e, k]))

    if determinants.all() > 0:
        line_range_ratio = numpy.empty(n_centroids)
        for k in range(n_centroids):
            determinants_min = determinants[:, k].min()
            determinants_max = determinants[:, k].max()
            line_range_ratio[k] = determinants_max / determinants_min
        range_ratio = line_range_ratio.max() / line_range_ratio.min()
        if range_ratio > 1e2:
            y_scale = "log"
        else:
            y_scale = "linear"

    centroids_palette = style.darker_palette(n_centroids)
    epochs = numpy.arange(n_epochs) + 1

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    for k in range(n_centroids):
        axis.plot(
            epochs,
            determinants[:, k],
            color=centroids_palette[k],
            label="$k = {}$".format(k)
        )

    axis.set_xlabel("Epochs")
    axis.set_ylabel(y_label)

    axis.set_yscale(y_scale)

    axis.legend(loc="best")

    return figure, figure_name
