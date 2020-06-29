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

import matplotlib.colors
import numpy
import scipy
import seaborn
import sklearn
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scvae.analyses.figures import saving, style


def plot_heat_map(values, x_name, y_name, z_name=None, z_symbol=None,
                  z_min=None, z_max=None, symmetric=False, labels=None,
                  label_kind=None, center=None, name=None):

    figure_name = saving.build_figure_name("heat_map", name)
    n_examples, n_features = values.shape

    if symmetric and n_examples != n_features:
        raise ValueError(
            "Input cannot be symmetric, when it is not given as a 2-d square"
            "array or matrix."
        )

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)

    if not z_min:
        z_min = values.min()

    if not z_max:
        z_max = values.max()

    if z_symbol:
        z_name = "$" + z_symbol + "$"

    cbar_dict = {}

    if z_name:
        cbar_dict["label"] = z_name

    if not symmetric:
        aspect_ratio = n_examples / n_features
        square_cells = 1/5 < aspect_ratio and aspect_ratio < 5
    else:
        square_cells = True

    if labels is not None:
        x_indices = numpy.argsort(labels)
        y_name += " sorted"
        if label_kind:
            y_name += " by " + label_kind
    else:
        x_indices = numpy.arange(n_examples)

    if symmetric:
        y_indices = x_indices
        x_name = y_name
    else:
        y_indices = numpy.arange(n_features)

    seaborn.set(style="white")
    seaborn.heatmap(
        values[x_indices][:, y_indices],
        vmin=z_min, vmax=z_max, center=center,
        xticklabels=False, yticklabels=False,
        cbar=True, cbar_kws=cbar_dict, cmap=style.STANDARD_COLOUR_MAP,
        square=square_cells,
        ax=axis
    )
    style.reset_plot_look()

    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)

    return figure, figure_name


def plot_matrix(feature_matrix, plot_distances=False, center_value=None,
                example_label=None, feature_label=None, value_label=None,
                sorting_method=None, distance_metric="Euclidean",
                labels=None, label_kind=None, class_palette=None,
                feature_indices_for_plotting=None, hide_dendrogram=False,
                name_parts=None):

    figure_name = saving.build_figure_name(name_parts)
    n_examples, n_features = feature_matrix.shape

    if plot_distances:
        center_value = None
        feature_label = None
        value_label = "Pairwise {} distances in {} space".format(
            distance_metric,
            value_label
        )

    if not plot_distances and feature_indices_for_plotting is None:
        feature_indices_for_plotting = numpy.arange(n_features)

    if sorting_method == "labels" and labels is None:
        raise ValueError("No labels provided to sort after.")

    if labels is not None and not class_palette:
        raise ValueError("No class palette provided.")

    # Distances (if needed)
    distances = None
    if plot_distances or sorting_method == "hierarchical_clustering":
        distances = sklearn.metrics.pairwise_distances(
            feature_matrix,
            metric=distance_metric.lower()
        )

    # Figure initialisation
    figure = pyplot.figure()

    axis_heat_map = figure.add_subplot(1, 1, 1)
    left_most_axis = axis_heat_map

    divider = make_axes_locatable(axis_heat_map)
    axis_colour_map = divider.append_axes("right", size="5%", pad=0.1)

    axis_labels = None
    axis_dendrogram = None

    if labels is not None:
        axis_labels = divider.append_axes("left", size="5%", pad=0.01)
        left_most_axis = axis_labels

    if sorting_method == "hierarchical_clustering" and not hide_dendrogram:
        axis_dendrogram = divider.append_axes("left", size="20%", pad=0.01)
        left_most_axis = axis_dendrogram

    # Label colours
    if labels is not None:
        label_colours = [
            tuple(colour) if isinstance(colour, list) else colour
            for colour in [class_palette[l] for l in labels]
        ]
        unique_colours = [
            tuple(colour) if isinstance(colour, list) else colour
            for colour in class_palette.values()
        ]
        value_for_colour = {
            colour: i for i, colour in enumerate(unique_colours)
        }
        label_colour_matrix = numpy.array(
            [value_for_colour[colour] for colour in label_colours]
        ).reshape(n_examples, 1)
        label_colour_map = matplotlib.colors.ListedColormap(unique_colours)
    else:
        label_colour_matrix = None
        label_colour_map = None

    # Heat map aspect ratio
    if not plot_distances:
        square_cells = False
    else:
        square_cells = True

    seaborn.set(style="white")

    # Sorting and optional dendrogram
    if sorting_method == "labels":
        example_indices = numpy.argsort(labels)

        if not label_kind:
            label_kind = "labels"

        if example_label:
            example_label += " sorted by " + label_kind

    elif sorting_method == "hierarchical_clustering":
        linkage = scipy.cluster.hierarchy.linkage(
            scipy.spatial.distance.squareform(distances, checks=False),
            metric="average"
        )
        dendrogram = seaborn.matrix.dendrogram(
            distances,
            linkage=linkage,
            metric=None,
            method="ward",
            axis=0,
            label=False,
            rotate=True,
            ax=axis_dendrogram
        )
        example_indices = dendrogram.reordered_ind

        if example_label:
            example_label += " sorted by hierarchical clustering"

    elif sorting_method is None:
        example_indices = numpy.arange(n_examples)

    else:
        raise ValueError(
            "`sorting_method` should be either \"labels\""
            " or \"hierarchical clustering\""
        )

    # Heat map of values
    if plot_distances:
        plot_values = distances[example_indices][:, example_indices]
    else:
        plot_values = feature_matrix[example_indices][
            :, feature_indices_for_plotting]

    if scipy.sparse.issparse(plot_values):
        plot_values = plot_values.A

    colour_bar_dictionary = {}

    if value_label:
        colour_bar_dictionary["label"] = value_label

    seaborn.heatmap(
        plot_values, center=center_value,
        xticklabels=False, yticklabels=False,
        cbar=True, cbar_kws=colour_bar_dictionary, cbar_ax=axis_colour_map,
        square=square_cells, ax=axis_heat_map
    )

    # Colour labels
    if axis_labels:
        seaborn.heatmap(
            label_colour_matrix[example_indices],
            xticklabels=False, yticklabels=False,
            cbar=False,
            cmap=label_colour_map,
            ax=axis_labels
        )

    style.reset_plot_look()

    # Axis labels
    if example_label:
        left_most_axis.set_ylabel(example_label)

    if feature_label:
        axis_heat_map.set_xlabel(feature_label)

    return figure, figure_name


def plot_correlation_matrix(correlation_matrix, axis_label=None, name=None):

    figure_name = saving.build_figure_name(name)

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)

    colour_bar_dictionary = {"label": "Pearson correlation coefficient"}

    seaborn.set(style="white")
    seaborn.heatmap(
        correlation_matrix,
        vmin=-1, vmax=1, center=0,
        xticklabels=False, yticklabels=False,
        cbar=True, cbar_kws=colour_bar_dictionary,
        square=True, ax=axis
    )
    style.reset_plot_look()

    if axis_label:
        axis.set_xlabel(axis_label)
        axis.set_ylabel(axis_label)

    return figure, figure_name
