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

import numpy
import scipy
import seaborn
from matplotlib import pyplot

from scvae.analyses.figures import saving, style
from scvae.analyses.figures.utilities import _covariance_matrix_as_ellipse
from scvae.utilities import normalise_string, capitalise_string


def plot_values(values, colour_coding=None, colouring_data_set=None,
                centroids=None, sampled_values=None, class_name=None,
                feature_index=None, figure_labels=None, example_tag=None,
                name="scatter"):

    figure_name = name

    if figure_labels:
        title = figure_labels.get("title")
        x_label = figure_labels.get("x label")
        y_label = figure_labels.get("y label")
    else:
        title = "none"
        x_label = "$x$"
        y_label = "$y$"

    if not title:
        title = "none"

    figure_name += "-" + normalise_string(title)

    if colour_coding:
        colour_coding = normalise_string(colour_coding)
        figure_name += "-" + colour_coding
        if "predicted" in colour_coding:
            if colouring_data_set.prediction_specifications:
                figure_name += "-" + (
                    colouring_data_set.prediction_specifications.name)
            else:
                figure_name += "unknown_prediction_method"
        if colouring_data_set is None:
            raise ValueError("Colouring data set not given.")

    if sampled_values is not None:
        figure_name += "-samples"

    values = values.copy()[:, :2]
    if scipy.sparse.issparse(values):
        values = values.A

    # Randomise examples in values to remove any prior order
    n_examples, __ = values.shape
    random_state = numpy.random.RandomState(117)
    shuffled_indices = random_state.permutation(n_examples)
    values = values[shuffled_indices]

    # Adjust marker size based on number of examples
    style._adjust_marker_size_for_scatter_plots(n_examples)

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    colour_map = seaborn.dark_palette(style.STANDARD_PALETTE[0], as_cmap=True)

    alpha = 1
    if sampled_values is not None:
        alpha = 0.5

    if colour_coding and (
            "labels" in colour_coding
            or "ids" in colour_coding
            or "class" in colour_coding
            or colour_coding == "batches"):

        if colour_coding == "predicted_cluster_ids":
            labels = colouring_data_set.predicted_cluster_ids
            class_names = numpy.unique(labels).tolist()
            number_of_classes = len(class_names)
            class_palette = None
            label_sorter = None
        elif colour_coding == "predicted_labels":
            labels = colouring_data_set.predicted_labels
            class_names = colouring_data_set.predicted_class_names
            number_of_classes = colouring_data_set.number_of_predicted_classes
            class_palette = colouring_data_set.predicted_class_palette
            label_sorter = colouring_data_set.predicted_label_sorter
        elif colour_coding == "predicted_superset_labels":
            labels = colouring_data_set.predicted_superset_labels
            class_names = colouring_data_set.predicted_superset_class_names
            number_of_classes = (
                colouring_data_set.number_of_predicted_superset_classes)
            class_palette = colouring_data_set.predicted_superset_class_palette
            label_sorter = colouring_data_set.predicted_superset_label_sorter
        elif "superset" in colour_coding:
            labels = colouring_data_set.superset_labels
            class_names = colouring_data_set.superset_class_names
            number_of_classes = colouring_data_set.number_of_superset_classes
            class_palette = colouring_data_set.superset_class_palette
            label_sorter = colouring_data_set.superset_label_sorter
        elif colour_coding == "batches":
            labels = colouring_data_set.batch_indices.flatten()
            class_names = colouring_data_set.batch_names
            number_of_classes = colouring_data_set.number_of_batches
            class_palette = None
            label_sorter = None
        else:
            labels = colouring_data_set.labels
            class_names = colouring_data_set.class_names
            number_of_classes = colouring_data_set.number_of_classes
            class_palette = colouring_data_set.class_palette
            label_sorter = colouring_data_set.label_sorter

        if not class_palette:
            index_palette = style.lighter_palette(number_of_classes)
            class_palette = {
                class_name: index_palette[i] for i, class_name in
                enumerate(sorted(class_names, key=label_sorter))
            }

        # Examples are shuffled, so should their labels be
        labels = labels[shuffled_indices]

        if ("labels" in colour_coding or "ids" in colour_coding
                or colour_coding == "batches"):
            colours = []
            classes = set()

            for i, label in enumerate(labels):
                colour = class_palette[label]
                colours.append(colour)

                # Plot one example for each class to add labels
                if label not in classes:
                    classes.add(label)
                    axis.scatter(
                        values[i, 0],
                        values[i, 1],
                        color=colour,
                        label=label,
                        alpha=alpha
                    )

            axis.scatter(values[:, 0], values[:, 1], c=colours, alpha=alpha)

            class_handles, class_labels = axis.get_legend_handles_labels()

            if class_labels:
                class_labels, class_handles = zip(*sorted(
                    zip(class_labels, class_handles),
                    key=(
                        lambda t: label_sorter(t[0])) if label_sorter else None
                ))
                class_label_maximum_width = max(map(len, class_labels))
                if class_label_maximum_width <= 5 and number_of_classes <= 20:
                    axis.legend(
                        class_handles, class_labels,
                        loc="best"
                    )
                else:
                    if number_of_classes <= 20:
                        class_label_columns = 2
                    else:
                        class_label_columns = 3
                    axis.legend(
                        class_handles,
                        class_labels,
                        bbox_to_anchor=(-0.1, 1.05, 1.1, 0.95),
                        loc="lower left",
                        ncol=class_label_columns,
                        mode="expand",
                        borderaxespad=0.,
                    )

        elif "class" in colour_coding:
            colours = []
            figure_name += "-" + normalise_string(str(class_name))
            ordered_indices_set = {
                str(class_name): [],
                "Remaining": []
            }

            for i, label in enumerate(labels):
                if label == class_name:
                    colour = class_palette[label]
                    ordered_indices_set[str(class_name)].append(i)
                else:
                    colour = style.NEUTRAL_COLOUR
                    ordered_indices_set["Remaining"].append(i)
                colours.append(colour)

            colours = numpy.array(colours)

            z_order_index = 1
            for label, ordered_indices in sorted(ordered_indices_set.items()):
                if label == "Remaining":
                    z_order = 0
                else:
                    z_order = z_order_index
                    z_order_index += 1
                ordered_values = values[ordered_indices]
                ordered_colours = colours[ordered_indices]
                axis.scatter(
                    ordered_values[:, 0],
                    ordered_values[:, 1],
                    c=ordered_colours,
                    label=label,
                    alpha=alpha,
                    zorder=z_order
                )

                handles, labels = axis.get_legend_handles_labels()
                labels, handles = zip(*sorted(
                    zip(labels, handles),
                    key=lambda t: label_sorter(t[0]) if label_sorter else None
                ))
                axis.legend(
                    handles,
                    labels,
                    bbox_to_anchor=(-0.1, 1.05, 1.1, 0.95),
                    loc="lower left",
                    ncol=2,
                    mode="expand",
                    borderaxespad=0.
                )

    elif colour_coding == "count_sum":

        n = colouring_data_set.count_sum[shuffled_indices].flatten()
        scatter_plot = axis.scatter(
            values[:, 0],
            values[:, 1],
            c=n,
            cmap=colour_map,
            alpha=alpha
        )
        colour_bar = figure.colorbar(scatter_plot)
        colour_bar.outline.set_linewidth(0)
        colour_bar.set_label("Total number of {}s per {}".format(
            colouring_data_set.terms["item"],
            colouring_data_set.terms["example"]
        ))

    elif colour_coding == "feature":
        if feature_index is None:
            raise ValueError("Feature number not given.")
        if feature_index > colouring_data_set.number_of_features:
            raise ValueError("Feature number higher than number of features.")

        feature_name = colouring_data_set.feature_names[feature_index]
        figure_name += "-{}".format(normalise_string(feature_name))

        f = colouring_data_set.values[shuffled_indices, feature_index]
        if scipy.sparse.issparse(f):
            f = f.A
        f = f.squeeze()

        scatter_plot = axis.scatter(
            values[:, 0],
            values[:, 1],
            c=f,
            cmap=colour_map,
            alpha=alpha
        )
        colour_bar = figure.colorbar(scatter_plot)
        colour_bar.outline.set_linewidth(0)
        colour_bar.set_label(feature_name)

    elif colour_coding is None:
        axis.scatter(
            values[:, 0], values[:, 1], c="k",
            alpha=alpha, edgecolors="none")

    else:
        raise ValueError(
            "Colour coding `{}` not found.".format(colour_coding))

    if centroids:
        prior_centroids = centroids["prior"]

        if prior_centroids:
            n_centroids = prior_centroids["probabilities"].shape[0]
        else:
            n_centroids = 0

        if n_centroids > 1:
            centroids_palette = style.darker_palette(n_centroids)
            classes = numpy.arange(n_centroids)

            means = prior_centroids["means"]
            covariance_matrices = prior_centroids["covariance_matrices"]

            for k in range(n_centroids):
                axis.scatter(
                    means[k, 0],
                    means[k, 1],
                    s=60,
                    marker="x",
                    color="black",
                    linewidth=3
                )
                axis.scatter(
                    means[k, 0],
                    means[k, 1],
                    marker="x",
                    facecolor=centroids_palette[k],
                    edgecolors="black"
                )
                ellipse_fill, ellipse_edge = _covariance_matrix_as_ellipse(
                    covariance_matrices[k],
                    means[k],
                    colour=centroids_palette[k]
                )
                axis.add_patch(ellipse_edge)
                axis.add_patch(ellipse_fill)

    if sampled_values is not None:

        sampled_values = sampled_values.copy()[:, :2]
        if scipy.sparse.issparse(sampled_values):
            sampled_values = sampled_values.A

        sample_colour_map = seaborn.blend_palette(
            ("white", "purple"), as_cmap=True)

        x_limits = axis.get_xlim()
        y_limits = axis.get_ylim()

        axis.hexbin(
            sampled_values[:, 0], sampled_values[:, 1],
            gridsize=75,
            cmap=sample_colour_map,
            linewidths=0., edgecolors="none",
            zorder=-100
        )

        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)

    # Reset marker size
    style.reset_plot_look()

    return figure, figure_name


def plot_variable_correlations(values, variable_names=None,
                               colouring_data_set=None,
                               name="variable_correlations"):

    figure_name = saving.build_figure_name(name)
    n_examples, n_features = values.shape

    random_state = numpy.random.RandomState(117)
    shuffled_indices = random_state.permutation(n_examples)
    values = values[shuffled_indices]

    if colouring_data_set:
        labels = colouring_data_set.labels
        class_names = colouring_data_set.class_names
        number_of_classes = colouring_data_set.number_of_classes
        class_palette = colouring_data_set.class_palette
        label_sorter = colouring_data_set.label_sorter

        if not class_palette:
            index_palette = style.lighter_palette(number_of_classes)
            class_palette = {
                class_name: index_palette[i] for i, class_name in
                enumerate(sorted(class_names, key=label_sorter))
            }

        labels = labels[shuffled_indices]

        colours = []

        for label in labels:
            colour = class_palette[label]
            colours.append(colour)

    else:
        colours = style.NEUTRAL_COLOUR

    figure, axes = pyplot.subplots(
        nrows=n_features,
        ncols=n_features,
        figsize=[1.5 * n_features] * 2
    )

    for i in range(n_features):
        for j in range(n_features):
            axes[i, j].scatter(values[:, i], values[:, j], c=colours, s=1)

            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            if i == n_features - 1:
                axes[i, j].set_xlabel(variable_names[j])

        axes[i, 0].set_ylabel(variable_names[i])

    return figure, figure_name


def plot_variable_label_correlations(variable_vector, variable_name,
                                     colouring_data_set,
                                     name="variable_label_correlations"):

    figure_name = saving.build_figure_name(name)
    n_examples = variable_vector.shape[0]

    class_names_to_class_ids = numpy.vectorize(
        lambda class_name:
        colouring_data_set.class_name_to_class_id[class_name]
    )
    class_ids_to_class_names = numpy.vectorize(
        lambda class_name:
        colouring_data_set.class_id_to_class_name[class_name]
    )

    labels = colouring_data_set.labels
    class_names = colouring_data_set.class_names
    number_of_classes = colouring_data_set.number_of_classes
    class_palette = colouring_data_set.class_palette
    label_sorter = colouring_data_set.label_sorter

    if not class_palette:
        index_palette = style.lighter_palette(number_of_classes)
        class_palette = {
            class_name: index_palette[i] for i, class_name in
            enumerate(sorted(class_names, key=label_sorter))
        }

    random_state = numpy.random.RandomState(117)
    shuffled_indices = random_state.permutation(n_examples)
    variable_vector = variable_vector[shuffled_indices]

    labels = labels[shuffled_indices]
    label_ids = numpy.expand_dims(class_names_to_class_ids(labels), axis=-1)
    colours = [class_palette[label] for label in labels]

    unique_class_ids = numpy.unique(label_ids)
    unique_class_names = class_ids_to_class_names(unique_class_ids)

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.scatter(variable_vector, label_ids, c=colours, s=1)

    axis.set_yticks(unique_class_ids)
    axis.set_yticklabels(unique_class_names)

    axis.set_xlabel(variable_name)
    axis.set_ylabel(capitalise_string(colouring_data_set.terms["class"]))

    return figure, figure_name
