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

import copy
import os
from time import time

import numpy
import scipy

from scvae.analyses import figures
from scvae.analyses.figures import style
from scvae.analyses.figures.utilities import _axis_label_for_symbol
from scvae.analyses.decomposition import (
    decompose,
    DECOMPOSITION_METHOD_NAMES,
    DECOMPOSITION_METHOD_LABEL
)
from scvae.data.utilities import save_values
from scvae.defaults import defaults
from scvae.utilities import (
    format_duration,
    normalise_string, proper_string, capitalise_string
)

MAXIMUM_NUMBER_OF_EXAMPLES_FOR_HEAT_MAPS = 10000
MAXIMUM_NUMBER_OF_FEATURES_FOR_HEAT_MAPS = 10000
MAXIMUM_NUMBER_OF_EXAMPLES_FOR_DENDROGRAM = 1000

MAXIMUM_NUMBER_OF_FEATURES_FOR_TSNE = 100
MAXIMUM_NUMBER_OF_EXAMPLES_FOR_TSNE = 200000
MAXIMUM_NUMBER_OF_PCA_COMPONENTS_BEFORE_TSNE = 50


def analyse_distributions(data_set, colouring_data_set=None, cutoffs=None,
                          preprocessed=False, analysis_level="normal",
                          export_options=None, analyses_directory=None):

    if not colouring_data_set:
        colouring_data_set = data_set

    if analysis_level is None:
        analysis_level = defaults["analyses"]["analysis_level"]

    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]
    distribution_directory = os.path.join(analyses_directory, "histograms")

    data_set_title = data_set.kind + " set"
    data_set_name = data_set.kind
    if data_set.version != "original":
        data_set_title = data_set.version + " " + data_set_title
        data_set_name = None

    data_set_discreteness = data_set.discreteness and not preprocessed

    print("Plotting distributions for {}.".format(data_set_title))

    # Class distribution
    if (data_set.number_of_classes and data_set.number_of_classes < 100
            and colouring_data_set == data_set):
        distribution_time_start = time()
        figure, figure_name = figures.plot_class_histogram(
            labels=data_set.labels,
            class_names=data_set.class_names,
            class_palette=data_set.class_palette,
            normed=True,
            scale="linear",
            label_sorter=data_set.label_sorter,
            name=data_set_name
        )
        figures.save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=distribution_directory
        )
        distribution_duration = time() - distribution_time_start
        print("    Class distribution plotted and saved ({}).".format(
            format_duration(distribution_duration)))

    # Superset class distribution
    if data_set.label_superset and colouring_data_set == data_set:
        distribution_time_start = time()
        figure, figure_name = figures.plot_class_histogram(
            labels=data_set.superset_labels,
            class_names=data_set.superset_class_names,
            class_palette=data_set.superset_class_palette,
            normed=True,
            scale="linear",
            label_sorter=data_set.superset_label_sorter,
            name=[data_set_name, "superset"]
        )
        figures.save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=distribution_directory
        )
        distribution_duration = time() - distribution_time_start
        print("    Superset class distribution plotted and saved ({}).".format(
            format_duration(distribution_duration)))

    # Count distribution
    if scipy.sparse.issparse(data_set.values):
        series = data_set.values.data
        excess_zero_count = data_set.values.size - series.size
    else:
        series = data_set.values.reshape(-1)
        excess_zero_count = 0
    distribution_time_start = time()
    for x_scale in ["linear", "log"]:
        figure, figure_name = figures.plot_histogram(
            series=series,
            excess_zero_count=excess_zero_count,
            label=data_set.terms["value"].capitalize() + "s",
            discrete=data_set_discreteness,
            normed=True,
            x_scale=x_scale,
            y_scale="log",
            name=["counts", data_set_name]
        )
        figures.save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=distribution_directory
        )
    distribution_duration = time() - distribution_time_start
    print("    Count distribution plotted and saved ({}).".format(
        format_duration(distribution_duration)))

    # Count distributions with cut-off
    if (analysis_level == "extensive" and cutoffs
            and data_set.example_type == "counts"):
        distribution_time_start = time()
        for cutoff in cutoffs:
            figure, figure_name = figures.plot_cutoff_count_histogram(
                series=series,
                excess_zero_count=excess_zero_count,
                cutoff=cutoff,
                normed=True,
                scale="log",
                name=data_set_name
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=distribution_directory + "-counts"
            )
        distribution_duration = time() - distribution_time_start
        print(
            "    Count distributions with cut-offs plotted and saved ({})."
            .format(format_duration(distribution_duration))
        )

    # Count sum distribution
    distribution_time_start = time()
    figure, figure_name = figures.plot_histogram(
        series=data_set.count_sum,
        label="Total number of {}s per {}".format(
            data_set.terms["item"], data_set.terms["example"]
        ),
        normed=True,
        y_scale="log",
        name=["count sum", data_set_name]
    )
    figures.save_figure(
        figure=figure,
        name=figure_name,
        options=export_options,
        directory=distribution_directory
    )
    distribution_duration = time() - distribution_time_start
    print("    Count sum distribution plotted and saved ({}).".format(
        format_duration(distribution_duration)))

    # Count distributions and count sum distributions for each class
    if analysis_level == "extensive" and colouring_data_set.labels is not None:

        class_count_distribution_directory = distribution_directory
        if data_set.version == "original":
            class_count_distribution_directory += "-classes"

        if colouring_data_set.label_superset:
            labels = colouring_data_set.superset_labels
            class_names = colouring_data_set.superset_class_names
            class_palette = colouring_data_set.superset_class_palette
            label_sorter = colouring_data_set.superset_label_sorter
        else:
            labels = colouring_data_set.labels
            class_names = colouring_data_set.class_names
            class_palette = colouring_data_set.class_palette
            label_sorter = colouring_data_set.label_sorter

        if not class_palette:
            index_palette = style.lighter_palette(
                colouring_data_set.number_of_classes)
            class_palette = {
                class_name: index_palette[i] for i, class_name
                in enumerate(sorted(class_names, key=label_sorter))
            }

        distribution_time_start = time()
        for class_name in class_names:

            class_indices = labels == class_name

            if not class_indices.any():
                continue

            values_label = data_set.values[class_indices]

            if scipy.sparse.issparse(values_label):
                series = values_label.data
                excess_zero_count = values_label.size - series.size
            else:
                series = data_set.values.reshape(-1)
                excess_zero_count = 0

            figure, figure_name = figures.plot_histogram(
                series=series,
                excess_zero_count=excess_zero_count,
                label=data_set.terms["value"].capitalize() + "s",
                discrete=data_set_discreteness,
                normed=True,
                y_scale="log",
                colour=class_palette[class_name],
                name=["counts", data_set_name, "class", class_name]
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=class_count_distribution_directory
            )

        distribution_duration = time() - distribution_time_start
        print(
            "    Count distributions for each class plotted and saved ({})."
            .format(format_duration(distribution_duration))
        )

        distribution_time_start = time()
        for class_name in class_names:

            class_indices = labels == class_name
            if not class_indices.any():
                continue

            figure, figure_name = figures.plot_histogram(
                series=data_set.count_sum[class_indices],
                label="Total number of {}s per {}".format(
                    data_set.terms["item"], data_set.terms["example"]
                ),
                normed=True,
                y_scale="log",
                colour=class_palette[class_name],
                name=["count sum", data_set_name, "class", class_name]
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=class_count_distribution_directory
            )

        distribution_duration = time() - distribution_time_start
        print(
            "    "
            "Count sum distributions for each class plotted and saved ({})."
            .format(format_duration(distribution_duration))
        )

    print()


def analyse_matrices(data_set, plot_distances=False, name=None,
                     export_options=None, analyses_directory=None):

    if plot_distances:
        base_name = "distances"
    else:
        base_name = "heat_maps"

    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]
    analyses_directory = os.path.join(analyses_directory, base_name)

    if not name:
        name = []
    elif not isinstance(name, list):
        name = [name]

    name.insert(0, base_name)

    # Subsampling indices (if necessary)
    random_state = numpy.random.RandomState(57)
    shuffled_indices = random_state.permutation(data_set.number_of_examples)

    # Feature selection for plotting (if necessary)
    feature_indices_for_plotting = None
    if (not plot_distances and data_set.number_of_features
            > MAXIMUM_NUMBER_OF_FEATURES_FOR_HEAT_MAPS):
        feature_variances = data_set.values.var(axis=0)
        if isinstance(feature_variances, numpy.matrix):
            feature_variances = feature_variances.A.squeeze()
        feature_indices_for_plotting = numpy.argsort(feature_variances)[
            -MAXIMUM_NUMBER_OF_FEATURES_FOR_HEAT_MAPS:]
        feature_indices_for_plotting.sort()

    # Class palette
    class_palette = data_set.class_palette
    if data_set.labels is not None and not class_palette:
        index_palette = style.lighter_palette(data_set.number_of_classes)
        class_palette = {
            class_name: tuple(index_palette[i]) for i, class_name in
            enumerate(sorted(data_set.class_names,
                             key=data_set.label_sorter))
        }

    # Axis labels
    example_label = data_set.terms["example"].capitalize() + "s"
    feature_label = data_set.terms["feature"].capitalize() + "s"
    value_label = data_set.terms["value"].capitalize() + "s"

    version = data_set.version
    symbol = None
    value_name = "values"

    if version in ["z", "x"]:
        symbol = "$\\mathbf{{{}}}$".format(version)
        value_name = "component"
    elif version in ["y"]:
        symbol = "${}$".format(version)
        value_name = "value"

    if version in ["y", "z"]:
        feature_label = " ".join([symbol, value_name + "s"])

    if plot_distances:
        if version in ["y", "z"]:
            value_label = symbol
        else:
            value_label = version

    if feature_indices_for_plotting is not None:
        feature_label = "{} most varying {}".format(
            len(feature_indices_for_plotting),
            feature_label.lower()
        )

    plot_string = "Plotting heat map for {} values."
    if plot_distances:
        plot_string = "Plotting pairwise distances in {} space."
    print(plot_string.format(data_set.version))

    sorting_methods = ["hierarchical_clustering"]

    if data_set.labels is not None:
        sorting_methods.insert(0, "labels")

    for sorting_method in sorting_methods:

        distance_metrics = [None]

        if plot_distances or sorting_method == "hierarchical_clustering":
            distance_metrics = ["Euclidean", "cosine"]

        for distance_metric in distance_metrics:

            start_time = time()

            if (sorting_method == "hierarchical_clustering"
                    and data_set.number_of_examples
                    > MAXIMUM_NUMBER_OF_EXAMPLES_FOR_DENDROGRAM):
                sample_size = MAXIMUM_NUMBER_OF_EXAMPLES_FOR_DENDROGRAM
            elif (data_set.number_of_examples
                    > MAXIMUM_NUMBER_OF_EXAMPLES_FOR_HEAT_MAPS):
                sample_size = MAXIMUM_NUMBER_OF_EXAMPLES_FOR_HEAT_MAPS
            else:
                sample_size = None

            indices = numpy.arange(data_set.number_of_examples)

            if sample_size:
                indices = shuffled_indices[:sample_size]
                example_label = "{} randomly sampled {}".format(
                    sample_size, data_set.terms["example"] + "s")

            figure, figure_name = figures.plot_matrix(
                feature_matrix=data_set.values[indices],
                plot_distances=plot_distances,
                example_label=example_label,
                feature_label=feature_label,
                value_label=value_label,
                sorting_method=sorting_method,
                distance_metric=distance_metric,
                labels=(
                    data_set.labels[indices]
                    if data_set.labels is not None else None
                ),
                label_kind=data_set.terms["class"],
                class_palette=class_palette,
                feature_indices_for_plotting=feature_indices_for_plotting,
                name_parts=name + [
                    data_set.version,
                    distance_metric,
                    sorting_method
                ]
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=analyses_directory
            )

            duration = time() - start_time

            plot_kind_string = "Heat map for {} values".format(
                data_set.version)

            if plot_distances:
                plot_kind_string = "{} distances in {} space".format(
                    distance_metric.capitalize(),
                    data_set.version
                )

            subsampling_string = ""

            if sample_size:
                subsampling_string = "{} {} randomly sampled examples".format(
                    "for" if plot_distances else "of", sample_size)

            sort_string = "sorted using {}".format(
                sorting_method.replace("_", " ")
            )

            if (not plot_distances
                    and sorting_method == "hierarchical_clustering"):
                sort_string += " (with {} distances)".format(distance_metric)

            print("    " + " ".join([s for s in [
                plot_kind_string,
                subsampling_string,
                sort_string,
                "plotted and saved",
                "({})".format(format_duration(duration))
            ] if s]) + ".")

    print()


def analyse_decompositions(data_sets, other_data_sets=None, centroids=None,
                           colouring_data_set=None,
                           sampled_data_set=None,
                           decomposition_methods=None,
                           highlight_feature_indices=None,
                           symbol=None, title="data set", specifier=None,
                           analysis_level=None, export_options=None,
                           analyses_directory=None):

    if analysis_level is None:
        analysis_level = defaults["analyses"]["analysis_level"]

    centroids_original = centroids

    if isinstance(data_sets, dict):
        data_sets = list(data_sets.values())

    if not isinstance(data_sets, (list, tuple)):
        data_sets = [data_sets]

    if other_data_sets is None:
        other_data_sets = [None] * len(data_sets)
    elif not isinstance(other_data_sets, (list, tuple)):
        other_data_sets = [other_data_sets]

    if len(data_sets) != len(other_data_sets):
        raise ValueError(
            "Lists of data sets and alternative data sets do not have the "
            "same length."
        )

    specification = None

    base_symbol = symbol

    original_title = title

    if decomposition_methods is None:
        decomposition_methods = [defaults["decomposition_method"]]
    elif not isinstance(decomposition_methods, (list, tuple)):
        decomposition_methods = [decomposition_methods]
    else:
        decomposition_methods = decomposition_methods.copy()
    decomposition_methods.insert(0, None)

    if highlight_feature_indices is None:
        highlight_feature_indices = defaults["analyses"][
            "highlight_feature_indices"]
    elif not isinstance(highlight_feature_indices, (list, tuple)):
        highlight_feature_indices = [highlight_feature_indices]
    else:
        highlight_feature_indices = highlight_feature_indices.copy()

    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]

    for data_set, other_data_set in zip(data_sets, other_data_sets):

        if data_set.values.shape[1] <= 1:
            continue

        title = original_title
        name = normalise_string(title)

        if specifier:
            specification = specifier(data_set)

        if specification:
            name += "-" + str(specification)
            title += " for " + specification

        title += " set"

        if not colouring_data_set:
            colouring_data_set = data_set

        if data_set.version in ["z", "z1"]:
            centroids = copy.deepcopy(centroids_original)
        else:
            centroids = None

        if other_data_set:
            title = "{} set values in {}".format(
                other_data_set.version, title)
            name = other_data_set.version + "-" + name

        decompositions_directory = os.path.join(analyses_directory, name)

        for decomposition_method in decomposition_methods:

            other_values = None
            sampled_values = None

            if other_data_set:
                other_values = other_data_set.values

            if sampled_data_set:
                sampled_values = sampled_data_set.values

            if not decomposition_method:
                if data_set.number_of_features == 2:
                    values_decomposed = data_set.values
                    other_values_decomposed = other_values
                    sampled_values_decomposed = sampled_values
                    centroids_decomposed = centroids
                else:
                    continue
            else:
                decomposition_method = proper_string(
                    decomposition_method, DECOMPOSITION_METHOD_NAMES)

                values_decomposed = data_set.values
                other_values_decomposed = other_values
                sampled_values_decomposed = sampled_values
                centroids_decomposed = centroids

                other_value_sets_decomposed = {}
                if other_values is not None:
                    other_value_sets_decomposed["other"] = other_values
                if sampled_values is not None:
                    other_value_sets_decomposed["sampled"] = sampled_values
                if not other_value_sets_decomposed:
                    other_value_sets_decomposed = None

                if decomposition_method == "t-SNE":
                    if (data_set.number_of_examples
                            > MAXIMUM_NUMBER_OF_EXAMPLES_FOR_TSNE):
                        print(
                            "The number of examples for {}".format(
                                title),
                            "is too large to decompose it",
                            "using {}. Skipping.".format(decomposition_method)
                        )
                        print()
                        continue

                    elif (data_set.number_of_features >
                            MAXIMUM_NUMBER_OF_FEATURES_FOR_TSNE):
                        number_of_pca_components_before_tsne = min(
                            MAXIMUM_NUMBER_OF_PCA_COMPONENTS_BEFORE_TSNE,
                            data_set.number_of_examples - 1
                        )
                        print(
                            "The number of features for {}".format(
                                title),
                            "is too large to decompose it",
                            "using {} in due time.".format(
                                decomposition_method)
                        )
                        print(
                            "Decomposing {} to {} components using PCA "
                            "beforehand.".format(
                                title,
                                number_of_pca_components_before_tsne
                            )
                        )
                        decompose_time_start = time()
                        (
                            values_decomposed,
                            other_value_sets_decomposed,
                            centroids_decomposed
                        ) = decompose(
                            values_decomposed,
                            other_value_sets=other_value_sets_decomposed,
                            centroids=centroids_decomposed,
                            method="pca",
                            number_of_components=(
                                number_of_pca_components_before_tsne)
                        )
                        decompose_duration = time() - decompose_time_start
                        print("{} pre-decomposed ({}).".format(
                            capitalise_string(title),
                            format_duration(decompose_duration)
                        ))

                    else:
                        if scipy.sparse.issparse(values_decomposed):
                            values_decomposed = values_decomposed.A
                        if scipy.sparse.issparse(other_values_decomposed):
                            other_values_decomposed = other_values_decomposed.A
                        if scipy.sparse.issparse(sampled_values_decomposed):
                            sampled_values_decomposed = (
                                sampled_values_decomposed.A)

                print("Decomposing {} using {}.".format(
                    title, decomposition_method))
                decompose_time_start = time()
                (
                    values_decomposed,
                    other_value_sets_decomposed,
                    centroids_decomposed
                ) = decompose(
                    values_decomposed,
                    other_value_sets=other_value_sets_decomposed,
                    centroids=centroids_decomposed,
                    method=decomposition_method,
                    number_of_components=2
                )
                decompose_duration = time() - decompose_time_start
                print("{} decomposed ({}).".format(
                    capitalise_string(title),
                    format_duration(decompose_duration)
                ))
                print()

                if other_value_sets_decomposed:
                    other_values_decomposed = other_value_sets_decomposed.get(
                        "other")
                    sampled_values_decomposed = (
                        other_value_sets_decomposed.get("sampled"))

            if base_symbol:
                symbol = base_symbol
            else:
                symbol = specification

            x_label = _axis_label_for_symbol(
                symbol=symbol,
                coordinate=1,
                decomposition_method=decomposition_method,
            )
            y_label = _axis_label_for_symbol(
                symbol=symbol,
                coordinate=2,
                decomposition_method=decomposition_method,
            )

            figure_labels = {
                "title": decomposition_method,
                "x label": x_label,
                "y label": y_label
            }

            if other_data_set:
                plot_values_decomposed = other_values_decomposed
                example_names = other_data_set.example_names
            else:
                plot_values_decomposed = values_decomposed
                example_names = data_set.example_names

            if plot_values_decomposed is None:
                print("No values to plot.\n")
                return

            if decomposition_method:

                table_name_parts = [
                    name, normalise_string(decomposition_method)]
                if sampled_values_decomposed is not None:
                    table_name_parts.append("samples")
                table_name = "-".join(table_name_parts)

                feature_names = [
                    "{}{}".format(
                        DECOMPOSITION_METHOD_LABEL[decomposition_method],
                        coordinate
                    )
                    for coordinate in [1, 2]
                ]

                print("Saving decomposed {}.".format(title))
                saving_time_start = time()

                save_values(
                    values=plot_values_decomposed,
                    name=table_name,
                    row_names=example_names,
                    column_names=feature_names,
                    directory=decompositions_directory)

                saving_duration = time() - saving_time_start
                print("Decomposed {} saved ({}).".format(
                    title,
                    format_duration(saving_duration)))
                print()

            print("Plotting {}{}.".format(
                "decomposed " if decomposition_method else "",
                title
            ))

            # No colour-coding
            plot_time_start = time()
            figure, figure_name = figures.plot_values(
                plot_values_decomposed,
                centroids=centroids_decomposed,
                figure_labels=figure_labels,
                example_tag=data_set.terms["example"],
                name=name
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=decompositions_directory
            )
            plot_duration = time() - plot_time_start
            print("    {} plotted and saved ({}).".format(
                capitalise_string(title),
                format_duration(plot_duration)
            ))

            # Samples
            if sampled_data_set:
                plot_time_start = time()
                figure, figure_name = figures.plot_values(
                    plot_values_decomposed,
                    centroids=centroids_decomposed,
                    sampled_values=sampled_values_decomposed,
                    figure_labels=figure_labels,
                    example_tag=data_set.terms["example"],
                    name=name
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print("    {} (with samples) plotted and saved ({}).".format(
                    capitalise_string(title),
                    format_duration(plot_duration)
                ))

            # Labels
            if colouring_data_set.labels is not None:
                plot_time_start = time()
                figure, figure_name = figures.plot_values(
                    plot_values_decomposed,
                    colour_coding="labels",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    example_tag=data_set.terms["example"],
                    name=name
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print("    {} (with labels) plotted and saved ({}).".format(
                    capitalise_string(title), format_duration(plot_duration)))

                # Superset labels
                if colouring_data_set.superset_labels is not None:
                    plot_time_start = time()
                    figure, figure_name = figures.plot_values(
                        plot_values_decomposed,
                        colour_coding="superset labels",
                        colouring_data_set=colouring_data_set,
                        centroids=centroids_decomposed,
                        figure_labels=figure_labels,
                        example_tag=data_set.terms["example"],
                        name=name
                    )
                    figures.save_figure(
                        figure=figure,
                        name=figure_name,
                        options=export_options,
                        directory=decompositions_directory
                    )
                    plot_duration = time() - plot_time_start
                    print(
                        "    "
                        "{} (with superset labels) plotted and saved ({})."
                        .format(
                            capitalise_string(title),
                            format_duration(plot_duration)
                        )
                    )

                # For each class
                if analysis_level == "extensive":
                    if colouring_data_set.number_of_classes <= 10:
                        plot_time_start = time()
                        for class_name in colouring_data_set.class_names:
                            figure, figure_name = figures.plot_values(
                                plot_values_decomposed,
                                colour_coding="class",
                                colouring_data_set=colouring_data_set,
                                centroids=centroids_decomposed,
                                class_name=class_name,
                                figure_labels=figure_labels,
                                example_tag=data_set.terms["example"],
                                name=name
                            )
                            figures.save_figure(
                                figure=figure,
                                name=figure_name,
                                options=export_options,
                                directory=decompositions_directory
                            )
                        plot_duration = time() - plot_time_start
                        print(
                            "    {} (for each class) plotted and saved ({})."
                            .format(
                                capitalise_string(title),
                                format_duration(plot_duration)
                            )
                        )

                    if (colouring_data_set.superset_labels is not None
                            and data_set.number_of_superset_classes <= 10):
                        plot_time_start = time()
                        for superset_class_name in (
                                colouring_data_set.superset_class_names):
                            figure, figure_name = figures.plot_values(
                                plot_values_decomposed,
                                colour_coding="superset class",
                                colouring_data_set=colouring_data_set,
                                centroids=centroids_decomposed,
                                class_name=superset_class_name,
                                figure_labels=figure_labels,
                                example_tag=data_set.terms["example"],
                                name=name
                            )
                            figures.save_figure(
                                figure=figure,
                                name=figure_name,
                                options=export_options,
                                directory=decompositions_directory
                            )
                        plot_duration = time() - plot_time_start
                        print(
                            "    {} (for each superset class) plotted and "
                            "saved ({}).".format(
                                capitalise_string(title),
                                format_duration(plot_duration)
                            )
                        )

            # Batches
            if colouring_data_set.has_batches:
                plot_time_start = time()
                figure, figure_name = figures.plot_values(
                    plot_values_decomposed,
                    colour_coding="batches",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    example_tag=data_set.terms["example"],
                    name=name,
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print(
                    "    "
                    "{} (with batches) plotted and saved ({})."
                    .format(
                        capitalise_string(title),
                        format_duration(plot_duration)
                    )
                )

            # Cluster IDs
            if colouring_data_set.has_predicted_cluster_ids:
                plot_time_start = time()
                figure, figure_name = figures.plot_values(
                    plot_values_decomposed,
                    colour_coding="predicted cluster IDs",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    example_tag=data_set.terms["example"],
                    name=name,
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print(
                    "    "
                    "{} (with predicted cluster IDs) plotted and saved ({})."
                    .format(
                        capitalise_string(title),
                        format_duration(plot_duration)
                    )
                )

            # Predicted labels
            if colouring_data_set.has_predicted_labels:
                plot_time_start = time()
                figure, figure_name = figures.plot_values(
                    plot_values_decomposed,
                    colour_coding="predicted labels",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    example_tag=data_set.terms["example"],
                    name=name,
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print(
                    "    "
                    "{} (with predicted labels) plotted and saved ({})."
                    .format(
                        capitalise_string(title),
                        format_duration(plot_duration)
                    )
                )

            if colouring_data_set.has_predicted_superset_labels:
                plot_time_start = time()
                figure, figure_name = figures.plot_values(
                    plot_values_decomposed,
                    colour_coding="predicted superset labels",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    example_tag=data_set.terms["example"],
                    name=name,
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print(
                    "    {} (with predicted superset labels) plotted and saved"
                    " ({}).".format(
                        capitalise_string(title),
                        format_duration(plot_duration)
                    )
                )

            # Count sum
            plot_time_start = time()
            figure, figure_name = figures.plot_values(
                plot_values_decomposed,
                colour_coding="count sum",
                colouring_data_set=colouring_data_set,
                centroids=centroids_decomposed,
                figure_labels=figure_labels,
                example_tag=data_set.terms["example"],
                name=name
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=decompositions_directory
            )
            plot_duration = time() - plot_time_start
            print("    {} (with count sum) plotted and saved ({}).".format(
                capitalise_string(title),
                format_duration(plot_duration)
            ))

            # Features
            for feature_index in highlight_feature_indices:
                plot_time_start = time()
                figure, figure_name = figures.plot_values(
                    plot_values_decomposed,
                    colour_coding="feature",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    feature_index=feature_index,
                    figure_labels=figure_labels,
                    example_tag=data_set.terms["example"],
                    name=name
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print("    {} (with {}) plotted and saved ({}).".format(
                    capitalise_string(title),
                    data_set.feature_names[feature_index],
                    format_duration(plot_duration)
                ))

            print()


def analyse_centroid_probabilities(centroids, name=None,
                                   analysis_level=None,
                                   export_options=None,
                                   analyses_directory=None):

    if name:
        name = normalise_string(name)
    if analysis_level is None:
        analysis_level = defaults["analyses"]["analysis_level"]
    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]

    print("Plotting centroid probabilities.")
    plot_time_start = time()

    posterior_probabilities = None
    prior_probabilities = None

    if "posterior" in centroids and centroids["posterior"]:
        posterior_probabilities = centroids["posterior"]["probabilities"]
        n_centroids = len(posterior_probabilities)
    if "prior" in centroids and centroids["prior"]:
        prior_probabilities = centroids["prior"]["probabilities"]
        n_centroids = len(prior_probabilities)

    centroids_palette = style.darker_palette(n_centroids)
    x_label = "$k$"
    if prior_probabilities is not None:
        if posterior_probabilities is not None:
            y_label = _axis_label_for_symbol(
                symbol="\\pi",
                distribution=normalise_string("posterior"),
                suffix="^k")
            if name:
                plot_name = [name, "posterior", "prior"]
            else:
                plot_name = ["posterior", "prior"]
        else:
            y_label = _axis_label_for_symbol(
                symbol="\\pi",
                distribution=normalise_string("prior"),
                suffix="^k")
            if name:
                plot_name = [name, "prior"]
            else:
                plot_name = "prior"
    elif posterior_probabilities is not None:
        y_label = _axis_label_for_symbol(
            symbol="\\pi",
            distribution=normalise_string("posterior"),
            suffix="^k")
        if name:
            plot_name = [name, "posterior"]
        else:
            plot_name = "posterior"

    figure, figure_name = figures.plot_probabilities(
        posterior_probabilities,
        prior_probabilities,
        x_label=x_label,
        y_label=y_label,
        palette=centroids_palette,
        uniform=False,
        name=plot_name
    )
    figures.save_figure(
        figure=figure,
        name=figure_name,
        options=export_options,
        directory=analyses_directory
    )

    plot_duration = time() - plot_time_start
    print("Centroid probabilities plotted and saved ({}).".format(
        format_duration(plot_duration)))


def analyse_predictions(evaluation_set, analyses_directory=None):

    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]

    print("Saving predictions.")

    predictions_directory = os.path.join(analyses_directory, "predictions")

    table_name = "predictions"

    if evaluation_set.prediction_specifications:
        table_name += "-" + (
            evaluation_set.prediction_specifications.name)
    else:
        table_name += "-unknown_prediction_method"

    if evaluation_set.has_predicted_cluster_ids:
        saving_time_start = time()
        save_values(
            values=evaluation_set.predicted_cluster_ids,
            name="{}-predicted_cluster_ids".format(table_name),
            row_names=evaluation_set.example_names,
            column_names=["Cluster ID"],
            directory=predictions_directory)
        saving_duration = time() - saving_time_start
        print("    Predicted cluster IDs saved ({}).".format(
            format_duration(saving_duration)))

    if evaluation_set.has_predicted_labels:
        saving_time_start = time()
        save_values(
            values=evaluation_set.predicted_labels,
            name="{}-predicted_labels".format(table_name),
            row_names=evaluation_set.example_names,
            column_names=[evaluation_set.terms["class"].capitalize()],
            directory=predictions_directory)
        saving_duration = time() - saving_time_start
        print("    Predicted labels saved ({}).".format(
            format_duration(saving_duration)))

    if evaluation_set.has_predicted_superset_labels:
        saving_time_start = time()
        save_values(
            values=evaluation_set.predicted_superset_labels,
            name="{}-predicted_superset_labels".format(table_name),
            row_names=evaluation_set.example_names,
            column_names=[evaluation_set.terms["class"].capitalize()],
            directory=predictions_directory)
        saving_duration = time() - saving_time_start
        print("    Predicted superset labels saved ({}).".format(
            format_duration(saving_duration)))

    print()
