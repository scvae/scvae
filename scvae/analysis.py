# ======================================================================== #
#
# Copyright (c) 2017 - 2019 scVAE authors
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
import gzip
import os
import pickle
import re
from time import time

import matplotlib.patches
import matplotlib.lines
import matplotlib.colors
import numpy
import pandas
import PIL
import scipy
import seaborn
import sklearn.metrics.cluster
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from auxiliary import (
    loadNumberOfEpochsTrained, loadLearningCurves, loadAccuracies,
    loadCentroids, loadKLDivergences,
    checkRunID,
    formatTime, formatDuration,
    normaliseString, properString, capitaliseString, subheading
)
from data.auxiliary import standard_deviation
from data.sparse import sparsity
from miscellaneous.decomposition import (
    decompose,
    DECOMPOSITION_METHOD_NAMES,
    DECOMPOSITION_METHOD_LABEL,
    DEFAULT_DECOMPOSITION_METHOD
)

STANDARD_PALETTE = seaborn.color_palette('Set2', 8)
STANDARD_COLOUR_MAP = seaborn.cubehelix_palette(light=.95, as_cmap=True)
NEUTRAL_COLOUR = (0.7, 0.7, 0.7)

DEFAULT_SMALL_MARKER_SIZE_IN_SCATTER_PLOTS = 1
DEFAULT_LARGE_MARKER_SIZE_IN_SCATTER_PLOTS = 3
DEFAULT_MARKER_SIZE_IN_SCATTER_PLOTS = (
    DEFAULT_SMALL_MARKER_SIZE_IN_SCATTER_PLOTS)

DEFAULT_MARKER_SIZE_IN_LEGENDS = 4

FIGURE_EXTENSION = ".png"
IMAGE_EXTENSION = ".png"

PUBLICATION_FIGURE_EXTENSION = ".tiff"
PUBLICATION_DPI = 350
PUBLICATION_COPIES = {
    "full_width": {
        "figure_width": 17.8 / 2.54
    },
    "three_quarters_width": {
        "figure_width": numpy.mean([8.6, 17.8]) / 2.54
    },
    "half_width": {
        "figure_width": 8.6 / 2.54
    }
}

MAXIMUM_SIZE_FOR_NORMAL_STATISTICS_COMPUTATION = 5e8
MAXIMUM_NUMBER_OF_EXAMPLES_BEFORE_SAMPLING_SILHOUETTE_SCORE = 20000

MAXIMUM_NUMBER_OF_BINS_FOR_HISTOGRAMS = 20000

MAXIMUM_NUMBER_OF_VALUES_FOR_HEAT_MAPS = 5000 * 25000
MAXIMUM_NUMBER_OF_EXAMPLES_FOR_HEAT_MAPS = 10000
MAXIMUM_NUMBER_OF_FEATURES_FOR_HEAT_MAPS = 10000
MAXIMUM_NUMBER_OF_EXAMPLES_FOR_DENDROGRAM = 1000

MAXIMUM_NUMBER_OF_FEATURES_FOR_TSNE = 100
MAXIMUM_NUMBER_OF_EXAMPLES_FOR_TSNE = 200000
MAXIMUM_NUMBER_OF_PCA_COMPONENTS_BEFORE_TSNE = 50

MAXIMUM_NUMBER_OF_EXAMPLES_FOR_LARGE_POINTS_IN_SCATTER_PLOTS = 1000

DEFAULT_NUMBER_OF_RANDOM_EXAMPLES = 100

EVALUATION_SUBSET_MAXIMUM_NUMBER_OF_EXAMPLES = 25
EVALUATION_SUBSET_MAXIMUM_NUMBER_OF_EXAMPLES_PER_CLASS = 3

PROFILE_COMPARISON_COUNT_CUT_OFF = 10.5

DEFAULT_CUTOFFS = range(1, 10)

ANALYSIS_GROUPS = {
    "simple": ["metrics", "images", "learning_curves", "accuracies"],
    "default": ["kl_heat_maps", "profile_comparisons", "distributions",
                "distances", "decompositions", "latent_space"],
    "complete": ["heat_maps", "latent_distributions", "latent_correlations",
                 "feature_value_standard_deviations"]
}
ANALYSIS_GROUPS["default"] += ANALYSIS_GROUPS["simple"]
ANALYSIS_GROUPS["complete"] += ANALYSIS_GROUPS["default"]

CLUSTERING_METRICS = {}


def analyse_data(data_sets,
                 decomposition_methods=None,
                 highlight_feature_indices=None,
                 analyses=None, analysis_level="normal",
                 export_options=None, results_directory="results"):

    results_directory = os.path.join(results_directory, "data")

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    analyses = _parse_analyses(analyses)

    if not isinstance(data_sets, list):
        data_sets = [data_sets]

    if "metrics" in analyses:

        print(subheading("Metrics"))

        print("Calculating metrics for data set.")
        metrics_time_start = time()

        data_set_statistics = []
        histogram_statistics = []
        number_of_examples = {}
        number_of_features = 0

        for data_set in data_sets:
            number_of_examples[data_set.kind] = data_set.number_of_examples
            if data_set.kind == "full":
                number_of_features = data_set.number_of_features
                histogram_statistics.extend([
                    summary_statistics(
                        series, name=series_name, skip_sparsity=True)
                    for series_name, series in {
                        "count sum": data_set.count_sum
                    }.items()
                ])
            data_set_statistics.append(summary_statistics(
                data_set.values, name=data_set.kind, tolerance=0.5))

        metrics_duration = time() - metrics_time_start
        print("Metrics calculated ({}).".format(
            formatDuration(metrics_duration)))

        metrics_path = os.path.join(results_directory, "data_set_metrics.log")
        metrics_saving_time_start = time()

        metrics_string_parts = [
            "Timestamp: {}".format(formatTime(metrics_saving_time_start)),
            "Features: {}".format(number_of_features),
            "Examples: {}".format(number_of_examples["full"]),
            "\n".join([
                "{} examples: {}".format(capitaliseString(kind), number)
                for kind, number in number_of_examples.items()
                if kind != "full"
            ]),
            "\n" + format_summary_statistics(data_set_statistics)
        ]
        metrics_string = "\n".join(metrics_string_parts) + "\n"

        with open(metrics_path, "w") as metrics_file:
            metrics_file.write(metrics_string)

        metrics_saving_duration = time() - metrics_saving_time_start
        print("Metrics saved ({}).".format(formatDuration(
            metrics_saving_duration)))
        print()

        print(format_summary_statistics(data_set_statistics))
        print()

        print(format_summary_statistics(histogram_statistics, name="Series"))
        print()

    for data_set in data_sets:

        print(subheading("Analyses of {} set".format(data_set.kind)))

        if "images" in analyses and data_set.example_type == "images":
            print("Saving image of {} random examples from {} set.".format(
                DEFAULT_NUMBER_OF_RANDOM_EXAMPLES, data_set.kind))
            image_time_start = time()
            image, image_name = combine_images_from_data_set(
                data_set,
                number_of_random_examples=DEFAULT_NUMBER_OF_RANDOM_EXAMPLES,
                name=data_set.kind
            )
            save_image(
                image=image,
                name=image_name,
                directory=results_directory
            )
            image_duration = time() - image_time_start
            print("Image saved ({}).".format(formatDuration(image_duration)))
            print()

        if "distributions" in analyses:
            analyse_distributions(
                data_set,
                cutoffs=DEFAULT_CUTOFFS,
                analysis_level=analysis_level,
                export_options=export_options,
                results_directory=results_directory
            )

        if "heat_maps" in analyses:
            analyse_matrices(
                data_set,
                name=[data_set.kind],
                results_directory=results_directory
            )

        if "distances" in analyses:
            analyse_matrices(
                data_set,
                name=[data_set.kind],
                plot_distances=True,
                results_directory=results_directory
            )

        if "decompositions" in analyses:
            analyse_decompositions(
                data_set,
                decomposition_methods=decomposition_methods,
                highlight_feature_indices=highlight_feature_indices,
                symbol="x",
                title="original space",
                specifier=lambda data_set: data_set.kind,
                analysis_level=analysis_level,
                export_options=export_options,
                results_directory=results_directory
            )

        if "feature_value_standard_deviations" in analyses:

            print("Computing and plotting feature value standard deviations:")

            feature_value_standard_deviations_directory = os.path.join(
                results_directory,
                "feature_value_standard_deviations"
            )

            time_start = time()

            feature_value_standard_deviations = data_set.values.std(axis=0)

            if isinstance(feature_value_standard_deviations, numpy.matrix):
                feature_value_standard_deviations = (
                    feature_value_standard_deviations.A)

            feature_value_standard_deviations = (
                feature_value_standard_deviations.squeeze())

            duration = time() - time_start
            print(
                "    Feature value standard deviations computed({})."
                .format(formatDuration(duration))
            )

            # Feature value standard_deviations
            time_start = time()

            figure, figure_name = plot_series(
                series=feature_value_standard_deviations,
                x_label=data_set.tags["feature"] + "s",
                y_label="{} standard deviations".format(
                    data_set.tags["type"]),
                sort=True,
                scale="log",
                name=["feature value standard deviations", data_set.kind]
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=feature_value_standard_deviations_directory
            )

            duration = time() - time_start
            print(
                "    Feature value standard deviations plotted and saved({})."
                .format(formatDuration(duration))
            )

            # Distribution of feature value standard deviations
            time_start = time()

            figure, figure_name = plot_histogram(
                series=feature_value_standard_deviations,
                label="{} {} standard deviations".format(
                    data_set.tags["feature"], data_set.tags["type"]
                ),
                normed=True,
                x_scale="linear",
                y_scale="log",
                name=["feature value standard deviations", data_set.kind]
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=feature_value_standard_deviations_directory
            )

            duration = time() - time_start
            print(
                "    Feature value standard deviation distribution plotted "
                "and saved ({}).".format(formatDuration(duration))
            )

            print()


def analyse_model(model, run_id=None,
                  analyses=None, analysis_level="normal",
                  export_options=None, results_directory="results"):

    if run_id:
        run_id = checkRunID(run_id)

    number_of_epochs_trained = loadNumberOfEpochsTrained(model, run_id=run_id)
    epochs_string = "e_" + str(number_of_epochs_trained)

    results_directory = _build_path_for_result_directory(
        base_directory=results_directory,
        model_name=model.name,
        run_id=run_id,
        subdirectories=[epochs_string]
    )

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    analyses = _parse_analyses(analyses)

    if "learning_curves" in analyses:

        print(subheading("Learning curves"))

        print("Plotting learning curves.")
        learning_curves_time_start = time()

        learning_curves = loadLearningCurves(
            model=model,
            data_set_kinds=["training", "validation"],
            run_id=run_id
        )

        figure, figure_name = plot_learning_curves(learning_curves, model.type)
        save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=results_directory
        )

        if "VAE" in model.type:
            loss_sets = [["lower_bound", "reconstruction_error"]]

            if model.type == "GMVAE":
                loss_sets.append("kl_divergence_z")
                loss_sets.append("kl_divergence_y")
            else:
                loss_sets.append("kl_divergence")

            for loss_set in loss_sets:
                figure, figure_name = plot_separate_learning_curves(
                    learning_curves,
                    loss=loss_set
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=results_directory
                )

        learning_curves_duration = time() - learning_curves_time_start
        print("Learning curves plotted and saved ({}).".format(
            formatDuration(learning_curves_duration)))
        print()

    if "accuracies" in analyses:

        accuracies_time_start = time()

        accuracies = loadAccuracies(
            model=model,
            run_id=run_id,
            data_set_kinds=["training", "validation"]
        )

        if accuracies is not None:

            print(subheading("Accuracies"))
            print("Plotting accuracies.")

            figure, figure_name = plot_accuracies(accuracies)
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=results_directory
            )

            superset_accuracies = loadAccuracies(
                model=model,
                data_set_kinds=["training", "validation"],
                superset=True,
                run_id=run_id
            )

            if superset_accuracies is not None:
                figure, figure_name = plot_accuracies(
                    superset_accuracies,
                    name="superset"
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=results_directory
                )

            accuracies_duration = time() - accuracies_time_start
            print("Accuracies plotted and saved ({}).".format(
                formatDuration(accuracies_duration)))
            print()

    if "kl_heat_maps" in analyses and "VAE" in model.type:

        print(subheading("KL divergence"))

        print("Plotting logarithm of KL divergence heat map.")
        heat_map_time_start = time()

        kl_neurons = loadKLDivergences(model=model, run_id=run_id)

        kl_neurons = numpy.sort(kl_neurons, axis=1)

        figure, figure_name = plot_kl_divergence_evolution(
            kl_neurons,
            scale="log"
        )
        save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=results_directory
        )

        heat_map_duration = time() - heat_map_time_start
        print("Heat map plotted and saved ({}).".format(
            formatDuration(heat_map_duration)))
        print()

    if "latent_distributions" in analyses and model.type == "GMVAE":

        print(subheading("Latent distributions"))

        centroids = loadCentroids(
            model=model,
            data_set_kinds="validation",
            run_id=run_id
        )

        centroids_directory = os.path.join(
            results_directory, "centroids_evolution")

        # TODO Optionally move to separate analyse function
        for distribution, distribution_centroids in centroids.items():
            if distribution_centroids:

                centroids_time_start = time()

                centroid_probabilities = distribution_centroids[
                    "probabilities"]
                centroid_means = distribution_centroids["means"]
                centroid_covariance_matrices = distribution_centroids[
                    "covariance_matrices"]

                __, n_clusters, latent_size = centroid_means.shape

                if n_clusters <= 1:
                    continue

                print("Plotting evolution of latent {} parameters.".format(
                    distribution))

                if latent_size > 2:
                    _, distribution_centroids_decomposed = decompose(
                        centroid_means[-1],
                        centroids=distribution_centroids
                    )
                    decomposed = True
                else:
                    distribution_centroids_decomposed = distribution_centroids
                    decomposed = False

                centroid_means_decomposed = (
                    distribution_centroids_decomposed["means"])

                figure, figure_name = plot_evolution_of_centroid_probabilities(
                    centroid_probabilities,
                    distribution=distribution
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=centroids_directory
                )

                figure, figure_name = plot_evolution_of_centroid_means(
                    centroid_means_decomposed,
                    distribution=distribution,
                    decomposed=decomposed
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=centroids_directory
                )

                figure, figure_name = (
                    plot_evolution_of_centroid_covariance_matrices(
                        centroid_covariance_matrices,
                        distribution=distribution
                    )
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=centroids_directory
                )

                centroids_duration = time() - centroids_time_start
                print(
                    "Evolution of latent {} parameters plotted and saved ({})"
                    .format(distribution, formatDuration(centroids_duration))
                )
                print()


def analyse_intermediate_results(epoch, learning_curves=None, epoch_start=None,
                                 model_type=None, latent_values=None,
                                 data_set=None, centroids=None,
                                 model_name=None, run_id=None,
                                 results_directory="results"):

    if run_id:
        run_id = checkRunID(run_id)

    results_directory = _build_path_for_result_directory(
        base_directory=results_directory,
        model_name=model_name,
        run_id=run_id,
        subdirectories=["intermediate"]
    )

    print("Plotting learning curves.")
    learning_curves_time_start = time()

    figure, figure_name = plot_learning_curves(
        learning_curves,
        model_type,
        epoch_offset=epoch_start
    )
    save_figure(
        figure=figure,
        name=figure_name,
        directory=results_directory
    )

    learning_curves_duration = time() - learning_curves_time_start
    print("Learning curves plotted and saved ({}).".format(
        formatDuration(learning_curves_duration)))

    if latent_values is not None:

        latent_set_name = "latent {} set".format(data_set.kind)

        print("Plotting {} with centroids for epoch {}.".format(
            latent_set_name, epoch + 1))

        if latent_values.shape[1] == 2:
            decomposition_method = ""
            latent_values_decomposed = latent_values
            centroids_decomposed = centroids
        else:
            decomposition_method = "PCA"

            print("Decomposing", latent_set_name, "using PCA.")
            decompose_time_start = time()

            latent_values_decomposed, centroids_decomposed = decompose(
                latent_values,
                centroids=centroids,
                method="PCA",
                number_of_components=2
            )

            decompose_duration = time() - decompose_time_start
            print("{} decomposed ({}).".format(
                capitaliseString(latent_set_name),
                formatDuration(decompose_duration))
            )

        symbol = "z"

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

        plot_time_start = time()

        epoch_name = "epoch-{}".format(epoch + 1)
        name = "latent_space-" + epoch_name

        if data_set.labels is not None:
            figure, figure_name = plot_values(
                latent_values_decomposed,
                colour_coding="labels",
                colouring_data_set=data_set,
                centroids=centroids_decomposed,
                figure_labels=figure_labels,
                name=name
            )
            save_figure(
                figure=figure,
                name=figure_name,
                directory=results_directory
            )
            if data_set.label_superset is not None:
                figure, figure_name = plot_values(
                    latent_values_decomposed,
                    colour_coding="superset labels",
                    colouring_data_set=data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    name=name
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    directory=results_directory
                )
        else:
            figure, figure_name = plot_values(
                latent_values_decomposed,
                centroids=centroids_decomposed,
                figure_labels=figure_labels,
                name=name
            )
            save_figure(
                figure=figure,
                name=figure_name,
                directory=results_directory
            )

        if centroids:
            analyse_centroid_probabilities(
                centroids, epoch_name,
                results_directory=results_directory
            )

        plot_duration = time() - plot_time_start
        print(
            "{} plotted and saved ({}).".format(
                capitaliseString(latent_set_name),
                formatDuration(plot_duration)
            )
        )


def analyse_results(evaluation_set, reconstructed_evaluation_set,
                    latent_evaluation_sets, model, run_id=None,
                    decomposition_methods=None,
                    evaluation_subset_indices=None,
                    highlight_feature_indices=None, prediction_details=None,
                    early_stopping=False, best_model=False,
                    analyses=None, analysis_level="normal",
                    export_options=None, results_directory="results"):

    if early_stopping and best_model:
        raise ValueError(
            "Early-stopping model and best model cannot be evaluated at the "
            "same time."
        )

    if run_id:
        run_id = checkRunID(run_id)

    if evaluation_subset_indices is None:
        evaluation_subset_indices = indices_for_evaluation_subset(
            evaluation_set)

    analyses = _parse_analyses(analyses)

    print("Setting up results analyses.")
    setup_time_start = time()

    number_of_epochs_trained = loadNumberOfEpochsTrained(
        model=model,
        run_id=run_id,
        early_stopping=early_stopping,
        best_model=best_model
    )

    # Comparison arrays
    if (("metrics" in analyses or "heat_maps" in analyses)
            and analysis_level == "extensive"):
        x_diff = reconstructed_evaluation_set.values - evaluation_set.values
        x_log_ratio = (
            numpy.log1p(reconstructed_evaluation_set.values)
            - numpy.log1p(evaluation_set.values)
        )

    # Directory path
    evaluation_directory_parts = ["e_" + str(number_of_epochs_trained)]

    if early_stopping:
        evaluation_directory_parts.append("early_stopping")
    elif best_model:
        evaluation_directory_parts.append("best_model")

    evaluation_directory_parts.append("mc_{}".format(
        model.number_of_monte_carlo_samples["evaluation"]))
    evaluation_directory_parts.append("iw_{}".format(
        model.number_of_importance_samples["evaluation"]))

    evaluation_directory = "-".join(evaluation_directory_parts)

    results_directory = _build_path_for_result_directory(
        base_directory=results_directory,
        model_name=model.name,
        run_id=run_id,
        subdirectories=[evaluation_directory]
    )

    if evaluation_set.kind != "test":
        results_directory = os.path.join(
            results_directory, evaluation_set.kind)

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    setup_duration = time() - setup_time_start
    print("Finished setting up ({}).".format(formatDuration(setup_duration)))
    print()

    if "metrics" in analyses:

        print(subheading("Metrics"))

        print("Loading results from model log directory.")
        loading_time_start = time()

        evaluation_eval = loadLearningCurves(
            model=model,
            data_set_kinds="evaluation",
            run_id=run_id,
            early_stopping=early_stopping,
            best_model=best_model
        )
        accuracy_eval = loadAccuracies(
            model=model,
            data_set_kinds="evaluation",
            run_id=run_id,
            early_stopping=early_stopping,
            best_model=best_model
        )
        superset_accuracy_eval = loadAccuracies(
            model=model,
            data_set_kinds="evaluation",
            superset=True,
            run_id=run_id,
            early_stopping=early_stopping,
            best_model=best_model
        )

        loading_duration = time() - loading_time_start
        print("Results loaded ({}).".format(formatDuration(loading_duration)))
        print()

        print("Calculating metrics for results.")
        metrics_time_start = time()

        evaluation_set_statistics = [
            summary_statistics(
                data_set.values, name=data_set.version, tolerance=0.5)
            for data_set in [evaluation_set, reconstructed_evaluation_set]
        ]

        if analysis_level == "extensive":
            evaluation_set_statistics.append(summary_statistics(
                numpy.abs(x_diff),
                name="differences",
                skip_sparsity=True
            ))
            evaluation_set_statistics.append(summary_statistics(
                numpy.abs(x_log_ratio),
                name="log-ratios",
                skip_sparsity=True
            ))

        clustering_metric_values = compute_clustering_metrics(evaluation_set)

        metrics_duration = time() - metrics_time_start
        print("Metrics calculated ({}).".format(
            formatDuration(metrics_duration)))

        metrics_saving_time_start = time()

        metrics_log_filename = "{}-metrics".format(evaluation_set.kind)
        metrics_log_path = os.path.join(
            results_directory, metrics_log_filename + ".log")
        metrics_dictionary_path = os.path.join(
            results_directory, metrics_log_filename + ".pkl.gz")

        # Build metrics string
        metrics_string_parts = [
            "Timestamp: {}".format(formatTime(metrics_saving_time_start)),
            "Number of epochs trained: {}".format(number_of_epochs_trained),
            "\nEvaluation:"
        ]
        if "VAE" in model.type:
            metrics_string_parts.extend([
                "    ELBO: {:.5g}.".format(evaluation_eval["lower_bound"][-1]),
                "    ENRE: {:.5g}.".format(
                    evaluation_eval["reconstruction_error"][-1])
            ])
            if model.type == "VAE":
                metrics_string_parts.append(
                    "    KL: {:.5g}.".format(
                        evaluation_eval["kl_divergence"][-1]))
            elif model.type == "GMVAE":
                metrics_string_parts.extend([
                    "    KL_z: {:.5g}.".format(
                        evaluation_eval["kl_divergence_z"][-1]),
                    "    KL_y: {:.5g}.".format(
                        evaluation_eval["kl_divergence_y"][-1])
                ])
        if accuracy_eval is not None:
            metrics_string_parts.append(
                "    Accuracy: {:6.2f} %.".format(100 * accuracy_eval[-1]))
        if superset_accuracy_eval is not None:
            metrics_string_parts.append(
                "    Accuracy (superset): {:6.2f} %.".format(
                    100 * superset_accuracy_eval[-1]))
        metrics_string_parts.append(
            "\n" + format_summary_statistics(evaluation_set_statistics))
        metrics_string = "\n".join(metrics_string_parts) + "\n"

        metrics_dictionary = {
            "timestamp": metrics_saving_time_start,
            "number of epochs trained": number_of_epochs_trained,
            "evaluation": evaluation_eval,
            "accuracy": accuracy_eval,
            "superset_accuracy": superset_accuracy_eval,
            "statistics": evaluation_set_statistics
        }

        with open(metrics_log_path, "w") as metrics_file:
            metrics_file.write(metrics_string)

        with gzip.open(metrics_dictionary_path, "w") as metrics_file:
            pickle.dump(metrics_dictionary, metrics_file)

        if prediction_details:

            prediction_log_filename = "{}-prediction-{}".format(
                evaluation_set.kind, prediction_details["id"])
            prediction_log_path = os.path.join(
                results_directory, prediction_log_filename + ".log")
            prediction_dictionary_path = os.path.join(
                results_directory, prediction_log_filename + ".pkl.gz")

            prediction_string_parts = [
                "Timestamp: {}".format(formatTime(
                    metrics_saving_time_start)),
                "Number of epochs trained: {}".format(
                    number_of_epochs_trained),
                "Prediction method: {}".format(prediction_details["method"])
            ]

            if prediction_details["number_of_classes"]:
                prediction_string_parts.append(
                    "Number of classes: {}".format(
                        prediction_details["number_of_classes"]
                    )
                )

            if prediction_details["training_set_name"]:
                prediction_string_parts.append(
                    "Training set: {}".format(
                        prediction_details["training_set_name"]
                    )
                )

            clustering_metrics_title_printed = False

            for metric_name, metric_set in clustering_metric_values.items():

                clustering_metric_name_printed = False

                for set_name, metric_value in metric_set.items():
                    if metric_value is not None:

                        if not clustering_metrics_title_printed:
                            prediction_string_parts.append(
                                "\nClustering metrics:")
                            clustering_metrics_title_printed = True

                        if not clustering_metric_name_printed:
                            prediction_string_parts.append(
                                "    {}:".format(
                                    capitaliseString(metric_name)
                                )
                            )
                            clustering_metric_name_printed = True

                        prediction_string_parts.append(
                            "        {}: {:.5g}.".format(
                                capitaliseString(set_name), metric_value
                            )
                        )

            prediction_string = "\n".join(prediction_string_parts) + "\n"

            prediction_dictionary = {
                "timestamp": metrics_saving_time_start,
                "number of epochs trained": number_of_epochs_trained,
                "prediction method": prediction_details["method"],
                "number of classes": prediction_details["number_of_classes"],
                "training set": prediction_details["training_set_name"],
                "clustering metric values": clustering_metric_values
            }

            with open(prediction_log_path, "w") as prediction_file:
                prediction_file.write(prediction_string)

            with gzip.open(prediction_dictionary_path, "w") as prediction_file:
                pickle.dump(prediction_dictionary, prediction_file)

        metrics_saving_duration = time() - metrics_saving_time_start
        print("Metrics saved ({}).".format(formatDuration(
            metrics_saving_duration)))
        print()

        print(format_summary_statistics(evaluation_set_statistics))
        print()

        for metric_name, metric_set in clustering_metric_values.items():

            clustering_metric_name_printed = False

            for set_name, metric_value in metric_set.items():
                if metric_value is not None:

                    if not clustering_metric_name_printed:
                        print("{}:".format(capitaliseString(metric_name)))
                        clustering_metric_name_printed = True

                    print("    {}: {:.5g}.".format(
                        capitaliseString(set_name), metric_value
                    ))

            print()

    # Only print subheading if necessary
    if ("images" in analyses
            and reconstructed_evaluation_set.example_type == "images"
            or "profile_comparisons" in analyses):
        print(subheading("Reconstructions"))

    if ("images" in analyses
            and reconstructed_evaluation_set.example_type == "images"):

        print("Saving image of {} random examples".format(
            DEFAULT_NUMBER_OF_RANDOM_EXAMPLES),
            "from reconstructed {} set.".format(
                evaluation_set.kind
            ))
        image_time_start = time()
        image, image_name = combine_images_from_data_set(
            reconstructed_evaluation_set,
            number_of_random_examples=DEFAULT_NUMBER_OF_RANDOM_EXAMPLES,
            name=reconstructed_evaluation_set.version
        )
        save_image(
            image=image,
            name=image_name,
            directory=results_directory
        )
        image_duration = time() - image_time_start
        print("Image saved ({}).".format(formatDuration(image_duration)))
        print()

    if "profile_comparisons" in analyses:

        print("Plotting profile comparisons.")
        profile_comparisons_time_start = time()

        image_comparisons_directory = os.path.join(
            results_directory, "image_comparisons")
        profile_comparisons_directory = os.path.join(
            results_directory, "profile_comparisons")

        y_cutoff = PROFILE_COMPARISON_COUNT_CUT_OFF

        for i in evaluation_subset_indices:

            observed_series = evaluation_set.values[i]
            expected_series = reconstructed_evaluation_set.values[i]
            example_name = str(evaluation_set.example_names[i])

            if evaluation_set.has_labels:
                example_label = str(evaluation_set.labels[i])

            if (reconstructed_evaluation_set.total_standard_deviations
                    is not None):
                expected_series_total_standard_deviations = (
                    reconstructed_evaluation_set.total_standard_deviations[i])
            else:
                expected_series_total_standard_deviations = None

            if (reconstructed_evaluation_set.explained_standard_deviations
                    is not None):
                expected_series_explained_standard_deviations = (
                    reconstructed_evaluation_set.explained_standard_deviations[
                        i])
            else:
                expected_series_explained_standard_deviations = None

            maximum_count = max(observed_series.max(), expected_series.max())

            if evaluation_set.has_labels:
                example_name_base_parts = [example_name, example_label]
            else:
                example_name_base_parts = [example_name]

            for sort_profile_comparison in [True, False]:
                for y_scale in ["linear"]:
                    example_name_parts = example_name_base_parts.copy()
                    example_name_parts.append(y_scale)
                    if sort_profile_comparison:
                        sort_name_part = "sorted"
                    else:
                        sort_name_part = "unsorted"
                    example_name_parts.append(sort_name_part)
                    figure, figure_name = plot_profile_comparison(
                        observed_series,
                        expected_series,
                        expected_series_total_standard_deviations,
                        expected_series_explained_standard_deviations,
                        x_name=evaluation_set.tags["feature"],
                        y_name=evaluation_set.tags["value"],
                        sort=sort_profile_comparison,
                        sort_by="expected",
                        sort_direction="descending",
                        x_scale="log",
                        y_scale=y_scale,
                        name=example_name_parts
                    )
                    save_figure(
                        figure=figure,
                        name=figure_name,
                        options=export_options,
                        directory=profile_comparisons_directory
                    )

            if maximum_count > 3 * y_cutoff:
                for y_scale in ["linear", "log", "both"]:
                    example_name_parts = example_name_base_parts.copy()
                    example_name_parts.append("cutoff")
                    example_name_parts.append(y_scale)
                    figure, figure_name = plot_profile_comparison(
                        observed_series,
                        expected_series,
                        expected_series_total_standard_deviations,
                        expected_series_explained_standard_deviations,
                        x_name=evaluation_set.tags["feature"],
                        y_name=evaluation_set.tags["value"],
                        sort=True,
                        sort_by="expected",
                        sort_direction="descending",
                        x_scale="log",
                        y_scale=y_scale,
                        y_cutoff=y_cutoff,
                        name=example_name_parts
                    )
                    save_figure(
                        figure=figure,
                        name=figure_name,
                        options=export_options,
                        directory=profile_comparisons_directory
                    )

            if evaluation_set.example_type == "images":
                example_name_parts = ["original"] + example_name_base_parts
                image, image_name = combine_images_from_data_set(
                    evaluation_set,
                    indices=[i],
                    name=example_name_parts
                )
                save_image(
                    image=image,
                    name=image_name,
                    directory=image_comparisons_directory
                )

            if reconstructed_evaluation_set.example_type == "images":
                example_name_parts = (
                    ["reconstructed"] + example_name_base_parts)
                image, image_name = combine_images_from_data_set(
                    reconstructed_evaluation_set,
                    indices=[i],
                    name=example_name_parts
                )
                save_image(
                    image=image,
                    name=image_name,
                    directory=image_comparisons_directory
                )

            if analysis_level == "limited":
                break

        profile_comparisons_duration = time() - profile_comparisons_time_start
        print("Profile comparisons plotted and saved ({}).".format(
            formatDuration(profile_comparisons_duration)))
        print()

    if "distributions" in analyses:
        print(subheading("Distributions"))
        analyse_distributions(
            reconstructed_evaluation_set,
            colouring_data_set=evaluation_set,
            preprocessed=evaluation_set.preprocessing_methods,
            analysis_level=analysis_level,
            export_options=export_options,
            results_directory=results_directory
        )

    if "decompositions" in analyses:

        print(subheading("Decompositions"))

        analyse_decompositions(
            reconstructed_evaluation_set,
            colouring_data_set=evaluation_set,
            decomposition_methods=decomposition_methods,
            highlight_feature_indices=highlight_feature_indices,
            prediction_details=prediction_details,
            symbol="\\tilde{{x}}",
            title="reconstruction space",
            analysis_level=analysis_level,
            export_options=export_options,
            results_directory=results_directory,
        )

        if analysis_level == "extensive":
            # Reconstructions plotted in original decomposed space
            analyse_decompositions(
                evaluation_set,
                reconstructed_evaluation_set,
                colouring_data_set=evaluation_set,
                decomposition_methods=decomposition_methods,
                highlight_feature_indices=highlight_feature_indices,
                prediction_details=prediction_details,
                symbol="x",
                title="original space",
                analysis_level=analysis_level,
                export_options=export_options,
                results_directory=results_directory,
            )

    if "heat_maps" in analyses:
        print(subheading("Heat maps"))
        analyse_matrices(
            reconstructed_evaluation_set,
            plot_distances=False,
            results_directory=results_directory
        )
        analyse_matrices(
            latent_evaluation_sets["z"],
            plot_distances=False,
            results_directory=results_directory
        )

        if (analysis_level == "extensive"
                and reconstructed_evaluation_set.number_of_values
                <= MAXIMUM_NUMBER_OF_VALUES_FOR_HEAT_MAPS):

            print("Plotting comparison heat maps.")
            heat_maps_directory = os.path.join(results_directory, "heat_maps")

            # Differences
            heat_maps_time_start = time()
            figure, figure_name = plot_heat_map(
                x_diff,
                labels=reconstructed_evaluation_set.labels,
                x_name=evaluation_set.tags["feature"].capitalize() + "s",
                y_name=evaluation_set.tags["example"].capitalize() + "s",
                z_name="Differences",
                z_symbol="\\tilde{{x}} - x",
                name="difference",
                center=0
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=heat_maps_directory
            )
            heat_maps_duration = time() - heat_maps_time_start
            print(
                "    Difference heat map plotted and saved ({})."
                .format(formatDuration(heat_maps_duration))
            )

            # log-ratios
            heat_maps_time_start = time()
            figure, figure_name = plot_heat_map(
                x_log_ratio,
                labels=reconstructed_evaluation_set.labels,
                x_name=evaluation_set.tags["feature"].capitalize() + "s",
                y_name=evaluation_set.tags["example"].capitalize() + "s",
                z_name="log-ratios",
                z_symbol="\\log \\frac{{\\tilde{{x}} + 1}}{{x + 1}}",
                name="log_ratio",
                center=0
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=heat_maps_directory
            )
            heat_maps_duration = time() - heat_maps_time_start
            print(
                "    log-ratio heat map plotted and saved ({})."
                .format(formatDuration(heat_maps_duration))
            )

    print()

    if "distances" in analyses:
        print(subheading("Distances"))
        analyse_matrices(
            reconstructed_evaluation_set,
            plot_distances=True,
            results_directory=results_directory
        )
        analyse_matrices(
            latent_evaluation_sets["z"],
            plot_distances=True,
            results_directory=results_directory
        )

    if "latent_space" in analyses and "VAE" in model.type:
        print(subheading("Latent space"))

        if model.latent_distribution_name == "gaussian mixture":
            print("Loading centroids from model log directory.")
            loading_time_start = time()
            centroids = loadCentroids(
                model=model,
                data_set_kinds="evaluation",
                run_id=run_id,
                early_stopping=early_stopping,
                best_model=best_model
            )
            loading_duration = time() - loading_time_start
            print("Centroids loaded ({}).".format(
                formatDuration(loading_duration)))
            print()
        else:
            centroids = None

        analyse_decompositions(
            latent_evaluation_sets,
            centroids=centroids,
            colouring_data_set=evaluation_set,
            decomposition_methods=decomposition_methods,
            highlight_feature_indices=highlight_feature_indices,
            prediction_details=prediction_details,
            title="latent space",
            specifier=lambda data_set: data_set.version,
            analysis_level=analysis_level,
            export_options=export_options,
            results_directory=results_directory,
        )

        if centroids:
            analyse_centroid_probabilities(
                centroids,
                analysis_level="normal",
                export_options=export_options,
                results_directory=results_directory
            )
            print()

    if "latent_correlations" in analyses and "VAE" in model.type:

        correlations_directory = os.path.join(
            results_directory, "latent_correlations")
        print(subheading("Latent correlations"))
        print("Plotting latent correlations.")

        for set_name in latent_evaluation_sets:
            correlations_time_start = time()
            latent_evaluation_set = latent_evaluation_sets[set_name]
            figure, figure_name = plot_variable_correlations(
                latent_evaluation_set.values,
                latent_evaluation_set.feature_names,
                latent_evaluation_set,
                name=["latent correlations", set_name]
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=correlations_directory
            )
            correlations_duration = time() - correlations_time_start
            print(
                "    Latent correlations for {} plotted ({})."
                .format(set_name, formatDuration(correlations_duration))
            )

        print()


def analyse_distributions(data_set, colouring_data_set=None, cutoffs=None,
                          preprocessed=False, analysis_level="normal",
                          export_options=None, results_directory="results"):

    if not colouring_data_set:
        colouring_data_set = data_set

    distribution_directory = os.path.join(results_directory, "histograms")

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
        figure, figure_name = plot_class_histogram(
            labels=data_set.labels,
            class_names=data_set.class_names,
            class_palette=data_set.class_palette,
            normed=True,
            scale="linear",
            label_sorter=data_set.label_sorter,
            name=data_set_name
        )
        save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=distribution_directory
        )
        distribution_duration = time() - distribution_time_start
        print("    Class distribution plotted and saved ({}).".format(
            formatDuration(distribution_duration)))

    # Superset class distribution
    if data_set.label_superset and colouring_data_set == data_set:
        distribution_time_start = time()
        figure, figure_name = plot_class_histogram(
            labels=data_set.superset_labels,
            class_names=data_set.superset_class_names,
            class_palette=data_set.superset_class_palette,
            normed=True,
            scale="linear",
            label_sorter=data_set.superset_label_sorter,
            name=[data_set_name, "superset"]
        )
        save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=distribution_directory
        )
        distribution_duration = time() - distribution_time_start
        print("    Superset class distribution plotted and saved ({}).".format(
            formatDuration(distribution_duration)))

    # Count distribution
    if scipy.sparse.issparse(data_set.values):
        series = data_set.values.data
        excess_zero_count = data_set.values.size - series.size
    else:
        series = data_set.values.reshape(-1)
        excess_zero_count = 0
    distribution_time_start = time()
    for x_scale in ["linear", "log"]:
        figure, figure_name = plot_histogram(
            series=series,
            excess_zero_count=excess_zero_count,
            label=data_set.tags["value"].capitalize() + "s",
            discrete=data_set_discreteness,
            normed=True,
            x_scale=x_scale,
            y_scale="log",
            name=["counts", data_set_name]
        )
        save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=distribution_directory
        )
    distribution_duration = time() - distribution_time_start
    print("    Count distribution plotted and saved ({}).".format(
        formatDuration(distribution_duration)))

    # Count distributions with cut-off
    if (analysis_level == "extensive" and cutoffs
            and data_set.example_type == "counts"):
        distribution_time_start = time()
        for cutoff in cutoffs:
            figure, figure_name = plot_cutoff_count_histogram(
                series=series,
                excess_zero_count=excess_zero_count,
                cutoff=cutoff,
                normed=True,
                scale="log",
                name=data_set_name
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=distribution_directory + "-counts"
            )
        distribution_duration = time() - distribution_time_start
        print(
            "    Count distributions with cut-offs plotted and saved ({})."
            .format(formatDuration(distribution_duration))
        )

    # Count sum distribution
    distribution_time_start = time()
    figure, figure_name = plot_histogram(
        series=data_set.count_sum,
        label="Total number of {}s per {}".format(
            data_set.tags["item"], data_set.tags["example"]
        ),
        normed=True,
        y_scale="log",
        name=["count sum", data_set_name]
    )
    save_figure(
        figure=figure,
        name=figure_name,
        options=export_options,
        directory=distribution_directory
    )
    distribution_duration = time() - distribution_time_start
    print("    Count sum distribution plotted and saved ({}).".format(
        formatDuration(distribution_duration)))

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
            index_palette = lighter_palette(
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

            figure, figure_name = plot_histogram(
                series=series,
                excess_zero_count=excess_zero_count,
                label=data_set.tags["value"].capitalize() + "s",
                discrete=data_set_discreteness,
                normed=True,
                y_scale="log",
                colour=class_palette[class_name],
                name=["counts", data_set_name, "class", class_name]
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=class_count_distribution_directory
            )

        distribution_duration = time() - distribution_time_start
        print(
            "    Count distributions for each class plotted and saved ({})."
            .format(formatDuration(distribution_duration))
        )

        distribution_time_start = time()
        for class_name in class_names:

            class_indices = labels == class_name
            if not class_indices.any():
                continue

            figure, figure_name = plot_histogram(
                series=data_set.count_sum[class_indices],
                label="Total number of {}s per {}".format(
                    data_set.tags["item"], data_set.tags["example"]
                ),
                normed=True,
                y_scale="log",
                colour=class_palette[class_name],
                name=["count sum", data_set_name, "class", class_name]
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=class_count_distribution_directory
            )

        distribution_duration = time() - distribution_time_start
        print(
            "    "
            "Count sum distributions for each class plotted and saved ({})."
            .format(formatDuration(distribution_duration))
        )

    print()


def analyse_matrices(data_set, plot_distances=False, name=None,
                     export_options=None, results_directory="results"):

    if plot_distances:
        base_name = "distances"
    else:
        base_name = "heat_maps"

    results_directory = os.path.join(results_directory, base_name)

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
        index_palette = lighter_palette(data_set.number_of_classes)
        class_palette = {
            class_name: tuple(index_palette[i]) for i, class_name in
            enumerate(sorted(data_set.class_names,
                             key=data_set.label_sorter))
        }

    # Axis labels
    example_label = data_set.tags["example"].capitalize() + "s"
    feature_label = data_set.tags["feature"].capitalize() + "s"
    value_label = data_set.tags["value"].capitalize() + "s"

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
                    sample_size, data_set.tags["example"] + "s")

            figure, figure_name = plot_matrix(
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
                label_kind=data_set.tags["class"],
                class_palette=class_palette,
                feature_indices_for_plotting=feature_indices_for_plotting,
                name_parts=name + [
                    data_set.version,
                    distance_metric,
                    sorting_method
                ]
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=results_directory
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
                "({})".format(formatDuration(duration))
            ] if s]) + ".")

    print()


def analyse_decompositions(data_sets, other_data_sets=None, centroids=None,
                           colouring_data_set=None,
                           decomposition_methods=None,
                           highlight_feature_indices=None,
                           prediction_details=None,
                           symbol=None, title="data set", specifier=None,
                           analysis_level="normal", export_options=None,
                           results_directory="results"):

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

    if decomposition_methods is None:
        decomposition_methods = [DEFAULT_DECOMPOSITION_METHOD]
    elif not isinstance(decomposition_methods, (list, tuple)):
        decomposition_methods = [decomposition_methods]
    else:
        decomposition_methods = decomposition_methods.copy()
    decomposition_methods.insert(0, None)

    if highlight_feature_indices is None:
        highlight_feature_indices = []
    elif not isinstance(highlight_feature_indices, (list, tuple)):
        highlight_feature_indices = [highlight_feature_indices]
    else:
        highlight_feature_indices = highlight_feature_indices.copy()

    for data_set, other_data_set in zip(data_sets, other_data_sets):

        if data_set.values.shape[1] <= 1:
            continue

        name = normaliseString(title)

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

        decompositions_directory = os.path.join(results_directory, name)

        for decomposition_method in decomposition_methods:

            if other_data_set:
                other_values = other_data_set.values
            else:
                other_values = None

            if not decomposition_method:
                if data_set.number_of_features == 2:
                    values_decomposed = data_set.values
                    other_values_decomposed = other_values
                    centroids_decomposed = centroids
                else:
                    continue
            else:
                decomposition_method = properString(
                    decomposition_method, DECOMPOSITION_METHOD_NAMES)

                values_decomposed = data_set.values
                other_values_decomposed = other_values
                centroids_decomposed = centroids

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
                            values_decomposed, other_values_decomposed,
                            centroids_decomposed
                        ) = decompose(
                            values_decomposed,
                            other_value_sets=other_values_decomposed,
                            centroids=centroids_decomposed,
                            method="pca",
                            number_of_components=(
                                number_of_pca_components_before_tsne)
                        )
                        decompose_duration = time() - decompose_time_start
                        print("{} pre-decomposed ({}).".format(
                            capitaliseString(title),
                            formatDuration(decompose_duration)
                        ))

                    else:
                        if scipy.sparse.issparse(values_decomposed):
                            values_decomposed = values_decomposed.A
                        if scipy.sparse.issparse(other_values_decomposed):
                            other_values_decomposed = values_decomposed.A

                print("Decomposing {} using {}.".format(
                    title, decomposition_method))
                decompose_time_start = time()
                (
                    values_decomposed, other_values_decomposed,
                    centroids_decomposed
                ) = decompose(
                    values_decomposed,
                    other_value_sets=other_values_decomposed,
                    centroids=centroids_decomposed,
                    method=decomposition_method,
                    number_of_components=2
                )
                decompose_duration = time() - decompose_time_start
                print("{} decomposed ({}).".format(
                    capitaliseString(title),
                    formatDuration(decompose_duration)
                ))
                print()

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
            else:
                plot_values_decomposed = values_decomposed

            if plot_values_decomposed is None:
                print("No values to plot.\n")
                return

            print("Plotting {}{}.".format(
                "decomposed " if decomposition_method else "",
                title
            ))

            # No colour-coding
            plot_time_start = time()
            figure, figure_name = plot_values(
                plot_values_decomposed,
                centroids=centroids_decomposed,
                figure_labels=figure_labels,
                example_tag=data_set.tags["example"],
                name=name
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=decompositions_directory
            )
            plot_duration = time() - plot_time_start
            print("    {} plotted and saved ({}).".format(
                capitaliseString(title),
                formatDuration(plot_duration)
            ))

            # Labels
            if colouring_data_set.labels is not None:
                plot_time_start = time()
                figure, figure_name = plot_values(
                    plot_values_decomposed,
                    colour_coding="labels",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    example_tag=data_set.tags["example"],
                    name=name
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print("    {} (with labels) plotted and saved ({}).".format(
                    capitaliseString(title), formatDuration(plot_duration)))

                # Superset labels
                if colouring_data_set.superset_labels is not None:
                    plot_time_start = time()
                    figure, figure_name = plot_values(
                        plot_values_decomposed,
                        colour_coding="superset labels",
                        colouring_data_set=colouring_data_set,
                        centroids=centroids_decomposed,
                        figure_labels=figure_labels,
                        example_tag=data_set.tags["example"],
                        name=name
                    )
                    save_figure(
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
                            capitaliseString(title),
                            formatDuration(plot_duration)
                        )
                    )

                # For each class
                if analysis_level == "extensive":
                    if colouring_data_set.number_of_classes <= 10:
                        plot_time_start = time()
                        for class_name in colouring_data_set.class_names:
                            figure, figure_name = plot_values(
                                plot_values_decomposed,
                                colour_coding="class",
                                colouring_data_set=colouring_data_set,
                                centroids=centroids_decomposed,
                                class_name=class_name,
                                figure_labels=figure_labels,
                                example_tag=data_set.tags["example"],
                                name=name
                            )
                            save_figure(
                                figure=figure,
                                name=figure_name,
                                options=export_options,
                                directory=decompositions_directory
                            )
                        plot_duration = time() - plot_time_start
                        print(
                            "    {} (for each class) plotted and saved ({})."
                            .format(
                                capitaliseString(title),
                                formatDuration(plot_duration)
                            )
                        )

                    if (colouring_data_set.superset_labels is not None
                            and data_set.number_of_superset_classes <= 10):
                        plot_time_start = time()
                        for superset_class_name in (
                                colouring_data_set.superset_class_names):
                            figure, figure_name = plot_values(
                                plot_values_decomposed,
                                colour_coding="superset class",
                                colouring_data_set=colouring_data_set,
                                centroids=centroids_decomposed,
                                class_name=superset_class_name,
                                figure_labels=figure_labels,
                                example_tag=data_set.tags["example"],
                                name=name
                            )
                            save_figure(
                                figure=figure,
                                name=figure_name,
                                options=export_options,
                                directory=decompositions_directory
                            )
                        plot_duration = time() - plot_time_start
                        print(
                            "    {} (for each superset class) plotted and "
                            "saved ({}).".format(
                                capitaliseString(title),
                                formatDuration(plot_duration)
                            )
                        )

            # Cluster IDs
            if colouring_data_set.has_predicted_cluster_ids:
                plot_time_start = time()
                figure, figure_name = plot_values(
                    plot_values_decomposed,
                    colour_coding="predicted cluster IDs",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    prediction_details=prediction_details,
                    example_tag=data_set.tags["example"],
                    name=name,
                )
                save_figure(
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
                        capitaliseString(title),
                        formatDuration(plot_duration)
                    )
                )

            # Predicted labels
            if colouring_data_set.has_predicted_labels:
                plot_time_start = time()
                figure, figure_name = plot_values(
                    plot_values_decomposed,
                    colour_coding="predicted labels",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    prediction_details=prediction_details,
                    example_tag=data_set.tags["example"],
                    name=name,
                )
                save_figure(
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
                        capitaliseString(title),
                        formatDuration(plot_duration)
                    )
                )

            if colouring_data_set.has_predicted_superset_labels:
                plot_time_start = time()
                figure, figure_name = plot_values(
                    plot_values_decomposed,
                    colour_coding="predicted superset labels",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    prediction_details=prediction_details,
                    example_tag=data_set.tags["example"],
                    name=name,
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print(
                    "    {} (with predicted superset labels) plotted and saved"
                    " ({}).".format(
                        capitaliseString(title),
                        formatDuration(plot_duration)
                    )
                )

            # Count sum
            plot_time_start = time()
            figure, figure_name = plot_values(
                plot_values_decomposed,
                colour_coding="count sum",
                colouring_data_set=colouring_data_set,
                centroids=centroids_decomposed,
                figure_labels=figure_labels,
                example_tag=data_set.tags["example"],
                name=name
            )
            save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=decompositions_directory
            )
            plot_duration = time() - plot_time_start
            print("    {} (with count sum) plotted and saved ({}).".format(
                capitaliseString(title),
                formatDuration(plot_duration)
            ))

            # Features
            for feature_index in highlight_feature_indices:
                plot_time_start = time()
                figure, figure_name = plot_values(
                    plot_values_decomposed,
                    colour_coding="feature",
                    colouring_data_set=colouring_data_set,
                    centroids=centroids_decomposed,
                    feature_index=feature_index,
                    figure_labels=figure_labels,
                    example_tag=data_set.tags["example"],
                    name=name
                )
                save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=decompositions_directory
                )
                plot_duration = time() - plot_time_start
                print("    {} (with {}) plotted and saved ({}).".format(
                    capitaliseString(title),
                    data_set.feature_names[feature_index],
                    formatDuration(plot_duration)
                ))

            print()


def analyse_centroid_probabilities(centroids, name=None,
                                   analysis_level="normal",
                                   export_options=None,
                                   results_directory="results"):

    print("Plotting centroid probabilities.")
    plot_time_start = time()

    if name:
        name = normaliseString(name)

    posterior_probabilities = None
    prior_probabilities = None

    if "posterior" in centroids and centroids["posterior"]:
        posterior_probabilities = centroids["posterior"]["probabilities"]
        n_centroids = len(posterior_probabilities)
    if "prior" in centroids and centroids["prior"]:
        prior_probabilities = centroids["prior"]["probabilities"]
        n_centroids = len(prior_probabilities)

    centroids_palette = darker_palette(n_centroids)
    x_label = "$k$"
    if prior_probabilities is not None:
        if posterior_probabilities is not None:
            y_label = _axis_label_for_symbol(
                symbol="\\pi",
                distribution=normaliseString("posterior"),
                suffix="^k")
            if name:
                plot_name = [name, "posterior", "prior"]
            else:
                plot_name = ["posterior", "prior"]
        else:
            y_label = _axis_label_for_symbol(
                symbol="\\pi",
                distribution=normaliseString("prior"),
                suffix="^k")
            if name:
                plot_name = [name, "prior"]
            else:
                plot_name = "prior"
    elif posterior_probabilities is not None:
        y_label = _axis_label_for_symbol(
            symbol="\\pi",
            distribution=normaliseString("posterior"),
            suffix="^k")
        if name:
            plot_name = [name, "posterior"]
        else:
            plot_name = "posterior"

    figure, figure_name = plot_probabilities(
        posterior_probabilities,
        prior_probabilities,
        x_label=x_label,
        y_label=y_label,
        palette=centroids_palette,
        uniform=False,
        name=plot_name
    )
    save_figure(
        figure=figure,
        name=figure_name,
        options=export_options,
        directory=results_directory
    )

    plot_duration = time() - plot_time_start
    print("Centroid probabilities plotted and saved ({}).".format(
        formatDuration(plot_duration)))


def indices_for_evaluation_subset(evaluation_set,
                                  maximum_number_of_examples_per_class=None,
                                  total_maximum_number_of_examples=None):

    if maximum_number_of_examples_per_class is None:
        maximum_number_of_examples_per_class = (
            EVALUATION_SUBSET_MAXIMUM_NUMBER_OF_EXAMPLES_PER_CLASS)

    if total_maximum_number_of_examples is None:
        total_maximum_number_of_examples = (
            EVALUATION_SUBSET_MAXIMUM_NUMBER_OF_EXAMPLES)

    random_state = numpy.random.RandomState(80)

    if evaluation_set.has_labels:

        if evaluation_set.label_superset:
            class_names = evaluation_set.superset_class_names
            labels = evaluation_set.superset_labels
        else:
            class_names = evaluation_set.class_names
            labels = evaluation_set.labels

        subset = set()

        for class_name in class_names:
            class_label_indices = numpy.argwhere(labels == class_name)
            random_state.shuffle(class_label_indices)
            subset.update(
                *class_label_indices[:maximum_number_of_examples_per_class])

    else:
        subset = numpy.random.permutation(evaluation_set.number_of_examples)[
            :total_maximum_number_of_examples]
        subset = set(subset)

    return subset


def summary_statistics(x, name="", tolerance=1e-3, skip_sparsity=False):

    batch_size = None

    if x.size > MAXIMUM_SIZE_FOR_NORMAL_STATISTICS_COMPUTATION:
        batch_size = 1000

    x_mean = x.mean()
    x_std = standard_deviation(x, ddof=1, batch_size=batch_size)

    x_min = x.min()
    x_max = x.max()

    x_dispersion = x_std**2 / x_mean

    if skip_sparsity:
        x_sparsity = numpy.nan
    else:
        x_sparsity = sparsity(x, tolerance=tolerance, batch_size=batch_size)

    statistics = {
        "name": name,
        "mean": x_mean,
        "standard deviation": x_std,
        "minimum": x_min,
        "maximum": x_max,
        "dispersion": x_dispersion,
        "sparsity": x_sparsity
    }

    return statistics


def format_summary_statistics(statistics_sets, name="Data set"):

    if not isinstance(statistics_sets, list):
        statistics_sets = [statistics_sets]

    name_width = max(
        [len(name)]
        + [len(statistics_set["name"]) for statistics_set in statistics_sets]
    )

    table_heading = "  ".join([
        "{:{}}".format(name, name_width),
        " mean ", "std. dev. ", "dispersion",
        " minimum ", " maximum ", "sparsity"
    ])

    table_rows = [table_heading]

    for statistics_set in statistics_sets:
        table_row_parts = [
            "{:{}}".format(statistics_set["name"], name_width),
            "{:<9.5g}".format(statistics_set["mean"]),
            "{:<9.5g}".format(statistics_set["standard deviation"]),
            "{:<9.5g}".format(statistics_set["dispersion"]),
            "{:<11.5g}".format(statistics_set["minimum"]),
            "{:<11.5g}".format(statistics_set["maximum"]),
            "{:<7.5g}".format(statistics_set["sparsity"]),
        ]
        table_row = "  ".join(table_row_parts)
        table_rows.append(table_row)

    table = "\n".join(table_rows)

    return table


def _exclude_classes_from_label_set(*label_sets, excluded_classes=None):

    if excluded_classes is None:
        excluded_classes = []

    labels = label_sets[0]
    other_label_sets = list(label_sets[1:])

    for excluded_class in excluded_classes:
        included_indices = labels != excluded_class
        labels = labels[included_indices]
        for i in range(len(other_label_sets)):
            other_label_sets[i] = other_label_sets[i][included_indices]

    if other_label_sets:
        return [labels] + other_label_sets
    else:
        return labels


def accuracy(labels, predicted_labels, excluded_classes=None):
    labels, predicted_labels = _exclude_classes_from_label_set(
        labels, predicted_labels, excluded_classes=excluded_classes)
    return numpy.mean(predicted_labels == labels)


def _register_clustering_metric(name, kind):
    def decorator(function):
        CLUSTERING_METRICS[name] = {
            "kind": kind,
            "function": function
        }
        return function
    return decorator


@_register_clustering_metric(name="adjusted Rand index", kind="supervised")
def adjusted_rand_index(labels, predicted_labels, excluded_classes=None):
    labels, predicted_labels = _exclude_classes_from_label_set(
        labels, predicted_labels, excluded_classes=excluded_classes)
    return sklearn.metrics.cluster.adjusted_rand_score(
        labels, predicted_labels)


@_register_clustering_metric(
    name="adjusted mutual information", kind="supervised")
def adjusted_mutual_information(labels, predicted_labels,
                                excluded_classes=None):
    labels, predicted_labels = _exclude_classes_from_label_set(
        labels, predicted_labels, excluded_classes=excluded_classes)
    return sklearn.metrics.cluster.adjusted_mutual_info_score(
        labels, predicted_labels)


@_register_clustering_metric(name="silhouette score", kind="unsupervised")
def silhouette_score(values, predicted_labels):
    number_of_predicted_classes = numpy.unique(predicted_labels).shape[0]
    number_of_examples = values.shape[0]

    if (number_of_predicted_classes < 2
            or number_of_predicted_classes > number_of_examples - 1):
        return numpy.nan

    sample_size = None

    if (number_of_examples
            > MAXIMUM_NUMBER_OF_EXAMPLES_BEFORE_SAMPLING_SILHOUETTE_SCORE):
        sample_size = (
            MAXIMUM_NUMBER_OF_EXAMPLES_BEFORE_SAMPLING_SILHOUETTE_SCORE)

    score = sklearn.metrics.silhouette_score(
        X=values,
        labels=predicted_labels,
        sample_size=sample_size
    )

    return score


def compute_clustering_metrics(evaluation_set):

    clustering_metric_values = {
        metric: {
            "clusters": None,
            "clusters; superset": None,
            "labels": None,
            "labels; superset": None
        }
        for metric in CLUSTERING_METRICS
    }

    for metric_name, metric_attributes in CLUSTERING_METRICS.items():

        metric_values = clustering_metric_values[metric_name]
        metric_kind = metric_attributes["kind"]
        metric_function = metric_attributes["function"]

        if metric_kind == "supervised":
            if evaluation_set.has_labels:
                if evaluation_set.has_predicted_cluster_ids:
                    metric_values["clusters"] = metric_function(
                        evaluation_set.labels,
                        evaluation_set.predicted_cluster_ids,
                        evaluation_set.excluded_classes
                    )
                if evaluation_set.has_predicted_labels:
                    metric_values["labels"] = metric_function(
                        evaluation_set.labels,
                        evaluation_set.predicted_labels,
                        evaluation_set.excluded_classes
                    )
            if evaluation_set.has_superset_labels:
                if evaluation_set.has_predicted_cluster_ids:
                    metric_values["clusters; superset"] = metric_function(
                        evaluation_set.superset_labels,
                        evaluation_set.predicted_cluster_ids,
                        evaluation_set.excluded_superset_classes
                    )
                if evaluation_set.has_predicted_superset_labels:
                    metric_values["labels; superset"] = metric_function(
                        evaluation_set.superset_labels,
                        evaluation_set.predicted_superset_labels,
                        evaluation_set.excluded_superset_classes
                    )
        elif metric_kind == "unsupervised":
            if evaluation_set.has_predicted_cluster_ids:
                metric_values["clusters"] = metric_function(
                    evaluation_set.values,
                    evaluation_set.predicted_cluster_ids
                )
            if evaluation_set.has_predicted_labels:
                metric_values["labels"] = metric_function(
                    evaluation_set.values,
                    evaluation_set.predicted_labels
                )
            if evaluation_set.has_predicted_superset_labels:
                metric_values["labels; superset"] = metric_function(
                    evaluation_set.values,
                    evaluation_set.predicted_superset_labels
                )

    return clustering_metric_values


def plot_class_histogram(labels, class_names=None, class_palette=None,
                         normed=False, scale="linear", label_sorter=None,
                         name=None):

    figure_name = "histogram"

    if normed:
        figure_name += "-normed"

    figure_name += "-classes"

    figure_name = build_figure_name(figure_name, name)

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    if class_names is None:
        class_names = numpy.unique(labels)

    n_classes = len(class_names)

    if not class_palette:
        index_palette = lighter_palette(n_classes)
        class_palette = {
            class_name: index_palette[i] for i, class_name in enumerate(sorted(
                class_names, key=label_sorter))
        }

    histogram = {
        class_name: {
            "index": i,
            "count": 0,
            "colour": class_palette[class_name]
        }
        for i, class_name in enumerate(sorted(
            class_names, key=label_sorter))
    }

    total_count_sum = 0

    for label in labels:
        histogram[label]["count"] += 1
        total_count_sum += 1

    indices = []
    class_names = []

    for class_name, class_values in sorted(histogram.items()):
        index = class_values["index"]
        count = class_values["count"]
        frequency = count / total_count_sum
        colour = class_values["colour"]
        indices.append(index)
        class_names.append(class_name)

        if normed:
            count_or_frequecny = frequency
        else:
            count_or_frequecny = count

        axis.bar(index, count_or_frequecny, color=colour)

    axis.set_yscale(scale)

    maximum_class_name_width = max([
        len(str(class_name)) for class_name in class_names
        if class_name not in ["No class"]
    ])
    if maximum_class_name_width > 5:
        y_ticks_rotation = 45
        y_ticks_horizontal_alignment = "right"
        y_ticks_rotation_mode = "anchor"
    else:
        y_ticks_rotation = 0
        y_ticks_horizontal_alignment = "center"
        y_ticks_rotation_mode = None
    pyplot.xticks(
        ticks=indices,
        labels=class_names,
        horizontalalignment=y_ticks_horizontal_alignment,
        rotation=y_ticks_rotation,
        rotation_mode=y_ticks_rotation_mode
    )

    axis.set_xlabel("Classes")

    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")

    return figure, figure_name


def plot_histogram(series, excess_zero_count=0, label=None, normed=False,
                   discrete=False, x_scale="linear", y_scale="linear",
                   colour=None, name=None):

    series = series.copy()

    figure_name = "histogram"

    if normed:
        figure_name += "-normed"

    figure_name = build_figure_name(figure_name, name)

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    series_length = len(series) + excess_zero_count

    series_max = series.max()

    if discrete and series_max < MAXIMUM_NUMBER_OF_BINS_FOR_HISTOGRAMS:
        number_of_bins = int(numpy.ceil(series_max)) + 1
        bin_range = numpy.array((-0.5, series_max + 0.5))
    else:
        if series_max < MAXIMUM_NUMBER_OF_BINS_FOR_HISTOGRAMS:
            number_of_bins = "auto"
        else:
            number_of_bins = MAXIMUM_NUMBER_OF_BINS_FOR_HISTOGRAMS
        bin_range = numpy.array((series.min(), series_max))

    if colour is None:
        colour = STANDARD_PALETTE[0]

    if x_scale == "log":
        series += 1
        bin_range += 1
        label += " (shifted one)"
        figure_name += "-log_values"

    y_log = y_scale == "log"

    histogram, bin_edges = numpy.histogram(
        series,
        bins=number_of_bins,
        range=bin_range
    )

    histogram[0] += excess_zero_count

    width = bin_edges[1] - bin_edges[0]
    bin_centres = bin_edges[:-1] + width / 2

    if normed:
        histogram = histogram / series_length

    axis.bar(
        bin_centres,
        histogram,
        width=width,
        log=y_log,
        color=colour,
        alpha=0.4
    )

    axis.set_xscale(x_scale)
    axis.set_xlabel(capitaliseString(label))

    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")

    return figure, figure_name


def plot_cutoff_count_histogram(series, excess_zero_count=0, cutoff=None,
                                normed=False, scale="linear", colour=None,
                                name=None):

    series = series.copy()

    figure_name = "histogram"

    if normed:
        figure_name += "-normed"

    figure_name += "-counts"
    figure_name = build_figure_name(figure_name, name)
    figure_name += "-cutoff-{}".format(cutoff)

    if not colour:
        colour = STANDARD_PALETTE[0]

    y_log = scale == "log"

    count_number = numpy.arange(cutoff + 1)
    # Array to count counts of a given count number
    count_number_count = numpy.zeros(cutoff + 1)

    for i in range(cutoff + 1):
        if count_number[i] < cutoff:
            c = (series == count_number[i]).sum()
        elif count_number[i] == cutoff:
            c = (series >= cutoff).sum()
        count_number_count[i] = c

    count_number_count[0] += excess_zero_count

    if normed:
        count_number_count /= count_number_count.sum()

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.bar(
        count_number,
        count_number_count,
        log=y_log,
        color=colour,
        alpha=0.4
    )

    axis.set_xlabel("Count bins")

    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")

    return figure, figure_name


def plot_series(series, x_label, y_label, sort=False, scale="linear",
                bar=False, colour=None, name=None):

    figure_name = build_figure_name("series", name)

    if not colour:
        colour = STANDARD_PALETTE[0]

    series_length = series.shape[0]

    x = numpy.linspace(0, series_length, series_length)

    y_log = scale == "log"

    if sort:
        # Sort descending
        series = numpy.sort(series)[::-1]
        x_label = "sorted " + x_label
        figure_name += "-sorted"

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    if bar:
        axis.bar(x, series, log=y_log, color=colour, alpha=0.4)
    else:
        axis.plot(x, series, color=colour)
        axis.set_yscale(scale)

    axis.set_xlabel(capitaliseString(x_label))
    axis.set_ylabel(capitaliseString(y_label))

    return figure, figure_name


def plot_learning_curves(curves, model_type, epoch_offset=0, name=None):

    figure_name = "learning_curves"
    figure_name = build_figure_name("learning_curves", name)

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
            return STANDARD_PALETTE[len(curves) * i + colour_index_offset]

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
    figure_name = build_figure_name("learning_curves", names)

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
            return STANDARD_PALETTE[len(curves) * i + colour_index_offset]

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


def plot_accuracies(accuracies, name=None):

    figure_name = build_figure_name("accuracies", name)
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    for accuracies_kind, accuracies in sorted(accuracies.items()):
        if accuracies is None:
            continue
        elif accuracies_kind == "training":
            line_style = "solid"
            colour = STANDARD_PALETTE[0]
        elif accuracies_kind == "validation":
            line_style = "dashed"
            colour = STANDARD_PALETTE[1]

        label = "{} set".format(capitaliseString(accuracies_kind))
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

    figure_name = build_figure_name("kl_divergence_evolution", name)
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
        cbar=True, cbar_kws=cbar_dict, cmap=STANDARD_COLOUR_MAP,
        ax=axis
    )

    axis.set_xlabel("Epochs")
    axis.set_ylabel("$i$")

    seaborn.despine(ax=axis)

    return figure, figure_name


def plot_evolution_of_centroid_probabilities(probabilities, distribution,
                                             linestyle="solid", name=None):

    distribution = normaliseString(distribution)

    y_label = _axis_label_for_symbol(
        symbol="\\pi",
        distribution=distribution,
        suffix="^k"
    )

    figure_name = "centroids_evolution-{}-probabilities".format(distribution)
    figure_name = build_figure_name(figure_name, name)

    n_epochs, n_centroids = probabilities.shape

    centroids_palette = darker_palette(n_centroids)
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


def plot_evolution_of_centroid_means(means, distribution, decomposed=False,
                                     name=None):

    symbol = "\\mu"
    if decomposed:
        decomposition_method = "PCA"
    else:
        decomposition_method = ""
    distribution = normaliseString(distribution)
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
    figure_name = build_figure_name(figure_name, name)

    n_epochs, n_centroids, latent_size = means.shape

    if latent_size > 2:
        raise ValueError("Dimensions of means should be 2.")

    centroids_palette = darker_palette(n_centroids)
    epochs = numpy.arange(n_epochs) + 1

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    colour_bar_scatter_plot = axis.scatter(
        means[:, 0, 0], means[:, 0, 1], c=epochs,
        cmap=seaborn.dark_palette(NEUTRAL_COLOUR, as_cmap=True),
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


def plot_evolution_of_centroid_covariance_matrices(
        covariance_matrices, distribution, name=None):

    distribution = normaliseString(distribution)
    figure_name = "centroids_evolution-{}-covariance_matrices".format(
        distribution)
    figure_name = build_figure_name(figure_name, name)

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

    centroids_palette = darker_palette(n_centroids)
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


def plot_profile_comparison(observed_series, expected_series,
                            expected_series_total_standard_deviations=None,
                            expected_series_explained_standard_deviations=None,
                            x_name="feature", y_name="value", sort=True,
                            sort_by="expected", sort_direction="ascending",
                            x_scale="linear", y_scale="linear", y_cutoff=None,
                            name=None):

    sort_by = normaliseString(sort_by)
    sort_direction = normaliseString(sort_direction)
    figure_name = build_figure_name("profile_comparison", name)

    if scipy.sparse.issparse(observed_series):
        observed_series = observed_series.A.squeeze()

    if scipy.sparse.issparse(expected_series_total_standard_deviations):
        expected_series_total_standard_deviations = (
            expected_series_total_standard_deviations.A.squeeze())

    if scipy.sparse.issparse(expected_series_explained_standard_deviations):
        expected_series_explained_standard_deviations = (
            expected_series_explained_standard_deviations.A.squeeze())

    observed_colour = STANDARD_PALETTE[0]

    expected_palette = seaborn.light_palette(STANDARD_PALETTE[1], 5)

    expected_colour = expected_palette[-1]
    expected_total_standard_deviations_colour = expected_palette[1]
    expected_explained_standard_deviations_colour = expected_palette[3]

    if sort:
        x_label = "{}s sorted {} by {} {}s [sort index]".format(
            capitaliseString(x_name), sort_direction, sort_by, y_name.lower())
    else:
        x_label = "{}s [original index]".format(capitaliseString(x_name))
    y_label = capitaliseString(y_name) + "s"

    observed_label = "Observed"
    expected_label = "Expected"
    expected_total_standard_deviations_label = "Total standard deviation"
    expected_explained_standard_deviations_label = (
        "Explained standard deviation")

    # Sorting
    if sort_by == "expected":
        sort_series = expected_series
        expected_marker = ""
        expected_line_style = "solid"
        expected_z_order = 3
        observed_marker = "o"
        observed_line_style = ""
        observed_z_order = 2
    elif sort_by == "observed":
        sort_series = observed_series
        expected_marker = "o"
        expected_line_style = ""
        expected_z_order = 2
        observed_marker = ""
        observed_line_style = "solid"
        observed_z_order = 3

    if sort:
        sort_indices = numpy.argsort(sort_series)
        if sort_direction == "descending":
            sort_indices = sort_indices[::-1]
        elif sort_direction != "ascending":
            raise ValueError(
                "Sort direction can either be ascending or descending.")
    else:
        sort_indices = slice(None)

    # Standard deviations
    if expected_series_total_standard_deviations is not None:
        with_total_standard_deviations = True
        expected_series_total_standard_deviations_lower = (
            expected_series - expected_series_total_standard_deviations)
        expected_series_total_standard_deviations_upper = (
            expected_series + expected_series_total_standard_deviations)
    else:
        with_total_standard_deviations = False

    if (expected_series_explained_standard_deviations is not None
            and expected_series_explained_standard_deviations.mean() > 0):
        with_explained_standard_deviations = True
        expected_series_explained_standard_deviations_lower = (
            expected_series - expected_series_explained_standard_deviations)
        expected_series_explained_standard_deviations_upper = (
            expected_series + expected_series_explained_standard_deviations)
    else:
        with_explained_standard_deviations = False

    # Figure
    if y_scale == "both":
        figure, axes = pyplot.subplots(nrows=2, sharex=True)
        figure.subplots_adjust(hspace=0.1)
        axis_upper = axes[0]
        axis_lower = axes[1]
        axis_upper.set_zorder = 1
        axis_lower.set_zorder = 0
    else:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        axes = [axis]

    handles = []
    feature_indices = numpy.arange(len(observed_series)) + 1

    for i, axis in enumerate(axes):
        observed_plot, = axis.plot(
            feature_indices,
            observed_series[sort_indices],
            label=observed_label,
            color=observed_colour,
            marker=observed_marker,
            linestyle=observed_line_style,
            zorder=observed_z_order
        )
        if i == 0:
            handles.append(observed_plot)
        expected_plot, = axis.plot(
            feature_indices,
            expected_series[sort_indices],
            label=expected_label,
            color=expected_colour,
            marker=expected_marker,
            linestyle=expected_line_style,
            zorder=expected_z_order
        )
        if i == 0:
            handles.append(expected_plot)
        if with_total_standard_deviations:
            axis.fill_between(
                feature_indices,
                expected_series_total_standard_deviations_lower[sort_indices],
                expected_series_total_standard_deviations_upper[sort_indices],
                color=expected_total_standard_deviations_colour,
                zorder=0
            )
            expected_plot_standard_deviations_values = (
                matplotlib.patches.Patch(
                    label=expected_total_standard_deviations_label,
                    color=expected_total_standard_deviations_colour
                )
            )
            if i == 0:
                handles.append(expected_plot_standard_deviations_values)
        if with_explained_standard_deviations:
            axis.fill_between(
                feature_indices,
                expected_series_explained_standard_deviations_lower[
                    sort_indices],
                expected_series_explained_standard_deviations_upper[
                    sort_indices],
                color=expected_explained_standard_deviations_colour,
                zorder=1
            )
            expected_plot_standard_deviations_expectations = (
                matplotlib.patches.Patch(
                    label=expected_explained_standard_deviations_label,
                    color=expected_explained_standard_deviations_colour
                )
            )
            if i == 0:
                handles.append(expected_plot_standard_deviations_expectations)

    if y_scale == "both":
        axis_upper.legend(
            handles=handles,
            loc="best"
        )
        seaborn.despine(ax=axis_upper)
        seaborn.despine(ax=axis_lower)

        axis_upper.set_yscale("log", nonposy="clip")
        axis_lower.set_yscale("linear")
        figure.text(0.04, 0.5, y_label, va="center", rotation="vertical")

        axis_lower.set_xscale(x_scale)
        axis_lower.set_xlabel(x_label)

        y_upper_min, y_upper_max = axis_upper.get_ylim()
        y_lower_min, y_lower_max = axis_lower.get_ylim()
        axis_upper.set_ylim(y_cutoff, y_upper_max)

        y_lower_min = max(-1, y_lower_min)
        axis_lower.set_ylim(y_lower_min, y_cutoff)

    else:
        axis.legend(
            handles=handles,
            loc="best"
        )
        seaborn.despine()

        axis.set_yscale(y_scale, nonposy="clip")
        axis.set_ylabel(y_label)

        axis.set_xscale(x_scale)
        axis.set_xlabel(x_label)

        y_min, y_max = axis.get_ylim()
        y_min = max(-1, y_min)

        if y_cutoff:
            if y_scale == "linear":
                y_max = y_cutoff
            elif y_scale == "log":
                y_min = y_cutoff

        axis.set_ylim(y_min, y_max)

    return figure, figure_name


def plot_heat_map(values, x_name, y_name, z_name=None, z_symbol=None,
                  z_min=None, z_max=None, symmetric=False, labels=None,
                  label_kind=None, center=None, name=None):

    figure_name = build_figure_name("heat_map", name)
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
        cbar=True, cbar_kws=cbar_dict, cmap=STANDARD_COLOUR_MAP,
        square=square_cells,
        ax=axis
    )
    reset_plot_look()

    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)

    return figure, figure_name


def plot_elbo_heat_map(data_frame, x_label, y_label, z_label=None,
                       z_symbol=None, z_min=None, z_max=None, name=None):

    figure_name = build_figure_name("ELBO_heat_map", name)
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)

    if not z_min:
        z_min = data_frame.values.min()
    if not z_max:
        z_max = data_frame.values.max()

    if z_symbol:
        z_label = "$" + z_symbol + "$"

    cbar_dict = {}
    if z_label:
        cbar_dict["label"] = z_label

    seaborn.set(style="white")
    seaborn.heatmap(
        data_frame,
        vmin=z_min, vmax=z_max,
        xticklabels=True, yticklabels=True,
        cbar=True, cbar_kws=cbar_dict,
        annot=True,
        fmt="-.6g",
        square=False,
        ax=axis
    )
    reset_plot_look()

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    return figure, figure_name


def plot_matrix(feature_matrix, plot_distances=False, center_value=None,
                example_label=None, feature_label=None, value_label=None,
                sorting_method=None, distance_metric="Euclidean",
                labels=None, label_kind=None, class_palette=None,
                feature_indices_for_plotting=None, hide_dendrogram=False,
                name_parts=None):

    figure_name = build_figure_name(name_parts)
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

    reset_plot_look()

    # Axis labels
    if example_label:
        left_most_axis.set_ylabel(example_label)

    if feature_label:
        axis_heat_map.set_xlabel(feature_label)

    return figure, figure_name


def plot_variable_correlations(values, variable_names=None,
                               colouring_data_set=None,
                               name="variable_correlations"):

    figure_name = build_figure_name(name)
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
            index_palette = lighter_palette(number_of_classes)
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
        colours = NEUTRAL_COLOUR

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


def plot_correlations(correlation_sets, x_key, y_key,
                      x_label=None, y_label=None, name=None):

    figure_name = build_figure_name("correlations", name)

    if not isinstance(correlation_sets, dict):
        correlation_sets = {"correlations": correlation_sets}

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    for correlation_set_name, correlation_set in correlation_sets.items():
        axis.scatter(
            correlation_set[x_key], correlation_set[y_key],
            label=correlation_set_name
        )

    if len(correlation_sets) > 1:
        axis.legend(loc="best")

    return figure, figure_name


def plot_model_metrics(metrics_sets, key, label=None,
                       primary_differentiator_key=None,
                       primary_differentiator_order=None,
                       secondary_differentiator_key=None,
                       secondary_differentiator_order=None,
                       palette=None, marker_styles=None, name=None):

    figure_name = build_figure_name("model_metrics", name)

    if not isinstance(metrics_sets, list):
        metrics_sets = [metrics_sets]

    if not palette:
        palette = STANDARD_PALETTE.copy()

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.set_xlabel(primary_differentiator_key.capitalize() + "s")
    axis.set_ylabel(label)

    x_positions = {}
    x_offsets = {}
    colours = {}

    x_gap = 3
    x_scale = len(secondary_differentiator_order) - 1 + 2 * x_gap

    for metrics_set in metrics_sets:

        y = numpy.array(metrics_set[key])

        if y.dtype == "object":
            continue

        y_mean = y.mean()
        y_ddof = 1 if y.size > 1 else 0
        y_sd = y.std(ddof=y_ddof)

        x_position_key = metrics_set[primary_differentiator_key]
        if x_position_key in x_positions:
            x_position = x_positions[x_position_key]
        else:
            try:
                index = primary_differentiator_order.index(x_position_key)
                x_position = index
            except (ValueError, IndexError):
                x_position = 0
            x_positions[x_position_key] = x_position

        x_offset_key = metrics_set[secondary_differentiator_key]
        if x_offset_key in x_offsets:
            x_offset = x_offsets[x_offset_key]
        else:
            try:
                index = secondary_differentiator_order.index(x_offset_key)
                x_offset = (index + x_gap - x_scale / 2) / x_scale
            except (ValueError, IndexError):
                x_offset = 0
            x_offsets[x_offset_key] = x_offset

        x = x_position + x_offset

        colour_key = x_offset_key
        if colour_key in colours:
            colour = colours[colour_key]
        else:
            try:
                index = secondary_differentiator_order.index(colour_key)
                colour = palette[index]
            except (ValueError, IndexError):
                colour = "black"
            colours[colour_key] = colour
            axis.errorbar(
                x=x,
                y=y_mean,
                yerr=y_sd,
                capsize=2,
                linestyle="",
                color=colour,
                label=colour_key
            )

        axis.errorbar(
            x=x,
            y=y_mean,
            yerr=y_sd,
            ecolor=colour,
            capsize=2,
            color=colour,
            marker="_",
            # markeredgecolor=darker_colour,
            # markersize=7
        )

    x_ticks = []
    x_tick_labels = []

    for model, x_position in x_positions.items():
        x_ticks.append(x_position)
        x_tick_labels.append(model)

    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_tick_labels)

    if len(metrics_sets) > 1:

        order = primary_differentiator_order + secondary_differentiator_order
        handles, labels = axis.get_legend_handles_labels()

        label_handles = {}

        for label, handle in zip(labels, handles):
            label_handles.setdefault(label, [])
            label_handles[label].append(handle)

        labels, handles = [], []

        for label, handle_set in label_handles.items():
            labels.append(label)
            handles.append(tuple(handle_set))

        labels, handles = zip(*sorted(
            zip(labels, handles),
            key=lambda l:
                [order.index(l[0]), l[0]] if l[0] in order
                else [len(order), l[0]]
        ))

        axis.legend(handles, labels, loc="best")

    return figure, figure_name


def plot_model_metric_sets(metrics_sets, x_key, y_key,
                           x_label=None, y_label=None,
                           primary_differentiator_key=None,
                           primary_differentiator_order=None,
                           secondary_differentiator_key=None,
                           secondary_differentiator_order=None,
                           special_cases=None, other_method_metrics=None,
                           palette=None, marker_styles=None, name=None):

    figure_name = build_figure_name("model_metric_sets", name)

    if other_method_metrics:
        figure_name += "-other_methods"

    if not isinstance(metrics_sets, list):
        metrics_sets = [metrics_sets]

    if not palette:
        palette = STANDARD_PALETTE.copy()

    if not marker_styles:
        marker_styles = [
            "X",  # cross
            "s",  # square
            "D",  # diamond
            "o",  # circle
            "P",  # plus
            "^",  # upright triangle
            "p",  # pentagon
            "*",  # star
        ]

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    colours = {}
    markers = {}

    for metrics_set in metrics_sets:

        x = numpy.array(metrics_set[x_key])
        y = numpy.array(metrics_set[y_key])

        if x.dtype == "object" or y.dtype == "object":
            continue

        x_mean = x.mean()
        x_ddof = 1 if x.size > 1 else 0
        x_sd = x.std(ddof=x_ddof)

        y_mean = y.mean()
        y_ddof = 1 if y.size > 1 else 0
        y_sd = y.std(ddof=y_ddof)

        colour_key = metrics_set[primary_differentiator_key]
        if colour_key in colours:
            colour = colours[colour_key]
        else:
            try:
                index = primary_differentiator_order.index(colour_key)
                colour = palette[index]
            except (ValueError, IndexError):
                colour = "black"
            colours[colour_key] = colour
            axis.errorbar(
                x=x_mean,
                y=y_mean,
                yerr=y_sd,
                xerr=x_sd,
                capsize=2,
                linestyle="",
                color=colour,
                label=colour_key,
                markersize=7
            )

        marker_key = metrics_set[secondary_differentiator_key]
        if marker_key in markers:
            marker = markers[marker_key]
        else:
            try:
                index = secondary_differentiator_order.index(marker_key)
                marker = marker_styles[index]
            except (ValueError, IndexError):
                marker = None
            markers[marker_key] = marker
            axis.errorbar(
                x_mean, y_mean,
                color="black", marker=marker, linestyle="none",
                label=marker_key
            )

        errorbar_colour = colour
        darker_colour = seaborn.dark_palette(colour, n_colors=4)[2]

        special_case_changes = special_cases.get(colour_key, {})
        special_case_changes.update(
            special_cases.get(marker_key, {})
        )

        for object_name, object_change in special_case_changes.items():

            if object_name == "errorbar_colour":
                if object_change == "darken":
                    errorbar_colour = darker_colour

        axis.errorbar(
            x=x_mean,
            y=y_mean,
            yerr=y_sd,
            xerr=x_sd,
            ecolor=errorbar_colour,
            capsize=2,
            color=colour,
            marker=marker,
            markeredgecolor=darker_colour,
            markersize=7
        )

    baseline_line_styles = [
        "dashed",
        "dotted",
        "dashdotted"
    ]

    if other_method_metrics:
        for method_name, metric_values in other_method_metrics.items():

            y_values = metric_values.get(y_key, None)

            if not y_values:
                continue

            y = numpy.array(y_values)

            y_mean = y.mean()

            if y.shape[0] > 1:
                y_sd = y.std(ddof=1)
            else:
                y_sd = None

            line_style = baseline_line_styles.pop(0)

            axis.axhline(
                y=y_mean,
                color=STANDARD_PALETTE[-1],
                linestyle=line_style,
                label=method_name,
                zorder=-1
            )

            if y_sd is not None:
                axis.axhspan(
                    ymin=y_mean - y_sd,
                    ymax=y_mean + y_sd,
                    facecolor=STANDARD_PALETTE[-1],
                    alpha=0.4,
                    edgecolor=None,
                    label=method_name,
                    zorder=-2
                )

    if len(metrics_sets) > 1:

        order = primary_differentiator_order + secondary_differentiator_order
        handles, labels = axis.get_legend_handles_labels()

        label_handles = {}

        for label, handle in zip(labels, handles):
            label_handles.setdefault(label, [])
            label_handles[label].append(handle)

        labels, handles = [], []

        for label, handle_set in label_handles.items():
            labels.append(label)
            handles.append(tuple(handle_set))

        labels, handles = zip(*sorted(
            zip(labels, handles),
            key=lambda l:
                [order.index(l[0]), l[0]] if l[0] in order
                else [len(order), l[0]]
        ))

        axis.legend(handles, labels, loc="best")

    return figure, figure_name


def plot_values(values, colour_coding=None, colouring_data_set=None,
                centroids=None, class_name=None, feature_index=None,
                figure_labels=None, prediction_details=None,
                example_tag=None, name="scatter"):

    figure_name = name

    if figure_labels:
        title = figure_labels["title"]
        x_label = figure_labels["x label"]
        y_label = figure_labels["y label"]
    else:
        title = "none"
        x_label = "$x$"
        y_label = "$y$"

    if not title:
        title = "none"

    figure_name += "-" + normaliseString(title)

    if colour_coding:
        colour_coding = normaliseString(colour_coding)
        figure_name += "-" + colour_coding
        if "predicted" in colour_coding:
            if prediction_details:
                figure_name += "-" + prediction_details["id"]
        if colouring_data_set is None:
            raise ValueError("Colouring data set not given.")

    values = values.copy()[:, :2]
    if scipy.sparse.issparse(values):
        values = values.A

    # Randomise examples in values to remove any prior order
    n_examples, __ = values.shape
    random_state = numpy.random.RandomState(117)
    shuffled_indices = random_state.permutation(n_examples)
    values = values[shuffled_indices]

    # Adjust point size based on number of examples
    if (n_examples
            <= MAXIMUM_NUMBER_OF_EXAMPLES_FOR_LARGE_POINTS_IN_SCATTER_PLOTS):
        marker_size = DEFAULT_LARGE_MARKER_SIZE_IN_SCATTER_PLOTS
    else:
        marker_size = DEFAULT_SMALL_MARKER_SIZE_IN_SCATTER_PLOTS
    _change_point_size_for_plots(marker_size)

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    colour_map = seaborn.dark_palette(STANDARD_PALETTE[0], as_cmap=True)

    if colour_coding and (
            "labels" in colour_coding
            or "ids" in colour_coding
            or "class" in colour_coding):

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
        else:
            labels = colouring_data_set.labels
            class_names = colouring_data_set.class_names
            number_of_classes = colouring_data_set.number_of_classes
            class_palette = colouring_data_set.class_palette
            label_sorter = colouring_data_set.label_sorter

        if not class_palette:
            index_palette = lighter_palette(number_of_classes)
            class_palette = {
                class_name: index_palette[i] for i, class_name in
                enumerate(sorted(class_names, key=label_sorter))
            }

        # Examples are shuffled, so should their labels be
        labels = labels[shuffled_indices]

        if "labels" in colour_coding or "ids" in colour_coding:
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
                        label=label
                    )

            axis.scatter(values[:, 0], values[:, 1], c=colours)

            class_handles, class_labels = axis.get_legend_handles_labels()

            if class_labels:
                class_labels, class_handles = zip(*sorted(
                    zip(class_labels, class_handles),
                    key=lambda t: label_sorter(t[0])
                ))
                class_label_maximum_width = max(*map(len, class_labels))
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
            figure_name += "-" + normaliseString(str(class_name))
            ordered_indices_set = {
                str(class_name): [],
                "Remaining": []
            }

            for i, label in enumerate(labels):
                if label == class_name:
                    colour = class_palette[label]
                    ordered_indices_set[str(class_name)].append(i)
                else:
                    colour = NEUTRAL_COLOUR
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
                    zorder=z_order
                )

                handles, labels = axis.get_legend_handles_labels()
                labels, handles = zip(*sorted(
                    zip(labels, handles),
                    key=lambda t: label_sorter(t[0])
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
            cmap=colour_map
        )
        colour_bar = figure.colorbar(scatter_plot)
        colour_bar.outline.set_linewidth(0)
        colour_bar.set_label("Total number of {}s per {}".format(
            colouring_data_set.tags["item"],
            colouring_data_set.tags["example"]
        ))

    elif colour_coding == "feature":
        if feature_index is None:
            raise ValueError("Feature number not given.")
        if feature_index > colouring_data_set.number_of_features:
            raise ValueError("Feature number higher than number of features.")

        feature_name = colouring_data_set.feature_names[feature_index]
        figure_name += "-{}".format(normaliseString(feature_name))

        f = colouring_data_set.values[shuffled_indices, feature_index]
        if scipy.sparse.issparse(f):
            f = f.A
        f = f.squeeze()

        scatter_plot = axis.scatter(
            values[:, 0],
            values[:, 1],
            c=f,
            cmap=colour_map
        )
        colour_bar = figure.colorbar(scatter_plot)
        colour_bar.outline.set_linewidth(0)
        colour_bar.set_label(feature_name)

    else:
        axis.scatter(values[:, 0], values[:, 1], c=[NEUTRAL_COLOUR])

    if centroids:
        prior_centroids = centroids["prior"]

        if prior_centroids:
            n_centroids = prior_centroids["probabilities"].shape[0]
        else:
            n_centroids = 0

        if n_centroids > 1:
            centroids_palette = darker_palette(n_centroids)
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
                ellipse_fill, ellipse_edge = covariance_matrix_as_ellipse(
                    covariance_matrices[k],
                    means[k],
                    colour=centroids_palette[k]
                )
                axis.add_patch(ellipse_edge)
                axis.add_patch(ellipse_fill)

    # Reset marker size
    reset_plot_look()

    return figure, figure_name


def plot_probabilities(posterior_probabilities, prior_probabilities,
                       x_label=None, y_label=None, palette=None,
                       uniform=False, name=None):

    figure_name = build_figure_name("probabilities", name)

    if posterior_probabilities is None and prior_probabilities is None:
        raise ValueError("No posterior nor prior probabilities given.")

    n_centroids = 0
    if posterior_probabilities is not None:
        n_posterior_centroids = len(posterior_probabilities)
        n_centroids = max(n_centroids, n_posterior_centroids)
    if prior_probabilities is not None:
        n_prior_centroids = len(prior_probabilities)
        n_centroids = max(n_centroids, n_prior_centroids)

    if not x_label:
        x_label = "$k$"
    if not y_label:
        y_label = "$\\pi_k$"

    figure = pyplot.figure(figsize=(8, 6), dpi=80)
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    if not palette:
        palette = [STANDARD_PALETTE[0]] * n_centroids

    if posterior_probabilities is not None:
        for k in range(n_posterior_centroids):
            axis.bar(k, posterior_probabilities[k], color=palette[k])
        axis.set_ylabel("$\\pi_{\\phi}^k$")
        if prior_probabilities is not None:
            for k in range(n_posterior_centroids):
                axis.plot(
                    [k-0.4, k+0.4],
                    2 * [prior_probabilities[k]],
                    color="black",
                    linestyle="dashed"
                )
            prior_line = matplotlib.lines.Line2D(
                [], [],
                color="black",
                linestyle="dashed",
                label="$\\pi_{\\theta}^k$"
            )
            axis.legend(handles=[prior_line], loc="best", fontsize=18)
    elif prior_probabilities is not None:
        for k in range(n_prior_centroids):
            axis.bar(k, prior_probabilities[k], color=palette[k])
        axis.set_ylabel("$\\pi_{\\theta}^k$")

    axis.set_xlabel(x_label)

    return figure, figure_name


def covariance_matrix_as_ellipse(covariance_matrix, mean, colour,
                                 linestyle="solid", radius_stddev=1,
                                 label=None):

    eigenvalues, eigenvectors = numpy.linalg.eig(covariance_matrix)
    indices_sorted_ascending = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices_sorted_ascending]
    eigenvectors = eigenvectors[:, indices_sorted_ascending]

    lambda_1, lambda_2 = numpy.sqrt(eigenvalues)
    theta = numpy.degrees(numpy.arctan2(
        eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse_fill = matplotlib.patches.Ellipse(
        xy=mean,
        width=2 * radius_stddev * lambda_1,
        height=2 * radius_stddev * lambda_2,
        angle=theta,
        linewidth=2,
        linestyle=linestyle,
        facecolor="none",
        edgecolor=colour,
        label=label
    )
    ellipse_edge = matplotlib.patches.Ellipse(
        xy=mean,
        width=2 * radius_stddev * lambda_1,
        height=2 * radius_stddev * lambda_2,
        angle=theta,
        linewidth=3,
        linestyle=linestyle,
        facecolor="none",
        edgecolor="black",
        label=label
    )

    return ellipse_fill, ellipse_edge


def combine_images_from_data_set(data_set, indices=None,
                                 number_of_random_examples=None, name=None):

    image_name = build_figure_name("random_image_examples", name)
    random_state = numpy.random.RandomState(13)

    if indices is not None:
        n_examples = len(indices)
        if number_of_random_examples is not None:
            n_examples = min(n_examples, number_of_random_examples)
            indices = random_state.permutation(indices)[:n_examples]
    else:
        if number_of_random_examples is not None:
            n_examples = number_of_random_examples
        else:
            n_examples = DEFAULT_NUMBER_OF_RANDOM_EXAMPLES
        indices = random_state.permutation(
            data_set.number_of_examples)[:n_examples]

    if n_examples == 1:
        image_name = build_figure_name("image_example", name)
    else:
        image_name = build_figure_name("image_examples", name)

    width, height = data_set.feature_dimensions

    examples = data_set.values[indices]
    if scipy.sparse.issparse(examples):
        examples = examples.A
    examples = examples.reshape(n_examples, width, height)

    column = int(numpy.ceil(numpy.sqrt(n_examples)))
    row = int(numpy.ceil(n_examples / column))

    image = numpy.zeros((row * width, column * height))

    for m in range(n_examples):
        c = int(m % column)
        r = int(numpy.floor(m / column))
        rows = slice(r*width, (r+1)*width)
        columns = slice(c*height, (c+1)*height)
        image[rows, columns] = examples[m]

    return image, image_name


def save_image(image, name, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    minimum = image.min()
    maximum = image.max()
    if 0 < minimum and minimum < 1 and 0 < maximum and maximum < 1:
        rescaled_image = 255 * image
    else:
        rescaled_image = (255 / (maximum - minimum) * (image - minimum))

    image = PIL.Image.fromarray(rescaled_image.astype(numpy.uint8))

    name += IMAGE_EXTENSION
    image_path = os.path.join(directory, name)
    image.save(image_path)


def build_figure_name(base_name, other_names=None):

    if isinstance(base_name, list):
        if not other_names:
            other_names = []
        other_names.extend(base_name[1:])
        base_name = normaliseString(base_name[0])

    figure_name = base_name

    if other_names:
        if not isinstance(other_names, list):
            other_names = str(other_names)
            other_names = [other_names]
        else:
            other_names = [
                str(name) for name in other_names if name is not None]
        figure_name += "-" + "-".join(map(normaliseString, other_names))

    return figure_name


def save_figure(figure, name=None, options=None, directory=None):

    if name is None:
        name = "figure"

    if options is None:
        options = []

    if directory is None:
        directory = "."

    if not os.path.exists(directory):
        os.makedirs(directory)

    figure_path_base = os.path.join(directory, name)

    default_tight_layout = True
    tight_layout = default_tight_layout

    figure_width, figure_height = figure.get_size_inches()
    aspect_ratio = figure_width / figure_height

    figure.set_tight_layout(tight_layout)
    _adjust_figure_for_legend(figure)

    figure_path = figure_path_base + FIGURE_EXTENSION
    figure.savefig(figure_path)

    if "publication" in options:

        figure.set_tight_layout(default_tight_layout)
        figure.set_dpi(PUBLICATION_DPI)

        local_publication_copies = PUBLICATION_COPIES.copy()
        local_publication_copies["standard"] = {
            "figure_width": figure_width
        }

        for copy_name, copy_properties in local_publication_copies.items():

            figure.set_size_inches(figure_width, figure_height)

            publication_figure_width = copy_properties["figure_width"]
            publication_figure_height = (
                publication_figure_width / aspect_ratio)

            figure.set_size_inches(publication_figure_width,
                                   publication_figure_height)
            _adjust_figure_for_legend(figure)

            figure_path = "-".join([
                figure_path_base, "publication", copy_name
            ]) + PUBLICATION_FIGURE_EXTENSION

            figure.savefig(figure_path)

    pyplot.close(figure)


def _adjust_figure_for_legend(figure):

    for axis in figure.get_axes():
        legend = axis.get_legend()

        if legend and _legend_is_above_axis(legend, axis):

            renderer = figure.canvas.get_renderer()
            figure.draw(renderer=renderer)

            legend_size = legend.get_window_extent()
            legend_height_in_inches = legend_size.height / figure.get_dpi()

            figure_width, figure_height = figure.get_size_inches()
            figure_height += legend_height_in_inches
            figure.set_size_inches(figure_width, figure_height)


def _legend_is_above_axis(legend, axis):

    legend_bottom_vertical_position_relative_to_axis = (
        legend.get_bbox_to_anchor().transformed(
            axis.transAxes.inverted()).ymin)

    if legend_bottom_vertical_position_relative_to_axis >= 1:
        legend_is_above_axis = True
    else:
        legend_is_above_axis = False

    return legend_is_above_axis


def _axis_label_for_symbol(symbol, coordinate=None, decomposition_method=None,
                           distribution=None, prefix="", suffix=""):

    if decomposition_method:
        decomposition_method = properString(
            normaliseString(decomposition_method),
            DECOMPOSITION_METHOD_NAMES
        )
        decomposition_label = DECOMPOSITION_METHOD_LABEL[decomposition_method]
    else:
        decomposition_label = ""

    if decomposition_label:
        decomposition_label = "\\mathrm{{{}}}".format(decomposition_label)

    if coordinate:
        coordinate_text = "{{{} {}}}".format(decomposition_label, coordinate)
    else:
        coordinate_text = ""

    if distribution == "prior":
        distribution_symbol = "\\theta"
    elif distribution == "posterior":
        distribution_symbol = "\\phi"
    else:
        distribution_symbol = ""

    if distribution_symbol and coordinate_text:
        distribution_position = "_"
        coordinate_position = "^"
    elif distribution_symbol and not coordinate_text:
        distribution_position = "_"
        coordinate_position = ""
    elif not distribution_symbol and coordinate_text:
        distribution_position = ""
        coordinate_position = "_"
    else:
        distribution_position = ""
        coordinate_position = ""

    if coordinate_position == "^":
        coordinate_text = "{{(" + coordinate_text + ")}}"
    elif coordinate_position == "_":
        coordinate_text = "{{" + coordinate_text + "}}"

    axis_label = "$" + "".join([
        prefix, symbol,
        distribution_position,
        distribution_symbol,
        coordinate_position,
        coordinate_text,
        suffix
    ]) + "$"

    return axis_label


def _change_point_size_for_plots(marker_size):
    matplotlib.rc(group="lines", markersize=marker_size)
    matplotlib.rc(
        group="legend",
        markerscale=legend_marker_scale_from_marker_size(marker_size)
    )


def _build_path_for_result_directory(base_directory, model_name,
                                     run_id=None, subdirectories=None):

    results_directory = os.path.join(base_directory, model_name)

    if run_id:
        run_id = checkRunID(run_id)
        results_directory = os.path.join(
            results_directory,
            "run_{}".format(run_id)
        )

    if subdirectories:

        if not isinstance(subdirectories, list):
            subdirectories = [subdirectories]

        results_directory = os.path.join(results_directory, *subdirectories)

    return results_directory


def _parse_analyses(analyses):

    if not analyses:
        analyses = ANALYSIS_GROUPS["default"]

    resulting_analyses = set()

    for analysis in analyses:
        if analysis in ANALYSIS_GROUPS:
            group = analysis
            resulting_analyses.update(map(
                normaliseString, ANALYSIS_GROUPS[group]))
        elif analysis in ANALYSIS_GROUPS["complete"]:
            resulting_analyses.add(normaliseString(analysis))
        else:
            raise ValueError("Analysis `{}` not found.".format(analysis))

    resulting_analyses = list(resulting_analyses)

    return resulting_analyses


def lighter_palette(n):
    return seaborn.husl_palette(n, l=.75)  # noqa: E741


def darker_palette(n):
    return seaborn.husl_palette(n, l=.55)  # noqa: E741


def legend_marker_scale_from_marker_size(marker_size):
    return DEFAULT_MARKER_SIZE_IN_LEGENDS / marker_size


def reset_plot_look():
    seaborn.set(
        context="paper",
        style="ticks",
        palette=STANDARD_PALETTE,
        color_codes=False,
        rc={
            "lines.markersize": DEFAULT_MARKER_SIZE_IN_SCATTER_PLOTS,
            "legend.markerscale": legend_marker_scale_from_marker_size(
                DEFAULT_MARKER_SIZE_IN_SCATTER_PLOTS
            ),
            "figure.dpi": 200
        }
    )


reset_plot_look()
