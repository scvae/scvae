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

import gzip
import os
import pickle
from time import time

import numpy

from scvae.analyses import figures, images, metrics, subanalyses
from scvae.analyses.decomposition import decompose
from scvae.analyses.figures.utilities import _axis_label_for_symbol
from scvae.data.utilities import indices_for_evaluation_subset, save_values
from scvae.defaults import defaults
from scvae.models.utilities import (
    load_number_of_epochs_trained, load_learning_curves, load_accuracies,
    load_centroids, load_kl_divergences,
    check_run_id
)
from scvae.utilities import (
    format_time, format_duration,
    normalise_string, capitalise_string, subheading
)

ANALYSIS_GROUPS = {
    "simple": ["metrics", "images", "learning_curves", "latent_values",
               "predictions"],
    "standard": ["profile_comparisons", "distributions",
                 "decompositions", "latent_space"],
    "all": ["heat_maps", "distances", "feature_value_standard_deviations",
            "latent_distributions", "latent_correlations", "latent_features",
            "kl_heat_maps", "accuracies"]
}
ANALYSIS_GROUPS["standard"] += ANALYSIS_GROUPS["simple"]
ANALYSIS_GROUPS["all"] += ANALYSIS_GROUPS["standard"]

MAXIMUM_NUMBER_OF_VALUES_FOR_HEAT_MAPS = 5000 * 25000
MAXIMUM_NUMBER_OF_CORRELATED_VARIABLE_PAIRS_TO_PLOT = 25
MAXIMUM_NUMBER_OF_VARIABLES_FOR_CORRELATION_PLOT = 10
PROFILE_COMPARISON_COUNT_CUTOFF = 10.5
DEFAULT_CUTOFFS = range(1, 10)


def analyse_data(data_sets,
                 decomposition_methods=None,
                 highlight_feature_indices=None,
                 analyses_directory=None,
                 **kwargs):
    """Analyse data set and save results and plots.

    Arguments:
        data_sets (list(DataSet)): List of data sets to analyse.
        decomposition_methods (str or list(str)): Method(s) used to
            decompose data set values: ``"PCA"``, ``"SVD"``, ``"ICA"``,
            and/or ``"t-SNE"``.
        highlight_feature_indices (int or list(int)): Index or indices
            to highlight in decompositions.
        analyses_directory (str, optional): Directory where to save analyses.
    """

    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]
    analyses_directory = os.path.join(analyses_directory, "data")

    if not os.path.exists(analyses_directory):
        os.makedirs(analyses_directory)

    included_analyses = kwargs.get("included_analyses")
    if included_analyses is None:
        included_analyses = defaults["analyses"]["included_analyses"]
    included_analyses = _parse_analyses(included_analyses)

    analysis_level = kwargs.get("analysis_level")
    if analysis_level is None:
        analysis_level = defaults["analyses"]["analysis_level"]

    export_options = kwargs.get("export_options")

    if not isinstance(data_sets, list):
        data_sets = [data_sets]

    if "metrics" in included_analyses:

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
                    metrics.summary_statistics(
                        series, name=series_name, skip_sparsity=True)
                    for series_name, series in {
                        "count sum": data_set.count_sum
                    }.items()
                ])
            data_set_statistics.append(metrics.summary_statistics(
                data_set.values, name=data_set.kind, tolerance=0.5))

        metrics_duration = time() - metrics_time_start
        print("Metrics calculated ({}).".format(
            format_duration(metrics_duration)))

        metrics_path = os.path.join(analyses_directory, "data_set_metrics.log")
        metrics_saving_time_start = time()

        metrics_string_parts = [
            "Timestamp: {}".format(format_time(metrics_saving_time_start)),
            "Features: {}".format(number_of_features),
            "Examples: {}".format(number_of_examples["full"]),
            "\n".join([
                "{} examples: {}".format(capitalise_string(kind), number)
                for kind, number in number_of_examples.items()
                if kind != "full"
            ]),
            "\n" + metrics.format_summary_statistics(data_set_statistics)
        ]
        metrics_string = "\n".join(metrics_string_parts) + "\n"

        with open(metrics_path, "w") as metrics_file:
            metrics_file.write(metrics_string)

        metrics_saving_duration = time() - metrics_saving_time_start
        print("Metrics saved ({}).".format(format_duration(
            metrics_saving_duration)))
        print()

        print(metrics.format_summary_statistics(data_set_statistics))
        print()

        print(metrics.format_summary_statistics(
            histogram_statistics, name="Series"))
        print()

    for data_set in data_sets:

        print(subheading("Analyses of {} set".format(data_set.kind)))

        if "images" in included_analyses and data_set.example_type == "images":
            print("Saving image of {} random examples from {} set.".format(
                images.DEFAULT_NUMBER_OF_RANDOM_EXAMPLES_FOR_COMBINED_IMAGES,
                data_set.kind
            ))
            image_time_start = time()
            image, image_name = images.combine_images_from_data_set(
                data_set,
                number_of_random_examples=(
                    images.
                    DEFAULT_NUMBER_OF_RANDOM_EXAMPLES_FOR_COMBINED_IMAGES
                ),
                name=data_set.kind
            )
            images.save_image(
                image=image,
                name=image_name,
                directory=analyses_directory
            )
            image_duration = time() - image_time_start
            print("Image saved ({}).".format(format_duration(image_duration)))
            print()

        if "distributions" in included_analyses:
            subanalyses.analyse_distributions(
                data_set,
                cutoffs=DEFAULT_CUTOFFS,
                analysis_level=analysis_level,
                export_options=export_options,
                analyses_directory=analyses_directory
            )

        if "heat_maps" in included_analyses:
            subanalyses.analyse_matrices(
                data_set,
                name=[data_set.kind],
                analyses_directory=analyses_directory
            )

        if "distances" in included_analyses:
            subanalyses.analyse_matrices(
                data_set,
                plot_distances=True,
                name=[data_set.kind],
                export_options=export_options,
                analyses_directory=analyses_directory
            )

        if "decompositions" in included_analyses:
            subanalyses.analyse_decompositions(
                data_set,
                decomposition_methods=decomposition_methods,
                highlight_feature_indices=highlight_feature_indices,
                symbol="x",
                title="original space",
                specifier=lambda data_set: data_set.kind,
                analysis_level=analysis_level,
                export_options=export_options,
                analyses_directory=analyses_directory
            )

        if "feature_value_standard_deviations" in included_analyses:

            print("Computing and plotting feature value standard deviations:")

            feature_value_standard_deviations_directory = os.path.join(
                analyses_directory,
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
                .format(format_duration(duration))
            )

            # Feature value standard_deviations
            time_start = time()

            figure, figure_name = figures.plot_series(
                series=feature_value_standard_deviations,
                x_label=data_set.terms["feature"] + "s",
                y_label="{} standard deviations".format(
                    data_set.terms["type"]),
                sort=True,
                scale="log",
                name=["feature value standard deviations", data_set.kind]
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=feature_value_standard_deviations_directory
            )

            duration = time() - time_start
            print(
                "    Feature value standard deviations plotted and saved({})."
                .format(format_duration(duration))
            )

            # Distribution of feature value standard deviations
            time_start = time()

            figure, figure_name = figures.plot_histogram(
                series=feature_value_standard_deviations,
                label="{} {} standard deviations".format(
                    data_set.terms["feature"], data_set.terms["type"]
                ),
                normed=True,
                x_scale="linear",
                y_scale="log",
                name=["feature value standard deviations", data_set.kind]
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=feature_value_standard_deviations_directory
            )

            duration = time() - time_start
            print(
                "    Feature value standard deviation distribution plotted "
                "and saved ({}).".format(format_duration(duration))
            )

            print()


def analyse_model(model, run_id=None, analyses_directory=None, **kwargs):
    """Analyse trained model and save results and plots.

    Arguments:
        model ((GaussianMixture)VariationalAutoencoder): Model to analyse.
        run_id (str, optional): ID used to identify a certain run
            of ``model``.
        analyses_directory (str, optional): Directory where to save analyses.
    """

    if run_id is None:
        run_id = defaults["models"]["run_id"]
    if run_id:
        run_id = check_run_id(run_id)

    if not model.has_been_trained(run_id=run_id):
        raise Exception("Cannot analyse model when it has not been trained.")

    number_of_epochs_trained = load_number_of_epochs_trained(
        model, run_id=run_id)
    epochs_string = "e_" + str(number_of_epochs_trained)

    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]
    analyses_directory = _build_path_for_analyses_directory(
        base_directory=analyses_directory,
        model_name=model.name,
        run_id=run_id,
        subdirectories=[epochs_string]
    )

    if not os.path.exists(analyses_directory):
        os.makedirs(analyses_directory)

    included_analyses = kwargs.get("included_analyses")
    if included_analyses is None:
        included_analyses = defaults["analyses"]["included_analyses"]
    included_analyses = _parse_analyses(included_analyses)

    analysis_level = kwargs.get("analysis_level")
    if analysis_level is None:
        analysis_level = defaults["analyses"]["analysis_level"]

    export_options = kwargs.get("export_options")

    if "learning_curves" in included_analyses:

        print(subheading("Learning curves"))

        print("Plotting learning curves.")
        learning_curves_time_start = time()

        learning_curves = load_learning_curves(
            model=model,
            data_set_kinds=["training", "validation"],
            run_id=run_id
        )

        figure, figure_name = figures.plot_learning_curves(
            learning_curves,
            model_type=model.type
        )
        figures.save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=analyses_directory
        )

        if "VAE" in model.type:
            loss_sets = [["lower_bound", "reconstruction_error"]]

            if model.type == "GMVAE":
                loss_sets.append("kl_divergence_z")
                loss_sets.append("kl_divergence_y")
            else:
                loss_sets.append("kl_divergence")

            for loss_set in loss_sets:
                figure, figure_name = figures.plot_separate_learning_curves(
                    learning_curves,
                    loss=loss_set
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=analyses_directory
                )

        learning_curves_duration = time() - learning_curves_time_start
        print("Learning curves plotted and saved ({}).".format(
            format_duration(learning_curves_duration)))
        print()

    if "accuracies" in included_analyses:

        accuracies_time_start = time()

        accuracies = load_accuracies(
            model=model,
            run_id=run_id,
            data_set_kinds=["training", "validation"]
        )

        if accuracies is not None:

            print(subheading("Accuracies"))
            print("Plotting accuracies.")

            figure, figure_name = figures.plot_accuracy_evolution(accuracies)
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=analyses_directory
            )

            superset_accuracies = load_accuracies(
                model=model,
                data_set_kinds=["training", "validation"],
                superset=True,
                run_id=run_id
            )

            if superset_accuracies is not None:
                figure, figure_name = figures.plot_accuracy_evolution(
                    superset_accuracies,
                    name="superset"
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=analyses_directory
                )

            accuracies_duration = time() - accuracies_time_start
            print("Accuracies plotted and saved ({}).".format(
                format_duration(accuracies_duration)))
            print()

    if "kl_heat_maps" in included_analyses and "VAE" in model.type:

        print(subheading("KL divergence"))

        print("Plotting logarithm of KL divergence heat map.")
        heat_map_time_start = time()

        kl_neurons = load_kl_divergences(model=model, run_id=run_id)

        kl_neurons = numpy.sort(kl_neurons, axis=1)

        figure, figure_name = figures.plot_kl_divergence_evolution(
            kl_neurons,
            scale="log"
        )
        figures.save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=analyses_directory
        )

        heat_map_duration = time() - heat_map_time_start
        print("Heat map plotted and saved ({}).".format(
            format_duration(heat_map_duration)))
        print()

    if "latent_distributions" in included_analyses and model.type == "GMVAE":

        print(subheading("Latent distributions"))

        kind_centroids = load_centroids(
            model=model,
            data_set_kinds=["training", "validation"],
            run_id=run_id
        )

        for set_kind, centroids in kind_centroids.items():

            centroids_directory = os.path.join(
                analyses_directory,
                "-".join(["centroids_evolution", set_kind])
            )

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
                        distribution_centroids_decomposed = (
                            distribution_centroids)
                        decomposed = False

                    centroid_means_decomposed = (
                        distribution_centroids_decomposed["means"])

                    figure, figure_name = (
                        figures.plot_centroid_probabilities_evolution(
                            centroid_probabilities,
                            distribution=distribution
                        )
                    )
                    figures.save_figure(
                        figure=figure,
                        name=figure_name,
                        options=export_options,
                        directory=centroids_directory
                    )

                    figure, figure_name = (
                        figures.plot_centroid_means_evolution(
                            centroid_means_decomposed,
                            distribution=distribution,
                            decomposed=decomposed
                        )
                    )
                    figures.save_figure(
                        figure=figure,
                        name=figure_name,
                        options=export_options,
                        directory=centroids_directory
                    )

                    figure, figure_name = (
                        figures.plot_centroid_covariance_matrices_evolution(
                            centroid_covariance_matrices,
                            distribution=distribution
                        )
                    )
                    figures.save_figure(
                        figure=figure,
                        name=figure_name,
                        options=export_options,
                        directory=centroids_directory
                    )

                    centroids_duration = time() - centroids_time_start
                    print(
                        "Evolution of latent {} parameters plotted and saved "
                        "({})".format(
                            distribution, format_duration(centroids_duration))
                    )
                    print()


def analyse_intermediate_results(epoch, learning_curves=None, epoch_start=None,
                                 model_type=None, latent_values=None,
                                 data_set=None, centroids=None,
                                 model_name=None, run_id=None,
                                 analyses_directory=None):
    """Analyse reconstructions and latent values.

    Reconstructions and latent values from evaluating a model on a data
    set are analysed, and results and plots are saved.

    Arguments:
        evaluation_set (DataSet): Data set used to evaluate ``model``.
        reconstructed_evaluation_set (DataSet): Reconstructed data set
            from evaluating ``model`` on ``evaluation_set``.
        latent_evaluation_sets (dict(str, DataSet)): Dictionary of data
            sets of the two latent variables.
        model ((GaussianMixture)VariationalAutoencoder): Model
            evaluated on ``evaluation_set``.
        run_id (str, optional): ID used to identify a certain run
            of ``model``.
        sample_reconstruction_set (DataSet): Reconstruction data set
            from sampling ``model``.
        decomposition_methods (str or list(str)): Method(s) used to
            decompose data set values: ``"PCA"``, ``"SVD"``, ``"ICA"``,
            and/or ``"t-SNE"``.
        highlight_feature_indices (int or list(int)): Index or indices
            to highlight in decompositions.
        early_stopping (bool, optional): If ``True``, use parameters
            for ``model``, when early stopping triggered during
            training. Defaults to ``False``.
        best_model (bool, optional): If ``True``, use parameters for
            ``model``, which resulted in the best performance on
            validation set during training. Defaults to ``False``.
        analyses_directory (str, optional): Directory where to save analyses.
    """

    if run_id is None:
        run_id = defaults["models"]["run_id"]
    if run_id:
        run_id = check_run_id(run_id)

    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]
    analyses_directory = _build_path_for_analyses_directory(
        base_directory=analyses_directory,
        model_name=model_name,
        run_id=run_id,
        subdirectories=["intermediate"]
    )

    print("Plotting learning curves.")
    learning_curves_time_start = time()

    figure, figure_name = figures.plot_learning_curves(
        learning_curves,
        model_type=model_type,
        epoch_offset=epoch_start
    )
    figures.save_figure(
        figure=figure,
        name=figure_name,
        directory=analyses_directory
    )

    learning_curves_duration = time() - learning_curves_time_start
    print("Learning curves plotted and saved ({}).".format(
        format_duration(learning_curves_duration)))

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
                capitalise_string(latent_set_name),
                format_duration(decompose_duration))
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
            figure, figure_name = figures.plot_values(
                latent_values_decomposed,
                colour_coding="labels",
                colouring_data_set=data_set,
                centroids=centroids_decomposed,
                figure_labels=figure_labels,
                name=name
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                directory=analyses_directory
            )
            if data_set.label_superset is not None:
                figure, figure_name = figures.plot_values(
                    latent_values_decomposed,
                    colour_coding="superset labels",
                    colouring_data_set=data_set,
                    centroids=centroids_decomposed,
                    figure_labels=figure_labels,
                    name=name
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    directory=analyses_directory
                )
        else:
            figure, figure_name = figures.plot_values(
                latent_values_decomposed,
                centroids=centroids_decomposed,
                figure_labels=figure_labels,
                name=name
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                directory=analyses_directory
            )

        if centroids:
            subanalyses.analyse_centroid_probabilities(
                centroids, epoch_name,
                analyses_directory=analyses_directory
            )

        plot_duration = time() - plot_time_start
        print(
            "{} plotted and saved ({}).".format(
                capitalise_string(latent_set_name),
                format_duration(plot_duration)
            )
        )


def analyse_results(evaluation_set, reconstructed_evaluation_set,
                    latent_evaluation_sets, model, run_id=None,
                    sample_reconstruction_set=None,
                    decomposition_methods=None,
                    highlight_feature_indices=None,
                    early_stopping=False, best_model=False,
                    analyses_directory=None, **kwargs):

    if early_stopping and best_model:
        raise ValueError(
            "Early-stopping model and best model cannot be evaluated at the "
            "same time."
        )

    if run_id is None:
        run_id = defaults["models"]["run_id"]
    if run_id:
        run_id = check_run_id(run_id)

    evaluation_subset_indices = kwargs.get("evaluation_subset_indices")
    if evaluation_subset_indices is None:
        evaluation_subset_indices = indices_for_evaluation_subset(
            evaluation_set)

    included_analyses = kwargs.get("included_analyses")
    if included_analyses is None:
        included_analyses = defaults["analyses"]["included_analyses"]
    included_analyses = _parse_analyses(included_analyses)

    analysis_level = kwargs.get("analysis_level")
    if analysis_level is None:
        analysis_level = defaults["analyses"]["analysis_level"]

    export_options = kwargs.get("export_options")

    print("Setting up results analyses.")
    setup_time_start = time()

    number_of_epochs_trained = load_number_of_epochs_trained(
        model=model,
        run_id=run_id,
        early_stopping=early_stopping,
        best_model=best_model
    )

    # Comparison arrays
    if (("metrics" in included_analyses or "heat_maps" in included_analyses)
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

    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]
    analyses_directory = _build_path_for_analyses_directory(
        base_directory=analyses_directory,
        model_name=model.name,
        run_id=run_id,
        subdirectories=[evaluation_directory]
    )

    if evaluation_set.kind != "test":
        analyses_directory = os.path.join(
            analyses_directory, evaluation_set.kind)

    if not os.path.exists(analyses_directory):
        os.makedirs(analyses_directory)

    setup_duration = time() - setup_time_start
    print("Finished setting up ({}).".format(format_duration(setup_duration)))
    print()

    if "metrics" in included_analyses:

        print(subheading("Metrics"))

        print("Loading results from model log directory.")
        loading_time_start = time()

        evaluation_eval = load_learning_curves(
            model=model,
            data_set_kinds="evaluation",
            run_id=run_id,
            early_stopping=early_stopping,
            best_model=best_model
        )
        accuracy_eval = load_accuracies(
            model=model,
            data_set_kinds="evaluation",
            run_id=run_id,
            early_stopping=early_stopping,
            best_model=best_model
        )
        superset_accuracy_eval = load_accuracies(
            model=model,
            data_set_kinds="evaluation",
            superset=True,
            run_id=run_id,
            early_stopping=early_stopping,
            best_model=best_model
        )

        loading_duration = time() - loading_time_start
        print("Results loaded ({}).".format(format_duration(loading_duration)))
        print()

        print("Calculating metrics for results.")
        metrics_time_start = time()

        evaluation_set_statistics = [
            metrics.summary_statistics(
                data_set.values, name=data_set.version, tolerance=0.5)
            for data_set in [evaluation_set, reconstructed_evaluation_set]
        ]

        if analysis_level == "extensive":
            evaluation_set_statistics.append(metrics.summary_statistics(
                numpy.abs(x_diff),
                name="differences",
                skip_sparsity=True
            ))
            evaluation_set_statistics.append(metrics.summary_statistics(
                numpy.abs(x_log_ratio),
                name="log-ratios",
                skip_sparsity=True
            ))

        clustering_metric_values = metrics.compute_clustering_metrics(
            evaluation_set)

        metrics_duration = time() - metrics_time_start
        print("Metrics calculated ({}).".format(
            format_duration(metrics_duration)))

        metrics_saving_time_start = time()

        metrics_log_filename = "{}-metrics".format(evaluation_set.kind)
        metrics_log_path = os.path.join(
            analyses_directory, metrics_log_filename + ".log")
        metrics_dictionary_path = os.path.join(
            analyses_directory, metrics_log_filename + ".pkl.gz")

        # Build metrics string
        metrics_string_parts = [
            "Timestamp: {}".format(format_time(metrics_saving_time_start)),
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
            "\n" + metrics.format_summary_statistics(
                evaluation_set_statistics))
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

        if evaluation_set.prediction_specifications:
            prediction_specifications = (
                evaluation_set.prediction_specifications)

            prediction_log_filename = "{}-prediction-{}".format(
                evaluation_set.kind, prediction_specifications.name)
            prediction_log_path = os.path.join(
                analyses_directory, prediction_log_filename + ".log")
            prediction_dictionary_path = os.path.join(
                analyses_directory, prediction_log_filename + ".pkl.gz")

            prediction_string_parts = [
                "Timestamp: {}".format(format_time(
                    metrics_saving_time_start)),
                "Number of epochs trained: {}".format(
                    number_of_epochs_trained),
                "Prediction method: {}".format(
                    prediction_specifications.method),
                "Number of classes: {}".format(
                    prediction_specifications.number_of_clusters
                )
            ]

            if prediction_specifications.training_set_kind:
                prediction_string_parts.append(
                    "Training set: {}".format(
                        prediction_specifications.training_set_kind
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
                                    capitalise_string(metric_name)
                                )
                            )
                            clustering_metric_name_printed = True

                        prediction_string_parts.append(
                            "        {}: {:.5g}.".format(
                                capitalise_string(set_name), metric_value
                            )
                        )

            prediction_string = "\n".join(prediction_string_parts) + "\n"

            prediction_dictionary = {
                "timestamp": metrics_saving_time_start,
                "number of epochs trained": number_of_epochs_trained,
                "prediction method": prediction_specifications.method,
                "number of classes": (
                    prediction_specifications.number_of_clusters),
                "training set": prediction_specifications.training_set_kind,
                "clustering metric values": clustering_metric_values
            }

            with open(prediction_log_path, "w") as prediction_file:
                prediction_file.write(prediction_string)

            with gzip.open(prediction_dictionary_path, "w") as prediction_file:
                pickle.dump(prediction_dictionary, prediction_file)

        metrics_saving_duration = time() - metrics_saving_time_start
        print("Metrics saved ({}).".format(format_duration(
            metrics_saving_duration)))
        print()

        print(metrics.format_summary_statistics(evaluation_set_statistics))
        print()

        for metric_name, metric_set in clustering_metric_values.items():

            clustering_metric_name_printed = False

            for set_name, metric_value in metric_set.items():
                if metric_value is not None:

                    if not clustering_metric_name_printed:
                        print("{}:".format(capitalise_string(metric_name)))
                        clustering_metric_name_printed = True

                    print("    {}: {:.5g}.".format(
                        capitalise_string(set_name), metric_value
                    ))

            if clustering_metric_name_printed:
                print()

    # Only print subheading if necessary
    if ("images" in included_analyses
            and reconstructed_evaluation_set.example_type == "images"
            or "profile_comparisons" in included_analyses):
        print(subheading("Reconstructions"))

    if ("images" in included_analyses
            and reconstructed_evaluation_set.example_type == "images"):

        print("Saving image of {} random examples".format(
            images.DEFAULT_NUMBER_OF_RANDOM_EXAMPLES_FOR_COMBINED_IMAGES),
            "from reconstructed {} set.".format(
                evaluation_set.kind
            ))
        image_time_start = time()
        image, image_name = images.combine_images_from_data_set(
            reconstructed_evaluation_set,
            number_of_random_examples=(
                images.DEFAULT_NUMBER_OF_RANDOM_EXAMPLES_FOR_COMBINED_IMAGES),
            name=reconstructed_evaluation_set.version
        )
        images.save_image(
            image=image,
            name=image_name,
            directory=analyses_directory
        )
        image_duration = time() - image_time_start
        print("Image saved ({}).".format(format_duration(image_duration)))
        print()

    if "profile_comparisons" in included_analyses:

        print("Plotting profile comparisons.")
        profile_comparisons_time_start = time()

        image_comparisons_directory = os.path.join(
            analyses_directory, "image_comparisons")
        profile_comparisons_directory = os.path.join(
            analyses_directory, "profile_comparisons")

        y_cutoff = PROFILE_COMPARISON_COUNT_CUTOFF

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
                    figure, figure_name = figures.plot_profile_comparison(
                        observed_series,
                        expected_series,
                        expected_series_total_standard_deviations,
                        expected_series_explained_standard_deviations,
                        x_name=evaluation_set.terms["feature"],
                        y_name=evaluation_set.terms["value"],
                        sort=sort_profile_comparison,
                        sort_by="expected",
                        sort_direction="descending",
                        x_scale="log",
                        y_scale=y_scale,
                        name=example_name_parts
                    )
                    figures.save_figure(
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
                    figure, figure_name = figures.plot_profile_comparison(
                        observed_series,
                        expected_series,
                        expected_series_total_standard_deviations,
                        expected_series_explained_standard_deviations,
                        x_name=evaluation_set.terms["feature"],
                        y_name=evaluation_set.terms["value"],
                        sort=True,
                        sort_by="expected",
                        sort_direction="descending",
                        x_scale="log",
                        y_scale=y_scale,
                        y_cutoff=y_cutoff,
                        name=example_name_parts
                    )
                    figures.save_figure(
                        figure=figure,
                        name=figure_name,
                        options=export_options,
                        directory=profile_comparisons_directory
                    )

            if evaluation_set.example_type == "images":
                example_name_parts = ["original"] + example_name_base_parts
                image, image_name = images.combine_images_from_data_set(
                    evaluation_set,
                    indices=[i],
                    name=example_name_parts
                )
                images.save_image(
                    image=image,
                    name=image_name,
                    directory=image_comparisons_directory
                )

            if reconstructed_evaluation_set.example_type == "images":
                example_name_parts = (
                    ["reconstructed"] + example_name_base_parts)
                image, image_name = images.combine_images_from_data_set(
                    reconstructed_evaluation_set,
                    indices=[i],
                    name=example_name_parts
                )
                images.save_image(
                    image=image,
                    name=image_name,
                    directory=image_comparisons_directory
                )

            if analysis_level == "limited":
                break

        profile_comparisons_duration = time() - profile_comparisons_time_start
        print("Profile comparisons plotted and saved ({}).".format(
            format_duration(profile_comparisons_duration)))
        print()

    if "distributions" in included_analyses:
        print(subheading("Distributions"))
        subanalyses.analyse_distributions(
            reconstructed_evaluation_set,
            colouring_data_set=evaluation_set,
            preprocessed=evaluation_set.preprocessing_methods,
            analysis_level=analysis_level,
            export_options=export_options,
            analyses_directory=analyses_directory
        )

    if "decompositions" in included_analyses:

        print(subheading("Decompositions"))

        subanalyses.analyse_decompositions(
            reconstructed_evaluation_set,
            colouring_data_set=evaluation_set,
            sampled_data_set=sample_reconstruction_set,
            decomposition_methods=decomposition_methods,
            highlight_feature_indices=highlight_feature_indices,
            symbol="\\tilde{{x}}",
            title="reconstructions",
            analysis_level=analysis_level,
            export_options=export_options,
            analyses_directory=analyses_directory,
        )

        if sample_reconstruction_set:
            subanalyses.analyse_decompositions(
                evaluation_set,
                sampled_data_set=sample_reconstruction_set,
                decomposition_methods=decomposition_methods,
                highlight_feature_indices=highlight_feature_indices,
                symbol="x",
                title="originals",
                analysis_level=analysis_level,
                export_options=export_options,
                analyses_directory=analyses_directory,
            )

        if analysis_level == "extensive":
            # Reconstructions plotted in original decomposed space
            subanalyses.analyse_decompositions(
                evaluation_set,
                reconstructed_evaluation_set,
                colouring_data_set=evaluation_set,
                decomposition_methods=decomposition_methods,
                highlight_feature_indices=highlight_feature_indices,
                symbol="x",
                title="originals",
                analysis_level=analysis_level,
                export_options=export_options,
                analyses_directory=analyses_directory,
            )

    if "heat_maps" in included_analyses:
        print(subheading("Heat maps"))
        subanalyses.analyse_matrices(
            reconstructed_evaluation_set,
            plot_distances=False,
            analyses_directory=analyses_directory
        )
        subanalyses.analyse_matrices(
            latent_evaluation_sets["z"],
            plot_distances=False,
            analyses_directory=analyses_directory
        )

        if (analysis_level == "extensive"
                and reconstructed_evaluation_set.number_of_values
                <= MAXIMUM_NUMBER_OF_VALUES_FOR_HEAT_MAPS):

            print("Plotting comparison heat maps.")
            heat_maps_directory = os.path.join(analyses_directory, "heat_maps")

            # Differences
            heat_maps_time_start = time()
            figure, figure_name = figures.plot_heat_map(
                x_diff,
                labels=reconstructed_evaluation_set.labels,
                x_name=evaluation_set.terms["feature"].capitalize() + "s",
                y_name=evaluation_set.terms["example"].capitalize() + "s",
                z_name="Differences",
                z_symbol="\\tilde{{x}} - x",
                name="difference",
                center=0
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=heat_maps_directory
            )
            heat_maps_duration = time() - heat_maps_time_start
            print(
                "    Difference heat map plotted and saved ({})."
                .format(format_duration(heat_maps_duration))
            )

            # log-ratios
            heat_maps_time_start = time()
            figure, figure_name = figures.plot_heat_map(
                x_log_ratio,
                labels=reconstructed_evaluation_set.labels,
                x_name=evaluation_set.terms["feature"].capitalize() + "s",
                y_name=evaluation_set.terms["example"].capitalize() + "s",
                z_name="log-ratios",
                z_symbol="\\log \\frac{{\\tilde{{x}} + 1}}{{x + 1}}",
                name="log_ratio",
                center=0
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=heat_maps_directory
            )
            heat_maps_duration = time() - heat_maps_time_start
            print(
                "    log-ratio heat map plotted and saved ({})."
                .format(format_duration(heat_maps_duration))
            )
        print()

    if "distances" in included_analyses:
        print(subheading("Distances"))
        subanalyses.analyse_matrices(
            reconstructed_evaluation_set,
            plot_distances=True,
            analyses_directory=analyses_directory
        )
        subanalyses.analyse_matrices(
            latent_evaluation_sets["z"],
            plot_distances=True,
            analyses_directory=analyses_directory
        )

    if "predictions" in included_analyses and evaluation_set.has_predictions:
        print(subheading("Predictions"))
        subanalyses.analyse_predictions(
            evaluation_set, analyses_directory=analyses_directory)

    if "latent_values" in included_analyses and "VAE" in model.type:
        print(subheading("Latent values"))
        print("Saving latent values.")
        for set_name, latent_evaluation_set in latent_evaluation_sets.items():
            saving_time_start = time()
            save_values(
                values=latent_evaluation_set.values,
                name="latent_values-{}".format(set_name),
                row_names=latent_evaluation_set.example_names,
                column_names=latent_evaluation_set.feature_names,
                directory=analyses_directory)
            saving_duration = time() - saving_time_start
            print("    Latent values for {} set saved ({}).".format(
                latent_evaluation_set.version,
                format_duration(saving_duration)))

    if "latent_space" in included_analyses and "VAE" in model.type:
        print(subheading("Latent space"))

        if "gaussian mixture" in model.latent_distribution_name:
            print("Loading centroids from model log directory.")
            loading_time_start = time()
            centroids = load_centroids(
                model=model,
                data_set_kinds="evaluation",
                run_id=run_id,
                early_stopping=early_stopping,
                best_model=best_model
            )
            loading_duration = time() - loading_time_start
            print("Centroids loaded ({}).".format(
                format_duration(loading_duration)))
            print()
        else:
            centroids = None

        subanalyses.analyse_decompositions(
            latent_evaluation_sets,
            centroids=centroids,
            colouring_data_set=evaluation_set,
            decomposition_methods=decomposition_methods,
            highlight_feature_indices=highlight_feature_indices,
            title="latent space",
            specifier=lambda data_set: data_set.version,
            analysis_level=analysis_level,
            export_options=export_options,
            analyses_directory=analyses_directory,
        )

        if centroids:
            subanalyses.analyse_centroid_probabilities(
                centroids,
                analysis_level="normal",
                export_options=export_options,
                analyses_directory=analyses_directory
            )
            print()

    if "latent_correlations" in included_analyses and "VAE" in model.type:

        correlations_directory = os.path.join(
            analyses_directory, "latent_correlations")
        print(subheading("Latent correlations"))
        print("Plotting latent correlations.")

        for set_name, latent_evaluation_set in latent_evaluation_sets.items():

            correlations_time_start = time()
            latent_correlation_matrix = metrics.correlation_matrix(
                latent_evaluation_set.values, axis="features")
            figure, figure_name = figures.plot_correlation_matrix(
                latent_correlation_matrix,
                axis_label="Latent units",
                name=["latent correlation matrix", set_name])
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=correlations_directory)
            correlations_duration = time() - correlations_time_start
            print(
                "    Latent correlation matrix for {} plotted ({})."
                .format(set_name, format_duration(correlations_duration)))

            correlations_time_start = time()
            most_correlated_latent_pairs = (
                metrics.most_correlated_variable_pairs_from_correlation_matrix(
                    latent_correlation_matrix,
                    n_limit=(
                        MAXIMUM_NUMBER_OF_CORRELATED_VARIABLE_PAIRS_TO_PLOT)))
            for latent_pair in most_correlated_latent_pairs:
                figure, figure_name = figures.plot_values(
                    latent_evaluation_set.values[:, latent_pair],
                    colour_coding="labels",
                    colouring_data_set=latent_evaluation_set,
                    figure_labels={
                        "x label": _axis_label_for_symbol(
                            symbol="z", coordinate=latent_pair[0] + 1),
                        "y label": _axis_label_for_symbol(
                            symbol="z", coordinate=latent_pair[1] + 1)
                    },
                    name="latent_correlations-{}-pair_{}_{}".format(
                        set_name, *latent_pair))
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=correlations_directory)
            correlations_duration = time() - correlations_time_start
            print(
                "    Most correlated latent pairs for {} plotted ({})."
                .format(set_name, format_duration(correlations_duration)))

            if latent_evaluation_set.number_of_features <= (
                    MAXIMUM_NUMBER_OF_VARIABLES_FOR_CORRELATION_PLOT):
                correlations_time_start = time()
                figure, figure_name = figures.plot_variable_correlations(
                    latent_evaluation_set.values,
                    latent_evaluation_set.feature_names,
                    colouring_data_set=latent_evaluation_set,
                    name=["latent correlations", set_name])
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=correlations_directory)
                correlations_duration = time() - correlations_time_start
                print(
                    "    Latent correlations for {} plotted ({})."
                    .format(set_name, format_duration(correlations_duration)))

            if latent_evaluation_set.has_labels:
                correlations_time_start = time()
                for latent_dimension in range(
                        latent_evaluation_set.number_of_features):
                    figure, figure_name = (
                        figures.plot_variable_label_correlations(
                            latent_evaluation_set.values[:, latent_dimension],
                            variable_name=_axis_label_for_symbol(
                                symbol="z", coordinate=latent_dimension + 1),
                            colouring_data_set=latent_evaluation_set,
                            name=(
                                "latent_correlations-{}-labels-"
                                "latent_dimension_{}".format(
                                    set_name, latent_dimension))))
                    figures.save_figure(
                        figure=figure,
                        name=figure_name,
                        options=export_options,
                        directory=correlations_directory)
                correlations_duration = time() - correlations_time_start
                print(
                    "    Labels correlated with latent dimensions for {} "
                    "plotted ({}).".format(
                        set_name, format_duration(correlations_duration)))

        print()

    if "latent_features" in included_analyses and "VAE" in model.type:

        latent_features_directory = os.path.join(
            analyses_directory, "latent_features")
        print(subheading("Latent features"))

        kl_divergences = load_kl_divergences(
            model=model,
            data_set_kind="evaluation",
            run_id=run_id,
            early_stopping=early_stopping,
            best_model=best_model
        )

        if kl_divergences is not None:

            sorted_kl_divergences_indices = kl_divergences.argsort()
            latent_factor_1 = sorted_kl_divergences_indices[0]
            latent_factor_2 = sorted_kl_divergences_indices[1]

        else:

            print(
                "The KL divergences for the evaluated latent features used "
                "to determine the two primary latent features are not "
                "available. The two most varying latent features are used "
                "instead."
                "\n"
            )

            latent_variances = latent_evaluation_sets["z"].values.var(axis=0)
            sorted_latent_variances_indices = latent_variances.argsort()
            latent_factor_1 = sorted_latent_variances_indices[0]
            latent_factor_2 = sorted_latent_variances_indices[1]

        print("Plotting latent features.")

        latent_features_time_start = time()
        figure, figure_name = figures.plot_values(
            latent_evaluation_sets["z"].values[
                :, [latent_factor_1, latent_factor_2]],
            colour_coding="labels",
            colouring_data_set=latent_evaluation_sets["z"],
            figure_labels={
                "x label": _axis_label_for_symbol(
                    symbol="z", coordinate=1),
                "y label": _axis_label_for_symbol(
                    symbol="z", coordinate=2)
            },
            name="latent_features-pair")
        figures.save_figure(
            figure=figure,
            name=figure_name,
            options=export_options,
            directory=latent_features_directory)
        latent_features_duration = time() - latent_features_time_start
        print(
            "    Second latent feature against first one plotted ({})."
            .format(format_duration(latent_features_duration)))

        if latent_evaluation_sets["z"].has_labels:
            latent_features_time_start = time()
            figure, figure_name = (
                figures.plot_variable_label_correlations(
                    latent_evaluation_sets["z"].values[:, latent_factor_1],
                    variable_name=_axis_label_for_symbol(
                        symbol="z", coordinate=1),
                    colouring_data_set=latent_evaluation_sets["z"],
                    name="latent_factor-labels"))
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=latent_features_directory)
            latent_features_duration = time() - latent_features_time_start
            print(
                "    Labels against first latent feature plotted ({})."
                .format(format_duration(latent_features_duration)))

        print()


def _build_path_for_analyses_directory(base_directory, model_name,
                                       run_id=None, subdirectories=None):

    analyses_directory = os.path.join(base_directory, model_name)

    if run_id is None:
        run_id = defaults["models"]["run_id"]
    if run_id:
        run_id = check_run_id(run_id)
        analyses_directory = os.path.join(
            analyses_directory,
            "run_{}".format(run_id)
        )

    if subdirectories:

        if not isinstance(subdirectories, list):
            subdirectories = [subdirectories]

        analyses_directory = os.path.join(analyses_directory, *subdirectories)

    return analyses_directory


def _parse_analyses(included_analyses=None):

    if not included_analyses:
        resulting_analyses = []
    else:
        if not isinstance(included_analyses, (list, tuple)):
            included_analyses = [included_analyses]

        resulting_analyses = set()

        for analysis in included_analyses:
            if analysis in ANALYSIS_GROUPS:
                group = analysis
                resulting_analyses.update(map(
                    normalise_string, ANALYSIS_GROUPS[group]))
            elif analysis in ANALYSIS_GROUPS["all"]:
                resulting_analyses.add(normalise_string(analysis))
            else:
                raise ValueError("Analysis `{}` not found.".format(analysis))

        resulting_analyses = list(resulting_analyses)

    return resulting_analyses
