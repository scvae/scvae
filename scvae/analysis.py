# ======================================================================== #
# 
# Copyright (c) 2017 - 2018 scVAE authors
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
from numpy import nan

import scipy

from miscellaneous.decomposition import (
    decompose,
    DECOMPOSITION_METHOD_NAMES,
    DECOMPOSITION_METHOD_LABEL
)

import sklearn.metrics.cluster

from matplotlib import pyplot
import matplotlib
import matplotlib.patches
import matplotlib.lines
import matplotlib.gridspec
import matplotlib.colors
from matplotlib.ticker import LogFormatterSciNotation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn

from PIL import Image

from pandas import DataFrame

from data import createLabelSorter, standard_deviation, sparsity

import os
import gzip
import pickle

import copy
import re

from time import time
from auxiliary import (
    loadNumberOfEpochsTrained, loadLearningCurves, loadAccuracies,
    loadCentroids, loadKLDivergences,
    checkRunID,
    formatTime, formatDuration,
    normaliseString, properString, capitaliseString, subheading
)
from miscellaneous.prediction import PREDICTION_METHOD_NAMES

import warnings

standard_palette = seaborn.color_palette('Set2', 8)
standard_colour_map = seaborn.cubehelix_palette(light = .95, as_cmap = True)
neutral_colour = (0.7, 0.7, 0.7)

lighter_palette = lambda N: seaborn.husl_palette(N, l = .75)
darker_palette = lambda N: seaborn.husl_palette(N, l = .55)

default_small_marker_size_in_scatter_plots = 1
default_large_marker_size_in_scatter_plots = 3
default_marker_size_in_scatter_plots = \
    default_small_marker_size_in_scatter_plots

default_marker_size_in_legends = 4

legend_marker_scale_from_marker_size = lambda marker_size: \
    default_marker_size_in_legends / marker_size

reset_plot_look = lambda: seaborn.set(
    context = "paper",
    style = "ticks",
    palette = standard_palette,
    color_codes = False,
    rc = {
        "lines.markersize": default_marker_size_in_scatter_plots,
        "legend.markerscale": legend_marker_scale_from_marker_size(
            default_marker_size_in_scatter_plots
        ),
        "figure.dpi": 200
    }
)
reset_plot_look()

figure_extension = ".png"
image_extension = ".png"

publication_figure_extension = ".tiff"
publication_dpi = 350
publication_copies = {
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

maximum_size_for_normal_statistics_computation = 5e8
maximum_number_of_examples_before_sampling_silhouette_score = 20000

maximum_number_of_bins_for_histograms = 20000

maximum_number_of_values_for_heat_maps = 5000 * 25000
maximum_number_of_examples_for_heat_maps = 10000
maximum_number_of_features_for_heat_maps = 10000
maximum_number_of_examples_for_dendrogram = 1000

maximum_number_of_features_for_t_sne = 100
maximum_number_of_examples_for_t_sne = 200000
maximum_number_of_pca_components_before_tsne = 50

maximum_number_of_examples_for_large_points_in_scatter_plots = 1000

number_of_random_examples = 100

evaluation_subset_maximum_number_of_examples = 25
evaluation_subset_maximum_number_of_examples_per_class = 3

profile_comparison_count_cut_off = 10.5

maximum_count_scales = [1, 5]
axis_limits_scales = [1, 5]

default_cutoffs = range(1, 10)

analysis_groups = {
    "simple": ["metrics", "images", "learning_curves", "accuracies"],
    "default": ["kl_heat_maps", "profile_comparisons", "distributions",
        "distances", "decompositions", "latent_space"],
    "complete": ["heat_maps", "latent_distributions", "latent_correlations",
        "feature_value_standard_deviations"]
}
analysis_groups["default"] += analysis_groups["simple"]
analysis_groups["complete"] += analysis_groups["default"]

def analyseData(data_sets,
    decomposition_methods = ["PCA"], highlight_feature_indices = [],
    analyses = ["default"], analysis_level = "normal",
    export_options = [], results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, "data")
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    analyses = parseAnalyses(analyses)
    
    if not isinstance(data_sets, list):
        data_sets = [data_sets]
    
    # Metrics
    
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
                    statistics(series, series_name, skip_sparsity = True)
                    for series_name, series in {
                        "count sum": data_set.count_sum
                    }.items()
                ])
            data_set_statistics.append(
                statistics(data_set.values, data_set.kind, tolerance = 0.5)
            )
        
        metrics_duration = time() - metrics_time_start
        print("Metrics calculated ({}).".format(
            formatDuration(metrics_duration)))
        
        ## Saving
        
        metrics_path = os.path.join(results_directory, "data_set_metrics.log")
        
        metrics_saving_time_start = time()
        
        with open(metrics_path, "w") as metrics_file:
            metrics_string = "Timestamp: {}\n".format(
                formatTime(metrics_saving_time_start)) \
                + "\n" \
                + "Features: {}\n".format(number_of_features) \
                + "Examples: {}\n".format(number_of_examples["full"]) \
                + "\n".join([
                    "{} examples: {}".format(capitaliseString(kind), number)
                    for kind, number in number_of_examples.items()
                    if kind != "full"
                ]) + "\n" \
                + "\n" \
                + "Metrics:\n" \
                + formatStatistics(data_set_statistics) \
                + "\n" \
                + "Series:\n" \
                + formatStatistics(histogram_statistics)
            metrics_file.write(metrics_string)
        
        metrics_saving_duration = time() - metrics_saving_time_start
        print("Metrics saved ({}).".format(formatDuration(
            metrics_saving_duration)))
        
        print()
        
        ## Displaying
        
        print(formatStatistics(data_set_statistics))
        print()
        
        print(formatStatistics(histogram_statistics, name = "Series"))
        print()
    
    # Loop over data sets
    
    for data_set in data_sets:
        
        print(subheading("Analyses of {} set".format(data_set.kind)))
        
        # Examples for data set
        
        if "images" in analyses and data_set.example_type == "images":
            print("Saving image of {} random examples from {} set.".format(
                number_of_random_examples, data_set.kind))
            image_time_start = time()
            image, image_name = combineImagesFromDataSet(
                data_set,
                number_of_random_examples,
                name = data_set.kind
            )
            saveImage(image, image_name, results_directory)
            image_duration = time() - image_time_start
            print("Image saved ({}).".format(formatDuration(image_duration)))
            print()
        
        # Distributions
        
        if "distributions" in analyses:
            analyseDistributions(
                data_set,
                cutoffs = default_cutoffs,
                analysis_level = analysis_level,
                export_options = export_options,
                results_directory = results_directory
            )
        
        # Heat map for data set
        
        if "heat_maps" in analyses:
            analyseMatrices(
                data_set,
                name=[data_set.kind],
                results_directory=results_directory
            )
        
        # Distance matrices
    
        if "distances" in analyses:
            analyseMatrices(
                data_set,
                name=[data_set.kind],
                plot_distances=True,
                results_directory=results_directory
            )
        
        # Decompositions
        
        if "decompositions" in analyses:
            analyseDecompositions(
                data_set,
                decomposition_methods = decomposition_methods,
                highlight_feature_indices = highlight_feature_indices,
                symbol = "x",
                title = "original space",
                specifier = lambda data_set: data_set.kind,
                analysis_level = analysis_level,
                export_options = export_options,
                results_directory = results_directory
            )
        
        # Feature value standard deviations
        
        if "feature_value_standard_deviations" in analyses:
            
            print("Computing and plotting feature value standard deviations:")
            
            feature_value_standard_deviations_directory = os.path.join(
                results_directory,
                "feature_value_standard_deviations"
            )
            
            time_start = time()
            
            feature_value_standard_deviations = data_set.values.std(axis=0)
            
            if isinstance(feature_value_standard_deviations, numpy.matrix):
                feature_value_standard_deviations = \
                    feature_value_standard_deviations.A
            
            feature_value_standard_deviations = \
                feature_value_standard_deviations.squeeze()
            
            duration = time() - time_start
            print("    Feature value standard deviations computed",
                      "({}).".format(formatDuration(duration)))
            
            ## Feature value standard_deviations
            
            time_start = time()
            
            figure, figure_name = plotSeries(
                series = feature_value_standard_deviations,
                x_label = data_set.tags["feature"] + "s",
                y_label = "{} standard deviations".format(
                    data_set.tags["type"]),
                sort = True,
                scale = "log",
                name = ["feature value standard deviations", data_set.kind]
            )
            saveFigure(figure, figure_name, export_options,
                feature_value_standard_deviations_directory)
            
            duration = time() - time_start
            print("    Feature value standard deviations plotted and saved",
                      "({}).".format(formatDuration(duration)))
            
            ## Distribution of feature value standard deviations
            
            time_start = time()
            
            figure, figure_name = plotHistogram(
                series = feature_value_standard_deviations,
                label = "{} {} standard deviations".format(
                    data_set.tags["feature"], data_set.tags["type"]
                ),
                normed = True,
                x_scale = "linear",
                y_scale = "log",
                name = ["feature value standard deviations", data_set.kind]
            )
            saveFigure(figure, figure_name, export_options,
                feature_value_standard_deviations_directory)
            
            duration = time() - time_start
            print("    Feature value standard deviation distribution",
                      "plotted and saved ({}).".format(
                          formatDuration(duration)))
            
            print()

def analyseModel(model, run_id = None, analyses = ["default"],
    analysis_level = "normal", export_options = [],
    results_directory = "results"):
    
    if run_id:
        run_id = checkRunID(run_id)
    
    # Setup
    
    number_of_epochs_trained = loadNumberOfEpochsTrained(model, run_id=run_id)
    epochs_string = "e_" + str(number_of_epochs_trained)
    
    results_directory = buildPathForResultDirectory(
        base_directory = results_directory,
        model_name = model.name,
        run_id = run_id,
        subdirectories = [epochs_string]
    )
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    analyses = parseAnalyses(analyses)
    
    # Learning curves
    
    if "learning_curves" in analyses:
        
        print(subheading("Learning curves"))
            
        print("Plotting learning curves.")
        learning_curves_time_start = time()
        
        learning_curves = loadLearningCurves(
            model = model,
            data_set_kinds = ["training", "validation"],
            run_id = run_id
        )
        
        figure, figure_name = plotLearningCurves(learning_curves, model.type)
        saveFigure(figure, figure_name, export_options, results_directory)
        
        if "video" in export_options:
            print("Plotting learning-curve evolutions for video.")
            for epoch in range(number_of_epochs_trained):
                figure, figure_name = plotLearningCurves(
                    learning_curves,
                    model.type,
                    epoch_slice = slice(epoch + 1),
                    global_y_lim = "video" in export_options
                )
                saveFigure(figure, figure_name, export_options,
                    os.path.join(results_directory, "learning_curve_evolution"))
        
        if model.type == "SNN":
            figure, figure_name = plotSeparateLearningCurves(learning_curves,
                loss = "log_likelihood")
            saveFigure(figure, figure_name, export_options, results_directory)
        elif "VAE" in model.type:
            figure, figure_name = plotSeparateLearningCurves(learning_curves,
                loss = ["lower_bound", "reconstruction_error"])
            saveFigure(figure, figure_name, export_options, results_directory)
            if model.type in ["GMVAE"]:
                figure, figure_name = plotSeparateLearningCurves(learning_curves,
                    loss = "kl_divergence_z")
                saveFigure(figure, figure_name, export_options,
                    results_directory)
                figure, figure_name = plotSeparateLearningCurves(
                    learning_curves,
                    loss = "kl_divergence_y"
                )
                saveFigure(figure, figure_name, export_options, results_directory)
            else:
                figure, figure_name = plotSeparateLearningCurves(learning_curves,
                    loss = "kl_divergence")
                saveFigure(figure, figure_name, export_options, results_directory)
    
        learning_curves_duration = time() - learning_curves_time_start
        print("Learning curves plotted and saved ({}).".format(
            formatDuration(learning_curves_duration)))
    
        print()
    
    # Accuracies
    
    if "accuracies" in analyses:
        
        accuracies_time_start = time()
        
        accuracies = loadAccuracies(
            model = model,
            run_id = run_id,
            data_set_kinds = ["training", "validation"]
        )
        
        if accuracies is not None:
            
            print(subheading("Accuracies"))
            
            print("Plotting accuracies.")
            
            figure, figure_name = plotAccuracies(accuracies)
            saveFigure(figure, figure_name, export_options, results_directory)
            
            superset_accuracies = loadAccuracies(
                model = model,
                data_set_kinds = ["training", "validation"],
                superset = True,
                run_id = run_id
            )
            
            if superset_accuracies is not None:
                figure, figure_name = plotAccuracies(superset_accuracies,
                    name = "superset")
                saveFigure(figure, figure_name, export_options, results_directory)
            
            accuracies_duration = time() - accuracies_time_start
            print("Accuracies plotted and saved ({}).".format(
                formatDuration(accuracies_duration)))
            
            print()
    
    # Heat map of KL for all latent neurons
    
    if "kl_heat_maps" in analyses and "VAE" in model.type:
        
        print(subheading("KL divergence"))
    
        print("Plotting logarithm of KL divergence heat map.")
        heat_map_time_start = time()
        
        KL_neurons = loadKLDivergences(model = model, run_id = run_id)
    
        KL_neurons = numpy.sort(KL_neurons, axis = 1)
        log_KL_neurons = numpy.log(KL_neurons)
        
        figure, figure_name = plotKLDivergenceEvolution(KL_neurons)
        saveFigure(figure, figure_name, export_options, results_directory)
        
        heat_map_duration = time() - heat_map_time_start
        print("Heat map plotted and saved ({}).".format(
            formatDuration(heat_map_duration)))
        
        print()
    
    # Latent distributions
    
    if "latent_distributions" in analyses and model.type == "GMVAE":
        
        print(subheading("Latent distributions"))
        
        centroids = loadCentroids(
            model = model,
            data_set_kinds = "validation",
            run_id = run_id
        )
        
        centroids_directory = os.path.join(results_directory, "centroids_evolution")
        
        for distribution, distribution_centroids in centroids.items():
            
            if distribution_centroids:
                
                centroids_time_start = time()
                
                centroid_probabilities = distribution_centroids["probabilities"]
                centroid_means = distribution_centroids["means"]
                centroid_covariance_matrices = \
                    distribution_centroids["covariance_matrices"]

                E, K, L = centroid_means.shape

                if K <= 1:
                    continue

                print("Plotting evolution of latent {} parameters.".format(
                    distribution))

                if L > 2:
                    _, distribution_centroids_decomposed = decompose(
                        centroid_means[-1],
                        centroids = distribution_centroids
                    )
                    decomposed = True
                else:
                    distribution_centroids_decomposed = distribution_centroids
                    decomposed = False
                
                centroid_means_decomposed = \
                    distribution_centroids_decomposed["means"]
                
                figure, figure_name = plotEvolutionOfCentroidProbabilities(
                    centroid_probabilities, distribution)
                saveFigure(figure, figure_name, export_options, centroids_directory)
                
                figure, figure_name = plotEvolutionOfCentroidMeans(
                    centroid_means_decomposed, distribution, decomposed)
                saveFigure(figure, figure_name, export_options, centroids_directory)
                
                figure, figure_name = plotEvolutionOfCentroidCovarianceMatrices(
                    centroid_covariance_matrices, distribution)
                saveFigure(figure, figure_name, export_options, centroids_directory)
                
                centroids_duration = time() - centroids_time_start
                print("Evolution of latent {} parameters plotted and saved ({})"\
                    .format(distribution, formatDuration(centroids_duration)))
                
                print()

        if "video" in export_options:
            print("Plotting evolution of latent class probabilities for video")
            centroid_prior_probabilities = centroids["prior"]["probabilities"]
            centroid_posterior_probabilities = centroids["posterior"]["probabilities"]
            for i in range(number_of_epochs_trained):
                figure, figure_name = plotEvolutionOfCentroidProbabilities(
                    centroid_prior_probabilities[:(i+1), :],
                    distribution = "prior",
                    figure = None,
                    linestyle = "dashed",
                    name = ["epoch", i]
                )                
                figure, figure_name = plotEvolutionOfCentroidProbabilities(
                    centroid_posterior_probabilities[:(i+1), :],
                    distribution = "posterior",
                    figure = figure,
                    linestyle = "solid",
                    name = ["epoch", i]
                )
                saveFigure(figure, figure_name, export_options, centroids_directory)

            print()

def analyseIntermediateResults(learning_curves = None, epoch_start = None,
    epoch = None, latent_values = None, data_set = None, centroids = None,
    model_name = None, run_id = None, model_type = None,
    results_directory = "results"):
    
    if run_id:
        run_id = checkRunID(run_id)
    
    # Setup
    
    export_options = []
    results_directory = buildPathForResultDirectory(
        base_directory = results_directory,
        model_name = model_name,
        run_id = run_id,
        subdirectories = ["intermediate"]
    )
    
    # Learning curves
    
    print("Plotting learning curves.")
    learning_curves_time_start = time()
    
    if epoch is not None:
        epoch_name = "epoch-{}".format(epoch + 1)
    else:
        epoch_name = None

    figure, figure_name = plotLearningCurves(
        learning_curves,
        model_type,
        epoch_offset = epoch_start
    )
    saveFigure(figure, figure_name, export_options, results_directory)
    
    learning_curves_duration = time() - learning_curves_time_start
    print("Learning curves plotted and saved ({}).".format(
        formatDuration(learning_curves_duration)))
    
    if latent_values is not None:
    
        # Latent variables
    
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
                centroids = centroids,
                method = "PCA",
                number_of_components = 2
            )

            decompose_duration = time() - decompose_time_start
            print("{} decomposed ({}).".format(
                capitaliseString(latent_set_name),
                formatDuration(decompose_duration))
            )
        
        symbol = "z"
    
        x_label = axisLabelForSymbol(
            symbol = symbol,
            coordinate = 1,
            decomposition_method = decomposition_method,
        )
        y_label = axisLabelForSymbol(
            symbol = symbol,
            coordinate = 2,
            decomposition_method = decomposition_method,
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
            figure, figure_name = plotValues(
                latent_values_decomposed,
                colour_coding = "labels",
                colouring_data_set = data_set,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
            )
            saveFigure(figure, figure_name, export_options = ["video"],
                results_directory = results_directory)
            if data_set.label_superset is not None:
                figure, figure_name = plotValues(
                    latent_values_decomposed,
                    colour_coding = "superset labels",
                    colouring_data_set = data_set,
                    centroids = centroids_decomposed,
                    figure_labels = figure_labels,
                    name = name
                )
                saveFigure(figure, figure_name, export_options = ["video"],    
                    results_directory = results_directory)
        else:
            figure, figure_name = plotValues(
                latent_values_decomposed,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
            )
            saveFigure(figure, figure_name, export_options = ["video"],
                results_directory = results_directory)
    
        if centroids:
            analyseCentroidProbabilities(
                centroids, epoch_name,
                export_options = ["video"],
                results_directory = results_directory)
    
        plot_duration = time() - plot_time_start
        print("{} plotted and saved ({}).".format(
            capitaliseString(latent_set_name), formatDuration(plot_duration)))

def analyseResults(evaluation_set, reconstructed_evaluation_set,
    latent_evaluation_sets, model, run_id = None,
    decomposition_methods = ["PCA"], evaluation_subset_indices = set(),
    highlight_feature_indices = [],
    prediction_details = None,
    early_stopping = False, best_model = False,
    analyses = ["default"], analysis_level = "normal",
    export_options = [], results_directory = "results"):
    
    if early_stopping and best_model:
        raise ValueError("Early-stopping model and best model cannot be"
            + " evaluated at the same time.")
    
    if run_id:
        run_id = checkRunID(run_id)
    
    # Setup
    
    print("Setting up results analyses.")
    setup_time_start = time()
    
    number_of_epochs_trained = loadNumberOfEpochsTrained(
        model = model,
        run_id = run_id,
        early_stopping = early_stopping,
        best_model = best_model
    )
    
    M = evaluation_set.number_of_examples
    
    analyses = parseAnalyses(analyses)
    
    ## Comparison arrays
    
    if ("metrics" in analyses or "heat_maps" in analyses) \
        and analysis_level == "extensive":
        
        x_diff = reconstructed_evaluation_set.values - evaluation_set.values
        x_log_ratio = numpy.log1p(reconstructed_evaluation_set.values) \
            - numpy.log1p(evaluation_set.values)
    
    ## Directory path
    
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
    
    results_directory = buildPathForResultDirectory(
        base_directory = results_directory,
        model_name = model.name,
        run_id = run_id,
        subdirectories = [evaluation_directory]
    )
    
    if evaluation_set.kind != "test":
        results_directory = os.path.join(results_directory, evaluation_set.kind)
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    setup_duration = time() - setup_time_start
    print("Finished setting up ({}).".format(formatDuration(setup_duration)))
    print()
    
    # Metrics
    
    if "metrics" in analyses:
    
        print(subheading("Metrics"))
        
        ## Loading
        
        print("Loading results from model log directory.")
        loading_time_start = time()
        
        evaluation_eval = loadLearningCurves(
            model = model,
            data_set_kinds = "evaluation",
            run_id = run_id,
            early_stopping = early_stopping,
            best_model = best_model
        )
        accuracy_eval = loadAccuracies(
            model = model,
            data_set_kinds = "evaluation",
            run_id = run_id,
            early_stopping = early_stopping,
            best_model = best_model
        )
        superset_accuracy_eval = loadAccuracies(
            model = model,
            data_set_kinds = "evaluation",
            superset = True,
            run_id = run_id,
            early_stopping = early_stopping,
            best_model = best_model
        )
        
        loading_duration = time() - loading_time_start
        print("Results loaded ({}).".format(formatDuration(loading_duration)))
        print()
        
        ## Calculating
        
        print("Calculating metrics for results.")
        metrics_time_start = time()
        
        ### Statistics
        evaluation_set_statistics = [
            statistics(data_set.values, data_set.version, tolerance = 0.5)
                for data_set in [evaluation_set, reconstructed_evaluation_set]
        ]
        
        ### Comparison statistics
        if analysis_level == "extensive":
            evaluation_set_statistics.append(statistics(numpy.abs(x_diff),
                "differences", skip_sparsity = True))
            evaluation_set_statistics.append(statistics(numpy.abs(x_log_ratio),
                "log-ratios", skip_sparsity = True))
        
        ### Count accuracies
        if analysis_level == "extensive":
            
            if evaluation_set.values.max() > 20:
                count_accuracy_method = "orders of magnitude"
            else:
                count_accuracy_method = None
            
            count_accuracies = computeCountAccuracies(
                evaluation_set.values,
                reconstructed_evaluation_set.values,
                method = count_accuracy_method
            )
        
        else:
            count_accuracies = None
        
        ### Clustering metrics
        
        clustering_metric_values = computeClusteringMetrics(evaluation_set)
        
        metrics_duration = time() - metrics_time_start
        print("Metrics calculated ({}).".format(
            formatDuration(metrics_duration)))
        
        ## Saving
        
        metrics_saving_time_start = time()
        
        metrics_log_filename = "{}-metrics".format(evaluation_set.kind)
        metrics_log_path = os.path.join(results_directory,
            metrics_log_filename + ".log")
        metrics_dictionary_path = os.path.join(results_directory,
            metrics_log_filename + ".pkl.gz")
        
        with open(metrics_log_path, "w") as metrics_file:
            metrics_string = "Timestamp: {}".format(
                formatTime(metrics_saving_time_start))
            metrics_string += "\n"
            metrics_string += "Number of epochs trained: {}".format(
                number_of_epochs_trained)
            metrics_string += "\n"*2
            metrics_string += "Evaluation:"
            metrics_string += "\n"
            if model.type == "SNN":
                metrics_string += \
                    "    log-likelihood: {:.5g}.\n".format(
                        evaluation_eval["log_likelihood"][-1])
            elif "VAE" in model.type:
                metrics_string += \
                    "    ELBO: {:.5g}.\n".format(
                        evaluation_eval["lower_bound"][-1]) + \
                    "    ENRE: {:.5g}.\n".format(
                        evaluation_eval["reconstruction_error"][-1])
                if model.type == "VAE":
                    metrics_string += \
                        "    KL: {:.5g}.\n".format(
                            evaluation_eval["kl_divergence"][-1])
                elif model.type in ["GMVAE"]:
                    metrics_string += \
                        "    KL_z: {:.5g}.\n".format(
                            evaluation_eval["kl_divergence_z"][-1]) + \
                        "    KL_y: {:.5g}.\n".format(
                            evaluation_eval["kl_divergence_y"][-1])
            if accuracy_eval is not None:
                metrics_string += "    Accuracy: {:6.2f} %.\n".format(
                    100 * accuracy_eval[-1])
            if superset_accuracy_eval is not None:
                metrics_string += "    Accuracy (superset): {:6.2f} %.\n"\
                    .format(100 * superset_accuracy_eval[-1])
            metrics_string += "\n" \
                + formatStatistics(evaluation_set_statistics) + "\n"
            if count_accuracies:
                metrics_string += "\n" + \
                    formatCountAccuracies(count_accuracies) + "\n"
            metrics_file.write(metrics_string)
        
        with gzip.open(metrics_dictionary_path, "w") as metrics_file:
            metrics_dictionary = {
                "timestamp": metrics_saving_time_start,
                "number of epochs trained": number_of_epochs_trained,
                "evaluation": evaluation_eval,
                "accuracy": accuracy_eval,
                "superset_accuracy": superset_accuracy_eval,
                "statistics": evaluation_set_statistics,
                "count accuracies": count_accuracies,
            }
            pickle.dump(metrics_dictionary, metrics_file)
        
        if prediction_details:
            
            prediction_log_filename = "{}-prediction-{}".format(
                evaluation_set.kind, prediction_details["id"])
            prediction_log_path = os.path.join(results_directory,
                prediction_log_filename + ".log")
            prediction_dictionary_path = os.path.join(results_directory,
                prediction_log_filename + ".pkl.gz")
            
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
            
            if prediction_details["decomposition_method"]:
                prediction_string_parts.append(
                    "Decomposition method: {}-d {}".format(
                        prediction_details["decomposition_dimensionality"],
                        prediction_details["decomposition_method"]
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
                "decomposition method": prediction_details["decomposition_method"],
                "decomposition dimensionality":
                    prediction_details["decomposition_dimensionality"],
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
        
        ## Displaying
        
        print(formatStatistics(evaluation_set_statistics))
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
        
        if count_accuracies:
            print(formatCountAccuracies(count_accuracies))
            print()
    
    # Reconstructions
    
    if "images" in analyses and \
        reconstructed_evaluation_set.example_type == "images" \
        or "profile_comparisons" in analyses:
        
        print(subheading("Reconstructions"))
    
    ## Examples
    
    if "images" in analyses and \
        reconstructed_evaluation_set.example_type == "images":
        
        print("Saving image of {} random examples".format(
            number_of_random_examples),
            "from reconstructed {} set.".format(
                evaluation_set.kind
            ))
        image_time_start = time()
        image, image_name = combineImagesFromDataSet(
            reconstructed_evaluation_set,
            number_of_random_examples,
            name = reconstructed_evaluation_set.version
        )
        saveImage(image, image_name, results_directory)
        image_duration = time() - image_time_start
        print("Image saved ({}).".format(formatDuration(image_duration)))
        print()

    ## Profile comparisons
    
    if "profile_comparisons" in analyses:
        
        print("Plotting profile comparisons.")
        
        image_comparisons_directory = os.path.join(
                results_directory, "image_comparisons")
        
        profile_comparisons_directory = os.path.join(
                results_directory, "profile_comparisons")
        
        profile_comparisons_time_start = time()
        
        y_cutoff = profile_comparison_count_cut_off
        
        for i in evaluation_subset_indices:
            
            observed_series = evaluation_set.values[i]
            expected_series = reconstructed_evaluation_set.values[i]
            example_name = str(evaluation_set.example_names[i])
            
            if evaluation_set.has_labels:
                example_label = str(evaluation_set.labels[i])
            
            if reconstructed_evaluation_set.total_standard_deviations \
                is not None:
                expected_series_total_standard_deviations = \
                    reconstructed_evaluation_set.total_standard_deviations[i]
            else:
                expected_series_total_standard_deviations = None
            
            if reconstructed_evaluation_set.explained_standard_deviations \
                is not None:
                
                expected_series_explained_standard_deviations = \
                    reconstructed_evaluation_set.explained_standard_deviations[i]
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
                    figure, figure_name = plotProfileComparison(
                        observed_series,
                        expected_series,
                        expected_series_total_standard_deviations,
                        expected_series_explained_standard_deviations,
                        x_name = evaluation_set.tags["feature"],
                        y_name = evaluation_set.tags["value"],
                        sort = sort_profile_comparison,
                        sort_by = "expected",
                        sort_direction = "descending",
                        x_scale = "log",
                        y_scale = y_scale,
                        name = example_name_parts
                    )
                    saveFigure(figure, figure_name, export_options,
                        profile_comparisons_directory)
            
            if maximum_count > 3 * y_cutoff:
                for y_scale in ["linear", "log", "both"]:
                    example_name_parts = example_name_base_parts.copy()
                    example_name_parts.append("cutoff")
                    example_name_parts.append(y_scale)
                    figure, figure_name = plotProfileComparison(
                        observed_series,
                        expected_series,
                        expected_series_total_standard_deviations,
                        expected_series_explained_standard_deviations,
                        x_name = evaluation_set.tags["feature"],
                        y_name = evaluation_set.tags["value"],
                        sort = True,
                        sort_by = "expected",
                        sort_direction = "descending",
                        x_scale = "log",
                        y_scale = y_scale,
                        y_cutoff = y_cutoff,
                        name = example_name_parts
                    )
                    saveFigure(figure, figure_name, export_options,
                        profile_comparisons_directory)
            
            # Plot image examples for subset
            if evaluation_set.example_type == "images":
                example_name_parts = ["original"] + example_name_base_parts
                image, image_name = combineImagesFromDataSet(
                    evaluation_set,
                    number_of_random_examples = 1,
                    indices = [i],
                    name = example_name_parts
                )
                saveImage(image, image_name, image_comparisons_directory)
            
            if reconstructed_evaluation_set.example_type == "images":
                example_name_parts = ["reconstructed"] + example_name_base_parts
                image, image_name = combineImagesFromDataSet(
                    reconstructed_evaluation_set,
                    number_of_random_examples = 1,
                    indices = [i],
                    name = example_name_parts
                )
                saveImage(image, image_name, image_comparisons_directory)
            
            if analysis_level == "limited":
                break
        
        profile_comparisons_duration = time() - profile_comparisons_time_start
        print("Profile comparisons plotted and saved ({}).".format(
            formatDuration(profile_comparisons_duration)))
        
        print()
    
    # Distributions
    
    evaluation_set_maximum_value = evaluation_set.values.max()
    
    if "distributions" in analyses:
        
        print(subheading("Distributions"))
        
        analyseDistributions(
            reconstructed_evaluation_set,
            colouring_data_set = evaluation_set,
            preprocessed = evaluation_set.preprocessing_methods,
            original_maximum_count = evaluation_set_maximum_value,
            analysis_level = analysis_level,
            export_options = export_options,
            results_directory = results_directory
        )
    
    ## Reconstructions decomposed
    
    if "decompositions" in analyses:
        
        print(subheading("Decompositions"))
        
        analyseDecompositions(
            reconstructed_evaluation_set,
            colouring_data_set = evaluation_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            prediction_details = prediction_details,
            symbol = "\\tilde{{x}}",
            pca_limits = evaluation_set.pca_limits,
            title = "reconstruction space",
            analysis_level = analysis_level,
            export_options = export_options,
            results_directory = results_directory,
        )
        
        if analysis_level == "extensive":
            
            ## Reconstructions plotted in original decomposed space
            analyseDecompositions(
                evaluation_set,
                reconstructed_evaluation_set,
                colouring_data_set = evaluation_set,
                decomposition_methods = decomposition_methods,
                highlight_feature_indices = highlight_feature_indices,
                prediction_details = prediction_details,
                symbol = "x",
                pca_limits = evaluation_set.pca_limits,
                title = "original space",
                analysis_level = analysis_level,
                export_options = export_options,
                results_directory = results_directory,
            )
    
    # Heat maps
    
    if "heat_maps" in analyses:
        
        print(subheading("Heat maps"))
        
        ## Reconstructions
        
        analyseMatrices(
            reconstructed_evaluation_set,
            plot_distances=False,
            results_directory=results_directory
        )
        
        ## Latent
        
        analyseMatrices(
            latent_evaluation_sets["z"],
            plot_distances=False,
            results_directory=results_directory
        )
        
        if analysis_level == "extensive" and \
            reconstructed_evaluation_set.number_of_values \
            <= maximum_number_of_values_for_heat_maps:
            
            print("Plotting comparison heat maps.")
            
            ## Differences
        
            heat_maps_time_start = time()
            
            figure, figure_name = plotHeatMap(
                x_diff,
                labels = reconstructed_evaluation_set.labels,
                x_name = evaluation_set.tags["feature"].capitalize() + "s",
                y_name = evaluation_set.tags["example"].capitalize() + "s",
                z_name = "Differences",
                z_symbol = "\\tilde{{x}} - x",
                name = "difference",
                center = 0
            )
            saveFigure(figure, figure_name, export_options, heat_maps_directory)
            
            heat_maps_duration = time() - heat_maps_time_start
            print("    Difference heat map plotted and saved ({})." \
                .format(formatDuration(heat_maps_duration)))
            
            ## log-ratios
            
            heat_maps_time_start = time()
            
            figure, figure_name = plotHeatMap(
                x_log_ratio,
                labels = reconstructed_evaluation_set.labels,
                x_name = evaluation_set.tags["feature"].capitalize() + "s",
                y_name = evaluation_set.tags["example"].capitalize() + "s",
                z_name = "log-ratios",
                z_symbol = "\\log \\frac{{\\tilde{{x}} + 1}}{{x + 1}}",
                name = "log_ratio",
                center = 0
            )
            saveFigure(figure, figure_name, export_options, heat_maps_directory)
            
            heat_maps_duration = time() - heat_maps_time_start
            print("    log-ratio heat map plotted and saved ({})." \
                .format(formatDuration(heat_maps_duration)))
    
    print()
    
    # Distance matrices
    
    if "distances" in analyses:
        
        print(subheading("Distances"))
        
        analyseMatrices(
            reconstructed_evaluation_set,
            plot_distances=True,
            results_directory=results_directory
        )
        
        analyseMatrices(
            latent_evaluation_sets["z"],
            plot_distances=True,
            results_directory=results_directory
        )
    
    # Latent space
    
    if "latent_space" in analyses and "VAE" in model.type:
        
        print(subheading("Latent space"))
        
        if model.latent_distribution_name == "gaussian mixture":
        
            print("Loading centroids from model log directory.")
            loading_time_start = time()
            
            centroids = loadCentroids(
                model = model,
                data_set_kinds = "evaluation",
                run_id = run_id,
                early_stopping = early_stopping,
                best_model = best_model
            )
            
            loading_duration = time() - loading_time_start
            print("Centroids loaded ({}).".format(
                formatDuration(loading_duration)))
            print()
        
        else:
            centroids = None
        
        analyseDecompositions(
            latent_evaluation_sets,
            centroids = centroids,
            colouring_data_set = evaluation_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            prediction_details = prediction_details,
            title = "latent space",
            specifier = lambda data_set: data_set.version,
            analysis_level = analysis_level,
            export_options = export_options,
            results_directory = results_directory,
        )
        
        if centroids:
            analyseCentroidProbabilities(
                centroids,
                analysis_level = "normal",
                results_directory = results_directory
            )
            
            print()
    
    # Latent correlations
    
    if "latent_correlations" in analyses and "VAE" in model.type:
        
        print(subheading("Latent correlations"))
        
        correlations_directory = os.path.join(results_directory,
            "latent_correlations")
        
        print("Plotting latent correlations.")
        
        for set_name in latent_evaluation_sets:
            correlations_time_start = time()
            
            latent_evaluation_set = latent_evaluation_sets[set_name]
            figure, figure_name = plotVariableCorrelations(
                latent_evaluation_set.values,
                latent_evaluation_set.feature_names,
                latent_evaluation_set,
                name = ["latent correlations", set_name]
            )
            saveFigure(figure, figure_name, export_options,
                correlations_directory)
            
            correlations_duration = time() - correlations_time_start
            print("    Latent correlations for {} plotted ({}).".format(
                set_name,
                formatDuration(correlations_duration)
            ))
        
        print()

def analyseDistributions(data_set, colouring_data_set = None,
    cutoffs = None, preprocessed = False, original_maximum_count = None,
    analysis_level = "normal", export_options = [],
    results_directory = "results"):
    
    # Setup
    
    if not colouring_data_set:
        colouring_data_set = data_set
    
    ## Naming
    
    data_set_title = data_set.kind + " set"
    data_set_name = data_set.kind
    
    distribution_directory = os.path.join(results_directory, "histograms")
    
    if data_set.version != "original":
        data_set_title = data_set.version + " " + data_set_title
        data_set_name = None
    
    ## Discreteness
    
    data_set_discreteness = data_set.discreteness and not preprocessed
    
    ## Maximum values for count histograms
    
    maximum_counts = {0: None}
    
    if original_maximum_count and data_set.example_type == "counts":
        
        for maximum_count_scale in maximum_count_scales:
            
            maximum_count_indcises = data_set.values \
                <= maximum_count_scale * original_maximum_count
            
            number_of_outliers = data_set.number_of_values \
                - maximum_count_indcises.sum()
            
            if number_of_outliers:
                maximum_count = numpy.ceil(
                    data_set.values[maximum_count_indcises].max())
                maximum_counts[maximum_count_scale] = maximum_count
    
    # Plotting
    
    print("Plotting distributions for {}.".format(data_set_title))
    
    ## Class distribution
    
    if data_set.number_of_classes and data_set.number_of_classes < 100 \
        and colouring_data_set == data_set:
        
        distribution_time_start = time()
        
        figure, figure_name = plotClassHistogram(
            labels = data_set.labels,
            class_names = data_set.class_names,
            class_palette = data_set.class_palette,
            normed = True,
            scale = "linear",
            label_sorter = data_set.label_sorter,
            name = data_set_name
        )
        saveFigure(figure, figure_name, export_options, distribution_directory)
        
        distribution_duration = time() - distribution_time_start
        print("    Class distribution plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
    
    if data_set.label_superset and colouring_data_set == data_set:
        
        distribution_time_start = time()
        
        figure, figure_name = plotClassHistogram(
            labels = data_set.superset_labels,
            class_names = data_set.superset_class_names,
            class_palette = data_set.superset_class_palette,
            normed = True,
            scale = "linear",
            label_sorter = data_set.superset_label_sorter,
            name = [data_set_name, "superset"]
        )
        saveFigure(figure, figure_name, export_options, distribution_directory)
        
        distribution_duration = time() - distribution_time_start
        print("    Superset class distribution plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
    
    ## Count distribution
    
    if scipy.sparse.issparse(data_set.values):
        series = data_set.values.data
        excess_zero_count = data_set.values.size - series.size
    else:
        series = data_set.values.reshape(-1)
        excess_zero_count = 0
    
    for maximum_count_scale, maximum_count in maximum_counts.items():
        
        distribution_time_start = time()
            
        if maximum_count:
            count_histogram_name = [
                "counts",
                data_set_name,
                "maximum_count_scale",
                maximum_count_scale
            ]
        else:
            count_histogram_name = ["counts", data_set_name]
        
        for x_scale in ["linear", "log"]:
            figure, figure_name = plotHistogram(
                series = series,
                excess_zero_count = excess_zero_count,
                label = data_set.tags["value"].capitalize() + "s",
                discrete = data_set_discreteness,
                normed = True,
                x_scale = x_scale,
                y_scale = "log",
                maximum_count = maximum_count,
                name = count_histogram_name
            )
            saveFigure(figure, figure_name, export_options,
                distribution_directory)
        
        if maximum_count:
            maximum_count_string = " (with a maximum count of {:d})".format(
                int(maximum_count))
        else:
            maximum_count_string = ""
        
        distribution_duration = time() - distribution_time_start
        print("    Count distribution{} plotted and saved ({})."\
            .format(maximum_count_string,
                formatDuration(distribution_duration)))
    
    ## Count distributions with cut-off

    if analysis_level == "extensive" and cutoffs and \
        data_set.example_type == "counts":
        
        distribution_time_start = time()

        for cutoff in cutoffs:
            figure, figure_name = plotCutOffCountHistogram(
                series = series,
                excess_zero_count = excess_zero_count,
                cutoff = cutoff,
                normed = True,
                scale = "log",
                name = data_set_name
            )
            saveFigure(figure, figure_name, export_options,
                distribution_directory + "-counts")

        distribution_duration = time() - distribution_time_start
        print("    Count distributions with cut-offs plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
    
    ## Count sum distribution
    
    distribution_time_start = time()
    
    figure, figure_name = plotHistogram(
        series = data_set.count_sum,
        label = "Total number of {}s per {}".format(
            data_set.tags["item"], data_set.tags["example"]
        ),
        normed = True,
        y_scale = "log",
        name = ["count sum", data_set_name]
    )
    saveFigure(figure, figure_name, export_options, distribution_directory)
    
    distribution_duration = time() - distribution_time_start
    print("    Count sum distribution plotted and saved ({})."\
        .format(formatDuration(distribution_duration)))
    
    ## Count distributions and count sum distributions for each class
    
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
        
        N = colouring_data_set.number_of_features
        
        distribution_time_start = time()
        
        if not class_palette:
            index_palette = lighter_palette(colouring_data_set.number_of_classes)
            class_palette = {class_name: index_palette[i] for i, class_name in
                             enumerate(sorted(class_names,
                             key = label_sorter))}

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
            
            figure, figure_name = plotHistogram(
                series = series,
                excess_zero_count = excess_zero_count,
                label = data_set.tags["value"].capitalize() + "s",
                discrete = data_set_discreteness,
                normed = True,
                y_scale = "log",
                colour = class_palette[class_name],
                name = ["counts", data_set_name, "class", class_name]
            )
            saveFigure(figure, figure_name, export_options,
                class_count_distribution_directory)
    
        distribution_duration = time() - distribution_time_start
        print("    Count distributions for each class plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
        
        distribution_time_start = time()
        
        for class_name in class_names:
            class_indices = labels == class_name
            if not class_indices.any():
                continue
            figure, figure_name = plotHistogram(
                series = data_set.count_sum[class_indices],
                label = "Total number of {}s per {}".format(
                    data_set.tags["item"], data_set.tags["example"]
                ),
                normed = True,
                y_scale = "log",
                colour = class_palette[class_name],
                name = ["count sum", data_set_name, "class", class_name]
            )
            saveFigure(figure, figure_name, export_options,
                class_count_distribution_directory)
    
        distribution_duration = time() - distribution_time_start
        print("    " + \
            "Count sum distributions for each class plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
    
    print()

def analyseMatrices(data_set, plot_distances=False,
    name=None, export_options=[], results_directory="results"):
    
    # Naming
    
    if plot_distances:
        base_name = "distances"
    else:
        base_name = "heat_maps"
    
    results_directory = os.path.join(results_directory, base_name)
    
    if not name:
        name = []
    elif not isinstance(name, list):
        name = [name]
    
    name.insert(0,base_name)
    
    # Subsampling indices (if necessary)
    
    random_state = numpy.random.RandomState(57)
    shuffled_indices = random_state.permutation(data_set.number_of_examples)
    
    # Feature selection for plotting (if necessary)
    
    feature_indices_for_plotting = None
    
    if not plot_distances and data_set.number_of_features \
        > maximum_number_of_features_for_heat_maps:
        
        feature_variances = data_set.values.var(axis=0)
        
        if isinstance(feature_variances, numpy.matrix):
            feature_variances = feature_variances.A.squeeze()
        
        feature_indices_for_plotting = numpy.argsort(feature_variances)\
            [-maximum_number_of_features_for_heat_maps:]
        feature_indices_for_plotting.sort()
    
    # Class palette
    
    class_palette = data_set.class_palette
    
    if data_set.labels is not None and not class_palette:
        index_palette = lighter_palette(data_set.number_of_classes)
        class_palette = {
            class_name: tuple(index_palette[i]) for i, class_name in
            enumerate(sorted(data_set.class_names,
                             key = data_set.label_sorter))
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
    
    # Loop over sorting methods
    
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
            
            if sorting_method == "hierarchical_clustering" \
                and data_set.number_of_examples \
                    > maximum_number_of_examples_for_dendrogram:
                    
                    sample_size = maximum_number_of_examples_for_dendrogram
            
            elif data_set.number_of_examples \
                > maximum_number_of_examples_for_heat_maps:
                
                sample_size = maximum_number_of_examples_for_heat_maps
            
            else:
                sample_size = None
            
            indices = numpy.arange(data_set.number_of_examples)
            
            if sample_size:
                indices = shuffled_indices[:sample_size]
                example_label = "{} randomly sampled {}".format(
                    sample_size, data_set.tags["example"] + "s"
                )
            
            figure, figure_name = plotMatrix(
                feature_matrix=data_set.values[indices],
                plot_distances=plot_distances,
                example_label=example_label,
                feature_label=feature_label,
                value_label=value_label,
                sorting_method=sorting_method,
                distance_metric=distance_metric,
                labels=data_set.labels[indices]
                    if data_set.labels is not None else None,
                label_kind=data_set.tags["class"],
                class_palette=class_palette,
                feature_indices_for_plotting=feature_indices_for_plotting,
                name_parts=name + [
                    data_set.version,
                    distance_metric,
                    sorting_method
                ]
            )
            saveFigure(figure, figure_name, export_options, results_directory)
            
            duration = time() - start_time
            
            plot_string_parts = []
            
            plot_kind_string = "Heat map for {} values".format(data_set.version)
            
            if plot_distances:
                plot_kind_string = "{} distances in {} space".format(
                    distance_metric.capitalize(),
                    data_set.version
                )
            
            subsampling_string = ""
            
            if sample_size:
                subsampling_string = "{} {} randomly sampled examples"\
                    .format("for" if plot_distances else "of", sample_size)
            
            sort_string = "sorted using {}".format(
                sorting_method.replace("_", " ")
            )
            
            if not plot_distances \
                and sorting_method == "hierarchical_clustering":
                    sort_string += " (with {} distances)".format(distance_metric)
            
            print("    " + " ".join([s for s in [
                plot_kind_string,
                subsampling_string,
                sort_string,
                "plotted and saved",
                "({})".format(formatDuration(duration))
            ] if s]) + ".")
        
    print()

def analyseDecompositions(data_sets, other_data_sets = [], centroids = None,
    colouring_data_set = None, decomposition_methods = ["PCA"],
    highlight_feature_indices = [],
    prediction_details = None,
    symbol = None, pca_limits = None,
    title = "data set", specifier = None,
    analysis_level = "normal", export_options = [],
    results_directory = "results"):
    
    centroids_original = centroids
    
    if isinstance(data_sets, dict):
        data_sets = list(data_sets.values())
    
    if not isinstance(data_sets, (list, tuple)):
        data_sets = [data_sets]
    
    if not isinstance(other_data_sets, (list, tuple)):
        other_data_sets = [other_data_sets]
    elif other_data_sets == []:
        other_data_sets = [None] * len(data_sets)
    
    if len(data_sets) != len(other_data_sets):
        raise ValueError("Lists of data sets and alternative data sets" +
            " do not have the same length.")
    
    ID = None
    
    base_symbol = symbol
    
    decomposition_methods = decomposition_methods.copy()
    decomposition_methods.insert(0, None)
    
    for data_set, other_data_set in zip(data_sets, other_data_sets):
        
        if data_set.values.shape[1] <= 1:
            continue
        
        name = normaliseString(title)
        
        if specifier:
            ID = specifier(data_set)
        
        if ID:
            name += "-" + str(ID)
            title_with_ID = title + " for " + ID
        else:
            title_with_ID = title
        
        title_with_ID += " set"
        
        if not colouring_data_set:
            colouring_data_set = data_set
        
        if data_set.version in ["z", "z1"]:
            centroids = copy.deepcopy(centroids_original)
        else:
            centroids = None
        
        if other_data_set:
            title_with_ID = "{} set values in {}".format(
                other_data_set.version, title_with_ID)
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
                decomposition_method = properString(decomposition_method,
                    DECOMPOSITION_METHOD_NAMES)
                
                values_decomposed = data_set.values
                other_values_decomposed = other_values
                centroids_decomposed = centroids
                
                if decomposition_method == "t-SNE":
                    if data_set.number_of_examples \
                        > maximum_number_of_examples_for_t_sne:
                        
                        print(
                            "The number of examples for {}".format(
                                title_with_ID),
                            "is too large to decompose it",
                            "using {}. Skipping.".format(decomposition_method)
                        )
                        print()
                        continue
                        
                    elif data_set.number_of_features > \
                        maximum_number_of_features_for_t_sne:
                        
                        number_of_pca_components_before_tsne = min(
                            maximum_number_of_pca_components_before_tsne,
                            data_set.number_of_examples - 1
                        )
                        
                        print(
                            "The number of features for {}".format(
                                title_with_ID),
                            "is too large to decompose it",
                            "using {} in due time.".format(
                                decomposition_method)
                        )
                        print("Decomposing {} to {} components using PCA"
                            .format(
                                title_with_ID,
                                number_of_pca_components_before_tsne
                            ),
                            "beforehand."
                        )
                        decompose_time_start = time()
                        
                        values_decomposed, other_values_decomposed, \
                            centroids_decomposed = decompose(
                                values_decomposed,
                                other_value_sets = other_values_decomposed,
                                centroids = centroids_decomposed,
                                method = "pca",
                                number_of_components = 
                                    number_of_pca_components_before_tsne
                            )
                        
                        decompose_duration = time() - decompose_time_start
                        print("{} pre-decomposed ({}).".format(
                            capitaliseString(title_with_ID),
                            formatDuration(decompose_duration)
                        ))
                    
                    else:
                        if scipy.sparse.issparse(values_decomposed):
                            values_decomposed = values_decomposed.A
                        if scipy.sparse.issparse(other_values_decomposed):
                            other_values_decomposed = values_decomposed.A
                
                print("Decomposing {} using {}.".format(
                    title_with_ID, decomposition_method))
                decompose_time_start = time()
                
                values_decomposed, other_values_decomposed, \
                    centroids_decomposed = decompose(
                        values_decomposed,
                        other_value_sets = other_values_decomposed,
                        centroids = centroids_decomposed,
                        method = decomposition_method,
                        number_of_components = 2
                    )
                
                decompose_duration = time() - decompose_time_start
                print("{} decomposed ({}).".format(
                    capitaliseString(title_with_ID),
                    formatDuration(decompose_duration)
                ))
                
                print()
            
            if base_symbol:
                symbol = base_symbol
            else:
                symbol = ID
            
            x_label = axisLabelForSymbol(
                symbol = symbol,
                coordinate = 1,
                decomposition_method = decomposition_method,
            )
            y_label = axisLabelForSymbol(
                symbol = symbol,
                coordinate = 2,
                decomposition_method = decomposition_method,
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
            
            axis_limits_set = {0: None}
            
            if pca_limits and normaliseString(decomposition_method) == "pca":
                
                values_x_min = plot_values_decomposed[:, 0].min()
                values_x_max = plot_values_decomposed[:, 0].max()
                values_y_min = plot_values_decomposed[:, 1].min()
                values_y_max = plot_values_decomposed[:, 1].max()
                
                for axis_limits_scale in axis_limits_scales:
                    
                    axis_limits = {
                        "x": {
                            "minimum": None,
                            "maximum": None
                        },
                        "y": {
                            "minimum": None,
                            "maximum": None
                        }
                    }
                    
                    axis_limits_changed = False
                    
                    # x minimum
                    
                    x_min_indices = plot_values_decomposed[:, 0] \
                        >= axis_limits_scale * pca_limits["PC1"]["minimum"]
                    x_min = plot_values_decomposed[x_min_indices][:, 0].min()
                                        
                    if x_min > values_x_min:
                        axis_limits_changed = True
                        axis_limits["x"]["minimum"] = numpy.floor(x_min)
                    
                    # x maximum
                    
                    x_max_indices = plot_values_decomposed[:, 0] \
                        <= axis_limits_scale * pca_limits["PC1"]["maximum"]
                    x_max = plot_values_decomposed[x_max_indices][:, 0].max()
                                        
                    if x_max < values_x_max:
                        axis_limits_changed = True
                        axis_limits["x"]["maximum"] = numpy.ceil(x_max)
                    
                    # y minimum
                    
                    y_min_indices = plot_values_decomposed[:, 1] \
                        >= axis_limits_scale * pca_limits["PC2"]["minimum"]
                    y_min = plot_values_decomposed[y_min_indices][:, 1].min()
                                        
                    if y_min > values_y_min:
                        axis_limits_changed = True
                        axis_limits["y"]["minimum"] = numpy.floor(y_min)
                    
                    # y maximum
                    
                    y_max_indices = plot_values_decomposed[:, 1] \
                        <= axis_limits_scale * pca_limits["PC2"]["maximum"]
                    y_max = plot_values_decomposed[y_max_indices][:, 1].max()
                                        
                    if y_max < values_y_max:
                        axis_limits_changed = True
                        axis_limits["y"]["maximum"] = numpy.ceil(y_max)
                    
                    # Add axis limits set if axis limits changed
                    
                    if axis_limits_changed:
                        axis_limits_set[axis_limits_scale] = axis_limits
                                
            # Loop over axis limits set
            
            for axis_limits_scale, axis_limits in axis_limits_set.items():
                
                if axis_limits:
                    plot_name = "{}-axis_limits_scale-{}".format(
                        name, axis_limits_scale)
                    axis_limits_string = " (axis limits for original set" \
                        + " scaled by {})".format(axis_limits_scale)
                else:
                    plot_name = name
                    axis_limits_string = ""
                
                # Plot data set
            
                print("Plotting {}{}{}.".format(
                    "decomposed " if decomposition_method else "",
                    title_with_ID,
                    axis_limits_string
                ))
            
                ## No colour-coding
            
                plot_time_start = time()
            
                figure, figure_name = plotValues(
                    plot_values_decomposed,
                    centroids = centroids_decomposed,
                    figure_labels = figure_labels,
                    axis_limits = axis_limits,
                    example_tag = data_set.tags["example"],
                    name = plot_name
                )
                saveFigure(figure, figure_name, export_options,
                    decompositions_directory)
            
                plot_duration = time() - plot_time_start
                print("    {} plotted and saved ({}).".format(
                    capitaliseString(title_with_ID),
                    formatDuration(plot_duration)
                ))
        
                # Labels
        
                if colouring_data_set.labels is not None:
                    plot_time_start = time()
            
                    figure, figure_name = plotValues(
                        plot_values_decomposed,
                        colour_coding = "labels",
                        colouring_data_set = colouring_data_set,
                        centroids = centroids_decomposed,
                        figure_labels = figure_labels,
                        axis_limits = axis_limits,
                        example_tag = data_set.tags["example"],
                        name = plot_name
                    )
                    saveFigure(figure, figure_name, export_options,
                        decompositions_directory)
        
                    plot_duration = time() - plot_time_start
                    print("    {} (with labels) plotted and saved ({}).".format(
                        capitaliseString(title_with_ID), formatDuration(plot_duration)))
                
                    if colouring_data_set.superset_labels is not None:
                        plot_time_start = time()
                    
                        figure, figure_name = plotValues(
                            plot_values_decomposed,
                            colour_coding = "superset labels",
                            colouring_data_set = colouring_data_set,
                            centroids = centroids_decomposed,
                            figure_labels = figure_labels,
                            axis_limits = axis_limits,
                            example_tag = data_set.tags["example"],
                            name = plot_name
                        )
                        saveFigure(figure, figure_name, export_options,
                            decompositions_directory)
                    
                        plot_duration = time() - plot_time_start
                        print("    " +
                            "{} (with superset labels) plotted and saved ({})."\
                                .format(
                                    capitaliseString(title_with_ID),
                                    formatDuration(plot_duration)
                            )
                        )
                    
                    if analysis_level == "extensive":
                        
                        if colouring_data_set.number_of_classes <= 10:
                            plot_time_start = time()
                        
                            for class_name in colouring_data_set.class_names:
                                figure, figure_name = plotValues(
                                    plot_values_decomposed,
                                    colour_coding = "class",
                                    colouring_data_set = colouring_data_set,
                                    centroids = centroids_decomposed,
                                    class_name = class_name,
                                    figure_labels = figure_labels,
                                    axis_limits = axis_limits,
                                    example_tag = data_set.tags["example"],
                                    name = plot_name
                                )
                                saveFigure(figure, figure_name, export_options,
                                    decompositions_directory)
                        
                            plot_duration = time() - plot_time_start
                            print("    {} (for each class) plotted and saved ({})."\
                                .format(
                                    capitaliseString(title_with_ID),
                                    formatDuration(plot_duration)
                            ))
                        
                        if colouring_data_set.superset_labels is not None \
                            and data_set.number_of_superset_classes <= 10:
                        
                            plot_time_start = time()
                        
                            for superset_class_name in \
                                colouring_data_set.superset_class_names:
                                figure, figure_name = plotValues(
                                    plot_values_decomposed,
                                    colour_coding = "superset class",
                                    colouring_data_set = colouring_data_set,
                                    centroids = centroids_decomposed,
                                    class_name = superset_class_name,
                                    figure_labels = figure_labels,
                                    axis_limits = axis_limits,
                                    example_tag = data_set.tags["example"],
                                    name = plot_name
                                )
                                saveFigure(figure, figure_name, export_options,
                                    decompositions_directory)
                        
                            plot_duration = time() - plot_time_start
                            print("    " +
                                "{} (for each superset class) plotted and saved ({})."\
                                    .format(
                                        capitaliseString(title_with_ID),
                                        formatDuration(plot_duration)
                            ))
                
                ## Predictions
                
                if colouring_data_set.has_predicted_cluster_ids:
                    plot_time_start = time()
                    
                    figure, figure_name = plotValues(
                        plot_values_decomposed,
                        colour_coding = "predicted cluster IDs",
                        colouring_data_set = colouring_data_set,
                        centroids = centroids_decomposed,
                        figure_labels = figure_labels,
                        prediction_details = prediction_details,
                        axis_limits = axis_limits,
                        example_tag = data_set.tags["example"],
                        name = plot_name,
                    )
                    saveFigure(figure, figure_name, export_options,
                        decompositions_directory)
                
                    plot_duration = time() - plot_time_start
                    print("    " +
                        "{} (with predicted cluster IDs) plotted and saved ({})."\
                            .format(
                                capitaliseString(title_with_ID),
                                formatDuration(plot_duration)
                        )
                    )
                
                if colouring_data_set.has_predicted_labels:
                    plot_time_start = time()
                    
                    figure, figure_name = plotValues(
                        plot_values_decomposed,
                        colour_coding = "predicted labels",
                        colouring_data_set = colouring_data_set,
                        centroids = centroids_decomposed,
                        figure_labels = figure_labels,
                        prediction_details = prediction_details,
                        axis_limits = axis_limits,
                        example_tag = data_set.tags["example"],
                        name = plot_name,
                    )
                    saveFigure(figure, figure_name, export_options,
                        decompositions_directory)
                    
                    plot_duration = time() - plot_time_start
                    print("    " +
                        "{} (with predicted labels) plotted and saved ({})."\
                            .format(
                                capitaliseString(title_with_ID),
                                formatDuration(plot_duration)
                        )
                    )
                
                if colouring_data_set.has_predicted_superset_labels:
                    plot_time_start = time()
                    
                    figure, figure_name = plotValues(
                        plot_values_decomposed,
                        colour_coding = "predicted superset labels",
                        colouring_data_set = colouring_data_set,
                        centroids = centroids_decomposed,
                        figure_labels = figure_labels,
                        prediction_details = prediction_details,
                        axis_limits = axis_limits,
                        example_tag = data_set.tags["example"],
                        name = plot_name,
                    )
                    saveFigure(figure, figure_name, export_options,
                        decompositions_directory)
                    
                    plot_duration = time() - plot_time_start
                    print("    " +
                        "{} (with predicted superset labels) plotted and saved ({})."\
                            .format(
                                capitaliseString(title_with_ID),
                                formatDuration(plot_duration)
                        )
                    )
                
                # Count sum
                
                plot_time_start = time()
        
                figure, figure_name = plotValues(
                    plot_values_decomposed,
                    colour_coding = "count sum",
                    colouring_data_set = colouring_data_set,
                    centroids = centroids_decomposed,
                    figure_labels = figure_labels,
                    axis_limits = axis_limits,
                    example_tag = data_set.tags["example"],
                    name = plot_name
                )
                saveFigure(figure, figure_name, export_options,
                    decompositions_directory)
        
                plot_duration = time() - plot_time_start
                print("    {} (with count sum) plotted and saved ({}).".format(
                    capitaliseString(title_with_ID),
                    formatDuration(plot_duration)
                ))
                
                # Features
                
                for feature_index in highlight_feature_indices:
            
                    plot_time_start = time()
            
                    figure, figure_name = plotValues(
                        plot_values_decomposed,
                        colour_coding = "feature",
                        colouring_data_set = colouring_data_set,
                        centroids = centroids_decomposed,
                        feature_index = feature_index,
                        figure_labels = figure_labels,
                        axis_limits = axis_limits,
                        example_tag = data_set.tags["example"],
                        name = plot_name
                    )
                    saveFigure(figure, figure_name, export_options,
                        decompositions_directory)
            
                    plot_duration = time() - plot_time_start
                    print("    {} (with {}) plotted and saved ({}).".format(
                        capitaliseString(title_with_ID),
                        data_set.feature_names[feature_index],
                        formatDuration(plot_duration)
                    ))
                
                print()

def analyseCentroidProbabilities(centroids, name = None,
    analysis_level = "normal", export_options = [], results_directory = "results"):
    
    print("Plotting centroid probabilities.")
    plot_time_start = time()
    
    if name:
        name = normaliseString(name)
    
    posterior_probabilities = None
    prior_probabilities = None

    if "posterior" in centroids and centroids["posterior"]:
        posterior_probabilities = centroids["posterior"]["probabilities"]
        K = len(posterior_probabilities)
    if "prior" in centroids and centroids["prior"]:
        prior_probabilities = centroids["prior"]["probabilities"]
        K = len(prior_probabilities)

    centroids_palette = darker_palette(K)
    x_label = "$k$"
    if prior_probabilities is not None:
        if posterior_probabilities is not None:
            y_label = axisLabelForSymbol(
                symbol = "\\pi",
                distribution = normaliseString("posterior"),
                suffix = "^k")
            if name:
                plot_name = [name, "posterior", "prior"]
            else:
                plot_name = ["posterior", "prior"]
        else:
            y_label = axisLabelForSymbol(
                symbol = "\\pi",
                distribution = normaliseString("prior"),
                suffix = "^k")
            if name:
                plot_name = [name, "prior"]
            else:
                plot_name = "prior"
    elif posterior_probabilities is not None:
        y_label = axisLabelForSymbol(
            symbol = "\\pi",
            distribution = normaliseString("posterior"),
            suffix = "^k")
        if name:
            plot_name = [name, "posterior"]
        else:
            plot_name = "posterior"

    figure, figure_name = plotProbabilities(
        posterior_probabilities,
        prior_probabilities,
        x_label = x_label,
        y_label = y_label,
        palette = centroids_palette,
        uniform = False,
        name = plot_name
    )
    saveFigure(figure, figure_name, export_options, results_directory)
    
    plot_duration = time() - plot_time_start
    print("Centroid probabilities plotted and saved ({}).".format(
        formatDuration(plot_duration)))

def evaluationSubsetIndices(evaluation_set,
    maximum_number_of_examples_per_class =
        evaluation_subset_maximum_number_of_examples_per_class,
    total_maximum_number_of_examples =
        evaluation_subset_maximum_number_of_examples):
    
    random_state = numpy.random.RandomState(80)
    
    M = evaluation_set.number_of_examples
    
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
            subset.update(*class_label_indices
                [:maximum_number_of_examples_per_class])
    
    else:
        subset = numpy.random.permutation(M)\
            [:total_maximum_number_of_examples]
        subset = set(subset)
    
    return subset

def statistics(x, name = "", tolerance = 1e-3, skip_sparsity = False):
    
    batch_size = None
    
    if x.size > maximum_size_for_normal_statistics_computation:
        batch_size = 1000
    
    x_mean = x.mean()
    x_std = standard_deviation(x, ddof=1, batch_size=batch_size)
    
    x_min  = x.min()
    x_max  = x.max()
    
    x_dispersion = x_std**2 / x_mean
    
    if skip_sparsity:
        x_sparsity = nan
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

def excludeClassesFromLabelSet(*label_sets, excluded_classes = []):
    
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

def accuracy(labels, predicted_labels, excluded_classes = []):
    labels, predicted_labels = excludeClassesFromLabelSet(
        labels, predicted_labels, excluded_classes = excluded_classes)
    return numpy.mean(predicted_labels == labels)

def adjusted_rand_index(labels, predicted_labels, excluded_classes = []):
    labels, predicted_labels = excludeClassesFromLabelSet(
        labels, predicted_labels, excluded_classes = excluded_classes)
    return sklearn.metrics.cluster.adjusted_rand_score(labels,
        predicted_labels)

def adjusted_mutual_information(labels, predicted_labels, excluded_classes = []):
    labels, predicted_labels = excludeClassesFromLabelSet(
        labels, predicted_labels, excluded_classes = excluded_classes)
    return sklearn.metrics.cluster.adjusted_mutual_info_score(labels,
        predicted_labels)

def silhouette_score(values, predicted_labels):
    
    number_of_predicted_classes = numpy.unique(predicted_labels).shape[0]
    number_of_examples = values.shape[0]
    
    if number_of_predicted_classes < 2 \
        or number_of_predicted_classes > number_of_examples - 1:
            return nan
    
    sample_size = None
    
    if number_of_examples \
        > maximum_number_of_examples_before_sampling_silhouette_score:
            sample_size \
                = maximum_number_of_examples_before_sampling_silhouette_score
    
    score = sklearn.metrics.silhouette_score(
        X=values,
        labels=predicted_labels,
        sample_size=sample_size
    )
    
    return score

def formatStatistics(statistics_sets, name = "Data set"):
    
    if type(statistics_sets) != list:
        statistics_sets = [statistics_sets]
    
    name_width = len(name)
    
    for statistics_set in statistics_sets:
        name_width = max(len(statistics_set["name"]), name_width)
    
    table_heading = "  ".join(["{:{}}".format(name, name_width),
        " mean ", "std. dev. ", "dispersion",
        " minimum ", " maximum ","sparsity"])
    
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

def computeCountAccuracies(x, x_tilde, method = None):
    """
    Compute accuracies for every count in original data set
    using reconstructed data set.
    """
    
    # Setting up
    
    count_accuracies = {}
    M, N = x.shape
    
    ## Round data sets to be able to compare
    x = numpy.round(x)
    x_tilde = numpy.round(x_tilde)
    
    ## Compute the max count value
    k_max = x.max().astype(int)
    
    if method == "orders of magnitude":
        
        log_k_max_floored = numpy.floor(numpy.log10(k_max)).astype(int)
        
        for l in range(log_k_max_floored + 1):
            
            x_scaled_floored = numpy.floor(x / pow(10, l))
            x_tilde_scaled_floored = numpy.floor(x_tilde / pow(10, l))
            
            k_max_scaled_floored = x_scaled_floored.max().astype(int)
            
            k_start = 1
            
            if l == 0:
                k_start = 0
            
            for k in range(k_start, min(10, k_max_scaled_floored + 1)):
                
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore",
                        category = scipy.sparse.SparseEfficiencyWarning
                    )
                    k_indices = x_scaled_floored == k
                
                k_size = k_indices.sum()
                
                if scipy.sparse.issparse(x):
                    k_indices = k_indices.tocoo()
                    k_indices = (k_indices.row, k_indices.col)
                
                k_sum = (x_tilde_scaled_floored[k_indices] == k).sum()
                
                if k_size != 0:
                    f = k_sum / k_size
                else:
                    f = numpy.nan
                
                k_real = k * pow(10, l)
                
                if l == 0:
                    k_string = str(k_real)
                else:
                    k_real_end = min(k_max, k_real + pow(10, l) - 1)
                    k_string = "{}-{}".format(k_real, k_real_end)
                
                count_accuracies[k_string] = f
    
    else:
        
        for k in range(k_max + 1):
            
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    category = scipy.sparse.SparseEfficiencyWarning
                )
                k_indices = x == k
            
            k_size = k_indices.sum()
            
            if scipy.sparse.issparse(x):
                k_indices = k_indices.tocoo()
                k_indices = (k_indices.row, k_indices.col)
            
            k_sum = (x_tilde[k_indices] == k).sum()
            
            if k_size != 0:
                f = k_sum / k_size
            else:
                f = numpy.nan
            
            count_accuracies[str(k)] = f
    
    return count_accuracies

def formatCountAccuracies(count_accuracies):
    
    k_width = 5
    
    for k in count_accuracies.keys():
        k_width = max(len(k), k_width)
    
    table_heading = "Count   Accuracy"
    table_rows = [table_heading]
    
    for k, f in count_accuracies.items():
        table_row_parts = [
            "{:^{}}".format(k, k_width),
            "{:6.2f}".format(100 * f)
        ]
        table_row = "    ".join(table_row_parts)
        table_rows.append(table_row)
    
    table = "\n".join(table_rows)
    
    return table

clustering_metrics = {
    "adjusted Rand index": {
        "kind": "supervised",
        "function": adjusted_rand_index
    },
    "adjusted mutual information": {
        "kind": "supervised",
        "function": adjusted_mutual_information
    },
    "silhouette score": {
        "kind": "unsupervised",
        "function": silhouette_score
    }
}

def computeClusteringMetrics(evaluation_set):
    
    clustering_metric_values = {
        metric: {
            "clusters": None,
            "clusters; superset": None,
            "labels": None,
            "labels; superset": None
        }
        for metric in clustering_metrics
    }
    
    for metric_name, metric_attributes in clustering_metrics.items():
        
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

def computePairwiseDistances(values, metric="euclidean"):
    return sklearn.metrics.pairwise_distances(values, metric=metric)

def plotClassHistogram(labels, class_names = None, class_palette = None,
    normed = False, scale = "linear", label_sorter = None, name = None):
    
    figure_name = "histogram"
    
    if normed:
        figure_name += "-normed"
    
    figure_name += "-classes"
    
    figure_name = figureName(figure_name, name)
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    if class_names is None:
        class_names = numpy.unique(labels)
    
    K = len(class_names)
    
    if not label_sorter:
        label_sorter = createLabelSorter()
    
    if not class_palette:
        index_palette = lighter_palette(K)
        class_palette = {class_name: index_palette[i] for i, class_name in
                         enumerate(sorted(class_names,
                         key = label_sorter))}
    
    histogram = {
        class_name: {
            "index": i,
            "count": 0,
            "colour": class_palette[class_name]
        }
        for i, class_name in enumerate(sorted(class_names,
            key = label_sorter))
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
        
        axis.bar(index, count_or_frequecny, color = colour)
    
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
    pyplot.xticks(indices, class_names,
        horizontalalignment = y_ticks_horizontal_alignment,
        rotation = y_ticks_rotation, rotation_mode = y_ticks_rotation_mode
    )
    
    axis.set_xlabel("Classes")
    
    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")
    
    seaborn.despine()
    
    return figure, figure_name

def plotHistogram(series, excess_zero_count = 0, label = None,
    maximum_count = None, normed = False, discrete = False,
    x_scale = "linear", y_scale = "linear", colour = None, name = None):
    
    series = series.copy()
    
    figure_name = "histogram"
    
    if normed:
        figure_name += "-normed"
    
    figure_name = figureName(figure_name, name)
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    series_length = len(series) + excess_zero_count
    
    if maximum_count:
        maximum_count_indcises = series <= maximum_count
        number_of_outliers = series.size - maximum_count_indcises.sum()
        series = series[maximum_count_indcises]
    
    series_max = series.max()
    
    if discrete and series_max < maximum_number_of_bins_for_histograms:
        number_of_bins = int(numpy.ceil(series_max)) + 1
        bin_range = numpy.array((-0.5, series_max + 0.5))
    else:
        if series_max < maximum_number_of_bins_for_histograms:
            number_of_bins = "auto"
        else:
            number_of_bins = maximum_number_of_bins_for_histograms
        bin_range = numpy.array((series.min(), series_max))
    
    if colour is None:
        colour = standard_palette[0]
    
    if x_scale == "log":
        series += 1
        bin_range += 1
        label += " (shifted one)"
        figure_name += "-log_values"
    
    y_log = y_scale == "log"
    
    histogram, bin_edges = numpy.histogram(series, bins = number_of_bins,
        range = bin_range)
    
    histogram[0] += excess_zero_count
    
    width = bin_edges[1] - bin_edges[0]
    bin_centres = bin_edges[:-1] + width / 2
    
    if normed:
        histogram = histogram / series_length
    
    axis.bar(bin_centres, histogram, width = width, log = y_log,
        color = colour, alpha = 0.4)
    
    axis.set_xscale(x_scale)
    
    axis.set_xlabel(capitaliseString(label))
    
    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")
    
    seaborn.despine()
    
    if maximum_count:
        axis.text(
            0.98, 0.98, "{} counts omitted".format(number_of_outliers),
            horizontalalignment = "right",
            verticalalignment = "top",
            fontsize = 10,
            transform = axis.transAxes
        )
    
    return figure, figure_name

def plotCutOffCountHistogram(series, excess_zero_count = 0, cutoff = None,
    normed = False, scale = "linear", colour = None, name = None):
    
    series = series.copy()
    
    figure_name = "histogram"
    
    if normed:
        figure_name += "-normed"
    
    figure_name += "-counts"
    figure_name = figureName(figure_name, name)
    figure_name += "-cutoff-{}".format(cutoff)
    
    if not colour:
        colour = standard_palette[0]
    
    y_log = scale == "log"
    
    k = numpy.arange(cutoff + 1)
    C = numpy.empty(cutoff + 1)
    
    for i in range(cutoff + 1):
        if k[i] < cutoff:
            c = (series == k[i]).sum()
        elif k[i] == cutoff:
            c = (series >= cutoff).sum()
        C[i] = c
    
    C[0] += excess_zero_count
    
    if normed:
        C /= C.sum()
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    axis.bar(k, C, log = y_log, color = colour, alpha = 0.4)
    
    axis.set_xlabel("Count bins")
    
    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")
    
    seaborn.despine()
    
    return figure, figure_name
    
def plotSeries(series, x_label, y_label, sort = False, scale = "linear",
    bar = False, colour = None, name = None):
    
    figure_name = figureName("series", name)
    
    if not colour:
        colour = standard_palette[0]
    
    D = series.shape[0]
    
    x = numpy.linspace(0, D, D)
    
    y_log = scale == "log"
    
    if sort:
        # Sort descending
        series = numpy.sort(series)[::-1]
        x_label = "sorted " + x_label
        figure_name += "-sorted"
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    if bar:
        axis.bar(x, series, log = y_log, color = colour, alpha = 0.4)
    else:
        axis.plot(x, series, color = colour)
        axis.set_yscale(scale)
    
    axis.set_xlabel(capitaliseString(x_label))
    axis.set_ylabel(capitaliseString(y_label))
    
    seaborn.despine()
    
    return figure, figure_name

def plotLearningCurves(curves, model_type, epoch_offset = 0, epoch_slice = slice(None), global_y_lim = False, name = None):
    
    figure_name = "learning_curves"
    
    figure_name = figureName("learning_curves", name)
    
    if epoch_slice.start is not None:
        figure_name += "-in_range-{}".format(epoch_slice.start + epoch_offset)
        if epoch_slice.stop is not None:
            figure_name += "_{}".format(epoch_slice.stop + epoch_offset)
        if epoch_slice.step is not None:
            figure_name += "_{}".format(epoch_slice.step)
    elif epoch_slice.stop is not None:
        figure_name += "-to_epoch-{}".format(epoch_slice.stop + epoch_offset)
        if epoch_slice.step is not None:
            figure_name += "_{}".format(epoch_slice.step)
    elif epoch_slice.step is not None:
        figure_name += "-per_step-{}".format(epoch_slice.step)

    x_label = "Epoch"
    y_label = "Nat"
    
    if model_type == "AE":
        figure = pyplot.figure()
        axis_1 = figure.add_subplot(1, 1, 1)
    elif model_type == "VAE":
        figure, (axis_1, axis_2) = pyplot.subplots(2, sharex = True,
            figsize = (6.4, 9.6))
        figure.subplots_adjust(hspace = 0.1)
    elif model_type == "GMVAE":
        figure, (axis_1, axis_2, axis_3) = pyplot.subplots(3, sharex = True,
            figsize = (6.4, 14.4))
        figure.subplots_adjust(hspace = 0.1)
    
    axis_1_lim = []
    axis_2_lim = []
    axis_3_lim = []

    for curve_set_name, curve_set in sorted(curves.items()):
        
        if curve_set_name == "training":
            line_style = "solid"
            colour_index_offset = 0
        elif curve_set_name == "validation":
            line_style = "dashed"
            colour_index_offset = 1
        
        curve_colour = lambda i: standard_palette[len(curves) * i
            + colour_index_offset]
        
        for curve_name, curve in sorted(curve_set.items()):
            if curve is None:
                continue
            
            if curve_name == "lower_bound":
                curve_name = "$\\mathcal{L}$"
                colour = curve_colour(0)
                axis = axis_1
                axis_1_lim = [
                    min(numpy.concatenate((curve, axis_1_lim))),
                    max(numpy.concatenate((curve, axis_1_lim)))
                ]
            elif curve_name == "reconstruction_error":
                curve_name = "$\\log p(x|z)$"
                colour = curve_colour(1)
                axis = axis_1
                axis_1_lim = [
                    min(numpy.concatenate((curve, axis_1_lim))),
                    max(numpy.concatenate((curve, axis_1_lim)))
                ]

            elif "kl_divergence" in curve_name:
                if curve_name == "kl_divergence":
                    index = ""
                    colour = curve_colour(0)
                    axis = axis_2
                    axis_2_lim = [
                        min(numpy.concatenate((curve, axis_2_lim))),
                        max(numpy.concatenate((curve, axis_2_lim)))
                    ]
                else:
                    latent_variable = curve_name.replace("kl_divergence_", "")
                    latent_variable = re.sub(r"(\w)(\d)", r"\1_\2", latent_variable)
                    index = "$_{" + latent_variable + "}$"
                    if latent_variable in ["z", "z_2"]:
                        colour = curve_colour(0)
                        axis = axis_2
                        axis_2_lim = [
                            min(numpy.concatenate((curve, axis_2_lim))),
                            max(numpy.concatenate((curve, axis_2_lim)))
                        ]
                    elif latent_variable == "z_1":
                        colour = curve_colour(1)
                        axis = axis_2
                        axis_2_lim = [
                            min(numpy.concatenate((curve, axis_2_lim))),
                            max(numpy.concatenate((curve, axis_2_lim)))
                        ]
                    elif latent_variable == "y":
                        colour = curve_colour(0)
                        axis = axis_3
                        axis_3_lim = [
                            min(numpy.concatenate((curve, axis_3_lim))),
                            max(numpy.concatenate((curve, axis_3_lim)))
                        ]
                curve_name = "KL" + index + "$(q||p)$"
            elif curve_name == "log_likelihood":
                curve_name = "$L$"
                axis = axis_1
                axis_1_lim = [
                    min(numpy.concatenate((curve, axis_1_lim))),
                    max(numpy.concatenate((curve, axis_1_lim)))
                ]
            epochs = numpy.arange(len(curve)) + epoch_offset + 1
            label = curve_name + " ({} set)".format(curve_set_name)
            axis.plot(
                epochs[epoch_slice],
                curve[epoch_slice],
                color = colour,
                linestyle = line_style,
                label = label
            )
    
    handles, labels = axis_1.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    
    axis_1.legend(handles, labels, loc = "best")
    
    if model_type == "AE":
        axis_1.set_xlabel(x_label)
        axis_1.set_ylabel(y_label)
        if global_y_lim:
            axis_1.set_ylim(axis_1_lim)
    elif model_type == "VAE":
        handles, labels = axis_2.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles),
            key = lambda t: t[0]))
        axis_2.legend(handles, labels, loc = "best")
        axis_1.set_ylabel("")
        axis_2.set_ylabel("")
        if global_y_lim:
            axis_1.set_ylim(axis_1_lim)
            axis_2.set_ylim(axis_2_lim)

        if model_type == "GMVAE":
            axis_3.legend(loc = "best")
            handles, labels = axis_3.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles),
                key = lambda t: t[0]))
            axis_3.legend(handles, labels, loc = "best")
            axis_3.set_xlabel(x_label)
            axis_3.set_ylabel("")
            if global_y_lim:
                axis_3.set_ylim(axis_3_lim)
        else:
            axis_2.set_xlabel(x_label)
        figure.text(-0.01, 0.5, y_label, va = "center", rotation = "vertical")
    
    seaborn.despine()
    
    return figure, figure_name

def plotSeparateLearningCurves(curves, loss, name = None):
    
    if not isinstance(loss, list):
        losses = [loss]
    else:
        losses = loss
    
    if not isinstance(name, list):
        names = [name]
    else:
        names = name
    
    names.extend(losses)
    
    figure_name = figureName("learning_curves", names)
    
    x_label = "Epoch"
    y_label = "Nat"
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    for curve_set_name, curve_set in sorted(curves.items()):
        
        if curve_set_name == "training":
            line_style = "solid"
            colour_index_offset = 0
        elif curve_set_name == "validation":
            line_style = "dashed"
            colour_index_offset = 1
        
        curve_colour = lambda i: standard_palette[len(curves) * i
            + colour_index_offset]
        
        for curve_name, curve in sorted(curve_set.items()):
            if curve is None or curve_name not in losses:
                continue
            if curve_name == "lower_bound":
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
                    latent_variable = re.sub(r"(\w)(\d)", r"\1_\2", latent_variable)
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
            axis.plot(epochs, curve, color = colour, linestyle = line_style,
                label = label)
    
    handles, labels = axis.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    
    axis.legend(handles, labels, loc = "best")
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    seaborn.despine()
    
    return figure, figure_name

def plotAccuracies(accuracies, name = None):
    
    figure_name = figureName("accuracies", name)
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    for accuracies_kind, accuracies in sorted(accuracies.items()):
        
        if accuracies is None:
            continue
        
        if accuracies_kind == "training":
            line_style = "solid"
            colour = standard_palette[0]
        elif accuracies_kind == "validation":
            line_style = "dashed"
            colour = standard_palette[1]
        
        label = "{} set".format(capitaliseString(accuracies_kind))
        
        epochs = numpy.arange(len(accuracies)) + 1
        axis.plot(epochs, 100 * accuracies, color = colour,
            linestyle = line_style, label = label)
    
    handles, labels = axis.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    
    axis.legend(handles, labels, loc = "best")
    
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracies")
    
    seaborn.despine()
    
    return figure, figure_name

def plotKLDivergenceEvolution(KL_neurons, scale = "log", name = None):
    
    E, L = KL_neurons.shape
    
    KL_neurons = numpy.sort(KL_neurons, axis = 1)
    
    if scale == "log":
        KL_neurons = numpy.log(KL_neurons)
        scale_label = "$\log$ "
    else:
        scale_label = ""
    
    figure_name = figureName("kl_divergence_evolution", name)
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    cbar_dict = {"label": scale_label + "KL$(p_i|q_i)$"}
    
    number_of_epoch_labels = 10
    if E > 2 * number_of_epoch_labels:
        epoch_label_frequency = int(numpy.floor(E / number_of_epoch_labels))
    else:
        epoch_label_frequency = True
    
    epochs = numpy.arange(E) + 1
    
    seaborn.heatmap(
        DataFrame(KL_neurons.T, columns = epochs),
        xticklabels = epoch_label_frequency,
        yticklabels = False,
        cbar = True, cbar_kws = cbar_dict, cmap = standard_colour_map,
        ax = axis
    )
    
    axis.set_xlabel("Epochs")
    axis.set_ylabel("$i$")
    
    seaborn.despine(ax = axis)
    
    return figure, figure_name

def plotEvolutionOfCentroidProbabilities(probabilities, distribution, figure = None, linestyle = "solid", name = None):
    
    distribution = normaliseString(distribution)
    
    y_label = axisLabelForSymbol(
        symbol = "\\pi",
        distribution = distribution,
        suffix = "^k"
    )
    
    figure_name = "centroids_evolution-{}-probabilities".format(distribution)
    
    figure_name = figureName(figure_name, name)
    
    E, K = probabilities.shape
    
    centroids_palette = darker_palette(K)
    epochs = numpy.arange(E) + 1
    
    if figure is not None:
        # axis = figure.as_list()[0]
        figure_new = figure
        axis = figure_new.gca()
        axis.plot(1,0,color="black",linestyle=linestyle,label = distribution)
        axis.legend(loc = "upper left")
    else:
        figure_new = pyplot.figure()
        axis = figure_new.add_subplot(1, 1, 1)

    for k in range(K):
        axis.plot(
            epochs,
            probabilities[:, k],
            color = centroids_palette[k],
            linestyle = linestyle,
            label = "$k = {}$".format(k)
        )

    if figure is not None:
        axis.set_xlabel("Epochs")
        axis.set_ylabel(y_label)
        seaborn.despine()
        return figure_new, figure_name

    axis.plot(1,0,color="black",linestyle=linestyle,label = distribution)
    axis.legend(loc = "best")
    axis.set_xlabel("Epochs")
    axis.set_ylabel(y_label)
    
    seaborn.despine()
    
    return figure_new, figure_name

def plotEvolutionOfCentroidMeans(means, distribution, decomposed = False,
    name = None):
    
    symbol = "\\mu"
    if decomposed:
        decomposition_method = "PCA"
    else:
        decomposition_method = ""
    distribution = normaliseString(distribution)
    suffix = "(y = k)"
    
    x_label = axisLabelForSymbol(
        symbol = symbol,
        coordinate = 1,
        decomposition_method = decomposition_method,
        distribution = distribution,
        suffix = suffix
    )
    y_label = axisLabelForSymbol(
        symbol = symbol,
        coordinate = 2,
        decomposition_method = decomposition_method,
        distribution = distribution,
        suffix = suffix
    )
    
    figure_name = "centroids_evolution-{}-means".format(distribution)
    
    figure_name = figureName(figure_name, name)
    
    E, K, L = means.shape
    
    if L > 2:
        raise ValueError("Dimensions of means should be 2.")
    
    centroids_palette = darker_palette(K)
    epochs = numpy.arange(E) + 1
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    colour_bar_scatter_plot = axis.scatter(
        means[:, 0, 0], means[:, 0, 1], c = epochs,
        cmap = seaborn.dark_palette(neutral_colour, as_cmap = True),
        zorder = 0
    )
    
    for k in range(K):
        colour = centroids_palette[k]
        colour_map = seaborn.dark_palette(colour, as_cmap = True)
        axis.plot(means[:, k, 0], means[:, k, 1], color = colour,
            label = "$k = {}$".format(k), zorder = k + 1)
        axis.scatter(means[:, k, 0], means[:, k, 1], c = epochs,
            cmap = colour_map, zorder = K + k + 1)
    
    axis.legend(loc = "best")
    
    seaborn.despine()
    
    colour_bar = figure.colorbar(colour_bar_scatter_plot)
    colour_bar.outline.set_linewidth(0)
    colour_bar.set_label("Epochs")
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    return figure, figure_name

def plotEvolutionOfCentroidCovarianceMatrices(covariance_matrices, distribution,
    name = None):
    
    distribution = normaliseString(distribution)
    
    y_label = axisLabelForSymbol(
        symbol = "\\Sigma",
        distribution = distribution,
        prefix = "|",
        suffix = "(y = k)|"
    )
    
    figure_name = "centroids_evolution-{}-covariance_matrices".format(
        distribution)
    
    figure_name = figureName(figure_name, name)
    
    E, K, L, L = covariance_matrices.shape
    
    determinants = numpy.empty([E, K])

    for e in range(E):
        for k in range(K):
            determinants[e, k] = numpy.prod(numpy.diag(
                covariance_matrices[e, k]))
    
    if determinants.all() > 0:
        line_range_ratio = numpy.empty(K)
        for k in range(K):
            determinants_min = determinants[:, k].min()
            determinants_max = determinants[:, k].max()
            line_range_ratio[k] = determinants_max / determinants_min
        range_ratio = line_range_ratio.max() / line_range_ratio.min()
        if range_ratio > 1e2:
            y_scale = "log"
        else:
            y_scale = "linear"
    
    centroids_palette = darker_palette(K)
    epochs = numpy.arange(E) + 1
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    for k in range(K):
        axis.plot(epochs, determinants[:, k], color = centroids_palette[k],
            label = "$k = {}$".format(k))
    
    axis.set_xlabel("Epochs")
    axis.set_ylabel(y_label)
    
    axis.set_yscale(y_scale)
    
    axis.legend(loc = "best")
    
    seaborn.despine()
    
    return figure, figure_name

def plotProfileComparison(observed_series, expected_series,
    expected_series_total_standard_deviations = None,
    expected_series_explained_standard_deviations = None,
    x_name = "feature", y_name = "value", sort = True,
    sort_by = "expected", sort_direction = "ascending",
    x_scale = "linear", y_scale = "linear", y_cutoff = None,
    name = None):
    
    # Setup
    
    sort_by = normaliseString(sort_by)
    sort_direction = normaliseString(sort_direction)
    
    figure_name = figureName("profile_comparison", name)
    
    if scipy.sparse.issparse(observed_series):
        observed_series = observed_series.A.squeeze()
    
    if scipy.sparse.issparse(expected_series_total_standard_deviations):
        expected_series_total_standard_deviations = \
            expected_series_total_standard_deviations.A.squeeze()
    
    if scipy.sparse.issparse(expected_series_explained_standard_deviations):
        expected_series_explained_standard_deviations = \
            expected_series_explained_standard_deviations.A.squeeze()
    
    N = observed_series.shape[0]
    
    observed_colour = standard_palette[0]
    
    expected_palette = seaborn.light_palette(standard_palette[1], 5)
    
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
    expected_explained_standard_deviations_label = \
        "Explained standard deviation"
    
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
            raise ValueError("Sort direction can either be ascending or descending.")
    else:
        sort_indices = slice(None)
        
    # Standard deviations
    
    if expected_series_total_standard_deviations is not None:
        with_total_standard_deviations = True
        expected_series_total_standard_deviations_lower = expected_series - \
            expected_series_total_standard_deviations
        expected_series_total_standard_deviations_upper = expected_series + \
            expected_series_total_standard_deviations
    else:
        with_total_standard_deviations = False
    
    if expected_series_explained_standard_deviations is not None \
        and expected_series_explained_standard_deviations.mean() > 0:
        with_explained_standard_deviations = True
        expected_series_explained_standard_deviations_lower = expected_series - \
            expected_series_explained_standard_deviations
        expected_series_explained_standard_deviations_upper = expected_series + \
            expected_series_explained_standard_deviations
    else:
        with_explained_standard_deviations = False
    
    # Figure
    
    feature_indices = numpy.arange(N) + 1
    
    if y_scale == "both":
        figure, axes = pyplot.subplots(nrows = 2, sharex = True)
        figure.subplots_adjust(hspace = 0.1)
        axis_upper = axes[0]
        axis_lower = axes[1]
        axis_upper.set_zorder = 1
        axis_lower.set_zorder = 0
    else:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        axes = [axis]
    
    handles = []
    
    for i, axis in enumerate(axes):
        observed_plot, = axis.plot(
            feature_indices,
            observed_series[sort_indices],
            label = observed_label,
            color = observed_colour,
            marker = observed_marker,
            linestyle = observed_line_style,
            zorder = observed_z_order
        )
        if i == 0:
            handles.append(observed_plot)
        expected_plot, = axis.plot(
            feature_indices,
            expected_series[sort_indices],
            label = expected_label,
            color = expected_colour,
            marker = expected_marker,
            linestyle = expected_line_style,
            zorder = expected_z_order
        )
        if i == 0:
            handles.append(expected_plot)
        if with_total_standard_deviations:
            axis.fill_between(
                feature_indices,
                expected_series_total_standard_deviations_lower[sort_indices],
                expected_series_total_standard_deviations_upper[sort_indices],
                color = expected_total_standard_deviations_colour,
                zorder = 0
            )
            expected_plot_standard_deviations_values = \
                matplotlib.patches.Patch(
                label = expected_total_standard_deviations_label,
                color = expected_total_standard_deviations_colour
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
                color = expected_explained_standard_deviations_colour,
                zorder = 1
            )
            expected_plot_standard_deviations_expectations = \
                matplotlib.patches.Patch(
                label = expected_explained_standard_deviations_label,
                color = expected_explained_standard_deviations_colour
            )
            if i == 0:
                handles.append(expected_plot_standard_deviations_expectations)
    
    if y_scale == "both":
        axis_upper.legend(
            handles = handles,
            loc = "best"
        )
        
        seaborn.despine(ax = axis_upper)
        seaborn.despine(ax = axis_lower)
        
        axis_upper.set_yscale("log", nonposy = "clip")
        axis_lower.set_yscale("linear")
        figure.text(0.04, 0.5, y_label, va = "center", rotation = "vertical")
        
        axis_lower.set_xscale(x_scale)
        axis_lower.set_xlabel(x_label)
        
        y_upper_min, y_upper_max = axis_upper.get_ylim()
        y_lower_min, y_lower_max = axis_lower.get_ylim()
        axis_upper.set_ylim(y_cutoff, y_upper_max)
            
        y_lower_min = max(-1, y_lower_min)
        axis_lower.set_ylim(y_lower_min, y_cutoff)
        
        # if y_upper_max < 100:
        #     axis_upper.yaxis.set_major_formatter(CustomTicker())
        
    else:
        axis.legend(
            handles = handles,
            loc = "best"
        )
        
        seaborn.despine()
        
        axis.set_yscale(y_scale, nonposy = "clip")
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
        
        # if y_scale == "log" and y_max < 100:
        #     axis.yaxis.set_major_formatter(CustomTicker())
        
    return figure, figure_name

def plotHeatMap(values, x_name, y_name,
    z_name=None, z_symbol=None, z_min=None, z_max=None,
    symmetric=False, labels=None, label_kind=None, center=None,
    normalisation=None, normalisation_constants=None, name=None):
    
    figure_name = figureName("heat_map", name)
    
    M, N = values.shape
    
    if symmetric and M != N:
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
    
    if normalisation:
        values = normalisation["function"](values, normalisation_constants)
        if z_symbol:
            z_name = normalisation["label"](z_symbol)
        elif z_name:
            z_name = "Normalised " + z_name.lower()
    
    cbar_dict = {}
    
    if z_name:
        cbar_dict["label"] = z_name
    
    if not symmetric:
        aspect_ratio = M / N
        square_cells = 1/5 < aspect_ratio and aspect_ratio < 5
    else:
        square_cells = True
    
    if labels is not None:
        x_indices = numpy.argsort(labels)
        y_name += " sorted"
        if label_kind:
            y_name += " by " + label_kind
    else:
        x_indices = numpy.arange(M)
    
    if symmetric:
        y_indices = x_indices
        x_name = y_name
    else:
        y_indices = numpy.arange(N)
    
    seaborn.set(style = "white")
    
    seaborn.heatmap(
        values[x_indices][:, y_indices],
        vmin = z_min, vmax = z_max, center = center, 
        xticklabels = False, yticklabels = False,
        cbar = True, cbar_kws = cbar_dict, cmap = standard_colour_map,
        square = square_cells, ax = axis
    )
    
    reset_plot_look()
    
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)
    
    return figure, figure_name

def plotELBOHeatMap(data_frame, x_label, y_label, z_label = None, z_symbol = None,
    z_min = None, z_max = None, name = None):
    
    figure_name = figureName("ELBO_heat_map", name)
    
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
    
    seaborn.set(style = "white")
    
    seaborn.heatmap(
        data_frame,
        vmin = z_min, vmax = z_max,
        xticklabels = True, yticklabels = True,
        cbar = True, cbar_kws = cbar_dict, #cmap = standard_colour_map,
        annot = True, fmt = "-.6g",
        square = False, ax = axis
    )
    
    reset_plot_look()
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    return figure, figure_name

def plotMatrix(feature_matrix, plot_distances=False, center_value=None,
    example_label=None, feature_label=None, value_label=None,
    sorting_method=None, distance_metric="Euclidean",
    labels=None, label_kind=None, class_palette=None,
    feature_indices_for_plotting=None,
    hide_dendrogram=False, name_parts=None):
    
    # Setup
    
    figure_name = figureName(name_parts)
    
    M, N = feature_matrix.shape
    
    if plot_distances:
        center_value = None
        feature_label = None
        value_label = "Pairwise {} distances in {} space".format(
            distance_metric,
            value_label
        )
    
    if not plot_distances and feature_indices_for_plotting is None:
        feature_indices_for_plotting = numpy.arange(N)
    
    # Checks
    
    if sorting_method == "labels" and labels is None:
        raise ValueError("No labels provided to sort after.")
    
    if labels is not None and not class_palette:
        raise ValueError("No class palette provided.")
    
    # Distances (if needed)
    
    distances = None
    
    if plot_distances or sorting_method == "hierarchical_clustering":
        distances = computePairwiseDistances(
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
    
    if sorting_method is "hierarchical_clustering" and not hide_dendrogram:
        axis_dendrogram = divider.append_axes("left", size="20%", pad=0.01)
        left_most_axis = axis_dendrogram
    
    # Label colours
    
    if labels is not None:
        label_colours = [class_palette[l] for l in labels]
        unique_colours = class_palette.values()
        value_for_colour = {
            colour: i for i, colour in enumerate(unique_colours)
        }
        label_colour_matrix = numpy.array(
            [value_for_colour[colour] for colour in label_colours]
        ).reshape(M, 1)
        label_colour_map = matplotlib.colors.ListedColormap(unique_colours)
    else:
        label_colour_matrix = None
        label_colour_map = None
    
    # Heat map aspect ratio
    
    if not plot_distances:
        square_cells = False
    else:
        square_cells = True
    
    # Plots
    
    seaborn.set(style = "white")
    
    ## Sorting and optional dendrogram
    
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
        example_indices = numpy.arange(M)
    
    else:
        raise ValueError(
            "`sorting_method` should be either \"labels\""
            " or \"hierarchical clustering\""
        )
    
    ## Heat map of values
    
    if plot_distances:
        plot_values = distances[example_indices][:, example_indices]
    else:
        plot_values = feature_matrix[example_indices]\
            [:, feature_indices_for_plotting]
    
    if scipy.sparse.issparse(plot_values):
        plot_values = plot_values.A
    
    colour_bar_dictionary = {}
    
    if value_label:
        colour_bar_dictionary["label"] = value_label
    
    seaborn.heatmap(
        plot_values, center = center_value,
        xticklabels=False, yticklabels=False,
        cbar=True, cbar_kws=colour_bar_dictionary, cbar_ax=axis_colour_map,
        square=square_cells, ax=axis_heat_map
    )
    
    ## Colour labels
    
    if axis_labels:
        seaborn.heatmap(
            label_colour_matrix[example_indices],
            xticklabels=False, yticklabels=False,
            cbar=False,
            cmap=label_colour_map,
            ax=axis_labels
        )
    
    reset_plot_look()
    
    ## Axis labels
    
    if example_label:
        left_most_axis.set_ylabel(example_label)
    
    if feature_label:
        axis_heat_map.set_xlabel(feature_label)
    
    return figure, figure_name

def plotVariableCorrelations(values, variable_names = None,
    colouring_data_set = None, name = "variable_correlations"):
    
    # Setup
    
    figure_name = figureName(name)
    
    M, N = values.shape
    
    random_state = numpy.random.RandomState(117)
    shuffled_indices = random_state.permutation(M)
    values = values[shuffled_indices]
    
    if colouring_data_set:
        labels = colouring_data_set.labels
        class_names = colouring_data_set.class_names
        number_of_classes = colouring_data_set.number_of_classes
        class_palette = colouring_data_set.class_palette
        label_sorter = colouring_data_set.label_sorter
        
        if not label_sorter:
            label_sorter = createLabelSorter()
        
        if not class_palette:
            index_palette = lighter_palette(number_of_classes)
            class_palette = {
                class_name: index_palette[i] for i, class_name in
                enumerate(sorted(class_names, key = label_sorter))
            }
        
        labels = labels[shuffled_indices]
        
        colours = []
    
        for label in labels:
            colour = class_palette[label]
            colours.append(colour)
        
    else:
        colours = neutral_colour
    
    # Figure
    
    figure, axes = pyplot.subplots(
        nrows = N,
        ncols = N,
        figsize = [1.5 * N] * 2
    )
    
    for i in range(N):
        for j in range(N):
            axes[i, j].scatter(values[:, i], values[:, j], c = colours, s = 1)
            
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            
            if i == N - 1:
                axes[i, j].set_xlabel(variable_names[j])
        
        axes[i, 0].set_ylabel(variable_names[i])
    
    return figure, figure_name

def plotCorrelations(correlation_sets, x_key, y_key,
    x_label = None, y_label = None, name = None):
    
    # Setup
    
    figure_name = figureName("correlations", name)
    
    if not isinstance(correlation_sets, dict):
        correlation_sets = {"correlations": correlation_sets}
    
    # Figure
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.despine()
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    for correlation_set_name, correlation_set in correlation_sets.items():
        axis.scatter(
            correlation_set[x_key], correlation_set[y_key],
            label = correlation_set_name
        )
    
    if len(correlation_sets) > 1:
        axis.legend(loc = "best")
    
    return figure, figure_name

def plotModelMetrics(
        metrics_sets,
        key,
        label = None,
        primary_differentiator_key = None,
        primary_differentiator_order = None,
        secondary_differentiator_key = None,
        secondary_differentiator_order = None,
        palette = None,
        marker_styles = None,
        name = None
    ):
    
    # Setup
    
    figure_name = figureName("model_metrics", name)
    
    if not isinstance(metrics_sets, list):
        metrics_sets = [metrics_sets]
    
    if not palette:
        palette = standard_palette.copy()
    
    # Figure
    
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
                x = x,
                y = y_mean,
                yerr = y_sd,
                capsize = 2,
                linestyle = "",
                color = colour,
                label = colour_key
            )
        
        axis.errorbar(
            x = x,
            y = y_mean,
            yerr = y_sd,
            ecolor = colour,
            capsize = 2,
            color = colour,
            marker = "_",
            # markeredgecolor = darker_colour,
            # markersize = 7
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
            key = lambda l:
                [order.index(l[0]), l[0]] if l[0] in order
                else [len(order), l[0]]
        ))
        
        axis.legend(handles, labels, loc = "best")
    
    return figure, figure_name

def plotModelMetricSets(
        metrics_sets,
        x_key, y_key,
        x_label = None, y_label = None,
        primary_differentiator_key = None,
        primary_differentiator_order = None,
        secondary_differentiator_key = None,
        secondary_differentiator_order = None,
        special_cases = None,
        other_method_metrics = None,
        palette = None,
        marker_styles = None,
        name = None
    ):
    
    # Setup
    
    figure_name = figureName("model_metric_sets", name)
    
    if other_method_metrics:
        figure_name += "-other_methods"
    
    if not isinstance(metrics_sets, list):
        metrics_sets = [metrics_sets]
    
    if not palette:
        palette = standard_palette.copy()
    
    if not marker_styles:
        marker_styles = [
            "X", # cross
            "s", # square
            "D", # diamond
            "o", # circle
            "P", # plus
            "^", # upright triangle
            "p", # pentagon
            "*", # star
        ]
    
    # Figure
    
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
                x = x_mean,
                y = y_mean,
                yerr = y_sd,
                xerr = x_sd,
                capsize = 2,
                linestyle = "",
                color = colour,
                label = colour_key,
                markersize = 7
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
                color = "black", marker = marker, linestyle = "none",
                label = marker_key
            )
        
        errorbar_colour = colour
        darker_colour = seaborn.dark_palette(colour, n_colors = 4)[2]
        
        special_case_changes = special_cases.get(colour_key, {})
        special_case_changes.update(
            special_cases.get(marker_key, {})
        )
        
        for object_name, object_change in special_case_changes.items():
            
            if object_name == "errorbar_colour":
                if object_change == "darken":
                    errorbar_colour = darker_colour
        
        axis.errorbar(
            x = x_mean,
            y = y_mean,
            yerr = y_sd,
            xerr = x_sd,
            ecolor = errorbar_colour,
            capsize = 2,
            color = colour,
            marker = marker,
            markeredgecolor = darker_colour,
            markersize = 7
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
                y = y_mean,
                color = standard_palette[-1],
                linestyle = line_style,
                label = method_name,
                zorder = -1
            )
            
            if y_sd is not None:
                axis.axhspan(
                    ymin = y_mean - y_sd,
                    ymax = y_mean + y_sd,
                    facecolor = standard_palette[-1],
                    alpha = 0.4,
                    edgecolor = None,
                    label = method_name,
                    zorder = -2
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
            key = lambda l:
                [order.index(l[0]), l[0]] if l[0] in order
                else [len(order), l[0]]
        ))
        
        axis.legend(handles, labels, loc = "best")
    
    return figure, figure_name

def plotValues(values, colour_coding = None, colouring_data_set = None,
    centroids = None, class_name = None, feature_index = None,
    figure_labels = None, prediction_details = None,
    axis_limits = None, example_tag = None, name = "scatter"):
    
    # Setup
    
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
    
    legend = None
    
    # Values
    
    values = values.copy()
    
    # Randomise examples in values to remove any prior order
    M, N = values.shape
    random_state = numpy.random.RandomState(117)
    shuffled_indices = random_state.permutation(M)
    values = values[shuffled_indices]
    
    if axis_limits:
        
        original_x_min = values[:, 0].min()
        original_x_max = values[:, 0].max()
        original_y_min = values[:, 1].min()
        original_y_max = values[:, 1].max()

        if axis_limits["x"]["minimum"]:
            x_min = axis_limits["x"]["minimum"]
        else:
            x_min = original_x_min

        if axis_limits["x"]["maximum"]:
            x_max = axis_limits["x"]["maximum"]
        else:
            x_max = original_x_max

        if axis_limits["y"]["minimum"]:
            y_min = axis_limits["y"]["minimum"]
        else:
            y_min = original_y_min

        if axis_limits["y"]["maximum"]:
            y_max = axis_limits["y"]["maximum"]
        else:
            y_max = original_y_max
        
        include_indices = []
        outlier_indices = []
    
        for i in range(M):
            if values[i, 0] < x_min or values[i, 0] > x_max \
                or values[i, 1] < y_min or values[i, 1] > y_max:
                outlier_indices.append(i)
            else:
                include_indices.append(i)
    
        values = values[include_indices]
        shuffled_indices = shuffled_indices[include_indices]
        
        outliers_string = "{} {}s not shown".format(len(outlier_indices),
            example_tag)
    
    # Adjust point size based on number of examples
    
    if M <= maximum_number_of_examples_for_large_points_in_scatter_plots:
        marker_size = default_large_marker_size_in_scatter_plots
    else:
        marker_size = default_small_marker_size_in_scatter_plots
    
    changePointSizeForPlots(marker_size)
    
    # Figure
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.despine()
    
    # axis.set_aspect("equal", adjustable = "datalim")
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    colour_map = seaborn.dark_palette(standard_palette[0], as_cmap = True)
    
    if colour_coding and (
            "labels" in colour_coding or
            "ids" in colour_coding or
            "class" in colour_coding
        ):
        
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
            number_of_classes = \
                colouring_data_set.number_of_predicted_superset_classes
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
        
        if not label_sorter:
            label_sorter = createLabelSorter()
        
        if not class_palette:
            index_palette = lighter_palette(number_of_classes)
            class_palette = {
                class_name: index_palette[i] for i, class_name in
                enumerate(sorted(class_names, key = label_sorter))
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
                    axis.scatter(values[i, 0], values[i, 1], color = colour,
                        label = label)
            
            axis.scatter(values[:, 0], values[:, 1], c = colours)
            
            class_handles, class_labels = axis.get_legend_handles_labels()
            
            if class_labels:
                class_labels, class_handles = zip(*sorted(
                    zip(class_labels, class_handles),
                    key = lambda t: label_sorter(t[0])
                ))
                class_label_maximum_width = max(*map(len, class_labels))
                if class_label_maximum_width <= 5 and number_of_classes <= 20:
                    legend = axis.legend(
                        class_handles, class_labels,
                        loc = "best"
                    )
                else:
                    if number_of_classes <= 20:
                        class_label_columns = 2
                    else:
                        class_label_columns = 3
                    legend = axis.legend(
                        class_handles,
                        class_labels,
                        bbox_to_anchor = (-0.1, 1.05, 1.1, 0.95),
                        loc = "lower left",
                        ncol = class_label_columns,
                        mode = "expand",
                        borderaxespad = 0.,
                        # fontsize = "x-small"
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
                    colour = neutral_colour
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
                axis.scatter(ordered_values[:, 0], ordered_values[:, 1],
                    c = ordered_colours, label = label, zorder = z_order)
                
                handles, labels = axis.get_legend_handles_labels()
                labels, handles = zip(*sorted(zip(labels, handles),
                    key = lambda t: label_sorter(t[0])))
                legend = axis.legend(
                    handles,
                    labels, 
                    bbox_to_anchor = (-0.1, 1.05, 1.1, 0.95),
                    loc = "lower left",
                    ncol = 2,
                    mode = "expand",
                    borderaxespad = 0.
                )
    
    elif colour_coding == "count_sum":
        
        n = colouring_data_set.count_sum[shuffled_indices].flatten()
        scatter_plot = axis.scatter(values[:, 0], values[:, 1], c = n,
            cmap = colour_map)
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
        
        f = colouring_data_set.values[shuffled_indices,
            feature_index]
        if scipy.sparse.issparse(f):
            f = f.A
        f = f.squeeze()
        
        scatter_plot = axis.scatter(values[:, 0], values[:, 1], c = f,
            cmap = colour_map)
        colour_bar = figure.colorbar(scatter_plot)
        colour_bar.outline.set_linewidth(0)
        colour_bar.set_label(feature_name)
    
    else:
        axis.scatter(values[:, 0], values[:, 1], color = neutral_colour)
    
    if centroids:
        prior_centroids = centroids["prior"]
        
        if prior_centroids:
            K = prior_centroids["probabilities"].shape[0]
        else:
            K = 0
        
        if K > 1:
            centroids_palette = darker_palette(K)
            classes = numpy.arange(K)
            
            probabilities = prior_centroids["probabilities"]
            means = prior_centroids["means"]
            covariance_matrices = prior_centroids["covariance_matrices"]
            
            for k in range(K):
                axis.scatter(means[k, 0], means[k, 1], marker = "x",
                    color = "black", linewidth = 3, s = 60)
                axis.scatter(means[k, 0], means[k, 1], marker = "x",
                    facecolor = centroids_palette[k], edgecolors = "black")
                ellipse_fill, ellipse_edge = covarianceMatrixAsEllipse(
                    covariance_matrices[k],
                    means[k],
                    colour = centroids_palette[k]
                )
                axis.add_patch(ellipse_edge)
                axis.add_patch(ellipse_fill)
    
    if axis_limits:
    
        if not legend:
            legend = axis.legend(handles = [])
    
        legend.set_title(outliers_string, {"size": "small"})
    
    reset_plot_look()
    
    return figure, figure_name

def plotProbabilities(posterior_probabilities, prior_probabilities, 
    x_label = None, y_label = None,
    palette = None, uniform = False, name = None):
    
    figure_name = figureName("probabilities", name)
    
    if not x_label:
        x_label = "$k$"
    
    if not y_label:
        y_label = "$\\pi_k$"
    
    figure = pyplot.figure(figsize=(8, 6), dpi=80)
    axis = figure.add_subplot(1, 1, 1)
    
    if not palette:
        palette = [standard_palette[0]] * K
    
    if posterior_probabilities is not None:
        K = len(posterior_probabilities)
        for k in range(K):
            axis.bar(k, posterior_probabilities[k], color = palette[k])
        axis.set_ylabel("$\\pi_{\\phi}^k$")
        if prior_probabilities is not None: 
            for k in range(K):
                axis.plot([k-0.4, k+0.4], 2 * [prior_probabilities[k]],
                    color = "black",
                    linestyle = "dashed"
                )
            prior_line = matplotlib.lines.Line2D([], [], color="black", linestyle="dashed", label = "$\\pi_{\\theta}^k$")
            axis.legend(handles=[prior_line], loc="best", fontsize=18)
    elif prior_probabilities is not None:
        K = len(prior_probabilities)
        for k in range(K):
            axis.bar(k, prior_probabilities[k], color = palette[k])
        axis.set_ylabel("$\\pi_{\\theta}^k$")
    
    axis.set_xlabel(x_label)
    
    seaborn.despine()
    
    return figure, figure_name

def covarianceMatrixAsEllipse(covariance_matrix, mean, colour,
    linestyle = "solid", radius_stddev = 1, label = None):
    
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance_matrix)
    indices_sorted_ascending = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices_sorted_ascending]
    eigenvectors = eigenvectors[:, indices_sorted_ascending]
    
    lambda_1, lambda_2 = numpy.sqrt(eigenvalues)
    
    theta = numpy.degrees(numpy.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    ellipse_fill = matplotlib.patches.Ellipse(
        xy = mean,
        width  = 2 * radius_stddev * lambda_1,
        height = 2 * radius_stddev * lambda_2,
        angle = theta,
        linewidth = 2,
        linestyle = linestyle,
        facecolor = "none",
        edgecolor = colour,
        label = label
    )
    ellipse_edge = matplotlib.patches.Ellipse(
        xy = mean,
        width  = 2 * radius_stddev * lambda_1,
        height = 2 * radius_stddev * lambda_2,
        angle = theta,
        linewidth = 3,
        linestyle = linestyle,
        facecolor = "none",
        edgecolor = "black",
        label = label
    )
    
    return ellipse_fill, ellipse_edge

def combineImagesFromDataSet(data_set, number_of_random_examples = 100, indices = None, name = None):
    
    image_name = figureName("random_image_examples", name)
    
    random_state = numpy.random.RandomState(13)
    
    if not isinstance(indices, (list, tuple)):
        M = number_of_random_examples
        indices = random_state.permutation(data_set.number_of_examples)[:M]
        if M == 1:
            image_name = figureName("image_example", name)
        else:
            image_name = figureName("image_examples", name)
    else: 
        M = len(indices)
        if M == 1:
            image_name = figureName("image_example", name)
        else:
            image_name = figureName("image_examples", name)
    
    W, H = data_set.feature_dimensions
    
    examples = data_set.values[indices]
    if scipy.sparse.issparse(examples):
        examples = examples.A
    examples = examples.reshape(M, W, H)
    
    C = int(numpy.ceil(numpy.sqrt(M)))
    R = int(numpy.ceil(M / C))
    
    image = numpy.zeros((R * W, C * H))
    
    for m in range(M):
        c = int(m % C)
        r = int(numpy.floor(m / C))
        rows = slice(r*W, (r+1)*W)
        columns = slice(c*H, (c+1)*H)
        image[rows, columns] = examples[m]
    
    return image, image_name

def saveImage(image, image_name, results_directory):
    
    shape = image.shape
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    minimum = image.min()
    maximum = image.max()
    if 0 < minimum and minimum < 1 and 0 < maximum and maximum < 1:
        rescaled_image = 255 * image
    else:
        rescaled_image = (255 / (maximum - minimum) * (image - minimum))
    
    image = Image.fromarray(rescaled_image.astype(numpy.uint8))
    
    image_name += image_extension
    image_path = os.path.join(results_directory, image_name)
    image.save(image_path)

def figureName(base_name, other_names = None):
    
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
            other_names = [str(name) for name in other_names if name is not None]
        figure_name += "-" + "-".join(map(normaliseString, other_names))
    
    return figure_name

def saveFigure(figure, figure_name, export_options, results_directory):
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    figure_path_base = os.path.join(results_directory, figure_name)
    
    default_tight_layout = True
    tight_layout = default_tight_layout
    
    figure_width, figure_height = figure.get_size_inches()
    aspect_ratio = figure_width / figure_height
    
    if "video" in export_options:
        # This check was used to ensure figure frames for a video had the same
        # size, since whitespace were removed when saving, but since figures
        # now use tight layout instead, whitespace does not need to be
        # removed. The check is kept for later use.
        # tight_layout = False
        pass
    
    figure.set_tight_layout(tight_layout)
    adjustFigureForLegend(figure)
    
    figure_path = figure_path_base + figure_extension
    figure.savefig(figure_path)
    
    if "publication" in export_options:
        
        figure.set_tight_layout(default_tight_layout)
        figure.set_dpi(publication_dpi)
        
        local_publication_copies = publication_copies.copy()
        local_publication_copies["standard"] = {
            "figure_width": figure_width
        }
        
        for copy_name, copy_properties in local_publication_copies.items():
            
            figure.set_size_inches(figure_width, figure_height)
            
            publication_figure_width = copy_properties["figure_width"]
            publication_figure_height = publication_figure_width \
                / aspect_ratio
            
            figure.set_size_inches(publication_figure_width,
                                   publication_figure_height)
            adjustFigureForLegend(figure)
            
            figure_path = "-".join([
                figure_path_base, "publication", copy_name
            ]) + publication_figure_extension
            
            figure.savefig(figure_path)
    
    pyplot.close(figure)

def adjustFigureForLegend(figure):
    
    for axis in figure.get_axes():
        legend = axis.get_legend()
        
        if legend and legendIsAboveAxis(legend, axis):
            
            renderer = figure.canvas.get_renderer()
            figure.draw(renderer=renderer)
            
            legend_size = legend.get_window_extent()
            legend_height_in_inches = legend_size.height / figure.get_dpi()
            
            figure_width, figure_height = figure.get_size_inches()
            figure_height += legend_height_in_inches
            figure.set_size_inches(figure_width, figure_height)

def legendIsAboveAxis(legend, axis):
    
    legend_bottom_vertical_position_relative_to_axis = \
        legend.get_bbox_to_anchor().transformed(axis.transAxes.inverted()).ymin
    
    if legend_bottom_vertical_position_relative_to_axis >= 1:
        legend_is_above_axis = True
    else:
        legend_is_above_axis = False
    
    return legend_is_above_axis

def axisLabelForSymbol(symbol, coordinate = None, decomposition_method = None,
    distribution = None, prefix = "", suffix = ""):
    
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
        coordinate_text = "{{" + decomposition_label + " " + str(coordinate) + "}}"
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
    
    axis_label = "$" + prefix + symbol \
        + distribution_position + distribution_symbol \
        + coordinate_position + coordinate_text \
        + suffix + "$"
    
    return axis_label

def changePointSizeForPlots(marker_size):
    matplotlib.rc(group="lines", markersize=marker_size)
    matplotlib.rc(
        group="legend",
        markerscale=legend_marker_scale_from_marker_size(marker_size)
    )

class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos = None):
        if 0.1 <= x < 1:
            tick_label = "{:.1f}".format(x)
        elif 1 <= x < 100:
            tick_label = "{:.0f}".format(x)
        else:
            tick_label =  LogFormatterSciNotation.__call__(self, x, pos = None)
        print(tick_label)
        return tick_label

def buildPathForResultDirectory(base_directory, model_name,
    run_id = None, subdirectories = None):
    
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

def parseAnalyses(analyses):
    
    resulting_analyses = set()
    
    for analysis in analyses:
        if analysis in analysis_groups:
            group = analysis
            resulting_analyses.update(map(normaliseString,
                analysis_groups[group]))
        elif analysis in analysis_groups["complete"]:
            resulting_analyses.add(normaliseString(analysis))
        else:
            raise ValueError("Analysis `{}` not found.".format(analysis))
    
    resulting_analyses = list(resulting_analyses)
    
    return resulting_analyses
