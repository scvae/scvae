#!/usr/bin/env python3

import numpy
from numpy import nan

import scipy.sparse

from sklearn.decomposition import PCA, IncrementalPCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import adjusted_rand_score

from matplotlib import pyplot
import matplotlib.patches
import matplotlib.lines
import matplotlib.gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.ticker import LogFormatterSciNotation

import seaborn

from PIL import Image

from pandas import DataFrame

import os
import gzip
import pickle

import copy
import re

from time import time
from auxiliary import (
    loadNumberOfEpochsTrained, loadLearningCurves, loadAccuracies,
    loadCentroids, loadKLDivergences,
    formatTime, formatDuration,
    normaliseString, properString, heading
)
import warnings

standard_palette = seaborn.color_palette('Set2', 8)
standard_colour_map = seaborn.cubehelix_palette(light = .95, as_cmap = True)
neutral_colour = (0.7, 0.7, 0.7)

lighter_palette = lambda N: seaborn.hls_palette(N)
darker_palette = lambda N: seaborn.hls_palette(N, l = .4)

reset_plot_look = lambda: seaborn.set(
    context = "notebook",
    style = "ticks",
    palette = standard_palette
)
reset_plot_look()

figure_extension = ".png"
image_extension = ".png"

maximum_feature_size_for_analyses = 2000

maximum_number_of_values_for_heat_maps = 5000 * 25000

maximum_number_of_values_for_dense_matrix = 4e9
maximum_number_of_features_for_t_sne = 100
maximum_number_of_examples_for_t_sne = 50000
number_of_pca_components_before_tsne = 32

number_of_random_examples = 100

profile_comparison_maximum_number_of_examples = 25
profile_comparison_maximum_number_of_examples_per_class = 3
profile_comparison_count_cut_off = 10.5

maximum_count_scales = [1, 5]
axis_limits_scales = [1, 5]

default_cutoffs = range(1, 10)

analysis_groups = {
    "simple": ["metrics", "images", "learning_curves", "accuracies"],
    "default": ["kl_heat_maps", "profile_comparisons", "distributions",
        "decompositions", "latent_space"],
    "complete": ["heat_maps", "latent_distributions"]
}
analysis_groups["default"] += analysis_groups["simple"]
analysis_groups["complete"] += analysis_groups["default"]

def analyseData(data_sets,
    decomposition_methods = ["PCA"], highlight_feature_indices = [],
    analyses = ["default"], analysis_level = "normal",
    results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, "data")
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    analyses = parseAnalyses(analyses)
    
    # Metrics
    
    if "metrics" in analyses:
        
        heading("Metrics")
        
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
                + "\n".join(["{} examples: {}".format(kind.capitalize(), number)
                             for kind, number in number_of_examples.items()
                             if kind != "full"]) + "\n" \
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
        
        print(formatStatistics(histogram_statistics, name = "Series"))
    
    # Loop over data sets
    
    for data_set in data_sets:
        
        heading("Analyses of {} set".format(data_set.kind))
        
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
                results_directory = results_directory
            )
        
        # Heat map for data set
        
        if "heat_maps" in analyses and \
            data_set.number_of_values <= maximum_number_of_values_for_heat_maps:
            
            print("Plotting heat map for {} set.".format(data_set.kind))
            
            heat_maps_directory = os.path.join(results_directory, "heat_maps")
            
            heat_maps_time_start = time()
            
            if data_set.labels is not None:
                labels = data_set.labels
            else:
                labels = None
            
            figure, figure_name = plotHeatMap(
                data_set.values,
                labels = labels,
                normalisation = data_set.heat_map_normalisation,
                normalisation_constants = data_set.count_sum,
                x_name = data_set.tags["feature"].capitalize() + "s",
                y_name = data_set.tags["example"].capitalize() + "s",
                z_name = data_set.tags["value"].capitalize() + "s",
                z_symbol = "x",
                name = data_set.kind
            )
            saveFigure(figure, figure_name, heat_maps_directory)
            
            heat_maps_duration = time() - heat_maps_time_start
            print("Heat map for {} set plotted and saved ({})."\
                .format(data_set.kind, formatDuration(heat_maps_duration)))
            
            print()
        
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
                results_directory = results_directory
            )

def analyseModel(model, analyses = ["default"], analysis_level = "normal",
    for_video = False, results_directory = "results"):
    
    # Setup
    
    number_of_epochs_trained = loadNumberOfEpochsTrained(model)
    
    epochs_string = "e_" + str(number_of_epochs_trained)
    
    results_directory = os.path.join(results_directory, model.name, epochs_string)
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    analyses = parseAnalyses(analyses)
    
    # Learning curves
    
    if "learning_curves" in analyses:
        
        heading("Learning curves")
            
        print("Plotting learning curves.")
        learning_curves_time_start = time()
        
        learning_curves = loadLearningCurves(model,
            data_set_kinds = ["training", "validation"])
        
        figure, figure_name = plotLearningCurves(learning_curves, model.type)
        saveFigure(figure, figure_name, results_directory)
        
        if for_video:
            print("Plotting learning curve evolution for video")
            for epoch in range(number_of_epochs_trained):
                figure, figure_name = plotLearningCurves(
                    learning_curves,
                    model.type,
                    epoch_slice = slice(epoch + 1),
                    global_y_lim = for_video
                )
                saveFigure(figure, figure_name, os.path.join(results_directory, "learning_curve_evolution"))
        
        if model.type == "SNN":
            figure, figure_name = plotSeparateLearningCurves(learning_curves,
                loss = "log_likelihood")
            saveFigure(figure, figure_name, results_directory)
        elif "VAE" in model.type:
            figure, figure_name = plotSeparateLearningCurves(learning_curves,
                loss = ["lower_bound", "reconstruction_error"])
            saveFigure(figure, figure_name, results_directory)
            if model.type in ["GMVAE"]:
                figure, figure_name = plotSeparateLearningCurves(learning_curves,
                    loss = "kl_divergence_z")
                saveFigure(figure, figure_name, results_directory)
                figure, figure_name = plotSeparateLearningCurves(learning_curves,
                    loss = "kl_divergence_y")
                saveFigure(figure, figure_name, results_directory)
            else:
                figure, figure_name = plotSeparateLearningCurves(learning_curves,
                    loss = "kl_divergence")
                saveFigure(figure, figure_name, results_directory)
    
        learning_curves_duration = time() - learning_curves_time_start
        print("Learning curves plotted and saved ({}).".format(
            formatDuration(learning_curves_duration)))
    
        print()
    
    # Accuracies
    
    if "accuracies" in analyses:
        
        accuracies_time_start = time()
        
        accuracies = loadAccuracies(model,
            data_set_kinds = ["training", "validation"])
        
        if accuracies is not None:
            
            heading("Accuracies")
            
            print("Plotting accuracies.")
            
            figure, figure_name = plotAccuracies(accuracies)
            saveFigure(figure, figure_name, results_directory)
            
            superset_accuracies = loadAccuracies(model,
                data_set_kinds = ["training", "validation"], superset = True)
            
            if superset_accuracies is not None:
                figure, figure_name = plotAccuracies(superset_accuracies,
                    name = "superset")
                saveFigure(figure, figure_name, results_directory)
            
            accuracies_duration = time() - accuracies_time_start
            print("Accuracies plotted and saved ({}).".format(
                formatDuration(accuracies_duration)))
            
            print()
    
    # Heat map of KL for all latent neurons
    
    if "kl_heat_maps" in analyses and model.type == "VAE":
        
        heading("KL divergence")
    
        print("Plotting logarithm of KL divergence heat map.")
        heat_map_time_start = time()
        
        KL_neurons = loadKLDivergences(model)
    
        KL_neurons = numpy.sort(KL_neurons, axis = 1)
        log_KL_neurons = numpy.log(KL_neurons)
        
        figure, figure_name = plotKLDivergenceEvolution(KL_neurons)
        saveFigure(figure, figure_name, results_directory)
        
        heat_map_duration = time() - heat_map_time_start
        print("Heat map plotted and saved ({}).".format(
            formatDuration(heat_map_duration)))
        
        print()
    
    # Latent distributions
    
    if "latent_distributions" in analyses and model.type == "GMVAE":
        
        heading("Latent distributions")
        
        centroids = loadCentroids(model, data_set_kinds = "validation")
        
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
                saveFigure(figure, figure_name, centroids_directory)
                
                figure, figure_name = plotEvolutionOfCentroidMeans(
                    centroid_means_decomposed, distribution, decomposed)
                saveFigure(figure, figure_name, centroids_directory)
                
                figure, figure_name = plotEvolutionOfCentroidCovarianceMatrices(
                    centroid_covariance_matrices, distribution)
                saveFigure(figure, figure_name, centroids_directory)
                
                centroids_duration = time() - centroids_time_start
                print("Evolution of latent {} parameters plotted and saved ({})"\
                    .format(distribution, formatDuration(centroids_duration)))
                
                print()

        if for_video:
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
                saveFigure(figure,
                    figure_name,
                    centroids_directory,
                    for_video = for_video
                )

            print()

def analyseIntermediateResults(learning_curves = None, epoch_start = None,
    epoch = None, latent_values = None, data_set = None, centroids = None,
    model_name = None, model_type = None, results_directory = "results"):
    
    results_directory = os.path.join(results_directory, model_name,
        "intermediate")
    
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
    
    if epoch is not None:
        saveFigure(figure, figure_name, results_directory, for_video = False)
    else:
        saveFigure(figure, figure_name, results_directory)

    
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
                components = 2
            )

            decompose_duration = time() - decompose_time_start
            print("{} decomposed ({}).".format(
                latent_set_name.capitalize(),
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
            figure.axes[0].set_xbound([-8, 13])
            figure.axes[0].set_ybound([-5, 12])
            figure.axes[0].set_xlim([-8, 13])
            figure.axes[0].set_ylim([-5, 12])
            saveFigure(figure, figure_name, results_directory,
                for_video = True)
            if data_set.label_superset is not None:
                figure, figure_name = plotValues(
                    latent_values_decomposed,
                    colour_coding = "superset labels",
                    colouring_data_set = data_set,
                    centroids = centroids_decomposed,
                    figure_labels = figure_labels,
                    name = name
                )
                figure.axes[0].set_xbound([-8, 13])
                figure.axes[0].set_ybound([-5, 12])
                figure.axes[0].set_xlim([-8, 13])
                figure.axes[0].set_ylim([-5, 12])
                saveFigure(figure, figure_name, results_directory,
                    for_video = True)
        else:
            figure, figure_name = plotValues(
                latent_values_decomposed,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
            )
            figure.axes[0].set_xbound([-8, 13])
            figure.axes[0].set_ybound([-5, 12])
            figure.axes[0].set_xlim([-8, 13])
            figure.axes[0].set_ylim([-5, 12])
            saveFigure(figure, figure_name, results_directory,
                for_video = True)
    
        if centroids:
            analyseCentroidProbabilities(
                centroids, epoch_name,
                for_video = True,
                results_directory = results_directory)
    
        plot_duration = time() - plot_time_start
        print("{} plotted and saved ({}).".format(
            latent_set_name.capitalize(), formatDuration(plot_duration)))

def analyseResults(evaluation_set, reconstructed_evaluation_set,
    latent_evaluation_sets, model,
    decomposition_methods = ["PCA"], highlight_feature_indices = [],
    early_stopping = False, best_model = False,
    analyses = ["default"], analysis_level = "normal",
    results_directory = "results", **remaining_arguments):
    
    if early_stopping and best_model:
        raise ValueError("Early-stopping model and best model cannot be"
            + " evaluated at the same time.")
    
    # Setup
    
    print("Setting up results analyses.")
    setup_time_start = time()
    
    number_of_epochs_trained = loadNumberOfEpochsTrained(model,
        early_stopping = early_stopping, best_model = best_model)
    
    M = evaluation_set.number_of_examples
    
    analyses = parseAnalyses(analyses)
    
    ## Comparison arrays
    
    if "metrics" in analyses or "heat_maps" in analyses \
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
    
    results_directory = os.path.join(results_directory, model.name,
        evaluation_directory)
    
    if evaluation_set.kind != "test":
        results_directory = os.path.join(results_directory, evaluation_set.kind)
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    setup_duration = time() - setup_time_start
    print("Finished setting up ({}).".format(formatDuration(setup_duration)))
    print()
    
    # Metrics
    
    if "metrics" in analyses:
    
        heading("Metrics")
        
        ## Loading
        
        print("Loading results from model log directory.")
        loading_time_start = time()
        
        evaluation_eval = loadLearningCurves(model, "evaluation",
            early_stopping = early_stopping, best_model = best_model)
        accuracy_eval = loadAccuracies(model, "evaluation",
            early_stopping = early_stopping, best_model = best_model)
        superset_accuracy_eval = loadAccuracies(
            model, "evaluation", superset = True,
            early_stopping = early_stopping, best_model = best_model
        )
        
        loading_duration = time() - loading_time_start
        print("Results loaded ({}).".format(formatDuration(loading_duration)))
        print()
        
        ## Calculating
        
        print("Calculating metrics for results.")
        metrics_time_start = time()
        
        evaluation_set_statistics = [
            statistics(data_set.values, data_set.version, tolerance = 0.5)
                for data_set in [evaluation_set, reconstructed_evaluation_set]
        ]
        
        if analysis_level == "extensive":
            evaluation_set_statistics.append(statistics(numpy.abs(x_diff),
                "differences", skip_sparsity = True))
            evaluation_set_statistics.append(statistics(numpy.abs(x_log_ratio),
                "log-ratios", skip_sparsity = True))
        
        if evaluation_set.values.max() > 20:
            count_accuracy_method = "orders of magnitude"
        else:
            count_accuracy_method = None
        
        if analysis_level == "extensive":
            count_accuracies = computeCountAccuracies(
                evaluation_set.values,
                reconstructed_evaluation_set.values,
                method = count_accuracy_method
            )
        else:
            count_accuracies = None
        
        if evaluation_set.has_labels:
            
            if evaluation_set.has_predicted_cluster_ids:
                ARI_clusters = adjusted_rand_score(
                    evaluation_set.labels,
                    evaluation_set.predicted_cluster_ids
                )
            else:
                ARI_clusters = None
            
            if evaluation_set.has_predicted_labels:
                ARI_labels = adjusted_rand_score(
                    evaluation_set.labels,
                    evaluation_set.predicted_labels
                )
            else:
                ARI_labels = None
        
        else:
            ARI_clusters = None
            ARI_labels = None
        
        metrics_duration = time() - metrics_time_start
        print("Metrics calculated ({}).".format(
            formatDuration(metrics_duration)))
        
        ## Saving
        
        metrics_log_filename = "{}_metrics".format(evaluation_set.kind)
        metrics_log_path = os.path.join(results_directory,
            metrics_log_filename + ".log")
        metrics_dictionary_path = os.path.join(results_directory,
            metrics_log_filename + ".pkl.gz")
        
        metrics_saving_time_start = time()
        
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
                metrics_string += "    Accuracy (superset): {:6.2f} %.\n".format(
                    100 * superset_accuracy_eval[-1])
            if ARI_clusters:
                metrics_string += "    ARI (clusters): {:.5g}.\n".format(
                    ARI_clusters)
            if ARI_labels:
                metrics_string += "    ARI (labels): {:.5g}.\n".format(
                    ARI_labels)
            metrics_string += "\n" + formatStatistics(evaluation_set_statistics)
            if count_accuracies:
                metrics_string += "\n" + formatCountAccuracies(count_accuracies)
            metrics_file.write(metrics_string)
        
        with gzip.open(metrics_dictionary_path, "w") as metrics_file:
            metrics_dictionary = {
                "timestamp": metrics_saving_time_start,
                "number of epochs trained": number_of_epochs_trained,
                "evaluation": evaluation_eval,
                "accuracy": accuracy_eval,
                "superset_accuracy": superset_accuracy_eval,
                "ARI (clusters)": ARI_clusters,
                "ARI (labels)": ARI_labels,
                "statistics": evaluation_set_statistics,
                "count accuracies": count_accuracies,
            }
            pickle.dump(metrics_dictionary, metrics_file)
        
        metrics_saving_duration = time() - metrics_saving_time_start
        print("Metrics saved ({}).".format(formatDuration(
            metrics_saving_duration)))
            
        print()
        
        ## Displaying
        
        print(formatStatistics(evaluation_set_statistics))
        
        if ARI_clusters or ARI_labels:
            print("Adjusted rand index:")
            if ARI_clusters:
                print("    clusters: {:.5g}".format(ARI_clusters))
            if ARI_labels:
                print("    labels: {:.5g}".format(ARI_labels))
            print()
        
        if count_accuracies:
            print(formatCountAccuracies(count_accuracies))
    
    # Reconstructions
    
    if "images" in analyses or "profile_comparisons" in analyses:
        heading("Reconstructions")
    
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
        
        numpy.random.seed(80)
        
        print("Plotting profile comparisons.")
        
        image_comparisons_directory = os.path.join(
                results_directory, "image_comparisons")
        
        profile_comparisons_directory = os.path.join(
                results_directory, "profile_comparisons")
        
        profile_comparisons_time_start = time()
        
        y_cutoff = profile_comparison_count_cut_off
        
        if analysis_level == "limited":
            subset = [0]
        else:
            if evaluation_set.has_labels:
                
                subset = set()
            
                if evaluation_set.label_superset:
                    class_names = evaluation_set.superset_class_names
                    labels = evaluation_set.superset_labels
                else:
                    class_names = evaluation_set.class_names
                    labels = evaluation_set.labels
            
                class_counter = {}
            
                for class_name in class_names:
                    class_counter[class_name] = 0
            
                counter_max = profile_comparison_maximum_number_of_examples_per_class
            
                while any(map(lambda x: x < counter_max, class_counter.values())):
                    i = numpy.random.randint(0, M)
                    label = labels[i]
                    if class_counter[label] >= counter_max or i in subset:
                        continue
                    else:
                        class_counter[label] += 1
                        subset.add(i)
            else:
                subset = numpy.random.permutation(M)\
                    [:profile_comparison_maximum_number_of_examples]
        
        for i in subset:
            
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
                    saveFigure(figure, figure_name,
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
                    saveFigure(figure, figure_name,
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
        
        profile_comparisons_duration = time() - profile_comparisons_time_start
        print("Profile comparisons plotted and saved ({}).".format(
            formatDuration(profile_comparisons_duration)))
        
        print()
        
        numpy.random.seed()
    
    # Distributions
    
    evaluation_set_maximum_value = evaluation_set.values.max()
    
    if "distributions" in analyses:
        
        analyseDistributions(
            reconstructed_evaluation_set,
            colouring_data_set = evaluation_set,
            preprocessed = evaluation_set.preprocessing_methods,
            original_maximum_count = evaluation_set_maximum_value,
            analysis_level = analysis_level,
            results_directory = results_directory
        )
    
    ## Reconstructions decomposed
    
    if "decompositions" in analyses:
        
        analyseDecompositions(
            reconstructed_evaluation_set,
            colouring_data_set = evaluation_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            symbol = "\\tilde{{x}}",
            pca_limits = evaluation_set.pca_limits,
            title = "reconstruction space",
            analysis_level = analysis_level,
            results_directory = results_directory,
            **remaining_arguments
        )
        
        ## Reconstructions plotted in original decomposed space
        
        analyseDecompositions(
            evaluation_set,
            reconstructed_evaluation_set,
            colouring_data_set = evaluation_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            symbol = "x",
            pca_limits = evaluation_set.pca_limits,
            title = "original space",
            analysis_level = analysis_level,
            results_directory = results_directory,
            **remaining_arguments
        )
    
    # Heat maps
    
    if "heat_maps" in analyses and \
        reconstructed_evaluation_set.number_of_values \
        <= maximum_number_of_values_for_heat_maps:
        
        print("Plotting heat maps.")
        
        heat_maps_directory = os.path.join(results_directory, "heat_maps")
        
        ## Reconstructions
        
        heat_maps_time_start = time()
        
        figure, figure_name = plotHeatMap(
            reconstructed_evaluation_set.values,
            labels = reconstructed_evaluation_set.labels,
            normalisation =
                reconstructed_evaluation_set.heat_map_normalisation,
            normalisation_constants = evaluation_set.count_sum,
            x_name = evaluation_set.tags["feature"].capitalize() + "s",
            y_name = evaluation_set.tags["example"].capitalize() + "s",
            z_name = evaluation_set.tags["value"].capitalize() + "s",
            z_symbol = "\\tilde{{x}}",
            name = "reconstruction"
        )
        saveFigure(figure, figure_name, heat_maps_directory)
        
        heat_maps_duration = time() - heat_maps_time_start
        print("    Reconstruction heat map plotted and saved ({})." \
            .format(formatDuration(heat_maps_duration)))
        
        ## Differences
        
        if analysis_level == "extensive":
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
            saveFigure(figure, figure_name, heat_maps_directory)
            
            heat_maps_duration = time() - heat_maps_time_start
            print("    Difference heat map plotted and saved ({})." \
                .format(formatDuration(heat_maps_duration)))
    
        ## log-ratios
        
        if analysis_level == "extensive":
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
            saveFigure(figure, figure_name, heat_maps_directory)
            
            heat_maps_duration = time() - heat_maps_time_start
            print("    log-ratio heat map plotted and saved ({})." \
                .format(formatDuration(heat_maps_duration)))
    
    print()
    
    # Latent space
    
    if "latent_space" in analyses and "VAE" in model.type:
        
        heading("Latent space")
        
        if model.latent_distribution_name == "gaussian_mixture":
        
            print("Loading centroids from model log directory.")
            loading_time_start = time()
            
            centroids = loadCentroids(model, data_set_kinds = "evaluation",
                early_stopping = early_stopping, best_model = best_model)
            
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
            title = "latent space",
            specifier = lambda data_set: data_set.version,
            analysis_level = analysis_level,
            results_directory = results_directory,
            **remaining_arguments
        )
        
        if centroids:
            analyseCentroidProbabilities(
                centroids,
                analysis_level = "normal",
                results_directory = results_directory
            )
            
            print()

def analyseDistributions(data_set, colouring_data_set = None,
    cutoffs = None, preprocessed = False, original_maximum_count = None,
    analysis_level = "normal", results_directory = "results"):
    
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
        saveFigure(figure, figure_name, distribution_directory)
        
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
        saveFigure(figure, figure_name, distribution_directory)
        
        distribution_duration = time() - distribution_time_start
        print("    Superset class distribution plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
    
    ## Count distribution

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
        
        if isinstance(data_set.values, scipy.sparse.csr_matrix):
            values_series = data_set.values.data
        else:
            values_series = data_set.values.reshape(-1)
        
        figure, figure_name = plotHistogram(
            series = values_series,
            label = data_set.tags["value"].capitalize() + "s",
            discrete = data_set_discreteness,
            normed = True,
            y_scale = "log",
            maximum_count = maximum_count,
            name = count_histogram_name
        )
        saveFigure(figure, figure_name, distribution_directory)
        
        figure, figure_name = plotHistogram(
            series = values_series,
            label = data_set.tags["value"].capitalize() + "s",
            discrete = data_set_discreteness,
            normed = True,
            x_scale = "log",
            y_scale = "log",
            maximum_count = maximum_count,
            name = count_histogram_name
        )
        saveFigure(figure, figure_name, distribution_directory)
        
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
                series = data_set.values.reshape(-1),
                cutoff = cutoff,
                normed = True,
                scale = "log",
                name = data_set_name
            )
            saveFigure(figure, figure_name,
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
    saveFigure(figure, figure_name, distribution_directory)
    
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
            figure, figure_name = plotHistogram(
                series = data_set.values[class_indices].reshape(-1),
                label = data_set.tags["value"].capitalize() + "s",
                discrete = data_set_discreteness,
                normed = True,
                y_scale = "log",
                colour = class_palette[class_name],
                name = ["counts", data_set_name, "class", class_name]
            )
            saveFigure(figure, figure_name,
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
            saveFigure(figure, figure_name,
                class_count_distribution_directory)
    
        distribution_duration = time() - distribution_time_start
        print("    " + \
            "Count sum distributions for each class plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
    
    print()
    
def analyseDecompositions(data_sets, other_data_sets = [], centroids = None,
    colouring_data_set = None, decomposition_methods = ["PCA"],
    highlight_feature_indices = [], symbol = None, pca_limits = None,
    title = "data set", specifier = None,
    analysis_level = "normal", results_directory = "results",
    **remaining_arguments):
    
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
                    decomposition_method_names)
                
                values_decomposed = data_set.values
                other_values_decomposed = other_values
                centroids_decomposed = centroids
                
                if decomposition_method != "SVD" \
                    and isinstance(values_decomposed, scipy.sparse.csr_matrix):
                    
                    if data_set.number_of_values > \
                        maximum_number_of_values_for_dense_matrix:
                        
                        print(
                            "The number of values for {}".format(
                                title_with_ID),
                            "is too large to convert from sparse matrix",
                            "to dense array to decompose using {}.".format(
                                decomposition_method),
                            "Skipping."
                        )
                        print()
                        continue
                    
                    else:
                        print(
                            "Converting {} from sparse matrix".format(
                                title_with_ID),
                            "to dense array to decompose using {}".format(
                                decomposition_method)
                        )
                        conversion_time_start = time()
                        
                        values_decomposed = values_decomposed.A
                        
                        conversion_duration = time() - conversion_time_start
                        print("{} converted ({})".format(
                            title_with_ID.capitalize(),
                            formatDuration(conversion_duration)
                        ))
                
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
                        
                    if data_set.number_of_features > \
                        maximum_number_of_features_for_t_sne:
                        
                        print(
                            "The number of features for {}".format(
                                title_with_ID),
                            "is too large to decompose it",
                            "using {} in due time.".format(decomposition_method)
                        )
                        print("Decomposing {} to {} components using PCA".format(
                            title_with_ID, number_of_pca_components_before_tsne),
                            "beforehand.")
                        decompose_time_start = time()
                        
                        values_decomposed, other_values_decomposed, \
                            centroids_decomposed = decompose(
                                values_decomposed,
                                other_value_sets = other_values_decomposed,
                                centroids = centroids_decomposed,
                                method = "pca",
                                components = number_of_pca_components_before_tsne
                            )
                        
                        decompose_duration = time() - decompose_time_start
                        print("{} pre-decomposed ({}).".format(
                            title_with_ID.capitalize(),
                            formatDuration(decompose_duration)
                        ))
                
                print("Decomposing {} using {}.".format(
                    title_with_ID, decomposition_method))
                decompose_time_start = time()
                
                values_decomposed, other_values_decomposed, \
                    centroids_decomposed = decompose(
                        values_decomposed,
                        other_value_sets = other_values_decomposed,
                        centroids = centroids_decomposed,
                        method = decomposition_method,
                        components = 2
                    )
                
                decompose_duration = time() - decompose_time_start
                print("{} decomposed ({}).".format(
                    title_with_ID.capitalize(),
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
                saveFigure(figure, figure_name, decompositions_directory)
            
                plot_duration = time() - plot_time_start
                print("    {} plotted and saved ({}).".format(
                    title_with_ID.capitalize(),
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
                    saveFigure(figure, figure_name, decompositions_directory)
        
                    plot_duration = time() - plot_time_start
                    print("    {} (with labels) plotted and saved ({}).".format(
                        title_with_ID.capitalize(), formatDuration(plot_duration)))
                
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
                        saveFigure(figure, figure_name, decompositions_directory)
                    
                        plot_duration = time() - plot_time_start
                        print("    " +
                            "{} (with superset labels) plotted and saved ({})."\
                                .format(
                                    title_with_ID.capitalize(),
                                    formatDuration(plot_duration)
                            )
                        )
                    
                        plot_duration = time() - plot_time_start
                        print("    " +
                            "{} (with predicted labels) plotted and saved ({})."\
                                .format(
                                    title_with_ID.capitalize(),
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
                                saveFigure(figure, figure_name,
                                    decompositions_directory)
                        
                            plot_duration = time() - plot_time_start
                            print("    {} (for each class) plotted and saved ({})."\
                                .format(
                                    title_with_ID.capitalize(),
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
                                saveFigure(figure, figure_name,
                                    decompositions_directory)
                        
                            plot_duration = time() - plot_time_start
                            print("    " +
                                "{} (for each superset class) plotted and saved ({})."\
                                    .format(
                                        title_with_ID.capitalize(),
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
                        axis_limits = axis_limits,
                        example_tag = data_set.tags["example"],
                        name = plot_name,
                        **remaining_arguments
                    )
                    saveFigure(figure, figure_name, decompositions_directory)
                
                    plot_duration = time() - plot_time_start
                    print("    " +
                        "{} (with predicted cluster IDs) plotted and saved ({})."\
                            .format(
                                title_with_ID.capitalize(),
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
                        axis_limits = axis_limits,
                        example_tag = data_set.tags["example"],
                        name = plot_name,
                        **remaining_arguments
                    )
                    saveFigure(figure, figure_name, decompositions_directory)
                
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
                saveFigure(figure, figure_name, decompositions_directory)
        
                plot_duration = time() - plot_time_start
                print("    {} (with count sum) plotted and saved ({}).".format(
                    title_with_ID.capitalize(), formatDuration(plot_duration)))
                
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
                    saveFigure(figure, figure_name, decompositions_directory)
            
                    plot_duration = time() - plot_time_start
                    print("    {} (with {}) plotted and saved ({}).".format(
                        title_with_ID.capitalize(),
                        data_set.feature_names[feature_index],
                        formatDuration(plot_duration)
                    ))
                
                print()

def analyseCentroidProbabilities(centroids, name = None,
    analysis_level = "normal", for_video = False, results_directory = "results"):
    
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
    saveFigure(figure, figure_name, results_directory, for_video=for_video)
    
    plot_duration = time() - plot_time_start
    print("Centroid probabilities plotted and saved ({}).".format(
        formatDuration(plot_duration)))

def statistics(data_set, name = "", tolerance = 1e-3, skip_sparsity = False):
    
    x_mean = data_set.mean()
    x_std  = data_set.std(ddof = 1)
    x_min  = data_set.min()
    x_max  = data_set.max()
    
    x_dispersion = x_std**2 / x_mean
    
    if skip_sparsity:
        x_sparsity = nan
    else:
        if isinstance(data_set, scipy.sparse.csr_matrix):
            x_sparsity = 1 - (data_set >= tolerance).sum() \
                / numpy.prod(data_set.shape)
        else:
            x_sparsity = (data_set < tolerance).sum() / data_set.size
    
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

def formatStatistics(statistics_sets, name = "Data set"):
    
    if type(statistics_sets) != list:
        statistics_sets = [statistics_sets]
    
    name_width = len(name)
    
    for statistics_set in statistics_sets:
        name_width = max(len(statistics_set["name"]), name_width)
    
    statistics_string = "  ".join(["{:{}}".format(name, name_width),
        " mean ", "std. dev. ", "dispersion",
        " minimum ", " maximum ","sparsity"]) + "\n"
    
    for statistics_set in statistics_sets:
        string_parts = [
            "{:{}}".format(statistics_set["name"], name_width),
            "{:<9.5g}".format(statistics_set["mean"]),
            "{:<9.5g}".format(statistics_set["standard deviation"]),
            "{:<9.5g}".format(statistics_set["dispersion"]),
            "{:<11.5g}".format(statistics_set["minimum"]),
            "{:<11.5g}".format(statistics_set["maximum"]),
            "{:<7.5g}".format(statistics_set["sparsity"]),
        ]
        
        statistics_string += "  ".join(string_parts) + "\n"
    
    return statistics_string

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
                
                if isinstance(x, scipy.sparse.csr_matrix):
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
            
            if isinstance(x, scipy.sparse.csr_matrix):
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
    
    formatted_string = "Count   Accuracy\n"
    
    for k, f in count_accuracies.items():
        string_parts = [
            "{:^{}}".format(k, k_width),
            "{:6.2f}".format(100 * f)
        ]
        formatted_string += "    ".join(string_parts) + "\n"
    
    return formatted_string

decomposition_method_names = {
    "PCA": ["pca"],
    "SVD": ["svd"],
    "ICA": ["ica"],
    "t-SNE": ["t_sne", "tsne"], 
}

def decompose(values, other_value_sets = [], centroids = {}, method = "PCA",
    components = 2, random = False):
    
    method = normaliseString(method)
    
    if other_value_sets is not None \
        and not isinstance(other_value_sets, (list, tuple)):
        other_value_sets = [other_value_sets]
    
    if random:
        random_state = None
    else:
        random_state = 42
    
    if method == "pca":
        if values.shape[1] <= maximum_feature_size_for_analyses:
            model = PCA(n_components = components)
        else:
            model = IncrementalPCA(n_components = components, batch_size = 100)
    elif method == "svd":
        model = TruncatedSVD(n_components = components)
    elif method == "ica":
        model = FastICA(n_components = components)
    elif method == "t_sne":
        model = TSNE(n_components = components, random_state = random_state)
    else:
        raise ValueError("Method `{}` not found.".format(method))
    
    values_decomposed = model.fit_transform(values)
    
    if other_value_sets and method != "t_sne":
        other_value_sets_decomposed = []
        for other_values in other_value_sets:
            other_value_decomposed = model.transform(other_values)
            other_value_sets_decomposed.append(other_value_decomposed)
    else:
        other_value_sets_decomposed = None
    
    if other_value_sets_decomposed and len(other_value_sets_decomposed) == 1:
        other_value_sets_decomposed = other_value_sets_decomposed[0]
    
    # Only supports centroids without data sets as top levels
    if centroids and method == "pca":
        if "means" in centroids:
            centroids = {"unknown": centroids}
        W = model.components_
        centroids_decomposed = {}
        for distribution, distribution_centroids in centroids.items():
            if distribution_centroids:
                centroids_distribution_decomposed = {}
                for parameter, values in distribution_centroids.items():
                    if parameter == "means":
                        shape = numpy.array(values.shape)
                        L = shape[-1]
                        reshaped_values = values.reshape(-1, L)
                        decomposed_values = model.transform(reshaped_values)
                        shape[-1] = components
                        new_values = decomposed_values.reshape(shape)
                    elif parameter == "covariance_matrices":
                        shape = numpy.array(values.shape)
                        L = shape[-1]
                        reshaped_values = values.reshape(-1, L, L)
                        B = reshaped_values.shape[0]
                        decomposed_values = numpy.empty((B, 2, 2))
                        for i in range(B):
                            decomposed_values[i] = W @ reshaped_values[i] @ W.T
                        shape[-2:] = components
                        new_values = decomposed_values.reshape(shape)
                    else:
                        new_values = values
                    centroids_distribution_decomposed[parameter] = new_values
                centroids_decomposed[distribution] = \
                    centroids_distribution_decomposed
            else:
                centroids_decomposed[distribution] = None
        if "unknown" in centroids_decomposed:
            centroids_decomposed = centroids_decomposed["unknown"]
    else:
        centroids_decomposed = None
    
    if other_value_sets != [] and centroids != {}:
        return values_decomposed, other_value_sets_decomposed, \
            centroids_decomposed
    elif other_value_sets != []:
        return values_decomposed, other_value_sets_decomposed
    elif centroids != {}:
        return values_decomposed, centroids_decomposed
    else:
        return values_decomposed

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

# TODO Extend function to also work with separate zero count for sparse matrices
def plotHistogram(series, label = None, maximum_count = None,
    normed = False, discrete = False, x_scale = "linear", y_scale = "linear",
    colour = None, name = None):
    
    series = series.copy()
    
    figure_name = "histogram"
    
    if normed:
        figure_name += "-normed"
    
    figure_name = figureName(figure_name, name)
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    if maximum_count:
        maximum_count_indcises = series <= maximum_count
        number_of_outliers = series.size - maximum_count_indcises.sum()
        series = series[maximum_count_indcises]
    
    series_max = series.max()
    
    if discrete:
        if series_max < 1000:
            number_of_bins = int(numpy.ceil(series_max)) + 1
        else:
            number_of_bins = 1000 + 1
    else:
        number_of_bins = None
    
    if colour is None:
        colour = standard_palette[0]
    
    if x_scale == "log":
        series += 1
        label += " (shifted one)"
        figure_name += "-log_values"
    
    seaborn.distplot(series, bins = number_of_bins, norm_hist = normed,
        color = colour, kde = False, ax = axis)
    
    axis.set_xscale(x_scale)
    axis.set_yscale(y_scale)
    
    axis.set_xlabel(label.capitalize())
    
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

def plotCutOffCountHistogram(series, cutoff = None, normed = False,
    scale = "linear", colour = None, name = None):
    
    series = series.copy()
    
    figure_name = "histogram"
    
    if normed:
        figure_name += "-normed"
    
    figure_name += "-counts"
    figure_name = figureName(figure_name, name)
    figure_name += "-cutoff-{}".format(cutoff)
    
    if not colour:
        colour = standard_palette[0]
    
    k = numpy.arange(cutoff + 1)
    C = numpy.empty(cutoff + 1)
    
    for i in range(cutoff + 1):
        if k[i] < cutoff:
            c = (series == k[i]).sum()
        elif k[i] == cutoff:
            c = (series >= cutoff).sum()
        C[i] = c
    
    if normed:
        C /= C.sum()
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    axis.bar(k, C, color = colour, alpha = 0.4)
    
    axis.set_yscale(scale)
    
    axis.set_xlabel("Count bins")
    
    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")
    
    seaborn.despine()
    
    return figure, figure_name
    
def plotSeries(series, x_label, y_label, scale = "linear", bar = False,
    name = None):
    
    figure_name = figureName("series", name)
    
    D = series.shape[0]
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    x = linspace(0, D, D)
    
    if bar:
        axis.bar(x, series)
    else:
        axis.plot(x, series)
    
    axis.set_yscale(scale)
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
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
            if curve_name not in losses:
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
        
        if accuracies_kind == "training":
            line_style = "solid"
            colour = standard_palette[0]
        elif accuracies_kind == "validation":
            line_style = "dashed"
            colour = standard_palette[1]
        
        label = "{} set".format(accuracies_kind.capitalize())
        
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
    
    if isinstance(observed_series, scipy.sparse.csr_matrix):
        observed_series = observed_series.A[0]
    
    N = observed_series.shape[0]
    
    observed_colour = standard_palette[0]
    
    expected_palette = seaborn.light_palette(standard_palette[1], 5)
    
    expected_colour = expected_palette[-1]
    expected_total_standard_deviations_colour = expected_palette[1]
    expected_explained_standard_deviations_colour = expected_palette[3]
    
    if sort:
        x_label = "{}s sorted {} by {} {}s [sort index]".format(
            x_name.capitalize(), sort_direction, sort_by, y_name.lower())
    else:
        x_label = "{}s [original index]".format(x_name.capitalize())
    y_label = y_name.capitalize() + "s"
    
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
        
        axis_upper.set_yscale("log")
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
        
        axis.set_yscale(y_scale)
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

def plotHeatMap(values, x_name, y_name, z_name = None, z_symbol = None,
    z_min = None, z_max = None, labels = None, center = None,
    normalisation = None, normalisation_constants = None, name = None):
    
    figure_name = figureName("heat_map", name)
    
    M, N = values.shape
    
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
    
    aspect_ratio = M / N
    square_cells = 1/5 < aspect_ratio and aspect_ratio < 5
    
    if labels is not None:
        indices = numpy.argsort(labels)
        y_name += " sorted by class"
    else:
        indices = numpy.arange(M)
    
    seaborn.set(style = "white")
    
    seaborn.heatmap(
        values[indices],
        vmin = z_min, vmax = z_max, center = center, 
        xticklabels = False, yticklabels = False,
        cbar = True, cbar_kws = cbar_dict, cmap = standard_colour_map,
        square = square_cells, ax = axis
    )
    
    reset_plot_look()
    
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)
    
    return figure, figure_name

def plotValues(values, colour_coding = None, colouring_data_set = None,
    centroids = None, class_name = None, feature_index = None,
    figure_labels = None, axis_limits = None, example_tag = None,
    name = "scatter", **remaining_arguments):
    
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
            if "prediction_method" in remaining_arguments:
                figure_name += "-" + remaining_arguments["prediction_method"]
            if "number_of_classes" in remaining_arguments:
                figure_name += "_" + str(remaining_arguments["number_of_classes"])
        if colouring_data_set is None:
            raise ValueError("Colouring data set not given.")
    
    legend = None
    
    # Values
    
    values = values.copy()
    
    # Randomise examples in values to remove any prior order
    M, N = values.shape
    shuffled_indices = numpy.random.permutation(M)
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
    
    # Figure
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.despine()
    
    axis.set_aspect("equal", adjustable = "datalim")
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    colour_map = seaborn.dark_palette(standard_palette[0], as_cmap = True)
    
    if colour_coding and (
            "labels" in colour_coding or
            "ids" in colour_coding or
            "class" in colour_coding
        ):
        
        if "superset" in colour_coding:
            labels = colouring_data_set.superset_labels
            class_names = colouring_data_set.superset_class_names
            number_of_classes = colouring_data_set.number_of_superset_classes
            class_palette = colouring_data_set.superset_class_palette
            label_sorter = colouring_data_set.superset_label_sorter
        elif colour_coding == "predicted_cluster_ids":
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
                    axis.scatter(values[i, 0], values[i, 1], c = colour,
                        label = label)
            
            axis.scatter(values[:, 0], values[:, 1], c = colours)
            
            if number_of_classes < 20:
                handles, labels = axis.get_legend_handles_labels()
                if labels:
                    labels, handles = zip(*sorted(zip(labels, handles),
                        key = lambda t: label_sorter(t[0])))
                    legend = axis.legend(handles, labels, loc = "best")
        
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
                legend = axis.legend(handles, labels, loc = "best")
    
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
        if isinstance(f, scipy.sparse.csr_matrix):
            f = f.A
        f = f.flatten()
        
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
    
    numpy.random.seed(60)
    if not isinstance(indices, (list, tuple)):
        M = number_of_random_examples
        indices = numpy.random.permutation(data_set.number_of_examples)[:M]
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


    numpy.random.seed()
    
    W, H = data_set.feature_dimensions
    
    examples = data_set.values[indices]
    if isinstance(examples, scipy.sparse.csr_matrix):
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
    figure_name = base_name
    
    if other_names:
        if not isinstance(other_names, list):
            other_names = str(other_names)
            other_names = [other_names]
        else:
            other_names = [str(name) for name in other_names if name is not None]
        figure_name += "-" + "-".join(map(normaliseString, other_names))
    
    return figure_name

def saveFigure(figure, figure_name, results_directory, for_video=False):
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    figure_path = os.path.join(results_directory, figure_name) + figure_extension
    if for_video:
        bounding_box = None
        dpi = 'figure'
    else:
        bounding_box = 'tight'
        dpi = None

    figure.savefig(figure_path, bbox_inches = bounding_box, dpi = dpi)
    
    pyplot.close(figure)

def axisLabelForSymbol(symbol, coordinate = None, decomposition_method = None,
    distribution = None, prefix = "", suffix = ""):
    
    if decomposition_method:
        decomposition_method = normaliseString(decomposition_method)
    
    if decomposition_method == "pca":
        decomposition_label = "PC"
    elif decomposition_method == "ica":
        decomposition_label = "IC"
    elif decomposition_method == "t_sne":
        decomposition_label = "tSNE"
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

def betterModelExists(model):
    E_current = loadNumberOfEpochsTrained(model, best_model = False)
    E_best = loadNumberOfEpochsTrained(model, best_model = True)
    return E_best < E_current

def createLabelSorter(sorted_class_names = []):
    
    def labelSorter(label):
        label = str(label)
        
        K = len(sorted_class_names)
        
        if label in sorted_class_names:
            index = sorted_class_names.index(label)
            label = str(index) + "_" + label
        elif label == "Others":
            label = "ZZZ" + str(K) + "_" + label
        elif label.startswith("Unknown"):
            label = "ZZZ" + str(K + 1) + "_" + label
        elif label == "No class":
            label = "ZZZ" + str(K + 2) + "_" + label
        elif label == "Remaining":
            label = "ZZZ" +str(K + 3) + "_" + label
        
        return label
    
    return labelSorter

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
