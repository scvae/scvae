#!/usr/bin/env python3

import numpy
from numpy import nan

from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.manifold import TSNE

from tensorflow.tensorboard.backend.event_processing import event_multiplexer

from matplotlib import pyplot
import matplotlib.patches
import matplotlib.gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
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
    formatTime, formatDuration,
    normaliseString, properString,
    heading
)

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
number_of_random_examples = 100
number_of_profile_comparisons = 25

def analyseData(data_sets, decomposition_methods = ["PCA"],
    highlight_feature_indices = [], results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, "data")
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    # Metrics
    
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
    
    print(formatStatistics(data_set_statistics), end = "")
    
    print()
    
    print(formatStatistics(histogram_statistics, name = "Series"), end = "")
    
    print()
    
    # Loop over data sets
    
    for data_set in data_sets:
        
        heading("Plots for {} set".format(data_set.kind))
        
        # Examples for data set
        
        if data_set.example_type == "images":
            print("Saving image of {} random examples from {} set.".format(
                number_of_random_examples, data_set.kind))
            image_time_start = time()
            image, image_name = combineRandomImagesFromDataSet(
                data_set,
                number_of_random_examples,
                name = data_set.kind
            )
            saveImage(image, image_name, results_directory)
            image_duration = time() - image_time_start
            print("Image saved ({}).".format(formatDuration(image_duration)))
            print()
        
        # Distributions
        
        print("Plotting distribution for {} set.".format(data_set.kind))
        
        distribution_directory = os.path.join(results_directory, "histograms")
        
        ## Class distribution
        
        if data_set.number_of_classes and data_set.number_of_classes < 50:
            distribution_time_start = time()
            
            figure, figure_name = plotClassHistogram(
                labels = data_set.labels,
                class_names = data_set.class_names,
                class_palette = data_set.class_palette,
                normed = True,
                scale = "linear",
                name = data_set.kind
            )
            saveFigure(figure, figure_name, distribution_directory)
            
            distribution_duration = time() - distribution_time_start
            print("    Class distribution plotted and saved ({})."\
                .format(formatDuration(distribution_duration)))
            
            if data_set.label_superset:
                distribution_time_start = time()
                
                figure, figure_name = plotClassHistogram(
                    labels = data_set.superset_labels,
                    class_names = data_set.superset_class_names,
                    class_palette = data_set.superset_class_palette,
                    normed = True,
                    scale = "linear",
                    name = [data_set.kind, "superset"]
                )
                saveFigure(figure, figure_name, distribution_directory)
                
                distribution_duration = time() - distribution_time_start
                print("    Superset class distribution plotted and saved ({})."\
                    .format(formatDuration(distribution_duration)))
                
        ## Count distribution
        
        distribution_time_start = time()
        
        figure, figure_name = plotHistogram(
            series = data_set.values.reshape(-1),
            title = "Counts",
            discrete = data_set.discreteness,
            normed = True,
            scale = "log",
            name = data_set.kind
        )
        saveFigure(figure, figure_name, distribution_directory)
        
        distribution_duration = time() - distribution_time_start
        print("    Count distribution plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
            
        ## Count distribution with cut-off
        
        if data_set.example_type == "counts":
            distribution_time_start = time()
            
            for cutoff in range(1, 10):
                figure, figure_name = plotHistogram(
                    series = data_set.values.reshape(-1),
                    title = "Counts",
                    discrete = data_set.discreteness,
                    normed = True,
                    cutoff = cutoff,
                    scale = "log",
                    name = data_set.kind
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
            title = "Count sum",
            normed = True,
            scale = "log",
            name = data_set.kind
        )
        saveFigure(figure, figure_name, distribution_directory)
        
        distribution_duration = time() - distribution_time_start
        print("    Count sum distribution plotted and saved ({})."\
            .format(formatDuration(distribution_duration)))
        
        print()
        
        # Heat map for data set
        
        if data_set.number_of_values <= maximum_number_of_values_for_heat_maps:
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
                transformation = data_set.heat_map_transformation,
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
        
        analyseDecompositions(
            data_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            symbol = "x",
            title = "original space",
            specifier = lambda data_set: data_set.kind,
            results_directory = results_directory
        )

def analyseModel(model, results_directory = "results"):
    
    # Setup
    
    if model.type in ["VAE", "IWVAE", "CVAE", "GMVAE", "GMVAE_alt"]:
        model_name = model.testing_name
    else:
        model_name = model.name
    
    results_directory = os.path.join(results_directory, model_name)
    
    # Learning curves
    
    heading("Learning curves")
    
    print("Plotting learning curves.")
    learning_curves_time_start = time()
    
    learning_curves = loadLearningCurves(model,
        data_set_kinds = ["training", "validation"])
    
    figure, figure_name = plotLearningCurves(learning_curves, model.type)
    saveFigure(figure, figure_name, results_directory)
    
    learning_curves_duration = time() - learning_curves_time_start
    print("Learning curves plotted and saved ({}).".format(
        formatDuration(learning_curves_duration)))
    
    print()
    
    # Accuracies
    
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
    
    if "AE" in model.type and model.type not in ["CVAE", "GMVAE", "GMVAE_alt"]:
        
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
    
    # Evolution of latent centroids
    
    if "AE" in model.type and "mixture" in model.latent_distribution_name:
        
        heading("Latent distributions")
    
        centroids = loadCentroids(model, data_set_kinds = "validation")
        
        centroids_directory = os.path.join(results_directory, "centroids_evolution")
        
        for distribution, distribution_centroids in centroids.items():
            
            if distribution_centroids:
                
                print("Plotting evolution of latent {} parameters.".format(
                    distribution))
                centroids_time_start = time()
                
                centroid_probabilities = distribution_centroids["probabilities"]
                centroid_means = distribution_centroids["means"]
                centroid_covariance_matrices = \
                    distribution_centroids["covariance_matrices"]
                
                E, K, L = centroid_means.shape
                
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

def analyseAllModels(models_summaries, results_directory = "results"):
    
    # Learning curves
    
    print("Plotting learning curves for all run models.")
    
    figure, figure_name = plotLearningCurvesForModels(models_summaries)
    saveFigure(figure, figure_name, results_directory)
    
    print()
    
    # Learning curves
    
    print("Plotting evaluation for all run models.")
    
    figure, figure_name = plotEvaluationsForModels(models_summaries)
    saveFigure(figure, figure_name, results_directory)
    
    print()

def analyseIntermediateResults(learning_curves = None, epoch_start = None,
    epoch = None, latent_values = None, data_set = None, centroids = None,
    model_name = None, model_type = None, results_directory = "results"):
    
    results_directory = os.path.join(results_directory, model_name)
    intermediate_directory = os.path.join(results_directory, "intermediate")
    
    # Learning curves
    
    print("Plotting learning curves.")
    learning_curves_time_start = time()
    
    figure, figure_name = plotLearningCurves(learning_curves, model_type,
        epoch_offset = epoch_start)
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
            saveFigure(figure, figure_name, intermediate_directory)
            if data_set.label_superset is not None:
                figure, figure_name = plotValues(
                    latent_values_decomposed,
                    colour_coding = "superset labels",
                    colouring_data_set = data_set,
                    centroids = centroids_decomposed,
                    figure_labels = figure_labels,
                    name = name
                )
                saveFigure(figure, figure_name, intermediate_directory)
        else:
            figure, figure_name = plotValues(
                latent_values_decomposed,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
            )
            saveFigure(figure, figure_name, intermediate_directory)
    
        if centroids:
            analyseCentroidProbabilities(centroids, epoch_name, intermediate_directory)
    
        plot_duration = time() - plot_time_start
        print("{} plotted and saved ({}).".format(
            latent_set_name.capitalize(), formatDuration(plot_duration)))

def analyseResults(test_set, reconstructed_test_set, latent_test_sets, model,
    decomposition_methods = ["PCA"], highlight_feature_indices = [],
    results_directory = "results"):
    
    # Setup
    
    if model.type in ["VAE", "IWVAE", "CVAE", "GMVAE", "GMVAE_alt"]:
        model_name = model.testing_name
    else:
        model_name = model.name
    
    results_directory = os.path.join(results_directory, model_name)
    
    M = test_set.number_of_examples
    
    # Loading
    
    evaluation_test = loadLearningCurves(model, "test")
    accuracy_test = loadAccuracies(model, "test")
    superset_accuracy_test = loadAccuracies(model, "test", superset = True)
    number_of_epochs_trained = loadNumberOfEpochsTrained(model)
    
    # Metrics

    heading("Metrics")
    
    print("Calculating metrics for results.")
    metrics_time_start = time()

    test_set_statistics = [
        statistics(data_set.values, data_set.version, tolerance = 0.5)
            for data_set in [test_set, reconstructed_test_set]
    ]

    x_diff = reconstructed_test_set.values - test_set.values
    test_set_statistics.append(statistics(numpy.abs(x_diff), "differences",
        skip_sparsity = True))

    x_log_ratio = numpy.log((reconstructed_test_set.values + 1) \
        / (test_set.values + 1))
    test_set_statistics.append(statistics(numpy.abs(x_log_ratio), "log-ratios",
        skip_sparsity = True))

    if test_set.values.max() > 20:
        count_accuracy_method = "orders of magnitude"
    else:
        count_accuracy_method = None

    count_accuracies = computeCountAccuracies(
        test_set.values,
        reconstructed_test_set.values,
        method = count_accuracy_method
    )

    metrics_duration = time() - metrics_time_start
    print("Metrics calculated ({}).".format(
        formatDuration(metrics_duration)))

    ## Saving

    metrics_log_path = os.path.join(results_directory, "test_metrics.log")
    metrics_dictionary_path = os.path.join(results_directory,
        "test_metrics.pkl.gz")

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
                    evaluation_test["log_likelihood"][-1])
        elif "AE" in model.type:
            metrics_string += \
                "    ELBO: {:.5g}.\n".format(
                    evaluation_test["lower_bound"][-1]) + \
                "    ENRE: {:.5g}.\n".format(
                    evaluation_test["reconstruction_error"][-1])
            if model.type not in ["CVAE", "GMVAE", "GMVAE_alt"]:
                metrics_string += \
                    "    KL: {:.5g}.\n".format(
                        evaluation_test["kl_divergence"][-1])
            elif model.type in ["GMVAE", "GMVAE_alt"]:
                metrics_string += \
                    "    KL_z: {:.5g}.\n".format(
                        evaluation_test["kl_divergence_z"][-1]) + \
                    "    KL_y: {:.5g}.\n".format(
                        evaluation_test["kl_divergence_y"][-1])
            elif model.type == "CVAE":
                metrics_string += \
                    "    KL_z1: {:.5g}.\n".format(
                        evaluation_test["kl_divergence_z1"][-1]) + \
                    "    KL_z2: {:.5g}.\n".format(
                        evaluation_test["kl_divergence_z2"][-1]) + \
                    "    KL_y: {:.5g}.\n".format(
                        evaluation_test["kl_divergence_y"][-1])
        if accuracy_test is not None:
            metrics_string += "    Accuracy: {:6.2f} %.\n".format(
                100 * accuracy_test[-1])
        if superset_accuracy_test is not None:
            metrics_string += "    Accuracy (superset): {:6.2f} %.\n".format(
                100 * superset_accuracy_test[-1])
        metrics_string += "\n" + formatStatistics(test_set_statistics)
        metrics_string += "\n" + formatCountAccuracies(count_accuracies)
        metrics_file.write(metrics_string)

    with gzip.open(metrics_dictionary_path, "w") as metrics_file:
        metrics_dictionary = {
            "timestamp": metrics_saving_time_start,
            "number of epochs trained": number_of_epochs_trained,
            "evaluation": evaluation_test,
            "statistics": statistics,
            "count accuracies": count_accuracies
        }
        pickle.dump(metrics_dictionary, metrics_file)
    
    metrics_saving_duration = time() - metrics_saving_time_start
    print("Metrics saved ({}).".format(formatDuration(
        metrics_saving_duration)))
        
    print()
    
    ## Displaying
    
    print(formatStatistics(test_set_statistics))
    print(formatCountAccuracies(count_accuracies))
    
    # Reconstructions
    
    heading("Reconstructions")
    
    ## Examples
    
    if reconstructed_test_set.example_type == "images":
        print("Saving image of {} random examples".format(
            number_of_random_examples), "from reconstructed test set.")
        image_time_start = time()
        image, image_name = combineRandomImagesFromDataSet(
            reconstructed_test_set,
            number_of_random_examples,
            name = reconstructed_test_set.version
        )
        saveImage(image, image_name, results_directory)
        image_duration = time() - image_time_start
        print("Image saved ({}).".format(formatDuration(image_duration)))
        print()

    ## Profile comparisons
    
    numpy.random.seed(80)
    
    print("Plotting profile comparisons.")
    
    profile_comparisons_directory = os.path.join(
            results_directory, "profile_comparisons")
    
    profile_comparisons_time_start = time()

    subset = numpy.random.randint(M, size = number_of_profile_comparisons)

    for i in subset:
        
        figure, figure_name = plotProfileComparison(
            test_set.values[i],
            reconstructed_test_set.values[i],
            x_name = test_set.tags["feature"].capitalize() + "s",
            y_name = test_set.tags["value"].capitalize() + "s",
            scale = "log",
            name = str(test_set.example_names[i])
        )
        saveFigure(figure, figure_name, profile_comparisons_directory)

    profile_comparisons_duration = time() - profile_comparisons_time_start
    print("Profile comparisons plotted and saved ({}).".format(
        formatDuration(profile_comparisons_duration)))

    print()
    
    numpy.random.seed()
    
    ## Reconstructions decomposed

    analyseDecompositions(
        reconstructed_test_set,
        colouring_data_set = test_set,
        decomposition_methods = decomposition_methods,
        highlight_feature_indices = highlight_feature_indices,
        symbol = "\\tilde{{x}}",
        title = "reconstruction space",
        results_directory = results_directory
    )
    
    ## Reconstructions plotted in original decomposed space
    
    analyseDecompositions(
        test_set,
        reconstructed_test_set,
        colouring_data_set = test_set,
        decomposition_methods = decomposition_methods,
        highlight_feature_indices = highlight_feature_indices,
        symbol = "x",
        title = "original space",
        results_directory = results_directory
    )
    
    if model.type == "GMVAE_alt":
        
        ## Reconstructions decomposed with predicted labels
        
        analyseDecompositions(
            test_set,
            colouring_data_set = reconstructed_test_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            symbol = "x",
            title = "original space with predicted labels",
            results_directory = results_directory
        )
        
        ## Reconstructions decomposed with predicted labels
        
        analyseDecompositions(
            reconstructed_test_set,
            colouring_data_set = reconstructed_test_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            symbol = "\\tilde{{x}}",
            title = "reconstruction space with predicted labels",
            results_directory = results_directory
        )
        
        ## Reconstructions plotted in original decomposed space
        ## with predicted labels
        
        analyseDecompositions(
            test_set,
            reconstructed_test_set,
            colouring_data_set = reconstructed_test_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            symbol = "x",
            title = "original space with predicted labels",
            results_directory = results_directory
        )

    # Heat maps
    
    if reconstructed_test_set.number_of_values \
        <= maximum_number_of_values_for_heat_maps:
        
        print("Plotting heat maps.")
        
        heat_maps_directory = os.path.join(results_directory, "heat_maps")
        
        ## Reconstructions
        
        heat_maps_time_start = time()
        
        figure, figure_name = plotHeatMap(
            reconstructed_test_set.values,
            labels = reconstructed_test_set.labels,
            transformation = reconstructed_test_set.heat_map_transformation,
            x_name = test_set.tags["feature"].capitalize() + "s",
            y_name = test_set.tags["example"].capitalize() + "s",
            z_name = test_set.tags["value"].capitalize() + "s",
            z_symbol = "\\tilde{{x}}",
            name = "reconstruction"
        )
        saveFigure(figure, figure_name, heat_maps_directory)
        
        heat_maps_duration = time() - heat_maps_time_start
        print("    Reconstruction heat map plotted and saved ({})." \
            .format(formatDuration(heat_maps_duration)))
        
        ## Differences
        
        heat_maps_time_start = time()
        
        figure, figure_name = plotHeatMap(
            x_diff,
            labels = reconstructed_test_set.labels,
            x_name = test_set.tags["feature"].capitalize() + "s",
            y_name = test_set.tags["example"].capitalize() + "s",
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
        
        heat_maps_time_start = time()
        
        figure, figure_name = plotHeatMap(
            x_log_ratio,
            labels = reconstructed_test_set.labels,
            x_name = test_set.tags["feature"].capitalize() + "s",
            y_name = test_set.tags["example"].capitalize() + "s",
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
    
    if latent_test_sets is not None:
        
        heading("Latent space")
        
        if "AE" in model.type:
            centroids = loadCentroids(model, data_set_kinds = "test")
        else:
            centroids = None
        
        analyseDecompositions(
            latent_test_sets,
            centroids = centroids,
            colouring_data_set = test_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            title = "latent space",
            specifier = lambda data_set: data_set.version,
            results_directory = results_directory
        )
        
        if model.type == "GMVAE_alt":
            analyseDecompositions(
                latent_test_sets,
                centroids = centroids,
                colouring_data_set = reconstructed_test_set,
                decomposition_methods = decomposition_methods,
                highlight_feature_indices = highlight_feature_indices,
                title = "latent space with predicted labels",
                specifier = lambda data_set: data_set.version,
                results_directory = results_directory
            )
        
        if centroids:
            analyseCentroidProbabilities(centroids,
                results_directory = results_directory)

def analyseDecompositions(data_sets, other_data_sets = [], centroids = None,
    colouring_data_set = None, decomposition_methods = ["PCA"],
    highlight_feature_indices = [], symbol = None,
    title = "data set", specifier = None,
    results_directory = "results"):
    
    centroids_original = centroids
    
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
        
        name = normaliseString(title)
        
        if specifier:
            ID = specifier(data_set)
        
        if ID:
            name += "-" + str(ID)
            title_with_ID = title + " for " + ID
        else:
            title_with_ID = title
        
        if not colouring_data_set:
            colouring_data_set = data_set
        
        if data_set.version in ["z", "z1"]:
            centroids = copy.deepcopy(centroids_original)
        else:
            centroids = None
        
        if other_data_set:
            title_with_ID = "{} values in {}".format(
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
                    centroids_decomposed = centroids
                    other_values_decomposed = other_values
                else:
                    continue
            else:    
                decomposition_method = properString(decomposition_method,
                    decomposition_method_names)
                
                print("Decomposing {} using {}.".format(
                    title_with_ID, decomposition_method))
                decompose_time_start = time()
                
                values_decomposed, other_values_decomposed, \
                    centroids_decomposed = decompose(
                        data_set.values,
                        other_value_sets = other_values,
                        centroids = centroids,
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
            
            # Plot data set
            
            print("Plotting {}{}.".format(
                "decomposed " if decomposition_method else "", title_with_ID))
            
            ## No colour-coding
            
            plot_time_start = time()
            
            figure, figure_name = plotValues(
                plot_values_decomposed,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
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
                    name = name
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
                        name = name
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
                            name = name
                        )
                        saveFigure(figure, figure_name, decompositions_directory)
                    
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
                            name = name
                        )
                        saveFigure(figure, figure_name, decompositions_directory)
                    
                    plot_duration = time() - plot_time_start
                    print("    " +
                        "{} (for each superset class) plotted and saved ({})."\
                            .format(
                                title_with_ID.capitalize(),
                                formatDuration(plot_duration)
                    ))
        
            # Count sum
        
            plot_time_start = time()
        
            figure, figure_name = plotValues(
                plot_values_decomposed,
                colour_coding = "count sum",
                colouring_data_set = colouring_data_set,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
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
                    name = name
                )
                saveFigure(figure, figure_name, decompositions_directory)
            
                plot_duration = time() - plot_time_start
                print("   {} (with {}) plotted and saved ({}).".format(
                    title_with_ID.capitalize(),
                    data_set.feature_names[feature_index],
                    formatDuration(plot_duration)
                ))
            
            print()

def analyseCentroidProbabilities(centroids, name = None,
    results_directory = "results"):
    
    print("Plotting centroid probabilities.")
    plot_time_start = time()
    
    if name:
        name = normaliseString(name)
    
    for distribution, distribution_centroids in centroids.items():
        if distribution_centroids:
            centroid_probabilities = \
                distribution_centroids["probabilities"]
            K = len(centroid_probabilities)
            centroids_palette = darker_palette(K)
            x_label = "$k$"
            y_label = axisLabelForSymbol(
                symbol = "\\pi",
                distribution = normaliseString(distribution),
                suffix = "^k"
            )
            if name:
                plot_name = [name, distribution]
            else:
                plot_name = distribution
            figure, figure_name = plotProbabilities(
                centroid_probabilities,
                x_label = x_label,
                y_label = y_label,
                palette = centroids_palette,
                uniform = False,
                name = plot_name
            )
            saveFigure(figure, figure_name, results_directory)
    
    plot_duration = time() - plot_time_start
    print("Centroid probabilities plotted and saved ({}.)".format(
        formatDuration(plot_duration)))

def statistics(data_set, name = "", tolerance = 1e-3, skip_sparsity = False):
    
    x_mean = data_set.mean()
    x_std  = data_set.std()
    x_min  = data_set.min()
    x_max  = data_set.max()
    
    x_dispersion = x_mean / x_std
    
    if skip_sparsity:
        x_sparsity = nan
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
            "{:<9.3g}".format(statistics_set["minimum"]),
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
    
    # Round data sets to be able to compare
    x = x.round()
    x_tilde = x_tilde.round()
    
    # Compute the max count value
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
            
                k_indices = x_scaled_floored == k
            
                k_sum = (x_tilde_scaled_floored[k_indices] == k).sum()
                k_size = k_indices.sum()
            
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
            
            k_indices = x == k
            
            k_sum = (x_tilde[k_indices] == k).sum()
            k_size = k_indices.sum()
            
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
    elif method == "ica":
        model = FastICA(n_components = components)
    elif method == "t_sne":
        model = TSNE(n_components = components, random_state = random_state)
    else:
        raise ValueError("method method not found.")
    
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
    if centroids and method != "t_sne":
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
                        shape[-1] = 2
                        new_values = decomposed_values.reshape(shape)
                    elif parameter == "covariance_matrices":
                        shape = numpy.array(values.shape)
                        L = shape[-1]
                        reshaped_values = values.reshape(-1, L, L)
                        B = reshaped_values.shape[0]
                        decomposed_values = numpy.empty((B, 2, 2))
                        for i in range(B):
                            decomposed_values[i] = W @ reshaped_values[i] @ W.T
                        shape[-2:] = 2
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
    normed = False, scale = "linear", name = None):
    
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
    
    if not class_palette:
        index_palette = lighter_palette(K)
        class_palette = {class_name: index_palette[i] for i, class_name in
                         enumerate(sorted(class_names))}
    
    histogram = {
        class_name: {
            "index": i,
            "count": 0,
            "colour": class_palette[class_name]
        }
        for i, class_name in enumerate(sorted(class_names))
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
    
    pyplot.xticks(indices, class_names)
    
    axis.set_xlabel("Classes")
    
    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")
    
    seaborn.despine()
    
    return figure, figure_name

def plotHistogram(series, title, cutoff = None, normed = False,
    discrete = False, scale = "linear", name = None):
    
    series = series.copy()
    
    figure_name = "histogram"
    
    if normed:
        figure_name += "-normed"
    
    figure_name += "-" + normaliseString(title)
    
    figure_name = figureName(figure_name, name)
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    if cutoff:
        figure_name += "-cutoff-{}".format(cutoff)
        clip_indices = series > cutoff
        series[clip_indices] = cutoff
    
    series_max = series.max()
    
    if discrete and series_max < 1000:
        number_of_bins = int(numpy.ceil(series_max)) + 1
    else:
        number_of_bins = None
    
    seaborn.distplot(series, bins = number_of_bins, norm_hist = normed,
        kde = False, ax = axis)
    
    axis.set_yscale(scale)
    
    axis.set_xlabel(title)
    
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

def plotLearningCurves(curves, model_type, epoch_offset = 0, name = None):
    
    figure_name = figureName("learning_curves", name)
    
    if model_type == "SNN":
        figure = pyplot.figure()
        axis_1 = figure.add_subplot(1, 1, 1)
    elif model_type in ["CVAE", "GMVAE", "GMVAE_alt"]:
        figure, (axis_1, axis_2, axis_3) = pyplot.subplots(3, sharex = True,
            figsize = (6.4, 14.4))
    elif "AE" in model_type:
        figure, (axis_1, axis_2) = pyplot.subplots(2, sharex = True,
            figsize = (6.4, 9.6))
    
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
                    latent_variable = re.sub(r"(\w)(\d)", r"\1_\2", latent_variable)
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
                curve_name = "KL" + index + "$(p||q)$"
            elif curve_name == "log_likelihood":
                curve_name = "$L$"
                axis = axis_1
            epochs = numpy.arange(len(curve)) + 1 + epoch_offset
            label = curve_name + " ({} set)".format(curve_set_name)
            axis.plot(epochs, curve, color = colour, linestyle = line_style,
                label = label)
    
    handles, labels = axis_1.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    
    axis_1.legend(handles, labels, loc = "best")
    
    if model_type == "SNN":
        axis_1.set_xlabel("Epoch")
    elif "AE" in model_type:
        handles, labels = axis_2.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        axis_2.legend(handles, labels, loc = "best")
        if model_type in ["CVAE", "GMVAE", "GMVAE_alt"]:
            axis_3.legend(loc = "best")
            handles, labels = axis_3.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            axis_3.legend(handles, labels, loc = "best")
            axis_3.set_xlabel("Epoch")
        else:
            axis_2.set_xlabel("Epoch")
    
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
        
        label = "{} set".format(accuracies_kind)
        
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

def plotEvolutionOfCentroidProbabilities(probabilities, distribution,
    name = None):
    
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
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    for k in range(K):
        axis.plot(epochs, probabilities[:, k], color = centroids_palette[k],
            label = "$k = {}$".format(k))
    
    axis.set_xlabel("Epochs")
    axis.set_ylabel(y_label)
    
    axis.legend(loc = "best")
    
    seaborn.despine()
    
    return figure, figure_name

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

def plotLearningCurvesForModels(models_summaries, name = None):
    
    figure_name = figureName("learning_curves", name)
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    for model_name, model_summary in models_summaries.items():
        if model_summary["type"] == "SNN":
            curve = model_summary["learning curves"]["validation"]["log_likelihood"]
        elif "AE" in model_summary["type"]:
            curve = model_summary["learning curves"]["validation"]["lower_bound"]
        epochs = numpy.arange(len(curve)) + 1
        label = model_summary["description"]
        axis.plot(epochs, curve, label = label)
    
    axis.legend(loc = "best")
    
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Lower bound for validation set")
    
    seaborn.despine()
    
    return figure, figure_name

def plotEvaluationsForModels(models_summaries, name = None):
    
    figure_name = figureName("evaluation", name)
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    log_likelihoods = []
    model_descriptions = []
    
    for model_description, model_summary in models_summaries.items():
        if model_summary["type"] == "SNN":
            log_likelihood = model_summary["test evaluation"]["log-likelihood"]
        elif "AE" in model_summary["type"]:
            log_likelihood = model_summary["test evaluation"]["ELBO"]
        log_likelihoods.append(log_likelihood)
        model_description = model_summary["description"]
        model_descriptions.append(model_description)
    
    model_indices = numpy.arange(len(models_summaries)) + 1
    axis.bar(model_indices, log_likelihoods)
    
    axis.set_xticks(model_indices)
    axis.set_xticklabels(model_descriptions, rotation = 45, ha = "right")
    
    axis.set_xlabel("Models")
    axis.set_ylabel("log-likelhood for validation set")
    
    seaborn.despine()
    
    return figure, figure_name

def plotProfileComparison(original_series, reconstructed_series, 
    x_name, y_name, scale = "linear", name = None):
    
    figure_name = figureName("profile_comparison", name)
    
    D = original_series.shape[0]
    
    sort_indices = numpy.argsort(original_series)[::-1]
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    index = numpy.linspace(0, D, D)
    axis.plot(index, original_series[sort_indices],
        color = standard_palette[0], label = 'Original', zorder = 1)
    axis.scatter(index, reconstructed_series[sort_indices],
        color = standard_palette[1], label = 'Reconstruction', zorder = 0)
    
    axis.legend()
    
    axis.set_xscale(scale)
    axis.set_yscale(scale)
    
    axis.set_xlabel(x_name + " sorted by original " + y_name.lower())
    axis.set_ylabel(y_name)
    
    seaborn.despine()
    
    return figure, figure_name

def plotHeatMap(values, x_name, y_name, z_name = None, z_symbol = None,
    z_min = None, z_max = None, labels = None, center = None,
    transformation = None, name = None):
    
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
    
    if transformation:
        values = transformation["function"](values)
        if z_symbol:
            z_name = transformation["label"](z_symbol)
        elif z_name:
            z_name = "Transformed " + z_name.lower()
    
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
    figure_labels = None, name = "scatter"):
    
    # Randomise examples in values to remove any prior order
    M, N = values.shape
    shuffled_indices = numpy.random.permutation(M)
    values = values[shuffled_indices]
    
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
        if colouring_data_set is None:
            raise ValueError("Colouring data set not given.")
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.despine()
    
    axis.set_aspect("equal", adjustable = "datalim")
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    colour_map = seaborn.dark_palette(standard_palette[0], as_cmap = True)
    
    if colour_coding and ("labels" in colour_coding or "class" in colour_coding):
        
        if "superset" in colour_coding:
            labels = colouring_data_set.superset_labels
            class_names = colouring_data_set.superset_class_names
            class_palette = colouring_data_set.superset_class_palette
            number_of_classes = colouring_data_set.number_of_superset_classes
        else:
            labels = colouring_data_set.labels
            class_names = colouring_data_set.class_names
            class_palette = colouring_data_set.class_palette
            number_of_classes = colouring_data_set.number_of_classes
        
        if not class_palette:
            index_palette = lighter_palette(number_of_classes)
            class_palette = {class_name: index_palette[i] for i, class_name in
                             enumerate(sorted(class_names))}
        
        # Examples are shuffled, so should their labels be
        labels = labels[shuffled_indices]
        
        if "labels" in colour_coding:
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
                labels, handles = zip(*sorted(zip(labels, handles),
                    key = lambda t: classLabelSortKey(t[0])))
                axis.legend(handles, labels, loc = "best")
        
        elif "class" in colour_coding:
            colours = []
            
            figure_name += "-" + normaliseString(str(class_name))
            
            ordered_indices_set = {
                str(class_name): [],
                "Others": []
            }
            
            for i, label in enumerate(labels):
                if label == class_name:
                    colour = class_palette[label]
                    ordered_indices_set[str(class_name)].append(i)
                else:
                    colour = neutral_colour
                    ordered_indices_set["Others"].append(i)
                colours.append(colour)
            
            colours = numpy.array(colours)
            
            z_order_index = 1
            for label, ordered_indices in sorted(ordered_indices_set.items()):
                if label == "Others":
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
                    key = lambda t: classLabelSortKey(t[0])))
                axis.legend(handles, labels, loc = "best")
    
    elif colour_coding == "count_sum":
        
        index_method = "shuffled"
        n = colouring_data_set.count_sum[shuffled_indices]
        scatter_plot = axis.scatter(values[:, 0], values[:, 1], c = n,
            cmap = colour_map)
        colour_bar = figure.colorbar(scatter_plot)
        colour_bar.outline.set_linewidth(0)
        colour_bar.set_label("Count sum")
    
    elif colour_coding == "feature":
        
        if feature_index is None:
            raise ValueError("Feature number not given.")
        
        if feature_index > colouring_data_set.number_of_features:
            raise ValueError("Feature number higher than number of features.")
        
        feature_name = colouring_data_set.feature_names[feature_index]
        figure_name += "-{}".format(normaliseString(feature_name))
        
        f = colouring_data_set.values[shuffled_indices,
            feature_index].reshape(-1, 1)
        
        scatter_plot = axis.scatter(values[:, 0], values[:, 1], c = f,
            cmap = colour_map)
        colour_bar = figure.colorbar(scatter_plot)
        colour_bar.outline.set_linewidth(0)
        colour_bar.set_label(feature_name)
    
    else:
        axis.scatter(values[:, 0], values[:, 1], color = standard_palette[0])
    
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
                    color = centroids_palette[k])
                ellipse = covarianceMatrixAsEllipse(
                    covariance_matrices[k],
                    means[k],
                    colour = centroids_palette[k]
                )
                axis.add_patch(ellipse)
    
    return figure, figure_name

def plotProbabilities(probabilities, x_label = None, y_label = None, 
    palette = None, uniform = False, name = None):
    
    figure_name = figureName("probabilities", name)
    
    if not x_label:
        x_label = "$k$"
    
    if not y_label:
        y_label = "$\pi_k$"
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    K = len(probabilities)
    
    if not palette:
        palette = [standard_palette[0]] * K
    
    for k in range(K):
        axis.bar(k, probabilities[k], color = palette[k])
    
    if uniform:
        axis.plot([0, K], 2 * [1 / K], color = neutral_colour,
            linestyle = "dashed")
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
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
    
    ellipse = matplotlib.patches.Ellipse(
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
    
    return ellipse

def combineRandomImagesFromDataSet(data_set, number_of_random_examples, name = None):
    
    image_name = figureName("random_examples", name)
    
    M = number_of_random_examples
    
    numpy.random.seed(60)
    indices = numpy.random.permutation(data_set.number_of_examples)[:M]
    numpy.random.seed()
    
    W, H = data_set.feature_dimensions
    examples = data_set.values[indices].reshape(M, W, H)
    
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
            other_names = [other_names]
        figure_name += "-" + "-".join(map(normaliseString, other_names))
    
    return figure_name

def saveFigure(figure, figure_name, results_directory):
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    figure_path = os.path.join(results_directory, figure_name) + figure_extension
    figure.savefig(figure_path, bbox_inches = 'tight')
    
    pyplot.close(figure)

def loadNumberOfEpochsTrained(model):
    
    # Seup
    
    ## Data set kind
    data_set_kind = "training"
    
    ## Loss depending on model type
    if model.type == "SNN":
        loss = "log_likelihood"
    elif "AE" in model.type:
        loss = "lower_bound"
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        model.log_directory)
    multiplexer.Reload()
    
    # Loading
    
    # Losses for every epochs
    scalars = multiplexer.Scalars(data_set_kind, "losses/" + loss)
    
    # First estimate of number of epochs
    E_1 = len(scalars)
    
    # Second estimate of number of epochs
    E_2 = 0
    
    for scalar in scalars:
        E_2 = max(E_2, scalar.step)
    
    assert E_1 == E_2
    
    return E_1

def loadLearningCurves(model, data_set_kinds = "all"):
    
    # Setup
    
    learning_curve_sets = {}
    
    ## Data set kinds
    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "test"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]
    
    ## Losses depending on model type
    if model.type == "SNN":
        losses = ["log_likelihood"]
    elif model.type == "CVAE":
        losses = ["lower_bound", "reconstruction_error",
            "kl_divergence_z1", "kl_divergence_z2", "kl_divergence_y"]
    elif model.type in ["GMVAE", "GMVAE_alt"]:
        losses = ["lower_bound", "reconstruction_error",
            "kl_divergence_z", "kl_divergence_y"]
    elif "AE" in model.type:
        losses = ["lower_bound", "reconstruction_error", "kl_divergence"]
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        model.log_directory)
    multiplexer.Reload()
    
    # Loading
    
    for data_set_kind in data_set_kinds:
        
        learning_curve_set = {}
        
        for loss in losses:
            
            scalars = multiplexer.Scalars(data_set_kind, "losses/" + loss)
            
            learning_curve = numpy.empty(len(scalars))
            
            if len(scalars) == 1:
                learning_curve[0] = scalars[0].value
            else:
                for scalar in scalars:
                    learning_curve[scalar.step - 1] = scalar.value
            
            learning_curve_set[loss] = learning_curve
        
        learning_curve_sets[data_set_kind] = learning_curve_set
    
    if len(data_set_kinds) == 1:
        learning_curve_sets = learning_curve_sets[data_set_kinds[0]]
    
    return learning_curve_sets

def loadAccuracies(model, data_set_kinds = "all", superset = False):
    
    # Setup
    
    accuracies = {}
    
    ## Data set kinds
    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "test"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        model.log_directory)
    multiplexer.Reload()
    
    ## Tag
    
    accuracy_tag = "accuracy"
    
    if superset:
        accuracy_tag = "superset_" + accuracy_tag
    
    # Loading
    
    errors = 0
    
    for data_set_kind in data_set_kinds:
        
        try:
            scalars = multiplexer.Scalars(data_set_kind, accuracy_tag)
        except KeyError:
            accuracies[data_set_kind] = None
            errors += 1
            continue
        
        accuracies_kind = numpy.empty(len(scalars))
        
        if len(scalars) == 1:
            accuracies_kind[0] = scalars[0].value
        else:
            for scalar in scalars:
                accuracies_kind[scalar.step - 1] = scalar.value
        
        accuracies[data_set_kind] = accuracies_kind
    
    if len(data_set_kinds) == 1:
        accuracies = accuracies[data_set_kinds[0]]
    
    if errors == len(data_set_kinds):
        accuracies = None
    
    return accuracies

def loadCentroids(model, data_set_kinds = "all"):
    
    # Setup
    
    centroids_sets = {}
    
    ## Data set kinds
    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "test"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]
    
    ## Exit if SNN model
    if model.type == "SNN":
        return None
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        model.log_directory)
    multiplexer.Reload()
    
    # Loading
    
    for data_set_kind in data_set_kinds:
        
        centroids_set = {}
        
        for distribution in ["prior", "posterior"]:
            
            try:
                scalars = multiplexer.Scalars(data_set_kind,
                    distribution + "/cluster_0/probability")
            except KeyError:
                centroids_set[distribution] = None
                continue
            
            # Number of epochs
            E = len(scalars)
            
            # Number of clusters
            if "mixture" in model.latent_distribution[distribution]["name"]:
                K = model.number_of_latent_clusters
            else:
                K = 1
            
            # Number of latent dimensions
            L = model.latent_size
            
            # Initialise
            z_probabilities = numpy.empty((E, K))
            z_means = numpy.empty((E, K, L))
            z_variances = numpy.empty((E, K, L))
            z_covariance_matrices = numpy.empty((E, K, L, L))
            
            # Looping
            for k in range(K):
                
                probability_scalars = multiplexer.Scalars(data_set_kind,
                    distribution + "/cluster_{}/probability".format(k))
                
                if len(probability_scalars) == 1:
                    z_probabilities[0][k] = probability_scalars[0].value
                else:
                    for scalar in probability_scalars:
                        z_probabilities[scalar.step - 1][k] = scalar.value
                
                for l in range(L):
                    
                    mean_scalars = multiplexer.Scalars(data_set_kind,
                        distribution + "/cluster_{}/mean/dimension_{}".format(
                            k, l))
                    
                    if len(mean_scalars) == 1:
                        z_means[0][k][l] = mean_scalars[0].value
                    else:
                        for scalar in mean_scalars:
                            z_means[scalar.step - 1][k][l] = scalar.value
                    
                    variance_scalars = multiplexer.Scalars(data_set_kind,
                        distribution + 
                            "/cluster_{}/variance/dimension_{}"
                                .format(k, l))
                    
                    if len(variance_scalars) == 1:
                        z_variances[0][k][l] = \
                            variance_scalars[0].value
                    else:
                        for scalar in variance_scalars:
                            z_variances[scalar.step - 1][k][l] = \
                                scalar.value
                
                for e in range(E):
                    z_covariance_matrices[e, k] = numpy.diag(z_variances[e, k])
            
            if data_set_kind == "test":
                z_probabilities = z_probabilities[0]
                z_means = z_means[0]
                z_covariance_matrices = z_covariance_matrices[0]
            
            centroids_set[distribution] = {
                "probabilities": z_probabilities,
                "means": z_means,
                "covariance_matrices": z_covariance_matrices
            }
        
        centroids_sets[data_set_kind] = centroids_set
    
    if len(data_set_kinds) == 1:
        centroids_sets = centroids_sets[data_set_kinds[0]]
    
    return centroids_sets

def loadKLDivergences(model):
    
    # Setup
    
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        model.log_directory)
    multiplexer.Reload()
    
    N_epochs = len(multiplexer.Scalars("training",
        "kl_divergence_neurons/0"))
    
    if "mixture" in model.latent_distribution_name:
        latent_size = 1
    else:
        latent_size = model.latent_size

    KL_neurons = numpy.empty([N_epochs, latent_size])
    
    # Loading
    
    for i in range(latent_size):
        scalars = multiplexer.Scalars("training",
            "kl_divergence_neurons/{}".format(i))
        
        for scalar in scalars:
            KL_neurons[scalar.step - 1][i] = scalar.value
    
    return KL_neurons

def axisLabelForSymbol(symbol, coordinate = None, decomposition_method = None,
    distribution = None, prefix = "", suffix = ""):
    
    if decomposition_method:
        decomposition_method = normaliseString(decomposition_method)
    
    if decomposition_method == "pca":
        decomposition_label = "PC"
    elif decomposition_method == "ica":
        decomposition_label = "IC"
    elif decomposition_method == "t_sne":
        decomposition_label = "$t$-SNE"
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

def classLabelSortKey(label):
    if label == "No class":
        label = "ZZZ" + label
    if label == "Others":
        label = "ZZZZ" + label
    return label
