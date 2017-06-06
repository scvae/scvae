#!/usr/bin/env python3

import numpy
from numpy import nan

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE

from tensorflow.tensorboard.backend.event_processing import event_multiplexer

from matplotlib import pyplot
import matplotlib.patches
import matplotlib.gridspec
import seaborn

from PIL import Image

import os
import gzip
import pickle

import copy
import re

from time import time
from auxiliary import formatTime, formatDuration, normaliseString, properString

standard_palette = seaborn.color_palette('Set2', 8)
lighter_palette = lambda N: seaborn.hls_palette(N)
darker_palette = lambda N: seaborn.hls_palette(N, l = .4)
seaborn.set(style='ticks', palette = standard_palette)

figure_extension = ".png"
image_extension = ".png"

number_of_random_examples = 100

def analyseData(data_sets, decomposition_methods = ["PCA"],
    highlight_feature_indices = [], results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, "data")
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    # Metrics
    
    print("Calculating metrics for data set.")
    metrics_time_start = time()
    
    x_statistics = []
    number_of_examples = {}
    number_of_features = 0
    
    for data_set in data_sets:
        number_of_examples[data_set.kind] = data_set.number_of_examples
        if data_set.kind == "full":
            number_of_features = data_set.number_of_features
        x_statistics.append(
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
            + formatStatistics(x_statistics)
        metrics_file.write(metrics_string)
    
    metrics_saving_duration = time() - metrics_saving_time_start
    print("Metrics saved ({}).".format(formatDuration(
        metrics_saving_duration)))
    
    print()
    
    ## Displaying
    
    print(formatStatistics(x_statistics), end = "")
    
    print()
    
    # Find test data set
    
    for data_set in data_sets:
        
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
        
        # Heat map for data set
        
        print("Plotting heat map for {} set.".format(data_set.kind))
        heat_maps_time_start = time()
        
        figure, figure_name = plotHeatMap(
            data_set.values,
            x_name = data_set.tags["example"].capitalize() + "s",
            y_name = data_set.tags["feature"].capitalize() + "s",
            name = data_set.kind
        )
        saveFigure(figure, figure_name, results_directory)
        
        heat_maps_duration = time() - heat_maps_time_start
        print("Heat map for {} set plotted and saved ({})." \
            .format(data_set.kind, formatDuration(heat_maps_duration)))
        
        print()
        
        # Decompositions
        
        analyseDecompositions(
            data_set,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            symbol = "$x$",
            title = "original space",
            specifier = lambda data_set: data_set.kind,
            results_directory = results_directory
        )

def analyseModel(model, results_directory = "results"):
    
    # Setup
    
    if model.type in ["VAE", "IWVAE", "CVAE"]:
        model_name = model.testing_name
    else:
        model_name = model.name
    
    results_directory = os.path.join(results_directory, model_name)
    
    # Learning curves
    
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
    
    # Heat map of KL for all latent neurons
    
    if "AE" in model.type and model.type != "CVAE":
        
        print("Plotting logarithm of KL divergence heat map.")
        heat_map_time_start = time()
        
        KL_neurons = loadKLDivergences(model)
    
        KL_neurons = numpy.sort(KL_neurons, axis = 1)
        log_KL_neurons = numpy.log(KL_neurons)
        
        figure, figure_name = plotHeatMap(
            log_KL_neurons, z_min = log_KL_neurons.min(),
            x_name = "Epoch", y_name = "$i$",
            z_name = "$\log$ KL$(p_i|q_i)$",
            name = "kl_divergence")
        saveFigure(figure, figure_name, results_directory)
        
        heat_map_duration = time() - heat_map_time_start
        print("Heat map plotted and saved ({}).".format(
            formatDuration(heat_map_duration)))
        
        print()
    
    # Evolution of latent centroids
    
    if "AE" in model.type and "mixture" in model.latent_distribution_name:
        
        centroids = loadCentroids(model, data_set_kinds = "validation")
        prior_centroids = centroids["prior"]
        
        if prior_centroids:
            
            print("Plotting evolution of latent prior parameters.")
            centroids_time_start = time()
            
            prior_centroid_probailities = prior_centroids["probabilities"]
            
            figure, figure_name = plotEvolutionOfLatentPriorProbabilities(
                prior_centroid_probailities)
            saveFigure(figure, figure_name, results_directory)
            
            figure, figure_name = plotEvolutionOfLatentPriorCentroids(centroids)
            saveFigure(figure, figure_name, results_directory)
            
            centroids_duration = time() - centroids_time_start
            print("Evolution of latent prior parameters plotted and saved".format(
                formatDuration(centroids_duration)))
            
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

def analyseIntermediateResults(latent_values, data_set, centroids, epoch,
    model_name, results_directory = "results"):
    
    results_directory = os.path.join(results_directory, model_name,
        "intermediate")
    
    latent_set_name = "latent {} set".format(data_set.kind)
    
    print("Plotting {} with centroids for epoch {}.".format(
        latent_set_name, epoch + 1))
    
    if latent_values.shape[1] == 2:
        symbol = "$z$"
        figure_labels = {
            "title": None,
            "x label": symbol + "$_1$",
            "y label": symbol + "$_2$"
        }
        
        latent_values_decomposed = latent_values
        centroids_decomposed = centroids
    else:    
        print("Decomposing", latent_set_name, "using PCA.")
        decompose_time_start = time()
        
        latent_values_decomposed, centroids_decomposed = decompose(
            latent_values,
            centroids,
            method = "PCA",
            components = 2
        )

        decompose_duration = time() - decompose_time_start
        print("{} decomposed ({}).".format(
            latent_set_name.capitalize(),
            formatDuration(decompose_duration))
        )
        
        figure_labels = {
            "title": "PCA",
            "x label": "PC 1",
            "y label": "PC 2"
        }
    
    plot_time_start = time()
    
    name = "latent_space-epoch-{}".format(epoch + 1)
    
    if data_set.labels is not None:
        figure, figure_name = plotValues(
            latent_values_decomposed,
            colour_coding = "labels",
            colouring_data_set = data_set,
            centroids = centroids_decomposed,
            figure_labels = figure_labels,
            name = name
        )
        saveFigure(figure, figure_name, results_directory)
        if data_set.label_superset is not None:
            figure, figure_name = plotValues(
                latent_values_decomposed,
                colour_coding = "label superset",
                colouring_data_set = data_set,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
            )
            saveFigure(figure, figure_name, results_directory)
    else:
        figure, figure_name = plotValues(
            latent_values_decomposed,
            centroids = centroids_decomposed,
            figure_labels = figure_labels,
            name = name
        )
        saveFigure(figure, figure_name, results_directory)
    
    plot_duration = time() - plot_time_start
    print("{} plotted and saved ({}).".format(
        latent_set_name.capitalize(), formatDuration(plot_duration)))

def analyseResults(test_set, reconstructed_test_set, latent_test_sets, model,
    decomposition_methods = ["PCA"], highlight_feature_indices = [],
    results_directory = "results"):
    
    # Setup
    
    if model.type in ["VAE", "IWVAE", "CVAE"]:
        model_name = model.testing_name
    else:
        model_name = model.name
    
    results_directory = os.path.join(results_directory, model_name)
    
    M = test_set.number_of_examples
    
    # Loading
    
    evaluation_test = loadLearningCurves(model, "test")
    number_of_epochs_trained = loadNumberOfEpochsTrained(model)
    
    # Metrics

    print("Calculating metrics for results.")
    metrics_time_start = time()

    x_statistics = [
        statistics(data_set.values, data_set.version, tolerance = 0.5)
            for data_set in [test_set, reconstructed_test_set]
    ]

    x_diff = test_set.values - reconstructed_test_set.values
    x_statistics.append(statistics(numpy.abs(x_diff), "differences",
        skip_sparsity = True))

    x_log_ratio = numpy.log((test_set.values + 1) \
        / (reconstructed_test_set.values + 1))
    x_statistics.append(statistics(numpy.abs(x_log_ratio), "log-ratios",
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
            if model.type != "CVAE":
                metrics_string += \
                    "    KL:   {:.5g}.\n".format(
                        evaluation_test["kl_divergence"][-1])
            else:
                metrics_string += \
                    "    KL_z1:   {:.5g}.\n".format(
                        evaluation_test["kl_divergence_z1"][-1]) + \
                    "    KL_z2:   {:.5g}.\n".format(
                        evaluation_test["kl_divergence_z2"][-1]) + \
                    "    KL_y:   {:.5g}.\n".format(
                        evaluation_test["kl_divergence_y"][-1])
        metrics_string += "\n" + formatStatistics(x_statistics)
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

    print(formatStatistics(x_statistics))
    print(formatCountAccuracies(count_accuracies))

    # Examples

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

    # Profile comparisons

    print("Plotting profile comparisons.")
    profile_comparisons_time_start = time()

    subset = numpy.random.randint(M, size = 10)

    for j, i in enumerate(subset):

        figure, figure_name = plotProfileComparison(
            test_set.values[i],
            reconstructed_test_set.values[i],
            x_name = test_set.tags["example"].capitalize() + "s",
            y_name = test_set.tags["feature"].capitalize() + "s",
            scale = "log",
            title = str(test_set.example_names[i]),
            name = str(j)
        )
        saveFigure(figure, figure_name, results_directory)

    profile_comparisons_duration = time() - profile_comparisons_time_start
    print("Profile comparisons plotted and saved ({}).".format(
        formatDuration(profile_comparisons_duration)))

    print()

    # Heat maps

    print("Plotting heat maps.")

    # Reconstructions

    heat_maps_time_start = time()

    figure, figure_name = plotHeatMap(
        reconstructed_test_set.values,
        x_name = test_set.tags["example"].capitalize() + "s",
        y_name = test_set.tags["feature"].capitalize() + "s",
        name = "reconstruction"
    )
    saveFigure(figure, figure_name, results_directory)

    heat_maps_duration = time() - heat_maps_time_start
    print("    Reconstruction heat map plotted and saved ({})." \
        .format(formatDuration(heat_maps_duration)))

    # Differences

    heat_maps_time_start = time()

    figure, figure_name = plotHeatMap(
        x_diff,
        x_name = test_set.tags["example"].capitalize() + "s",
        y_name = test_set.tags["feature"].capitalize() + "s",
        name = "difference",
        center = 0
    )
    saveFigure(figure, figure_name, results_directory)

    heat_maps_duration = time() - heat_maps_time_start
    print("    Difference heat map plotted and saved ({})." \
        .format(formatDuration(heat_maps_duration)))

    # log-ratios

    heat_maps_time_start = time()

    figure, figure_name = plotHeatMap(
        x_log_ratio,
        x_name = test_set.tags["example"].capitalize() + "s",
        y_name = test_set.tags["feature"].capitalize() + "s",
        name = "log_ratio",
        center = 0
    )
    saveFigure(figure, figure_name, results_directory)

    heat_maps_duration = time() - heat_maps_time_start
    print("    log-ratio heat map plotted and saved ({})." \
        .format(formatDuration(heat_maps_duration)))

    print()
    
    # Latent space
    
    if latent_test_sets is not None:
        
        if "AE" in model.type:
            centroids = loadCentroids(model, data_set_kinds = "test")
        else:
            centroids = None
        
        analyseDecompositions(
            latent_test_sets,
            colouring_data_set = test_set,
            decomposition_methods = decomposition_methods,
            centroids = centroids,
            highlight_feature_indices = highlight_feature_indices,
            symbol = "$z$",
            title = "latent space",
            specifier = lambda data_set: data_set.version,
            results_directory = results_directory
        )

def analyseDecompositions(data_sets, colouring_data_set = None,
    decomposition_methods = ["PCA"], centroids = None,
    highlight_feature_indices = [], symbol = "$x$",
    title = "data set", specifier = None,
    results_directory = "results"):
    
    centroids_original = centroids

    if not isinstance(data_sets, (list, tuple)):
        data_sets = [data_sets]
    
    ID = None
    
    for data_set in data_sets:
        
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

        decomposition_methods.insert(0, None)
    
        for decomposition_method in decomposition_methods:
        
            if not decomposition_method:
                if data_set.number_of_features == 2:
                    figure_labels = {
                        "title": None,
                        "x label": symbol + "$_1$",
                        "y label": symbol + "$_2$"
                    }
                
                    values_decomposed = data_set.values
                    centroids_decomposed = centroids
                else:
                    continue
            else:    
                decomposition_method = properString(decomposition_method,
                    decomposition_method_names)
                
                print("Decomposing {} using {}.".format(
                    title_with_ID, decomposition_method))
                decompose_time_start = time()
    
                values_decomposed, centroids_decomposed = decompose(
                    data_set.values,
                    centroids,
                    decomposition_method,
                    components = 2
                )
    
                decompose_duration = time() - decompose_time_start
                print("{} decomposed ({}).".format(
                    title_with_ID.capitalize(),
                    formatDuration(decompose_duration)
                ))
            
                if decomposition_method == "PCA":
                    figure_labels = {
                        "title": "PCA",
                        "x label": "PC 1",
                        "y label": "PC 2"
                    }
                elif decomposition_method == "ICA":
                    figure_labels = {
                        "title": "ICA",
                        "x label": "IC 1",
                        "y label": "IC 2"
                    }
                elif decomposition_method == "t-SNE":
                    figure_labels = {
                        "title": "$t$-SNE",
                        "x label": "$t$-SNE 1",
                        "y label": "$t$-SNE 2"
                    }
                
                print()
            
            # Plot data set
            
            print("Plotting {}{}.".format(
                "decomposed " if decomposition_method else "", title_with_ID))
                
            ## No colour-coding
            
            plot_time_start = time()
            
            figure, figure_name = plotValues(
                values_decomposed,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
            )
            saveFigure(figure, figure_name, results_directory)
    
            plot_duration = time() - plot_time_start
            print("    {} plotted and saved ({}).".format(
                title_with_ID.capitalize(),
                formatDuration(plot_duration)
            ))
        
            # Labels
        
            if data_set.labels is not None:
                plot_time_start = time()
            
                figure, figure_name = plotValues(
                    values_decomposed,
                    colour_coding = "labels",
                    colouring_data_set = colouring_data_set,
                    centroids = centroids_decomposed,
                    figure_labels = figure_labels,
                    name = name
                )
                saveFigure(figure, figure_name, results_directory)
        
                plot_duration = time() - plot_time_start
                print("    {} (with labels) plotted and saved ({}).".format(
                    title_with_ID.capitalize(), formatDuration(plot_duration)))
                
                if data_set.label_superset is not None:
                    plot_time_start = time()
                    
                    figure, figure_name = plotValues(
                        values_decomposed,
                        colour_coding = "label superset",
                        colouring_data_set = colouring_data_set,
                        centroids = centroids_decomposed,
                        figure_labels = figure_labels,
                        name = name
                    )
                    saveFigure(figure, figure_name, results_directory)
                    
                    plot_duration = time() - plot_time_start
                    print("    {} (with label superset) plotted and saved ({}).".format(
                        title_with_ID.capitalize(), formatDuration(plot_duration)))
        
            # Count sum
        
            plot_time_start = time()
        
            figure, figure_name = plotValues(
                values_decomposed,
                colour_coding = "count sum",
                colouring_data_set = colouring_data_set,
                centroids = centroids_decomposed,
                figure_labels = figure_labels,
                name = name
            )
            saveFigure(figure, figure_name, results_directory)
        
            plot_duration = time() - plot_time_start
            print("    {} (with count sum) plotted and saved ({}).".format(
                title_with_ID.capitalize(), formatDuration(plot_duration)))
        
            # Features
        
            for feature_index in highlight_feature_indices:
            
                plot_time_start = time()
            
                figure, figure_name = plotValues(
                    values_decomposed,
                    colour_coding = "feature",
                    colouring_data_set = colouring_data_set,
                    centroids = centroids_decomposed,
                    feature_index = feature_index,
                    figure_labels = figure_labels,
                    name = name
                )
                saveFigure(figure, figure_name, results_directory)
            
                plot_duration = time() - plot_time_start
                print("   {} (with {}) plotted and saved ({}).".format(
                    title_with_ID.capitalize(),
                    data_set.feature_names[feature_index],
                    formatDuration(plot_duration)
                ))
        
            print()

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

def formatStatistics(statistics_sets):
    
    if type(statistics_sets) != list:
        statistics_sets = [statistics_sets]
    
    name_width = 0
    
    for statistics_set in statistics_sets:
        name_width = max(len(statistics_set["name"]), name_width)
    
    statistics_string = "  ".join(["{:{}}".format("Data set", name_width),
        " mean ", "std. dev. ", "dispersion",
        " minimum ", " maximum ","sparsity"]) + "\n"
    
    for statistics_set in statistics_sets:
        string_parts = [
            "{:{}}".format(statistics_set["name"], name_width),
            "{:<9.5g}".format(statistics_set["mean"]),
            "{:<9.5g}".format(statistics_set["standard deviation"]),
            "{:<9.5g}".format(statistics_set["dispersion"]),
            "{:<9.3g}".format(statistics_set["minimum"]),
            "{:<9.3g}".format(statistics_set["maximum"]),
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

def decompose(values, centroids = None, method = "PCA",
    components = 2, random = False):
    
    method = normaliseString(method)
    
    if random:
        random_state = None
    else:
        random_state = 42
    
    if method == "pca":
        model = PCA(n_components = components)
    elif method == "ica":
        model = FastICA(n_components = components)
    elif method == "t_sne":
        model = TSNE(n_components = components, random_state = random_state)
    else:
        raise ValueError("method method not found.")
    
    values_decomposed = model.fit_transform(values)
    
    # Only supports centroids without data sets as top levels and no epoch
    # information
    if centroids and method == "pca":
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
    else:
        centroids_decomposed = None
    
    return values_decomposed, centroids_decomposed

def plotHistogram(series, x_label, scale = "linear", name = None):
    
    figure_name = "histogram"
    
    if name:
        figure_name = name + "_" + figure_name
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.distplot(series, kde = False, ax = axis)
    
    axis.set_yscale(scale)
    
    axis.set_xlabel(x_label)
    
    return figure, figure_name

def plotSeries(series, x_label, y_label, scale = "linear", bar = False,
    name = None):
    
    figure_name = "series"
    
    if name:
        figure_name = name + "_" + figure_name
    
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
    
    return figure, figure_name

def plotLearningCurves(curves, model_type, name = None):
    
    figure_name = "learning_curves"
    
    if name:
        figure_name = figure_name + "_" + name
    
    if model_type == "SNN":
        figure = pyplot.figure()
        axis_1 = figure.add_subplot(1, 1, 1)
    elif model_type == "CVAE":
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
                    if latent_variable == "z_2":
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
            epochs = numpy.arange(len(curve)) + 1
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
        if model_type == "CVAE":
            axis_3.legend(loc = "best")
            handles, labels = axis_3.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
            axis_3.legend(handles, labels, loc = "best")
            axis_3.set_xlabel("Epoch")
        else:
            axis_2.set_xlabel("Epoch")
    
    return figure, figure_name

def plotEvolutionOfLatentPriorProbabilities(prior_probabilities, name = None):
    
    figure_name = "prior_evolution-probabilities"
    
    if name:
        figure_name = figure_name + "_" + name
    
    E, K = prior_probabilities.shape
    
    centroids_palette = seaborn.hls_palette(K, l = .4)
    epochs = numpy.arange(E) + 1
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    for k in range(K):
        axis.plot(epochs, prior_probabilities[:, k], color = centroids_palette[k])
    
    return figure, figure_name

def plotEvolutionOfLatentPriorCentroids(centroids, name = None):
    
    figure_name = "prior_evolution-centroids"
    
    if name:
        figure_name = figure_name + "_" + name
    
    means = centroids["prior"]["means"]
    E, K, L = centroids["prior"]["means"].shape
    
    if L > 2:
        _, centroids_decomposed = decompose(means.reshape(-1, L), centroids)
    else:
        centroids_decomposed = centroids
    
    means = centroids_decomposed["prior"]["means"]
    covariance_matrices = centroids_decomposed["prior"]["covariance_matrices"]
    
    centroids_palette = numpy.array(seaborn.hls_palette(K, l = .4))
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    for k in range(K):
        for e in range(E):
            colour = (e / (2*E) + 0.5) * centroids_palette[k]
            axis.scatter(means[e, k, 0], means[e, k, 1],
                color = colour)
            if e in [0, E-1]:
                addCovariance(covariance_matrices[e, k], means[e, k], axis, colour)
    
    return figure, figure_name

def plotLearningCurvesForModels(models_summaries, name = None):
    
    figure_name = "learning_curves"
    
    if name:
        figure_name = figure_name + "_" + name
    
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
    
    return figure, figure_name

def plotEvaluationsForModels(models_summaries, name = None):
    
    figure_name = "evaluation"
    
    if name:
        figure_name = figure_name + "_" + name
    
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
    
    return figure, figure_name

def plotProfileComparison(original_series, reconstructed_series, 
    x_name, y_name, scale = "linear", title = None, name = None):
    
    figure_name = "profile_comparison"
    
    if name:
        figure_name = figure_name + "_" + name
    
    D = original_series.shape[0]
    
    sort_indices = numpy.argsort(original_series)[::-1]
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    x = numpy.linspace(0, D, D)
    axis.plot(x, original_series[sort_indices], color = standard_palette[0],
        label = 'Original', zorder = 1)
    axis.scatter(x, reconstructed_series[sort_indices], color = standard_palette[1],
        label = 'Reconstruction', zorder = 0)
    
    axis.legend()
    
    if title:
        axis.set_title(title)
    
    axis.set_xscale(scale)
    axis.set_yscale(scale)
    
    axis.set_xlabel(x_name + " sorted by original " + y_name.lower())
    axis.set_ylabel(y_name)
    
    return figure, figure_name

def plotHeatMap(data_set, x_name, y_name, z_name = None,
    z_min = None, z_max = None, center = None, name = None):
    
    figure_name = "heat_map"
    
    if name:
        figure_name = figure_name + "_" + name
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    cbar_dict = {}
    
    if z_name:
        cbar_dict["label"] = z_name
    
    seaborn.heatmap(
        data_set.T,
        vmin = z_min, vmax = z_max, center = center, 
        xticklabels = False, yticklabels = False,
        cbar = True, cbar_kws = cbar_dict, square = True, ax = axis
    )
    
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)
    
    return figure, figure_name

decomposition_method_names = {
    "PCA": ["pca"],
    "ICA": ["ica"],
    "t-SNE": ["t_sne", "tsne"], 
}

def plotValues(values, colour_coding = None, colouring_data_set = None,
    centroids = None, feature_index = None, figure_labels = None,
    name = "scatter"):
    
    figure_name = name
    
    if figure_labels:
        title = figure_labels["title"]
        x_label = figure_labels["x label"]
        y_label = figure_labels["y label"]
    else:
        title = None
        x_label = "$x$"
        y_label = "$y$"
    
    if title:
        figure_name += "-" + normaliseString(title)
    
    if colour_coding:
        colour_coding = normaliseString(colour_coding)
        figure_name += "-" + colour_coding
        if colouring_data_set is None:
            raise ValueError("Colouring data set not given.")
    
    figure = pyplot.figure()
    
    if centroids and centroids["prior"] \
        and centroids["prior"]["probabilities"].shape[0] > 1:
            axes = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            axis = pyplot.subplot(axes[0])
            axis_cat = pyplot.subplot(axes[1])
        # figure, (axis, axis_cat) = pyplot.subplots(1, 2, figsize = (16, 6))
    else:
        axis = figure.add_subplot(1, 1, 1)
    
    axis.set_aspect("equal")
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    if colour_coding == "labels":
        
        label_indices = dict()
        
        for index, label in enumerate(colouring_data_set.labels):
        
            if label not in label_indices:
                label_indices[label] = []
        
            label_indices[label].append(index)
        
        if colouring_data_set.label_palette:
            label_palette = colouring_data_set.label_palette
        else:
            hls_palette = seaborn.color_palette("hls", len(label_indices))
            label_palette = {label: hls_palette[i] for i, label in
                             enumerate(sorted(label_indices.keys()))}
        
        for label, indices in sorted(label_indices.items()):
            axis.scatter(values[indices, 0], values[indices, 1], label = label,
                color = label_palette[label])
        
        if len(label_indices) < 20:
            axis.legend(loc = "best")
    
    elif colour_coding == "label_superset":
        
        label_indices = dict()
        
        label_superset = colouring_data_set.label_superset
        label_superset_reverse = {v: k for k, vs in label_superset.items() for v in vs}
        
        for index, label in enumerate(colouring_data_set.labels):
            
            superset_label = label_superset_reverse[label]
            
            if superset_label not in label_indices:
                label_indices[superset_label] = []
        
            label_indices[superset_label].append(index)
        
        if colouring_data_set.label_palette:
            label_palette = {}
            for superset_label, labels_in_superset_label in label_superset.items():
                superset_label_colours = []
                for label_in_superset_label in labels_in_superset_label:
                    superset_label_colours.append(
                        colouring_data_set.label_palette[label_in_superset_label]
                    )
                label_palette[superset_label] = \
                    numpy.array(superset_label_colours).mean(axis = 0)
        else:
            index_palette = ligther_palette(len(label_indices))
            label_palette = {label: index_palette[i] for i, label in
                             enumerate(sorted(label_indices.keys()))}
        
        for label, indices in sorted(label_indices.items()):
            axis.scatter(values[indices, 0], values[indices, 1], label = label,
                color = label_palette[label])
        
        if len(label_indices) < 20:
            axis.legend(loc = "best")
    
    elif colour_coding == "count_sum":
        
        n_test = colouring_data_set.count_sum
        n_normalised = n_test / n_test.max()
        colours = numpy.array([standard_palette[0]]) * n_normalised
        
        axis.scatter(values[:, 0], values[:, 1], color = colours)
    
    elif colour_coding == "feature":
        
        if feature_index is None:
            raise ValueError("Feature number not given.")
        
        if feature_index > colouring_data_set.number_of_features:
            raise ValueError("Feature number higher than number of features.")
        
        figure_name += "-{}".format(
            normaliseString(colouring_data_set.feature_names[feature_index]))
        
        f_test = colouring_data_set.values[:, feature_index].reshape(-1, 1)
        
        if f_test.max() != 0:
            f_normalised = f_test / f_test.max()
        else:
            f_normalised = f_test
        
        colours = numpy.array([standard_palette[0]]) * f_normalised
        
        axis.scatter(values[:, 0], values[:, 1], color = colours)
    
    else:
        axis.scatter(values[:, 0], values[:, 1])
    
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
                axis_cat.barh(classes[k], probabilities[k],
                    color = centroids_palette[k])
                axis.scatter(means[k, 0], means[k, 1], marker = "x",
                    color = centroids_palette[k])
                addCovariance(covariance_matrices[k], means[k], axis,
                    colour = centroids_palette[k])
            
            axis_cat.set_yticks([])
    
    return figure, figure_name

def plotLatentSpace(latent_values, data_set = None, colour_coding = None,
    feature_index = None, figure_labels = None, name = None):
    
    figure_name = "latent_space"
    
    if name:
        figure_name += "-" + normaliseString(name)
    
    figure, figure_name = plotValues(
        values = latent_values,
        colour_coding = colour_coding,
        colouring_data_set = data_set,
        feature_index = feature_index,
        figure_labels = figure_labels,
        name = figure_name
    )
    
    return figure, figure_name

def addCovariance(covariance, mean, axis, colour, label = None, linestyle = "solid"):
    eigenvalues, eigenvectors = numpy.linalg.eig(covariance)
    index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    lambda_1, lambda_2 = numpy.sqrt(eigenvalues)
    alpha = numpy.rad2deg(numpy.arctan(eigenvectors[1, 0] / eigenvectors[0, 0]))
    ellipse = matplotlib.patches.Ellipse(xy = mean, width = 2 * lambda_1, height = 2 * lambda_2, angle = alpha, linewidth = 2, linestyle = linestyle, facecolor = "none", edgecolor = colour, label = label)
    axis.add_patch(ellipse)

def combineRandomImagesFromDataSet(data_set, number_of_random_examples, name = None):
    
    image_name = "random_examples"
    
    if name:
        image_name += "-" + normaliseString(name)
    
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

def saveFigure(figure, figure_name, results_directory, no_spine = True):
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    if no_spine:
        seaborn.despine()
    
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
