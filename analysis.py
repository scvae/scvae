#!/usr/bin/env python3

import numpy
from numpy import nan
from sklearn.decomposition import PCA

from tensorflow.tensorboard.backend.event_processing import event_multiplexer

from matplotlib import pyplot
import seaborn

import os
import gzip
import pickle

from time import time
from auxiliary import formatTime, formatDuration, normaliseString

palette = seaborn.color_palette('Set2', 8)
seaborn.set(style='ticks', palette = palette)

figure_extension = ".png"

pyplot.rcParams.update({'figure.max_open_warning': 0})

def analyseData(data_sets, results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, "data")
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    # Metrics
    
    print("Calculating metrics for data set.")
    metrics_time_start = time()
    
    x_statistics = [
        statistics(data_set.values, data_set.kind, tolerance = 0.5)
            for data_set in data_sets
    ]
    
    metrics_duration = time() - metrics_time_start
    print("Metrics calculated ({}).".format(
        formatDuration(metrics_duration)))
    
    ## Saving
    
    metrics_path = os.path.join(results_directory, "data_set_metrics.log")
    
    metrics_saving_time_start = time()
    
    with open(metrics_path, "w") as metrics_file:
        metrics_string = "Metrics:\n"
        metrics_string += formatStatistics(x_statistics)
        metrics_file.write(metrics_string)
    
    metrics_saving_duration = time() - metrics_saving_time_start
    print("Metrics saved ({}).".format(formatDuration(
        metrics_saving_duration)))
    
    print()
    
    ## Displaying
    
    print(formatStatistics(x_statistics), end = "")
    
    print()
    
    # Heat map for test set
    
    for data_set in data_sets:
        if data_set.kind == "test":
            x_test = data_set
    
    print("Plotting heat map for test set.")
    
    heat_maps_time_start = time()
    
    figure, name = plotHeatMap(
        x_test.values,
        x_name = x_test.tags["example"].capitalize() + "s",
        y_name = x_test.tags["feature"].capitalize() + "s",
        name = "test"
    )
    saveFigure(figure, name, results_directory)
    
    heat_maps_duration = time() - heat_maps_time_start
    print("Heat map for test set plotted and saved ({})." \
        .format(formatDuration(heat_maps_duration)))
    
    print()

def analyseModel(model, results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, model.name)
    
    # Learning curves
    
    print("Plotting learning curves.")
    learning_curves_time_start = time()
    
    learning_curves = loadLearningCurves(model,
        data_set_kinds = ["training", "validation"])
    
    figure, name = plotLearningCurves(learning_curves, model.type)
    saveFigure(figure, name, results_directory)
    
    learning_curves_duration = time() - learning_curves_time_start
    print("Learning curves plotted and saved ({}).".format(
        formatDuration(learning_curves_duration)))
    
    print()
    
    # Heat map of KL for all latent neurons
    
    if "AE" in model.type:
        
        print("Plotting logarithm of KL divergence heat map.")
        heat_map_time_start = time()
        
        KL_neurons = loadKLDivergences(model)
    
        KL_neurons = numpy.sort(KL_neurons, axis = 1)
        log_KL_neurons = numpy.log(KL_neurons)
        
        figure, name = plotHeatMap(
            log_KL_neurons, z_min = log_KL_neurons.min(),
            x_name = "Epoch", y_name = "$i$",
            z_name = "$\log$ KL$(p_i|q_i)$",
            name = "kl_divergence")
        saveFigure(figure, name, results_directory)
        
        heat_map_duration = time() - heat_map_time_start
        print("Heat map plotted and saved ({}).".format(
            formatDuration(heat_map_duration)))
        
        print()

def analyseAllModels(models_summaries, results_directory = "results"):
    
    # Learning curves
    
    print("Plotting learning curves for all run models.")
    
    figure, name = plotLearningCurvesForModels(models_summaries)
    saveFigure(figure, name, results_directory)
    
    print()
    
    # Learning curves
    
    print("Plotting evaluation for all run models.")
    
    figure, name = plotEvaluationsForModels(models_summaries)
    saveFigure(figure, name, results_directory)
    
    print()

def analyseResults(x_test, x_tilde_test, z_test, highlight_feature_indices,
    model, results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, model.name)
    
    M = x_test.number_of_examples
    
    # Loading
    
    evaluation_test = loadLearningCurves(model, "test")
    number_of_epochs_trained = loadNumberOfEpochsTrained(model)
    
    # Metrics
    
    print("Calculating metrics for results.")
    metrics_time_start = time()
    
    x_statistics = [
        statistics(data_set.values, data_set.version, tolerance = 0.5)
            for data_set in [x_test, x_tilde_test]
    ]
    
    x_diff = x_test.values - x_tilde_test.values
    x_statistics.append(statistics(numpy.abs(x_diff), "differences",
        skip_sparsity = True))
    
    x_log_ratio = numpy.log((x_test.values + 1) / (x_tilde_test.values + 1))
    x_statistics.append(statistics(numpy.abs(x_log_ratio), "log-ratios",
        skip_sparsity = True))
    
    if x_test.values.max() > 20:
        count_accuracy_method = "orders of magnitude"
    else:
        count_accuracy_method = None
    
    count_accuracies = computeCountAccuracies(
        x_test.values,
        x_tilde_test.values,
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
        elif "VAE" in model.type:
            metrics_string += \
                "    ELBO: {:.5g}.\n".format(
                    evaluation_test["lower_bound"][-1]) + \
                "    ENRE: {:.5g}.\n".format(
                    evaluation_test["reconstruction_error"][-1]) + \
                "    KL:   {:.5g}.\n".format(
                    evaluation_test["kl_divergence"][-1])
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
    
    # Profile comparisons
    
    print("Plotting profile comparisons.")
    profile_comparisons_time_start = time()
    
    subset = numpy.random.randint(M, size = 10)
    
    for j, i in enumerate(subset):
        
        figure, name = plotProfileComparison(
            x_test.values[i],
            x_tilde_test.values[i],
            x_name = x_test.tags["example"].capitalize() + "s",
            y_name = x_test.tags["feature"].capitalize() + "s",
            scale = "log",
            title = str(x_test.example_names[i]),
            name = str(j)
        )
        saveFigure(figure, name, results_directory)
    
    profile_comparisons_duration = time() - profile_comparisons_time_start
    print("Profile comparisons plotted and saved ({}).".format(
        formatDuration(profile_comparisons_duration)))
    
    print()
    
    # Heat maps
    
    print("Plotting heat maps.")
    
    # Reconstructions
    
    heat_maps_time_start = time()
    
    figure, name = plotHeatMap(
        x_tilde_test.values,
        x_name = x_test.tags["example"].capitalize() + "s",
        y_name = x_test.tags["feature"].capitalize() + "s",
        name = "reconstruction"
    )
    saveFigure(figure, name, results_directory)
    
    heat_maps_duration = time() - heat_maps_time_start
    print("    Reconstruction heat map plotted and saved ({})." \
        .format(formatDuration(heat_maps_duration)))
    
    # Differences
    
    heat_maps_time_start = time()
    
    figure, name = plotHeatMap(
        x_diff,
        x_name = x_test.tags["example"].capitalize() + "s",
        y_name = x_test.tags["feature"].capitalize() + "s",
        name = "difference",
        center = 0
    )
    saveFigure(figure, name, results_directory)
    
    heat_maps_duration = time() - heat_maps_time_start
    print("    Difference heat map plotted and saved ({})." \
        .format(formatDuration(heat_maps_duration)))
    
    # log-ratios
    
    heat_maps_time_start = time()
    
    figure, name = plotHeatMap(
        x_log_ratio,
        x_name = x_test.tags["example"].capitalize() + "s",
        y_name = x_test.tags["feature"].capitalize() + "s",
        name = "log_ratio",
        center = 0
    )
    saveFigure(figure, name, results_directory)
    
    heat_maps_duration = time() - heat_maps_time_start
    print("    log-ratio heat map plotted and saved ({})." \
        .format(formatDuration(heat_maps_duration)))
    
    print()
    
    # Latent space
    
    if z_test is not None:
        
        print("Plotting latent space.")
        
        # No colour-coding
        
        latent_space_time_start = time()
        
        figure, name = plotLatentSpace(z_test)
        saveFigure(figure, name, results_directory)
        
        latent_space_duration = time() - latent_space_time_start
        print("    Latent space plotted and saved ({}).".format(
            formatDuration(latent_space_duration)))
        
        # Labels
        
        if z_test.labels is not None:
            latent_space_time_start = time()
            
            figure, name = plotLatentSpace(z_test, colour_coding = "labels")
            saveFigure(figure, name, results_directory)
            
            latent_space_duration = time() - latent_space_time_start
            print("    Latent space (with labels) plotted and saved ({}).".format(
                formatDuration(latent_space_duration)))
        
        # Count sum
        
        latent_space_time_start = time()
        
        figure, name = plotLatentSpace(z_test, colour_coding = "count sum",
            test_set = x_test)
        saveFigure(figure, name, results_directory)
        
        latent_space_duration = time() - latent_space_time_start
        print("    Latent space (with count sum) plotted and saved ({}).".format(
            formatDuration(latent_space_duration)))
        
        # Features
        
        for feature_index in highlight_feature_indices:
        
            latent_space_time_start = time()
        
            figure, name = plotLatentSpace(z_test, colour_coding = "feature",
                feature_index = feature_index, test_set = x_test)
            saveFigure(figure, name, results_directory)
        
            latent_space_duration = time() - latent_space_time_start
            print("   ",
                "Latent space (with {}) plotted and saved ({}).".format(
                    x_test.feature_names[feature_index],
                    formatDuration(latent_space_duration)))

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

def plotLearningCurves(curves, model_type, name = None):
    
    figure_name = "learning_curves"
    
    if name:
        figure_name = figure_name + "_" + name
    
    if model_type == "SNN":
        figure = pyplot.figure()
        axis_1 = figure.add_subplot(1, 1, 1)
    elif "AE" in model_type:
        figure, (axis_1, axis_2) = pyplot.subplots(2, sharex = True,
            figsize = (6.4, 9.6))
    
    for i, (curve_set_name, curve_set) in enumerate(sorted(curves.items())):
        
        colour = palette[i]
        
        for curve_name, curve in sorted(curve_set.items()):
            if curve_name == "lower_bound":
                curve_name = "$\\mathcal{L}$"
                line_style = "solid"
                axis = axis_1
            elif curve_name == "reconstruction_error":
                curve_name = "$\\log p(x|z)$"
                line_style = "dashed"
                axis = axis_1
            elif curve_name == "kl_divergence":
                line_style = "dashed"
                curve_name = "KL$(p||q)$"
                axis = axis_2
            if curve_name == "log_likelihood":
                curve_name = "$L$"
                line_style = "solid"
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
        axis_2.legend(loc = "best")
        axis_2.set_xlabel("Epoch")
    
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
    axis.plot(x, original_series[sort_indices], color = palette[0],
        label = 'Original', zorder = 1)
    axis.scatter(x, reconstructed_series[sort_indices], color = palette[1],
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

def plotLatentSpace(latent_set, colour_coding = None, feature_index = None,
    test_set = None, name = None):
    
    figure_name = "latent_space"
    
    if colour_coding:
        figure_name += "-" + normaliseString(colour_coding)
    
    M = latent_set.number_of_examples
    L = latent_set.number_of_features
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    if L > 2:
        pca = PCA(n_components = 2)
        pca.fit(latent_set.values)
        values = pca.transform(latent_set.values)
        
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
    
    elif L == 2:
        values = latent_set.values
        
        axis.set_xlabel("$z_1$")
        axis.set_ylabel("$z_2$")
    
    if colour_coding == "labels":
        
        label_indices = dict()
        
        for i, label in enumerate(latent_set.labels):
        
            if label not in label_indices:
                label_indices[label] = []
        
            label_indices[label].append(i)
        
        latent_palette = seaborn.color_palette("hls", len(label_indices))
        
        for i, (label, indices) in enumerate(sorted(label_indices.items())):
            axis.scatter(values[indices, 0], values[indices, 1], label = label,
                color = latent_palette[i])
        
        if len(label_indices) < 20:
            axis.legend(loc = "best")
    
    elif colour_coding == "count sum":
        
        if test_set is None:
            raise ValueError("Test set not given.")
        
        n_test = test_set.count_sum
        
        n_normalised = n_test / n_test.max()
        
        colours = numpy.array([palette[0]]) * n_normalised
        
        axis.scatter(values[:, 0], values[:, 1], color = colours)
    
    elif colour_coding == "feature":
        
        if test_set is None:
            raise ValueError("Test set not given.")
        
        if feature_index is None:
            raise ValueError("Feature number not given.")
        
        if feature_index > test_set.number_of_features:
            raise ValueError("Feature number higher than number of features.")
        
        figure_name += "-{}".format(
            normaliseString(test_set.feature_names[feature_index]))
        
        f_test = test_set.values[:, feature_index].reshape(-1, 1)
        
        if f_test.max() != 0:
            f_normalised = f_test / f_test.max()
        else:
            f_normalised = f_test
        
        colours = numpy.array([palette[0]]) * f_normalised
        
        axis.scatter(values[:, 0], values[:, 1], color = colours)
    
    else:
        axis.scatter(values[:, 0], values[:, 1])
    
    if name:
        figure_name = figure_name + "_" + name
    
    return figure, figure_name

def saveFigure(figure, figure_name, results_directory, no_spine = True):
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    if no_spine:
        seaborn.despine()
    
    figure_path = os.path.join(results_directory, figure_name) + figure_extension
    figure.savefig(figure_path, bbox_inches = 'tight')

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
    elif "AE" in model.type:
        losses = ["lower_bound", "kl_divergence", "reconstruction_error"]
    
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
