#!/usr/bin/env python3

import numpy
from numpy import nan
from sklearn.decomposition import PCA

from tensorflow.tensorboard.backend.event_processing import event_multiplexer

from matplotlib import pyplot
import seaborn

import os

from time import time
from auxiliary import formatDuration, normaliseString

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
        metrics_string += convertStatisticsToString(x_statistics)
        metrics_file.write(metrics_string)
    
    metrics_saving_duration = time() - metrics_saving_time_start
    print("Metrics saved ({}).".format(formatDuration(
        metrics_saving_duration)))
    
    print()
    
    ## Displaying
    
    print(convertStatisticsToString(x_statistics), end = "")
    
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
    
    results_directory = os.path.join(results_directory, model.directory_suffix)
    
    learning_curves, KL_neurons = parseSummaries(model)
    
    # Learning curves
    
    print("Plotting learning curves.")
    
    figure, name = plotLearningCurves(learning_curves, model.type)
    saveFigure(figure, name, results_directory)
    
    print()
    
    # Heat map of KL for all latent neurons
    
    if KL_neurons is not None:
        
        print("Plotting logarithm of KL divergence heat map.")
        
        KL_neurons = numpy.sort(KL_neurons, axis = 1)
        log_KL_neurons = numpy.log(KL_neurons)
        
        figure, name = plotHeatMap(
            log_KL_neurons, z_min = log_KL_neurons.min(),
            x_name = "Epoch", y_name = "$i$",
            z_name = "$\log$ KL$(p_i|q_i)$",
            name = "kl_divergence")
        saveFigure(figure, name, results_directory)
        
        print()
    
    return learning_curves

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

def analyseResults(x_test, x_tilde_test, z_test, evaluation_test,
    model, results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, model.directory_suffix)
    
    M = x_test.number_of_examples
    
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
    
    metrics_duration = time() - metrics_time_start
    print("Metrics calculated ({}).".format(
        formatDuration(metrics_duration)))
    
    ## Saving
    
    metrics_path = os.path.join(results_directory, "test_metrics.log")
    
    metrics_saving_time_start = time()
    
    with open(metrics_path, "w") as metrics_file:
        metrics_string = "Evaluation:\n"
        if model.type == "SNN":
            metrics_string += \
                "    log-likelihood: {:.5g}.\n".format(evaluation_test["log-likelihood"])
        elif "VAE" in model.type:
            metrics_string += \
                "    ELBO: {:.5g}.\n".format(evaluation_test["ELBO"]) + \
                "    ENRE: {:.5g}.\n".format(evaluation_test["ENRE"]) + \
                "    KL:   {:.5g}.\n".format(evaluation_test["KL"])
        metrics_string += "\n" + convertStatisticsToString(x_statistics)
        metrics_file.write(metrics_string)
    
    metrics_saving_duration = time() - metrics_saving_time_start
    print("Metrics saved ({}).".format(formatDuration(
        metrics_saving_duration)))
    
    print()
    
    ## Displaying
    
    print(convertStatisticsToString(x_statistics), end = "")
    
    print()
    
    # Profile comparisons
    
    print("Plotting profile comparisons.")
    profile_comparisons_time_start = time()
    
    subset = numpy.random.randint(M, size = 10)
    
    for j, i in enumerate(subset):
        
        figure, name = plotProfileComparison(
            x_test.values[i], x_tilde_test.values[i],
            x_name = "Genes", y_name = "Counts", scale = "log",
            title = str(x_test.example_names[i]), name = str(j)
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
    print("    Reconstruction heat map for plotted and saved ({})." \
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
    print("    Difference heat map for plotted and saved ({})." \
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
    print("    log-ratio heat map for plotted and saved ({})." \
        .format(formatDuration(heat_maps_duration)))
    
    print()
    
    # Latent space
    
    if z_test is not None:
        
        print("Plotting latent space.")
        
        # Labels
        
        latent_space_time_start = time()
        
        if z_test.labels is None:
            figure, name = plotLatentSpace(z_test)
            saveFigure(figure, name, results_directory)
            
            latent_space_duration = time() - latent_space_time_start
            print("    Latent space (no labels) plotted and saved ({}).".format(
                formatDuration(latent_space_duration)))
        
        else:
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
        
        # Feature x
        
        feature_index = 0
        
        latent_space_time_start = time()
        
        figure, name = plotLatentSpace(z_test, colour_coding = "feature",
            feature_index = feature_index, test_set = x_test)
        saveFigure(figure, name, results_directory)
        
        latent_space_duration = time() - latent_space_time_start
        print("    Latent space (with feature {}) plotted and saved ({}).".format(
            feature_index + 1,
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

def convertStatisticsToString(statistics_sets):
    
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
    
    figure_name = "latent_space-" + normaliseString(colour_coding)
    
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
        
        for i, (label, indices) in enumerate(label_indices.items()):
            axis.scatter(values[indices, 0], values[indices, 1], label = label,
                color = latent_palette[i])
    
        axis.legend()
    
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
        
        f_normalised = f_test / f_test.max()
        
        colours = numpy.array([palette[0]]) * f_normalised
        
        axis.scatter(values[:, 0], values[:, 1], color = colours)
    
    else:
        axis.scatter(values[indices, 0], values[indices, 1])
    
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

def parseSummaries(model):
    
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        model.log_directory)
    multiplexer.Reload()
    
    # Learning curves
    
    data_set_kinds = ["training", "validation"]
    
    if model.type == "SNN":
        losses = ["log_likelihood"]
    elif "AE" in model.type:
        losses = ["lower_bound", "kl_divergence", "reconstruction_error"]
    
    learning_curve_sets = {}
    
    for data_set_kind in data_set_kinds:
        
        learning_curve_set = {}
        
        for loss in losses:
            
            scalars = multiplexer.Scalars(data_set_kind, "losses/" + loss)
            
            learning_curve = numpy.empty(len(scalars))
            
            for scalar in scalars:
                learning_curve[scalar.step - 1] = scalar.value
            
            learning_curve_set[loss] = learning_curve
        
        learning_curve_sets[data_set_kind] = learning_curve_set
    
    # KL divergence for all latent neurons
    
    try:
        N_epochs = len(multiplexer.Scalars("training",
            "kl_divergence_neurons/0"))
        
        KL_neurons = numpy.empty([N_epochs, model.latent_size])
        
        for i in range(model.latent_size):
            scalars = multiplexer.Scalars("training",
                "kl_divergence_neurons/{}".format(i))
            
            for scalar in scalars:
                KL_neurons[scalar.step - 1][i] = scalar.value
    except:
        KL_neurons = None
    
    return learning_curve_sets, KL_neurons
