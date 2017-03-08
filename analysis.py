#!/usr/bin/env python3

import numpy
from sklearn.decomposition import PCA

from tensorflow.python.summary import event_multiplexer

from matplotlib import pyplot
import seaborn

import os

from time import time

from pprint import pprint

palette = seaborn.color_palette('Set2', 8)
seaborn.set(style='ticks', palette = palette)

figure_extension = ".png"

pyplot.rcParams.update({'figure.max_open_warning': 0})

def analyseData(data_sets, results_directory = "results"):
    
    printSummaryStatistics([
        statistics(data_set.counts, data_set.kind, tolerance = 0.5)
        for data_set in data_sets
    ])

def analyseModel(model, results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, model.name)
    
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        model.log_directory)
    multiplexer.Reload()
    
    # Learning curves
    
    print("Plotting learning curves.")
    
    data_set_kinds = ["training", "validation"]
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
    
    figure, name = plotLearningCurves(learning_curve_sets)
    saveFigure(figure, name, results_directory)
    
    print()

def analyseResults(x_test, x_tilde_test, z_test,
    model, results_directory = "results", intensive_calculations = False):
    
    # Setup
    
    results_directory = os.path.join(results_directory, model.name)
    
    M = x_test.number_of_examples
    
    # Metrics
    
    print("Calculating metrics.")
    metrics_time_start = time()
    
    x_statistics = [
            statistics(data_set.counts, data_set.version, tolerance = 0.5)
            for data_set in [x_test, x_tilde_test]
    ]
    
    x_diff = x_test.counts - x_tilde_test.counts
    x_diff_abs = numpy.abs(x_diff)
    x_diff_abs_mean = x_diff_abs.mean()
    x_diff_abs_std = x_diff_abs.std()
    
    x_log_ratio = numpy.log((x_test.counts + 1) / (x_tilde_test.counts + 1))
    x_log_ratio_abs = numpy.abs(x_log_ratio)
    x_log_ratio_abs_mean = x_log_ratio_abs.mean()
    x_log_ratio_abs_std = x_log_ratio_abs.std()
    
    metrics_duration = time() - metrics_time_start
    print("Metrics calculated ({:.3g} s).".format(metrics_duration))
    
    print()
    
    printSummaryStatistics(x_statistics)
    
    print()
    
    print("Differences: mean: {:.5g}, std: {:.5g}.".format(
        x_diff_abs_mean, x_diff_abs_std))
    print("log-ratios:  mean: {:.5g}, std: {:.5g}.".format(
        x_log_ratio_abs_mean, x_log_ratio_abs_std))
    
    print()
    
    # Profile comparisons
    
    print("Plotting profile comparisons.")
    profile_comparisons_time_start = time()
    
    subset = numpy.random.randint(M, size = 10)
    
    for j, i in enumerate(subset):
        
        figure, name = plotProfileComparison(
            x_test.counts[i], x_tilde_test.counts[i],
            x_name = "Genes", y_name = "Counts", scale = "log",
            title = str(x_test.cells[i]), name = str(j)
        )
        saveFigure(figure, name, results_directory)
    
    profile_comparisons_duration = time() - profile_comparisons_time_start
    print("Profile comparisons plotted and saved ({:.3g} s)".format(
        profile_comparisons_duration))
    
    print()
    
    # Heat maps
    
    if intensive_calculations:
        print("Plotting heat maps.")
        
        # Difference
        
        heat_maps_time_start = time()
        
        figure, name = plotHeatMap(x_diff, x_name = "Cell", y_name = "Gene",
            center = 0, name = "difference")
        saveFigure(figure, name, results_directory)
        
        heat_maps_duration = time() - heat_maps_time_start
        print("    Difference heat map for plotted and saved ({:.3g} s)" \
            .format(heat_maps_duration))
        
        # log-ratios
        
        heat_maps_time_start = time()
        
        figure, name = plotHeatMap(x_log_ratio, x_name = "Cell", y_name = "Gene",
            center = 0, name = "log_ratio")
        saveFigure(figure, name, results_directory)
        
        heat_maps_duration = time() - heat_maps_time_start
        print("    log-ratio heat map for plotted and saved ({:.3g} s)" \
            .format(heat_maps_duration))
        
        print()
    
    # Latent space
    
    print("Plotting latent space.")
    latent_space_time_start = time()
    
    figure, name = plotLatentSpace(z_test)
    saveFigure(figure, name, results_directory)
    
    latent_space_duration = time() - latent_space_time_start
    print("Latent space plotted and saved ({:.3g} s)".format(
        latent_space_duration))

def statistics(data_set, name = "", tolerance = 1e-3):
    
    x_mean = data_set.mean()
    x_std  = data_set.std()
    x_min  = data_set.min()
    x_max  = data_set.max()
    
    x_dispersion = x_mean / x_std
    
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

def printSummaryStatistics(statistics_sets):
    
    if type(statistics_sets) != list:
        statistics_sets = [statistics_sets]
    
    name_width = 0
    
    for statistics_set in statistics_sets:
        name_width = max(len(statistics_set["name"]), name_width)
    
    print("  ".join(["{:{}}".format("Data set", name_width),
        " mean ", "std. dev. ", "dispersion",
        " minimum ", " maximum ","sparsity"]))
    
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
        
        print("  ".join(string_parts))

def plotLearningCurves(curves, name = None):
    
    figure_name = "learning_curves"
    
    if name:
        figure_name = figure_name + "_" + name
    
    figure, (axis_1, axis_2) = pyplot.subplots(2, sharex = True, figsize = (6.4, 9.6))


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
            epochs = numpy.arange(len(curve)) + 1
            label = curve_name + " ({} set)".format(curve_set_name)
            axis.plot(curve, color = colour, linestyle = line_style, label = label)
    
    handles, labels = axis_1.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    
    axis_1.legend(handles, labels, loc = "best")
    axis_2.legend(loc = "best")
    
    axis_2.set_xlabel("Epoch")
    
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
    axis.plot(x, original_series[sort_indices], label = 'Original', zorder = 1)
    axis.plot(x, reconstructed_series[sort_indices], label = 'Reconstruction',
        zorder = 0)
    
    axis.legend()
    
    if title:
        axis.set_title(title)
    
    axis.set_xscale(scale)
    axis.set_yscale(scale)
    
    axis.set_xlabel(x_name + " sorted by original " + y_name.lower())
    axis.set_ylabel(y_name)
    
    return figure, figure_name

def plotHeatMap(data_set, x_name, y_name, center = None, name = None):
    
    figure_name = "heat_map"
    
    if name:
        figure_name = figure_name + "_" + name
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    seaborn.heatmap(data_set.T, xticklabels = False, yticklabels = False,
        cbar = True, square = True, center = center, ax = axis)
    
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)
    
    return figure, figure_name

def plotLatentSpace(latent_set, name = None):
    
    figure_name = "latent_space"
    
    if name:
        figure_name = figure_name + "_" + name
    
    M, L = latent_set.shape
    
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    
    if L > 2:
        pca = PCA(n_components = 2)
        pca.fit(latent_set)
        latent_set = pca.transform(latent_set)
        
        axis.set_xlabel("PC 1")
        axis.set_ylabel("PC 2")
    elif L == 2:
        axis.set_xlabel("$z_1$")
        axis.set_ylabel("$z_2$")
    
    axis.scatter(latent_set[:, 0], latent_set[:, 1])
    
    return figure, figure_name

def saveFigure(figure, figure_name, results_directory, no_spine = True):
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    if no_spine:
        seaborn.despine()
    
    figure_path = os.path.join(results_directory, figure_name) + figure_extension
    figure.savefig(figure_path, bbox_inches = 'tight')
