#!/usr/bin/env python3

import os

import numpy

from matplotlib import pyplot
import seaborn

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
    pass

def analyseResults(x_test, x_tilde_test, z_test,
    model, results_directory = "results"):
    
    # Setup
    
    results_directory = os.path.join(results_directory, model.name)
    
    M = x_test.number_of_examples
    
    # Metrics
    
    printSummaryStatistics([
        statistics(data_set.counts, data_set.version, tolerance = 0.5)
        for data_set in [x_test, x_tilde_test]
    ])
    
    print()
    
    # Profile comparisons
    
    print("Creating profile comparisons.")
    
    subset = numpy.random.randint(M, size = 10)
    
    for j, i in enumerate(subset):
        
        figure, name = plotProfileComparison(
            x_test.counts[i], x_tilde_test.counts[i],
            x_name = "Genes", y_name = "Counts", scale = "log",
            title = str(x_test.cells[i]), name = str(j)
        )
        saveFigure(figure, name, results_directory)

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

def plotProfileComparison(original_series, reconstructed_series, 
    x_name, y_name, scale = "linear", title = None, name = None):
    
    figure_name = "profile_comparison"
    
    if name:
        figure_name = name + "_" + figure_name
    
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

def saveFigure(figure, figure_name, results_directory, no_spine = True):
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    if no_spine:
        seaborn.despine()
    
    figure_path = os.path.join(results_directory, figure_name) + figure_extension
    figure.savefig(figure_path, bbox_inches = 'tight')
