#!/usr/bin/env python3

import os

import numpy

from matplotlib import pyplot
import seaborn

palette = seaborn.color_palette('Set2', 8)
seaborn.set(style='ticks', palette = palette)

figure_extension = ".png"

pyplot.rcParams.update({'figure.max_open_warning': 0})

def analyseModel(log_directory, results_directory):
    pass

def analyseResults(test_set, reconstructed_test_set, latent_set, results_directory):
    
    plotProfileComparison(test_set.counts[0], reconstructed_test_set[0],
        x_label = "Genes sorted by original counts", y_label = "Counts",
        scale = "log", name = str(test_set.cells[0]), results_directory = results_directory)

def plotProfileComparison(original_series, reconstructed_series, x_label, y_label,
    scale = "linear", name = None, results_directory = ""):
    
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
    
    axis.set_xscale(scale)
    axis.set_yscale(scale)
    
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    
    saveFigure(figure, figure_name, results_directory)

def saveFigure(figure, figure_name, results_directory, no_spine = True):
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    if no_spine:
        seaborn.despine()
    
    figure_path = os.path.join(results_directory, figure_name) + figure_extension
    figure.savefig(figure_path, bbox_inches = 'tight')
