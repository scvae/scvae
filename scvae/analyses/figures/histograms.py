# ======================================================================== #
#
# Copyright (c) 2017 - 2020 scVAE authors
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

import matplotlib.lines
import numpy
import seaborn
from matplotlib import pyplot

from scvae.analyses.figures import saving, style
from scvae.utilities import capitalise_string

MAXIMUM_NUMBER_OF_BINS_FOR_HISTOGRAMS = 20000


def plot_class_histogram(labels, class_names=None, class_palette=None,
                         normed=False, scale="linear", label_sorter=None,
                         name=None):

    figure_name = "histogram"

    if normed:
        figure_name += "-normed"

    figure_name += "-classes"

    figure_name = saving.build_figure_name(figure_name, name)

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    if class_names is None:
        class_names = numpy.unique(labels)

    n_classes = len(class_names)

    if not class_palette:
        index_palette = style.lighter_palette(n_classes)
        class_palette = {
            class_name: index_palette[i] for i, class_name in enumerate(sorted(
                class_names, key=label_sorter))
        }

    histogram = {
        class_name: {
            "index": i,
            "count": 0,
            "colour": class_palette[class_name]
        }
        for i, class_name in enumerate(sorted(
            class_names, key=label_sorter))
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

        axis.bar(index, count_or_frequecny, color=colour)

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
    pyplot.xticks(
        ticks=indices,
        labels=class_names,
        horizontalalignment=y_ticks_horizontal_alignment,
        rotation=y_ticks_rotation,
        rotation_mode=y_ticks_rotation_mode
    )

    axis.set_xlabel("Classes")

    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")

    return figure, figure_name


def plot_histogram(series, excess_zero_count=0, label=None, normed=False,
                   discrete=False, x_scale="linear", y_scale="linear",
                   colour=None, name=None):

    series = series.copy()

    figure_name = "histogram"

    if normed:
        figure_name += "-normed"

    figure_name = saving.build_figure_name(figure_name, name)

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    series_length = len(series) + excess_zero_count

    series_max = series.max()

    if discrete and series_max < MAXIMUM_NUMBER_OF_BINS_FOR_HISTOGRAMS:
        number_of_bins = int(numpy.ceil(series_max)) + 1
        bin_range = numpy.array((-0.5, series_max + 0.5))
    else:
        if series_max < MAXIMUM_NUMBER_OF_BINS_FOR_HISTOGRAMS:
            number_of_bins = "auto"
        else:
            number_of_bins = MAXIMUM_NUMBER_OF_BINS_FOR_HISTOGRAMS
        bin_range = numpy.array((series.min(), series_max))

    if colour is None:
        colour = style.STANDARD_PALETTE[0]

    if x_scale == "log":
        series += 1
        bin_range += 1
        label += " (shifted one)"
        figure_name += "-log_values"

    y_log = y_scale == "log"

    histogram, bin_edges = numpy.histogram(
        series,
        bins=number_of_bins,
        range=bin_range
    )

    histogram[0] += excess_zero_count

    width = bin_edges[1] - bin_edges[0]
    bin_centres = bin_edges[:-1] + width / 2

    if normed:
        histogram = histogram / series_length

    axis.bar(
        bin_centres,
        histogram,
        width=width,
        log=y_log,
        color=colour,
        alpha=0.4
    )

    axis.set_xscale(x_scale)
    axis.set_xlabel(capitalise_string(label))

    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")

    return figure, figure_name


def plot_cutoff_count_histogram(series, excess_zero_count=0, cutoff=None,
                                normed=False, scale="linear", colour=None,
                                name=None):

    series = series.copy()

    figure_name = "histogram"

    if normed:
        figure_name += "-normed"

    figure_name += "-counts"
    figure_name = saving.build_figure_name(figure_name, name)
    figure_name += "-cutoff-{}".format(cutoff)

    if not colour:
        colour = style.STANDARD_PALETTE[0]

    y_log = scale == "log"

    count_number = numpy.arange(cutoff + 1)
    # Array to count counts of a given count number
    count_number_count = numpy.zeros(cutoff + 1)

    for i in range(cutoff + 1):
        if count_number[i] < cutoff:
            c = (series == count_number[i]).sum()
        elif count_number[i] == cutoff:
            c = (series >= cutoff).sum()
        count_number_count[i] = c

    count_number_count[0] += excess_zero_count

    if normed:
        count_number_count /= count_number_count.sum()

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.bar(
        count_number,
        count_number_count,
        log=y_log,
        color=colour,
        alpha=0.4
    )

    axis.set_xlabel("Count bins")

    if normed:
        axis.set_ylabel("Frequency")
    else:
        axis.set_ylabel("Number of counts")

    return figure, figure_name


def plot_probabilities(posterior_probabilities, prior_probabilities,
                       x_label=None, y_label=None, palette=None,
                       uniform=False, name=None):

    figure_name = saving.build_figure_name("probabilities", name)

    if posterior_probabilities is None and prior_probabilities is None:
        raise ValueError("No posterior nor prior probabilities given.")

    n_centroids = 0
    if posterior_probabilities is not None:
        n_posterior_centroids = len(posterior_probabilities)
        n_centroids = max(n_centroids, n_posterior_centroids)
    if prior_probabilities is not None:
        n_prior_centroids = len(prior_probabilities)
        n_centroids = max(n_centroids, n_prior_centroids)

    if not x_label:
        x_label = "$k$"
    if not y_label:
        y_label = "$\\pi_k$"

    figure = pyplot.figure(figsize=(8, 6), dpi=80)
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    if not palette:
        palette = [style.STANDARD_PALETTE[0]] * n_centroids

    if posterior_probabilities is not None:
        for k in range(n_posterior_centroids):
            axis.bar(k, posterior_probabilities[k], color=palette[k])
        axis.set_ylabel("$\\pi_{\\phi}^k$")
        if prior_probabilities is not None:
            for k in range(n_posterior_centroids):
                axis.plot(
                    [k-0.4, k+0.4],
                    2 * [prior_probabilities[k]],
                    color="black",
                    linestyle="dashed"
                )
            prior_line = matplotlib.lines.Line2D(
                [], [],
                color="black",
                linestyle="dashed",
                label="$\\pi_{\\theta}^k$"
            )
            axis.legend(handles=[prior_line], loc="best", fontsize=18)
    elif prior_probabilities is not None:
        for k in range(n_prior_centroids):
            axis.bar(k, prior_probabilities[k], color=palette[k])
        axis.set_ylabel("$\\pi_{\\theta}^k$")

    axis.set_xlabel(x_label)

    return figure, figure_name
