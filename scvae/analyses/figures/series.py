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

import matplotlib.patches
import numpy
import scipy
import seaborn
from matplotlib import pyplot

from scvae.analyses.figures import saving, style
from scvae.utilities import normalise_string, capitalise_string


def plot_series(series, x_label, y_label, sort=False, scale="linear",
                bar=False, colour=None, name=None):

    figure_name = saving.build_figure_name("series", name)

    if not colour:
        colour = style.STANDARD_PALETTE[0]

    series_length = series.shape[0]

    x = numpy.linspace(0, series_length, series_length)

    y_log = scale == "log"

    if sort:
        # Sort descending
        series = numpy.sort(series)[::-1]
        x_label = "sorted " + x_label
        figure_name += "-sorted"

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    if bar:
        axis.bar(x, series, log=y_log, color=colour, alpha=0.4)
    else:
        axis.plot(x, series, color=colour)
        axis.set_yscale(scale)

    axis.set_xlabel(capitalise_string(x_label))
    axis.set_ylabel(capitalise_string(y_label))

    return figure, figure_name


def plot_profile_comparison(observed_series, expected_series,
                            expected_series_total_standard_deviations=None,
                            expected_series_explained_standard_deviations=None,
                            x_name="feature", y_name="value", sort=True,
                            sort_by="expected", sort_direction="ascending",
                            x_scale="linear", y_scale="linear", y_cutoff=None,
                            name=None):

    sort_by = normalise_string(sort_by)
    sort_direction = normalise_string(sort_direction)
    figure_name = saving.build_figure_name("profile_comparison", name)

    if scipy.sparse.issparse(observed_series):
        observed_series = observed_series.A.squeeze()

    if scipy.sparse.issparse(expected_series_total_standard_deviations):
        expected_series_total_standard_deviations = (
            expected_series_total_standard_deviations.A.squeeze())

    if scipy.sparse.issparse(expected_series_explained_standard_deviations):
        expected_series_explained_standard_deviations = (
            expected_series_explained_standard_deviations.A.squeeze())

    observed_colour = style.STANDARD_PALETTE[0]
    expected_palette = seaborn.light_palette(style.STANDARD_PALETTE[1], 5)

    expected_colour = expected_palette[-1]
    expected_total_standard_deviations_colour = expected_palette[1]
    expected_explained_standard_deviations_colour = expected_palette[3]

    if sort:
        x_label = "{}s sorted {} by {} {}s [sort index]".format(
            capitalise_string(x_name), sort_direction, sort_by, y_name.lower())
    else:
        x_label = "{}s [original index]".format(capitalise_string(x_name))
    y_label = capitalise_string(y_name) + "s"

    observed_label = "Observed"
    expected_label = "Expected"
    expected_total_standard_deviations_label = "Total standard deviation"
    expected_explained_standard_deviations_label = (
        "Explained standard deviation")

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
            raise ValueError(
                "Sort direction can either be ascending or descending.")
    else:
        sort_indices = slice(None)

    # Standard deviations
    if expected_series_total_standard_deviations is not None:
        with_total_standard_deviations = True
        expected_series_total_standard_deviations_lower = (
            expected_series - expected_series_total_standard_deviations)
        expected_series_total_standard_deviations_upper = (
            expected_series + expected_series_total_standard_deviations)
    else:
        with_total_standard_deviations = False

    if (expected_series_explained_standard_deviations is not None
            and expected_series_explained_standard_deviations.mean() > 0):
        with_explained_standard_deviations = True
        expected_series_explained_standard_deviations_lower = (
            expected_series - expected_series_explained_standard_deviations)
        expected_series_explained_standard_deviations_upper = (
            expected_series + expected_series_explained_standard_deviations)
    else:
        with_explained_standard_deviations = False

    # Figure
    if y_scale == "both":
        figure, axes = pyplot.subplots(nrows=2, sharex=True)
        figure.subplots_adjust(hspace=0.1)
        axis_upper = axes[0]
        axis_lower = axes[1]
        axis_upper.set_zorder = 1
        axis_lower.set_zorder = 0
    else:
        figure = pyplot.figure()
        axis = figure.add_subplot(1, 1, 1)
        axes = [axis]

    handles = []
    feature_indices = numpy.arange(len(observed_series)) + 1

    for i, axis in enumerate(axes):
        observed_plot, = axis.plot(
            feature_indices,
            observed_series[sort_indices],
            label=observed_label,
            color=observed_colour,
            marker=observed_marker,
            linestyle=observed_line_style,
            zorder=observed_z_order
        )
        if i == 0:
            handles.append(observed_plot)
        expected_plot, = axis.plot(
            feature_indices,
            expected_series[sort_indices],
            label=expected_label,
            color=expected_colour,
            marker=expected_marker,
            linestyle=expected_line_style,
            zorder=expected_z_order
        )
        if i == 0:
            handles.append(expected_plot)
        if with_total_standard_deviations:
            axis.fill_between(
                feature_indices,
                expected_series_total_standard_deviations_lower[sort_indices],
                expected_series_total_standard_deviations_upper[sort_indices],
                color=expected_total_standard_deviations_colour,
                zorder=0
            )
            expected_plot_standard_deviations_values = (
                matplotlib.patches.Patch(
                    label=expected_total_standard_deviations_label,
                    color=expected_total_standard_deviations_colour
                )
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
                color=expected_explained_standard_deviations_colour,
                zorder=1
            )
            expected_plot_standard_deviations_expectations = (
                matplotlib.patches.Patch(
                    label=expected_explained_standard_deviations_label,
                    color=expected_explained_standard_deviations_colour
                )
            )
            if i == 0:
                handles.append(expected_plot_standard_deviations_expectations)

    if y_scale == "both":
        axis_upper.legend(
            handles=handles,
            loc="best"
        )
        seaborn.despine(ax=axis_upper)
        seaborn.despine(ax=axis_lower)

        axis_upper.set_yscale("log", nonposy="clip")
        axis_lower.set_yscale("linear")
        figure.text(0.04, 0.5, y_label, va="center", rotation="vertical")

        axis_lower.set_xscale(x_scale)
        axis_lower.set_xlabel(x_label)

        y_upper_min, y_upper_max = axis_upper.get_ylim()
        y_lower_min, y_lower_max = axis_lower.get_ylim()
        axis_upper.set_ylim(y_cutoff, y_upper_max)

        y_lower_min = max(-1, y_lower_min)
        axis_lower.set_ylim(y_lower_min, y_cutoff)

    else:
        axis.legend(
            handles=handles,
            loc="best"
        )
        seaborn.despine()

        y_scale_arguments = {}
        if y_scale == "log":
            y_scale_arguments["nonposy"] = "clip"
        axis.set_yscale(y_scale, **y_scale_arguments)
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

    return figure, figure_name
