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

import numpy
import seaborn
from matplotlib import pyplot

from scvae.analyses.figures import saving, style


def plot_elbo_heat_map(data_frame, x_label, y_label, z_label=None,
                       z_symbol=None, z_min=None, z_max=None, name=None):

    figure_name = saving.build_figure_name("ELBO_heat_map", name)
    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)

    if not z_min:
        z_min = data_frame.values.min()
    if not z_max:
        z_max = data_frame.values.max()

    if z_symbol:
        z_label = "$" + z_symbol + "$"

    cbar_dict = {}
    if z_label:
        cbar_dict["label"] = z_label

    seaborn.set(style="white")
    seaborn.heatmap(
        data_frame,
        vmin=z_min, vmax=z_max,
        xticklabels=True, yticklabels=True,
        cbar=True, cbar_kws=cbar_dict,
        annot=True,
        fmt="-.6g",
        square=False,
        ax=axis
    )
    style.reset_plot_look()

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    return figure, figure_name


def plot_correlations(correlation_sets, x_key, y_key,
                      x_label=None, y_label=None, name=None):

    figure_name = saving.build_figure_name("correlations", name)

    if not isinstance(correlation_sets, dict):
        correlation_sets = {"correlations": correlation_sets}

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    for correlation_set_name, correlation_set in correlation_sets.items():
        axis.scatter(
            correlation_set[x_key], correlation_set[y_key],
            label=correlation_set_name
        )

    if len(correlation_sets) > 1:
        axis.legend(loc="best")

    return figure, figure_name


def plot_model_metrics(metrics_sets, key, label=None,
                       primary_differentiator_key=None,
                       primary_differentiator_order=None,
                       secondary_differentiator_key=None,
                       secondary_differentiator_order=None,
                       palette=None, marker_styles=None, name=None):

    figure_name = saving.build_figure_name("model_metrics", name)

    if not isinstance(metrics_sets, list):
        metrics_sets = [metrics_sets]

    if not palette:
        palette = style.STANDARD_PALETTE.copy()

    figure = pyplot.figure()
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.set_xlabel(primary_differentiator_key.capitalize() + "s")
    axis.set_ylabel(label)

    x_positions = {}
    x_offsets = {}
    colours = {}

    x_gap = 3
    x_scale = len(secondary_differentiator_order) - 1 + 2 * x_gap

    for metrics_set in metrics_sets:

        y = numpy.array(metrics_set[key])

        if y.dtype == "object":
            continue

        y_mean = y.mean()
        y_ddof = 1 if y.size > 1 else 0
        y_sd = y.std(ddof=y_ddof)

        x_position_key = metrics_set[primary_differentiator_key]
        if x_position_key in x_positions:
            x_position = x_positions[x_position_key]
        else:
            try:
                index = primary_differentiator_order.index(x_position_key)
                x_position = index
            except (ValueError, IndexError):
                x_position = 0
            x_positions[x_position_key] = x_position

        x_offset_key = metrics_set[secondary_differentiator_key]
        if x_offset_key in x_offsets:
            x_offset = x_offsets[x_offset_key]
        else:
            try:
                index = secondary_differentiator_order.index(x_offset_key)
                x_offset = (index + x_gap - x_scale / 2) / x_scale
            except (ValueError, IndexError):
                x_offset = 0
            x_offsets[x_offset_key] = x_offset

        x = x_position + x_offset

        colour_key = x_offset_key
        if colour_key in colours:
            colour = colours[colour_key]
        else:
            try:
                index = secondary_differentiator_order.index(colour_key)
                colour = palette[index]
            except (ValueError, IndexError):
                colour = "black"
            colours[colour_key] = colour
            axis.errorbar(
                x=x,
                y=y_mean,
                yerr=y_sd,
                capsize=2,
                linestyle="",
                color=colour,
                label=colour_key
            )

        axis.errorbar(
            x=x,
            y=y_mean,
            yerr=y_sd,
            ecolor=colour,
            capsize=2,
            color=colour,
            marker="_",
            # markeredgecolor=darker_colour,
            # markersize=7
        )

    x_ticks = []
    x_tick_labels = []

    for model, x_position in x_positions.items():
        x_ticks.append(x_position)
        x_tick_labels.append(model)

    axis.set_xticks(x_ticks)
    axis.set_xticklabels(x_tick_labels)

    if len(metrics_sets) > 1:

        order = primary_differentiator_order + secondary_differentiator_order
        handles, labels = axis.get_legend_handles_labels()

        label_handles = {}

        for label, handle in zip(labels, handles):
            label_handles.setdefault(label, [])
            label_handles[label].append(handle)

        labels, handles = [], []

        for label, handle_set in label_handles.items():
            labels.append(label)
            handles.append(tuple(handle_set))

        labels, handles = zip(*sorted(
            zip(labels, handles),
            key=lambda l:
                [order.index(l[0]), l[0]] if l[0] in order
                else [len(order), l[0]]
        ))

        axis.legend(handles, labels, loc="best")

    return figure, figure_name


def plot_model_metric_sets(metrics_sets, x_key, y_key,
                           x_label=None, y_label=None,
                           primary_differentiator_key=None,
                           primary_differentiator_order=None,
                           secondary_differentiator_key=None,
                           secondary_differentiator_order=None,
                           special_cases=None, other_method_metrics=None,
                           palette=None, marker_styles=None, name=None):

    figure_name = saving.build_figure_name("model_metric_sets", name)

    if other_method_metrics:
        figure_name += "-other_methods"

    if not isinstance(metrics_sets, list):
        metrics_sets = [metrics_sets]

    if not palette:
        palette = style.STANDARD_PALETTE.copy()

    if not marker_styles:
        marker_styles = [
            "X",  # cross
            "s",  # square
            "D",  # diamond
            "o",  # circle
            "P",  # plus
            "^",  # upright triangle
            "p",  # pentagon
            "*",  # star
        ]

    figure = pyplot.figure(figsize=(9, 6))
    axis = figure.add_subplot(1, 1, 1)
    seaborn.despine()

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    colours = {}
    markers = {}

    for metrics_set in metrics_sets:

        x = numpy.array(metrics_set[x_key])
        y = numpy.array(metrics_set[y_key])

        if x.dtype == "object" or y.dtype == "object":
            continue

        x_mean = x.mean()
        x_ddof = 1 if x.size > 1 else 0
        x_sd = x.std(ddof=x_ddof)

        y_mean = y.mean()
        y_ddof = 1 if y.size > 1 else 0
        y_sd = y.std(ddof=y_ddof)

        colour_key = metrics_set[primary_differentiator_key]
        if colour_key in colours:
            colour = colours[colour_key]
        else:
            try:
                index = primary_differentiator_order.index(colour_key)
                colour = palette[index]
            except (ValueError, IndexError):
                colour = "black"
            colours[colour_key] = colour
            axis.errorbar(
                x=x_mean,
                y=y_mean,
                yerr=y_sd,
                xerr=x_sd,
                capsize=2,
                linestyle="",
                color=colour,
                label=colour_key,
                markersize=7
            )

        marker_key = metrics_set[secondary_differentiator_key]
        if marker_key in markers:
            marker = markers[marker_key]
        else:
            try:
                index = secondary_differentiator_order.index(marker_key)
                marker = marker_styles[index]
            except (ValueError, IndexError):
                marker = None
            markers[marker_key] = marker
            axis.errorbar(
                x_mean, y_mean,
                color="black", marker=marker, linestyle="none",
                label=marker_key
            )

        errorbar_colour = colour
        darker_colour = seaborn.dark_palette(colour, n_colors=4)[2]

        special_case_changes = special_cases.get(colour_key, {})
        special_case_changes.update(
            special_cases.get(marker_key, {})
        )

        for object_name, object_change in special_case_changes.items():

            if object_name == "errorbar_colour":
                if object_change == "darken":
                    errorbar_colour = darker_colour

        axis.errorbar(
            x=x_mean,
            y=y_mean,
            yerr=y_sd,
            xerr=x_sd,
            ecolor=errorbar_colour,
            capsize=2,
            color=colour,
            marker=marker,
            markeredgecolor=darker_colour,
            markersize=7
        )

    baseline_line_styles = [
        "dashed",
        "dotted",
        "dashdot",
        "solid"
    ]
    legend_outside = False

    if other_method_metrics:
        for method_name, metric_values in other_method_metrics.items():

            x_values = metric_values.get(x_key, None)
            y_values = metric_values.get(y_key, None)

            if not y_values:
                continue

            y = numpy.array(y_values)
            y_mean = y.mean()

            if y.shape[0] > 1:
                y_sd = y.std(ddof=1)
            else:
                y_sd = None

            if x_values:
                legend_outside = True

                x = numpy.array(x_values)
                x_mean = x.mean()

                if x.shape[0] > 1:
                    x_sd = x.std(ddof=1)
                else:
                    x_sd = None

                axis.errorbar(
                    x=x_mean,
                    y=y_mean,
                    yerr=y_sd,
                    xerr=x_sd,
                    ecolor=style.STANDARD_PALETTE[-1],
                    color=style.STANDARD_PALETTE[-1],
                    capsize=2,
                    linestyle="none",
                    label=method_name
                )

            else:
                line_style = baseline_line_styles.pop(0)

                axis.axhline(
                    y=y_mean,
                    color=style.STANDARD_PALETTE[-1],
                    linestyle=line_style,
                    label=method_name,
                    zorder=-1
                )

                if y_sd is not None:
                    axis.axhspan(
                        ymin=y_mean - y_sd,
                        ymax=y_mean + y_sd,
                        facecolor=style.STANDARD_PALETTE[-1],
                        alpha=0.1,
                        edgecolor=None,
                        label=method_name,
                        zorder=-2
                    )

    if len(metrics_sets) > 1:

        order = primary_differentiator_order + secondary_differentiator_order
        handles, labels = axis.get_legend_handles_labels()

        label_handles = {}

        for label, handle in zip(labels, handles):
            label_handles.setdefault(label, [])
            label_handles[label].append(handle)

        labels, handles = [], []

        for label, handle_set in label_handles.items():
            labels.append(label)
            handles.append(tuple(handle_set))

        labels, handles = zip(*sorted(
            zip(labels, handles),
            key=lambda l:
                [order.index(l[0]), l[0]] if l[0] in order
                else [len(order), l[0]]
        ))

        if legend_outside:
            legend_keywords = {
                "loc": 'center left',
                "bbox_to_anchor": (1.05, 0, .3, 1),
                "ncol": 1,
                # "mode": "expand",
                "borderaxespad": 0
            }
        else:
            legend_keywords = {"loc": "best"}

        axis.legend(handles, labels, **legend_keywords)

    return figure, figure_name
