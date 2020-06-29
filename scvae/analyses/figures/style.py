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

import matplotlib
import seaborn

STANDARD_PALETTE = seaborn.color_palette("Set2", 8)
STANDARD_COLOUR_MAP = seaborn.cubehelix_palette(light=.95, as_cmap=True)
NEUTRAL_COLOUR = (0.7, 0.7, 0.7)

MAXIMUM_NUMBER_OF_EXAMPLES_FOR_LARGE_POINTS_IN_SCATTER_PLOTS = 1000
DEFAULT_SMALL_MARKER_SIZE_IN_SCATTER_PLOTS = 1
DEFAULT_LARGE_MARKER_SIZE_IN_SCATTER_PLOTS = 3
DEFAULT_MARKER_SIZE_IN_SCATTER_PLOTS = (
    DEFAULT_SMALL_MARKER_SIZE_IN_SCATTER_PLOTS)
DEFAULT_MARKER_SIZE_IN_LEGENDS = 4


def lighter_palette(n):
    return seaborn.husl_palette(n, l=.75)  # noqa: E741


def darker_palette(n):
    return seaborn.husl_palette(n, l=.55)  # noqa: E741


def legend_marker_scale_from_marker_size(marker_size):
    return DEFAULT_MARKER_SIZE_IN_LEGENDS / marker_size


def reset_plot_look():
    seaborn.set(
        context="paper",
        style="ticks",
        palette=STANDARD_PALETTE,
        color_codes=False,
        rc={
            "lines.markersize": DEFAULT_MARKER_SIZE_IN_SCATTER_PLOTS,
            "legend.markerscale": legend_marker_scale_from_marker_size(
                DEFAULT_MARKER_SIZE_IN_SCATTER_PLOTS
            ),
            "figure.dpi": 200
        }
    )


def _adjust_marker_size_for_scatter_plots(n_examples):

    if (n_examples
            <= MAXIMUM_NUMBER_OF_EXAMPLES_FOR_LARGE_POINTS_IN_SCATTER_PLOTS):
        marker_size = DEFAULT_LARGE_MARKER_SIZE_IN_SCATTER_PLOTS
    else:
        marker_size = DEFAULT_SMALL_MARKER_SIZE_IN_SCATTER_PLOTS

    matplotlib.rc(group="lines", markersize=marker_size)
    matplotlib.rc(
        group="legend",
        markerscale=legend_marker_scale_from_marker_size(marker_size)
    )
