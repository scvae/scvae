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

import os

import numpy
from matplotlib import pyplot

from scvae.defaults import defaults
from scvae.utilities import normalise_string

FIGURE_EXTENSION = ".png"

PUBLICATION_FIGURE_EXTENSION = ".tiff"
PUBLICATION_DPI = 350
PUBLICATION_COPIES = {
    "full_width": {
        "figure_width": 17.8 / 2.54
    },
    "three_quarters_width": {
        "figure_width": numpy.mean([8.6, 17.8]) / 2.54
    },
    "half_width": {
        "figure_width": 8.6 / 2.54
    }
}


def build_figure_name(base_name, other_names=None):

    if isinstance(base_name, list):
        if not other_names:
            other_names = []
        other_names.extend(base_name[1:])
        base_name = normalise_string(base_name[0])

    figure_name = base_name

    if other_names:
        if not isinstance(other_names, list):
            other_names = str(other_names)
            other_names = [other_names]
        else:
            other_names = [
                str(name) for name in other_names if name is not None]
        figure_name += "-" + "-".join(map(normalise_string, other_names))

    return figure_name


def save_figure(figure, name=None, options=None, directory=None):

    if name is None:
        name = "figure"

    if options is None:
        options = defaults["analyses"]["export_options"]

    if directory is None:
        directory = defaults["analyses"]["directory"]

    if not os.path.exists(directory):
        os.makedirs(directory)

    figure_path_base = os.path.join(directory, name)

    default_tight_layout = True
    tight_layout = default_tight_layout

    figure_width, figure_height = figure.get_size_inches()
    aspect_ratio = figure_width / figure_height

    figure.set_tight_layout(tight_layout)
    _adjust_figure_for_legend(figure)

    figure_path = figure_path_base + FIGURE_EXTENSION
    figure.savefig(figure_path)

    if "publication" in options:

        figure.set_tight_layout(default_tight_layout)
        figure.set_dpi(PUBLICATION_DPI)

        local_publication_copies = PUBLICATION_COPIES.copy()
        local_publication_copies["standard"] = {
            "figure_width": figure_width
        }

        for copy_name, copy_properties in local_publication_copies.items():

            figure.set_size_inches(figure_width, figure_height)

            publication_figure_width = copy_properties["figure_width"]
            publication_figure_height = (
                publication_figure_width / aspect_ratio)

            figure.set_size_inches(publication_figure_width,
                                   publication_figure_height)
            _adjust_figure_for_legend(figure)

            figure_path = "-".join([
                figure_path_base, "publication", copy_name
            ]) + PUBLICATION_FIGURE_EXTENSION

            figure.savefig(figure_path)

    pyplot.close(figure)


def _adjust_figure_for_legend(figure):

    for axis in figure.get_axes():
        legend = axis.get_legend()

        if legend and _legend_is_above_axis(legend, axis):

            renderer = figure.canvas.get_renderer()
            figure.draw(renderer=renderer)

            legend_size = legend.get_window_extent()
            legend_height_in_inches = legend_size.height / figure.get_dpi()

            figure_width, figure_height = figure.get_size_inches()
            figure_height += legend_height_in_inches
            figure.set_size_inches(figure_width, figure_height)


def _legend_is_above_axis(legend, axis):

    legend_bottom_vertical_position_relative_to_axis = (
        legend.get_bbox_to_anchor().transformed(
            axis.transAxes.inverted()).ymin)

    if legend_bottom_vertical_position_relative_to_axis >= 1:
        legend_is_above_axis = True
    else:
        legend_is_above_axis = False

    return legend_is_above_axis
