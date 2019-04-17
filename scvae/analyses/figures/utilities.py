# ======================================================================== #
#
# Copyright (c) 2017 - 2019 scVAE authors
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

from scvae.analyses.decomposition import (
    DECOMPOSITION_METHOD_NAMES,
    DECOMPOSITION_METHOD_LABEL
)
from scvae.utilities import normalise_string, proper_string


def _axis_label_for_symbol(symbol, coordinate=None, decomposition_method=None,
                           distribution=None, prefix="", suffix=""):

    if decomposition_method:
        decomposition_method = proper_string(
            normalise_string(decomposition_method),
            DECOMPOSITION_METHOD_NAMES
        )
        decomposition_label = DECOMPOSITION_METHOD_LABEL[decomposition_method]
    else:
        decomposition_label = ""

    if decomposition_label:
        decomposition_label = "\\mathrm{{{}}}".format(decomposition_label)

    if coordinate:
        coordinate_text = "{{{} {}}}".format(decomposition_label, coordinate)
    else:
        coordinate_text = ""

    if distribution == "prior":
        distribution_symbol = "\\theta"
    elif distribution == "posterior":
        distribution_symbol = "\\phi"
    else:
        distribution_symbol = ""

    if distribution_symbol and coordinate_text:
        distribution_position = "_"
        coordinate_position = "^"
    elif distribution_symbol and not coordinate_text:
        distribution_position = "_"
        coordinate_position = ""
    elif not distribution_symbol and coordinate_text:
        distribution_position = ""
        coordinate_position = "_"
    else:
        distribution_position = ""
        coordinate_position = ""

    if coordinate_position == "^":
        coordinate_text = "{{(" + coordinate_text + ")}}"
    elif coordinate_position == "_":
        coordinate_text = "{{" + coordinate_text + "}}"

    axis_label = "$" + "".join([
        prefix, symbol,
        distribution_position,
        distribution_symbol,
        coordinate_position,
        coordinate_text,
        suffix
    ]) + "$"

    return axis_label


def _covariance_matrix_as_ellipse(covariance_matrix, mean, colour,
                                  linestyle="solid", radius_stddev=1,
                                  label=None):

    eigenvalues, eigenvectors = numpy.linalg.eig(covariance_matrix)
    indices_sorted_ascending = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices_sorted_ascending]
    eigenvectors = eigenvectors[:, indices_sorted_ascending]

    lambda_1, lambda_2 = numpy.sqrt(eigenvalues)
    theta = numpy.degrees(numpy.arctan2(
        eigenvectors[1, 0], eigenvectors[0, 0]))

    ellipse_fill = matplotlib.patches.Ellipse(
        xy=mean,
        width=2 * radius_stddev * lambda_1,
        height=2 * radius_stddev * lambda_2,
        angle=theta,
        linewidth=2,
        linestyle=linestyle,
        facecolor="none",
        edgecolor=colour,
        label=label
    )
    ellipse_edge = matplotlib.patches.Ellipse(
        xy=mean,
        width=2 * radius_stddev * lambda_1,
        height=2 * radius_stddev * lambda_2,
        angle=theta,
        linewidth=3,
        linestyle=linestyle,
        facecolor="none",
        edgecolor="black",
        label=label
    )

    return ellipse_fill, ellipse_edge
