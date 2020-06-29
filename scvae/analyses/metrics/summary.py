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

from scvae.data.sparse import sparsity
from scvae.data.utilities import standard_deviation

MAXIMUM_NUMBER_OF_VALUES_FOR_NORMAL_STATISTICS_COMPUTATION = 5e8


def summary_statistics(x, name="", tolerance=1e-3, skip_sparsity=False):

    batch_size = None

    if x.size > MAXIMUM_NUMBER_OF_VALUES_FOR_NORMAL_STATISTICS_COMPUTATION:
        batch_size = 1000

    x_mean = x.mean()
    x_std = standard_deviation(x, ddof=1, batch_size=batch_size)

    x_min = x.min()
    x_max = x.max()

    x_dispersion = x_std**2 / x_mean

    if skip_sparsity:
        x_sparsity = numpy.nan
    else:
        x_sparsity = sparsity(x, tolerance=tolerance, batch_size=batch_size)

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


def format_summary_statistics(statistics_sets, name="Data set"):

    if not isinstance(statistics_sets, list):
        statistics_sets = [statistics_sets]

    name_width = max(
        [len(name)]
        + [len(statistics_set["name"]) for statistics_set in statistics_sets]
    )

    table_heading = "  ".join([
        "{:{}}".format(name, name_width),
        " mean ", "std. dev. ", "dispersion",
        " minimum ", " maximum ", "sparsity"
    ])

    table_rows = [table_heading]

    for statistics_set in statistics_sets:
        table_row_parts = [
            "{:{}}".format(statistics_set["name"], name_width),
            "{:<9.5g}".format(statistics_set["mean"]),
            "{:<9.5g}".format(statistics_set["standard deviation"]),
            "{:<9.5g}".format(statistics_set["dispersion"]),
            "{:<11.5g}".format(statistics_set["minimum"]),
            "{:<11.5g}".format(statistics_set["maximum"]),
            "{:<7.5g}".format(statistics_set["sparsity"]),
        ]
        table_row = "  ".join(table_row_parts)
        table_rows.append(table_row)

    table = "\n".join(table_rows)

    return table
