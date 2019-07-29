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

__all__ = [
    "plot_histogram",
    "plot_cutoff_count_histogram",
    "plot_class_histogram",
    "plot_probabilities",
    "plot_learning_curves",
    "plot_separate_learning_curves",
    "plot_accuracy_evolution",
    "plot_kl_divergence_evolution",
    "plot_centroid_probabilities_evolution",
    "plot_centroid_means_evolution",
    "plot_centroid_covariance_matrices_evolution",
    "plot_matrix",
    "plot_correlation_matrix",
    "plot_heat_map",
    "plot_values",
    "plot_variable_correlations",
    "plot_variable_label_correlations",
    "plot_series",
    "plot_profile_comparison",
    "save_figure"
]

from scvae.analyses.figures.histograms import (
    plot_histogram, plot_cutoff_count_histogram,
    plot_class_histogram, plot_probabilities
)
from scvae.analyses.figures.learning_curves import (
    plot_learning_curves, plot_separate_learning_curves,
    plot_accuracy_evolution, plot_kl_divergence_evolution,
    plot_centroid_probabilities_evolution, plot_centroid_means_evolution,
    plot_centroid_covariance_matrices_evolution,
)
from scvae.analyses.figures.matrices import (
    plot_matrix, plot_correlation_matrix, plot_heat_map)
from scvae.analyses.figures.saving import save_figure
from scvae.analyses.figures.scatter import (
    plot_values, plot_variable_correlations, plot_variable_label_correlations
)
from scvae.analyses.figures.series import plot_series, plot_profile_comparison
from scvae.analyses.figures.style import reset_plot_look

reset_plot_look()
