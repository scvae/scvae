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

import copy
import gzip
import os
import pickle
import re
import statistics
import textwrap
from itertools import product
from string import ascii_uppercase

import numpy
import pandas
from scipy.stats import pearsonr

from scvae.analyses import figures
from scvae.analyses.figures.cross_model import (
    plot_correlations, plot_elbo_heat_map,
    plot_model_metrics, plot_model_metric_sets
)
from scvae.analyses.metrics import format_summary_statistics
from scvae.defaults import defaults
from scvae.utilities import (
    format_time,
    normalise_string, proper_string, capitalise_string,
    title, subtitle, heading, subheading
)

METRICS_BASENAME = "test-metrics"
PREDICTION_BASENAME = "test-prediction"

ZIPPED_PICKLE_EXTENSION = ".pkl.gz"
LOG_EXTENSION = ".log"

SORTED_COMPARISON_TABLE_COLUMN_NAMES = [
    "ID",
    "type",
    "likelihood",
    "sizes",
    "other",
    "clustering method",
    "runs",
    "version",
    "epochs",
    "ELBO",
    "adjusted Rand index",
    "adjusted mutual information",
    "silhouette score"
]

ABBREVIATIONS = {
    # Model specifications
    "ID": "#",
    "type": "T",
    "likelihood": "L",
    "sizes": "S",
    "other": "O",
    "clustering method": "CM",
    "runs": "R",
    "version": "V",
    "epochs": "E",
    # Model versions
    "end of training": "EOT",
    "optimal parameters": "OP",
    "early stopping": "ES",
    # Clustering metrics
    "adjusted Rand index": "ARI",
    "adjusted mutual information": "AMI",
    "silhouette score": "SS",
    # Data sets
    "superset": "sup"
}

CLUSTERING_METRICS = {
    "adjusted Rand index": {
        "kind": "supervised",
        "symbol": r"$R_\mathrm{adj}$"
    },
    "adjusted mutual information": {
        "kind": "supervised",
        "symbol": r"$\mathrm{AMI}$"
    },
    "silhouette score": {
        "kind": "unsupervised",
        "symbol": r"$s$"
    },
}

MODEL_TYPE_ORDER = [
    "VAE",
    "GMVAE",
    "FA"
]
LIKELIHOOD_DISRIBUTION_ORDER = [
    "P",
    "NB",
    "ZIP",
    "ZINB",
    "PCP",
    "CP"
]

FACTOR_ANALYSIS_MODEL_TYPE = "VAE(G, g: LFM)"
FACTOR_ANALYSIS_MODEL_TYPE_ALIAS = "FA"

BASELINE_METHOD_TYPE = "Factor Analysis + k-means"
BASELINE_METHOD_TYPE_ALIAS = "LFA-kM"

OTHER_METHOD_NAMES = {
    "k-means": ["k_means", "kmeans"],
    "Seurat": ["seurat"],
    "Factor Analysis": ["factor_analysis"]
}
OTHER_METHOD_SPECIFICATIONS = {
    "Seurat": {
        "directory name": "Seurat",
        "training set name": "full",
        "evaluation set name": "full"
    },
}

DATA_SET_NAME_REPLACEMENTS = {
    "10x": "10x",
    "10x_20k": "10x (20k samples)",
    "10x_arc_lira": "10x ARC LIRA",
    "development": "Development",
    r"dimm_sc_10x_(\w+)": lambda match: "3′ ({})".format(match.group(1)),
    "gtex": "GTEx",
    r"mnist_(\w+)": lambda match: "MNIST ({})".format(match.group(1)),
    r"sample_?(sparse)?": (
        lambda match: "Sample"
        if len(match.groups()) == 1
        else "Sample ({})".format(match.group(1))
    ),
    "tcga_kallisto": "TCGA (Kallisto)"
}
SPLIT_REPLACEMENTS = {
    r"split-(\w+)_(0\.\d+)": lambda match:
        "{} split ({:.3g} %)".format(
            match.group(1),
            100 * float(match.group(2))
        )
}
FEATURE_REPLACEMENTS = {
    "features_mapped": "feature mapping",
    r"keep_variances_above_([\d.]+)": lambda match:
        "features with variance above {}".format(
            int(float(match.group(1)))),
    r"keep_highest_variances_([\d.]+)": lambda match:
        "{} most varying features".format(int(float(match.group(1))))
}
EXAMPLE_FEATURE_REPLACEMENTS = {
    "macosko": "Macosko",
    "remove_zeros": "examples with only zeros removed",
    r"remove_count_sum_above_([\d.]+)": lambda match:
        "examples with count sum above {} removed".format(
            int(float(match.group(1))))
}
EXAMPLE_REPLACEMENTS = {
    r"keep_(\w+)": (
        lambda match: "{} examples".format(match.group(1).replace("_", ", "))),
    r"remove_(\w+)": (
        lambda match:
        "{} examples removed".format(match.group(1).replace("_", ", "))
    ),
    "excluded_classes": "excluded classes removed",
}
MODEL_REORDER_REPLACEMENTS = {
    r"(-sum)(-l_\d+-h_[\d_]+)": lambda match: "".join(reversed(match.groups()))
}
MODEL_REPLACEMENTS = {
    r"GMVAE/gaussian_mixture-c_(\d+)-?p?_?(\w+)?": lambda match:
        "GMVAE({})".format(match.group(1))
        if not match.group(2)
        else "GMVAE({}; {})".format(*match.groups()),
    r"VAE/([\w-]+)": lambda match: "VAE({})".format(match.group(1)),
    "-parameterised": ", PLP",
    r"-ia_(\w+)-ga_(\w+)": lambda match: ", {}".format(match.group(1))
        if match.group(1) == match.group(2)
        else ", i: {}, g: {}".format(*match.group(1, 2))
}
SECONDARY_MODEL_REPLACEMENTS = {
    r"gaussian_mixture-c_(\d+)": lambda match: "GM({})".format(match.group(1)),
    r"-ia_(\w+)": lambda match: ", i: {}".format(match.group(1)),
    r"-ga_(\w+)": lambda match: ", g: {}".format(match.group(1))
}
DISTRIBUTION_MODIFICATION_REPLACEMENTS = {
    "constrained_poisson": "CP",
    "zero_inflated_": "ZI",
    r"/(\w+)-k_(\d+)": lambda match: "/PC{}({})".format(
        match.group(1), match.group(2))
}
DISTRIBUTION_REPLACEMENTS = {
    "gaussian": "G",
    "bernoulli": "B",
    "poisson": "P",
    "negative_binomial": "NB",
    "lomax": "L",
    "pareto": "Pa",
}
NETWORK_REPLACEMENTS = {
    r"l_(\d+)-h_([\d_]+)": lambda match: "{}×{}".format(
        match.group(2).replace("_", "×"),
        match.group(1)
    )
}
SAMPLE_REPLACEMENTS = {
    r"-mc_(\d+)": (
        lambda match: "" if int(match.group(1)) == 1
        else "-{} MC samples".format(match.groups(1))
    ),
    r"-iw_(\d+)": (
        lambda match: "" if int(match.group(1)) == 1
        else "-{} IW samples".format(match.groups(1))
    )
}
MISCELLANEOUS_MODEL_REPLACEMENTS = {
    "sum": "CS",
    "-kl-": "-",
    "bn": "BN",
    "bc": "BC",
    r"dropout_([\d._]+)": lambda match: "dropout: {}".format(
        match.group(1).replace("_", ", ")),
    r"wu_(\d+)": lambda match: "WU({})".format(match.group(1)),
    r"klw_([\d.]+)": lambda match: "KLW: {}".format(match.group(1))
}
INBUILT_CLUSTERING_REPLACEMENTS = {
    r"model \(\d+ classes\)": "M"
}
CLUSTERING_METHOD_REPLACEMENTS = {
    r"(\w+) \((\d+) classes\)": r"\1(\2)",
    r"(\w+) \((\d+) components\)": r"\1(\2)",
    "k-means": "kM",
    "t-SNE": "tSNE"
}


def cross_analyse(analyses_directory,
                  data_set_included_strings=None,
                  data_set_excluded_strings=None,
                  model_included_strings=None,
                  model_excluded_strings=None,
                  prediction_included_strings=None,
                  prediction_excluded_strings=None,
                  additional_other_option=None,
                  no_prediction_methods_for_gmvae_in_plots=False,
                  epoch_cut_off=None,
                  other_methods=None,
                  export_options=None,
                  log_summary=None):

    if log_summary is None:
        log_summary = defaults["cross_analysis"]["log_summary"]

    search_strings_sets = [
        {
            "strings": data_set_included_strings,
            "kind": "data set",
            "inclusiveness": True,
            "abbreviation": "d"
        },
        {
            "strings": data_set_excluded_strings,
            "kind": "data set",
            "inclusiveness": False,
            "abbreviation": "D"
        },
        {
            "strings": model_included_strings,
            "kind": "model",
            "inclusiveness": True,
            "abbreviation": "m"
        },
        {
            "strings": model_excluded_strings,
            "kind": "model",
            "inclusiveness": False,
            "abbreviation": "M"
        },
        {
            "strings": prediction_included_strings,
            "kind": "prediction method",
            "inclusiveness": True,
            "abbreviation": "p"
        },
        {
            "strings": prediction_excluded_strings,
            "kind": "prediction method",
            "inclusiveness": False,
            "abbreviation": "P"
        }
    ]

    # Directory and filenames
    analyses_directory = os.path.normpath(analyses_directory) + os.sep
    cross_analysis_name_parts = []

    for search_strings_set in search_strings_sets:
        search_strings = search_strings_set["strings"]
        search_abbreviation = search_strings_set["abbreviation"]

        if search_strings:
            cross_analysis_name_parts.append("{}_{}".format(
                search_abbreviation,
                "_".join(s.replace("/", "") for s in search_strings)
            ))

    if additional_other_option:
        cross_analysis_name_parts.append("a_{}".format(
            additional_other_option))

    if epoch_cut_off:
        cross_analysis_name_parts.append("e_{}".format(epoch_cut_off))

    if cross_analysis_name_parts:
        cross_analysis_name = "-".join(cross_analysis_name_parts)
    else:
        cross_analysis_name = "all"

    cross_analysis_directory = os.path.join(
        analyses_directory,
        "cross_analysis",
        cross_analysis_name
    )

    if log_summary:
        log_filename = cross_analysis_name + LOG_EXTENSION
        log_path = os.path.join(cross_analysis_directory, log_filename)

    # Print filtering
    explanation_string_parts = []

    for search_strings_set in search_strings_sets:
        search_strings = search_strings_set["strings"]
        search_kind = search_strings_set["kind"]
        search_inclusiveness = search_strings_set["inclusiveness"]

        if search_strings:
            explanation_string_parts.append("{} {}s with: {}.".format(
                "Only including" if search_inclusiveness else "Excluding",
                search_kind,
                ", ".join(search_strings)
            ))

    if additional_other_option:
        explanation_string_parts.append(
            "Additional other option to use for models in model metrics "
            "plot: {}.".format(additional_other_option)
        )

    if epoch_cut_off:
        explanation_string_parts.append(
            "Excluding models trained for longer than {} epochs".format(
                epoch_cut_off
            )
        )

    explanation_string = "\n".join(explanation_string_parts)

    print(explanation_string)
    print()

    if log_summary:
        log_string_parts = [explanation_string + "\n"]

    metrics_sets = _metrics_sets_in_analyses_directory(
        analyses_directory,
        data_set_included_strings,
        data_set_excluded_strings,
        model_included_strings,
        model_excluded_strings
    )

    model_ids = _generate_model_ids()

    for data_set_path, models in metrics_sets.items():

        data_set_title = _data_set_title_from_data_set_name(data_set_path)
        print(title(data_set_title))

        if log_summary:
            log_string_parts.append(title(data_set_title, plain=True))

        summary_metrics_sets = {}
        correlation_sets = {}

        for model_name, runs in models.items():

            model_title = _model_title_from_model_name(model_name)
            model_id = next(model_ids)
            model_id_string = "ID: {}\n".format(model_id)
            print(subtitle(model_title))
            print(model_id_string)

            if log_summary:
                log_string_parts.append(
                    subtitle(model_title, plain=True)
                )
                log_string_parts.append(model_id_string)

            metrics_results = _parse_metrics_for_runs_and_versions_of_model(
                runs=runs,
                log_summary=log_summary,
                prediction_included_strings=prediction_included_strings,
                prediction_excluded_strings=prediction_excluded_strings,
                epoch_cut_off=epoch_cut_off
            )

            if log_summary and "log_string_parts" in metrics_results:
                log_string_parts.extend(
                    metrics_results["log_string_parts"]
                )

            # Update summary metrics sets
            model_summary_metrics_sets = metrics_results.get(
                "summary_metrics_sets", [])

            for model_summary_metrics_set in model_summary_metrics_sets:

                clustering_method = model_summary_metrics_set.get(
                    "clustering method", None
                )

                if clustering_method:
                    clustering_method_title = (
                        _clustering_method_title_from_clustering_method_name(
                            model_summary_metrics_set["clustering method"]
                        )
                    )
                else:
                    clustering_method_title = "---"

                set_title = "; ".join([
                    model_title,
                    clustering_method_title,
                    model_summary_metrics_set["runs"],
                    model_summary_metrics_set["version"]
                ])

                summary_metrics_set = {"ID": model_id}

                for field_name, field_value in (
                        model_summary_metrics_set.items()):
                    summary_metrics_set[field_name] = field_value

                summary_metrics_sets[set_title] = summary_metrics_set

            # Update correlation sets
            model_correlation_sets = metrics_results.get(
                "correlation_sets", [])

            for set_name, set_metrics in model_correlation_sets.items():
                if set_name in correlation_sets:
                    for metric_name, metric_values in set_metrics.items():
                        correlation_sets[set_name][metric_name].extend(
                            metric_values)
                else:
                    correlation_sets[set_name] = {}
                    for metric_name, metric_values in set_metrics.items():
                        correlation_sets[set_name][metric_name] = (
                            metric_values)

        if len(summary_metrics_sets) <= 1:
            continue

        if correlation_sets:
            correlation_string_parts = []
            correlation_table = {}

            for set_name in correlation_sets:
                if len(correlation_sets[set_name]["ELBO"]) < 2:
                    continue
                correlation_coefficient, _ = pearsonr(
                    correlation_sets[set_name]["ELBO"],
                    correlation_sets[set_name]["clustering metric"]
                )
                correlation_table[set_name] = {
                    "r": correlation_coefficient
                }

            if correlation_table:
                correlation_table = pandas.DataFrame(correlation_table).T
                correlation_string_parts.append(str(correlation_table))

            correlation_string = "\n".join(correlation_string_parts)

            print(subtitle("Metric correlations"))
            print(correlation_string + "\n")

            if log_summary:
                log_string_parts.append(
                    subtitle("Metric correlations", plain=True))
                log_string_parts.append(correlation_string + "\n")

            print("Plotting correlations.")
            figure, figure_name = plot_correlations(
                correlation_sets,
                x_key="ELBO",
                y_key="clustering metric",
                x_label="$\\mathcal{L}$",
                y_label="",
                name=data_set_path.replace(os.sep, "-")
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=cross_analysis_directory
            )
            print()

        # Comparisons
        if other_methods:
            set_other_method_metrics = _metrics_for_other_methods(
                data_set_directory=os.path.join(
                    analyses_directory,
                    data_set_path
                ),
                other_methods=other_methods,
                prediction_included_strings=prediction_included_strings,
                prediction_excluded_strings=prediction_excluded_strings
            )
        else:
            set_other_method_metrics = None

        model_field_names = set()

        for model_title in summary_metrics_sets:
            model_title_parts = model_title.split("; ")
            summary_metrics_sets[model_title].update({
                "type": model_title_parts.pop(0),
                "likelihood": model_title_parts.pop(0),
                "sizes": model_title_parts.pop(0),
                "version": ABBREVIATIONS[model_title_parts.pop(-1)],
                "runs": (
                    model_title_parts.pop(-1)
                    .replace("default run", "D")
                    .replace(" runs", "")
                ),
                "clustering method": model_title_parts.pop(-1),
                "other": "; ".join(model_title_parts)
            })
            model_field_names.update(
                summary_metrics_sets[model_title].keys()
            )

        model_field_names = sorted(
            list(model_field_names),
            key=_comparison_table_column_sorter
        )

        for summary_metrics_set in summary_metrics_sets.values():
            for field_name in model_field_names:
                summary_metrics_set.setdefault(field_name, None)

        # Network architecture
        architecture_lower_bounds = {}
        architecture_versions = {}

        for model_fields in summary_metrics_sets.values():
            type_match = model_fields["type"] == "VAE(G)"
            likelihood_match = model_fields["likelihood"] == "NB"
            other_match = model_fields["other"] == "BN"
            run_match = model_fields["runs"] == "D"
            if type_match and likelihood_match and other_match and run_match:

                lower_bound = model_fields["ELBO"]
                architecture = model_fields["sizes"]
                hidden_sizes, latent_size = architecture.rsplit(
                    "×", maxsplit=1)
                version = {
                    "version": model_fields["version"],
                    "epoch_number": model_fields["epochs"],
                }

                architecture_lower_bounds.setdefault(latent_size, {})
                architecture_versions.setdefault(latent_size, {})

                if hidden_sizes not in architecture_lower_bounds[latent_size]:
                    architecture_lower_bounds[latent_size][hidden_sizes] = (
                        lower_bound)
                    architecture_versions[latent_size][hidden_sizes] = version
                else:
                    previous_version = architecture_versions[latent_size][
                        hidden_sizes]
                    best_version = _best_variant(version, previous_version)
                    if version == best_version:
                        architecture_versions[latent_size][hidden_sizes] = (
                            version)
                        architecture_lower_bounds[latent_size][
                            hidden_sizes] = lower_bound

        if architecture_lower_bounds:
            architecture_lower_bounds = pandas.DataFrame(
                architecture_lower_bounds
            )
            architecture_lower_bounds = architecture_lower_bounds.reindex(
                columns=sorted(
                    architecture_lower_bounds.columns,
                    key=lambda s: int(s)
                )
            )
            architecture_lower_bounds = architecture_lower_bounds.reindex(
                index=sorted(
                    architecture_lower_bounds.index,
                    key=lambda s: numpy.prod(list(map(int, s.split("×"))))
                )
            )

            if architecture_lower_bounds.size > 1:
                figure, figure_name = plot_elbo_heat_map(
                    architecture_lower_bounds,
                    x_label="Latent dimension",
                    y_label="Number of hidden units",
                    z_symbol="\\mathcal{L}",
                    name=data_set_path.replace(os.sep, "-")
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=cross_analysis_directory
                )

        # Table
        comparisons = {
            model_title: {
                field_name: copy.deepcopy(field_value)
                for field_name, field_value in model_fields.items()
                if field_name in SORTED_COMPARISON_TABLE_COLUMN_NAMES
            }
            for model_title, model_fields in summary_metrics_sets.items()
        }
        comparison_field_names = [
            field_name for field_name in model_field_names
            if field_name in SORTED_COMPARISON_TABLE_COLUMN_NAMES
        ]

        sorted_comparison_items = sorted(
            comparisons.items(),
            key=lambda key_value_pair: numpy.mean(key_value_pair[-1]["ELBO"]),
            reverse=True
        )

        for model_title, model_fields in comparisons.items():
            for field_name, field_value in model_fields.items():

                if isinstance(field_value, list) and len(field_value) == 1:
                    field_value == field_value.pop()

                if isinstance(field_value, str):
                    continue
                elif not field_value:
                    string = ""
                elif isinstance(field_value, float):
                    string = "{:-.6g}".format(field_value)
                elif isinstance(field_value, int):
                    string = "{:d}".format(field_value)
                elif isinstance(field_value, list):
                    if field_value[0] is None:
                        string = "---"
                    else:
                        field_values = numpy.array(field_value)

                        mean = field_values.mean()
                        sd = field_values.std(ddof=1)

                        if field_values.dtype == int:
                            string = "{:.0f}±{:.3g}".format(mean, sd)
                        else:
                            string = "{:-.6g}±{:.3g}".format(mean, sd)
                else:
                    raise TypeError(
                        "`{}` not supported in comparison table."
                        .format(type(field_value))
                    )

                comparisons[model_title][field_name] = string

        common_comparison_fields = {}
        comparison_fields_to_remove = []

        for field_name in comparison_field_names:
            field_values = set()
            for model_fields in comparisons.values():
                field_values.add(model_fields.get(field_name, None))
            if len(field_values) == 1:
                for model_fields in comparisons.values():
                    model_fields.pop(field_name, None)
                comparison_fields_to_remove.append(field_name)
                field_value = field_values.pop()
                if field_value:
                    common_comparison_fields[field_name] = field_value

        for field_name in comparison_fields_to_remove:
            comparison_field_names.remove(field_name)

        common_comparison_fields_string_parts = []

        for field_name, field_value in common_comparison_fields.items():
            common_comparison_fields_string_parts.append(
                "{}: {}".format(capitalise_string(field_name), field_value)
            )

        common_comparison_fields_string = "\n".join(
            common_comparison_fields_string_parts)

        comparison_table_rows = []
        table_column_spacing = "  "

        comparison_table_column_widths = {}

        for field_name in comparison_field_names:
            comparison_table_column_widths[field_name] = max(
                [len(model_fields[field_name]) for model_fields in
                    comparisons.values()]
            )

        comparison_table_heading_parts = []

        for field_name in comparison_field_names:

            field_width = comparison_table_column_widths[field_name]

            if field_width == 0:
                continue

            if len(field_name) > field_width:
                for full_form, abbreviation in ABBREVIATIONS.items():
                    field_name = re.sub(full_form, abbreviation, field_name)
                field_name = textwrap.shorten(
                    capitalise_string(field_name),
                    width=field_width,
                    placeholder="…"
                )
            elif field_name == field_name.lower():
                field_name = capitalise_string(field_name)

            comparison_table_heading_parts.append(
                "{:{}}".format(field_name, field_width)
            )

        comparison_table_heading = table_column_spacing.join(
            comparison_table_heading_parts
        )
        comparison_table_toprule = "-" * len(comparison_table_heading)

        comparison_table_rows.append(comparison_table_heading)
        comparison_table_rows.append(comparison_table_toprule)

        for model_title, model_fields in sorted_comparison_items:

            sorted_model_field_items = sorted(
                model_fields.items(),
                key=lambda key_value_pair: (
                    comparison_field_names.index(key_value_pair[0]))
            )

            comparison_table_row_parts = [
                "{:{}}".format(
                    field_value,
                    comparison_table_column_widths[field_name]
                )
                for field_name, field_value in sorted_model_field_items
                if comparison_table_column_widths[field_name] > 0
            ]

            comparison_table_rows.append(
                table_column_spacing.join(comparison_table_row_parts)
            )

        comparison_table = "\n".join(comparison_table_rows)

        print(subtitle("Comparison"))
        print(comparison_table + "\n")
        print(common_comparison_fields_string + "\n")

        if set_other_method_metrics:
            other_methods_string_parts = ["Other methods:"]

            other_methods_metric_values = {}
            for set_name, other_method_metrics in (
                    set_other_method_metrics.items()):
                for method, method_metrics in other_method_metrics.items():
                    for metric_name, metric_values in method_metrics.items():

                        set_metric_name = metric_name

                        if set_name == "superset":
                            set_metric_name += " (superset)"

                        other_methods_metric_values.setdefault(method, {})
                        other_methods_metric_values[method].setdefault(
                            set_metric_name, metric_values
                        )

            for method, metric_values in other_methods_metric_values.items():

                other_methods_string_parts.append("    {}:".format(
                    method
                ))

                for metric_name, values in metric_values.items():
                    if len(values) > 1:
                        value_string = "{:.6g}±{:.6g}".format(
                            statistics.mean(values),
                            statistics.stdev(values)
                        )
                    elif len(values) == 1:
                        value_string = "{:.6g}".format(values[0])
                    else:
                        continue

                    other_methods_string_parts.append(
                        "        {}: {}".format(
                            metric_name,
                            value_string
                        )
                    )

            other_methods_string = "\n".join(other_methods_string_parts)
            print(other_methods_string + "\n")

        if log_summary:
            log_string_parts.append(subtitle("Comparison", plain=True))
            log_string_parts.append(comparison_table + "\n")
            log_string_parts.append(
                common_comparison_fields_string + "\n"
            )
            if set_other_method_metrics:
                log_string_parts.append(other_methods_string + "\n")

        # Model filter based on the most frequent architecture
        filter_field_names = ["sizes", "other"]

        model_filter_fields = {}

        for model_fields in summary_metrics_sets.values():
            runs = model_fields.get("runs", None)
            if not runs.isdigit():
                continue
            model_type = model_fields.get("type", None)
            if model_type:
                for filter_name in filter_field_names:
                    field_value = model_fields.get(filter_name, None)
                    if field_value:
                        model_filter_fields.setdefault(model_type, {})
                        model_filter_fields[model_type].setdefault(
                            filter_name, []
                        )
                        model_filter_fields[model_type][filter_name].append(
                            field_value)

        for model_type, filter_fields in model_filter_fields.items():
            for filter_name, filter_values in filter_fields.items():
                try:
                    mode = statistics.mode(filter_values)
                except statistics.StatisticsError as exception:
                    if "no unique mode" in str(exception):
                        mode = filter_values[0]
                model_filter_fields[model_type][filter_name] = mode

        optimised_metric_names = ["ELBO", "ENRE", "KL_z", "KL_y"]
        optimised_metric_symbols = {
            "ELBO": "$\\mathcal{L}$",
            "ENRE": "$\\log p(x|z)$",
            "KL_z": "KL$_z(q||p)$",
            "KL_y": "KL$_y(q||p)$",
        }

        supervised_clustering_metric_names = [
            n for n, d in CLUSTERING_METRICS.items()
            if d["kind"] == "supervised"
        ]
        unsupervised_clustering_metric_names = [
            n for n, d in CLUSTERING_METRICS.items()
            if d["kind"] == "unsupervised"
        ]

        # Collect metrics from relevant models
        model_likelihood_metrics = {}
        set_method_likelihood_metrics = {
            "standard": {},
            "superset": {},
            "unsupervised": {}
        }
        method_likelihood_variants = {}

        for model_title, model_fields in summary_metrics_sets.items():

            model_type = model_fields.get("type", None)

            if not model_type:
                print("No model type for model: {}".format(model_title))
                continue

            # Filter models
            discard_model = False

            if model_type in model_filter_fields:

                filter_fields = model_filter_fields[model_type]

                for filter_name, filter_value in filter_fields.items():

                    field_value = model_fields.get(filter_name, None)

                    if filter_name == "other":

                        filter_values = set(filter_value.split("; "))
                        field_values = set(field_value.split("; "))

                        if additional_other_option:
                            field_values.discard(additional_other_option)

                        filter_value = "; ".join(sorted(filter_values))
                        field_value = "; ".join(sorted(field_values))

                    if field_value != filter_value:
                        discard_model = True
                        break

            else:
                discard_model = True

            runs = model_fields["runs"]
            if runs == "D":
                discard_model = True

            if discard_model:
                continue

            # Extract method
            if model_type == FACTOR_ANALYSIS_MODEL_TYPE:
                model_type = FACTOR_ANALYSIS_MODEL_TYPE_ALIAS

            method_parts = [model_type]

            clustering_method = model_fields.get(
                "clustering method", None)

            if clustering_method:
                if clustering_method not in ["M", "---"]:
                    clustering_method = clustering_method.replace(", ", "-")
                    method_parts.append(clustering_method)

            if method_parts:
                method = "-".join(method_parts)
            else:
                method = "---"

            if (no_prediction_methods_for_gmvae_in_plots
                    and model_type.startswith("GMVAE")
                    and clustering_method and clustering_method != "M"):
                continue

            # Extract likelihood distribution
            likelihood = model_fields.get("likelihood")

            # Determine whether to update metrics if other version of
            # model exist and is worse than current version
            variant = {
                "other": model_fields.get("other", None),
                "version": model_fields.get("version", None),
                "epoch_number": model_fields.get("epochs", None),
            }

            likelihood_previous_variants = method_likelihood_variants.get(
                method, None
            )
            if likelihood_previous_variants:
                previous_variant = likelihood_previous_variants.get(
                    likelihood, None
                )
            else:
                previous_variant = None

            if previous_variant:
                best_variant = _best_variant(
                    variant, previous_variant,
                    additional_other_option=additional_other_option
                )
            else:
                best_variant = variant

            if variant != best_variant:
                continue

            method_likelihood_variants.setdefault(method, {})
            method_likelihood_variants[method][likelihood] = variant

            # Extract metrics
            metrics = {}

            for field_name, field_value in model_fields.items():

                field_name_parts = re.split(
                    pattern=r" \((.+)\)",
                    string=field_name,
                    maxsplit=1
                )

                field_name = field_name_parts[0]

                if field_name in supervised_clustering_metric_names:
                    if len(field_name_parts) == 3:
                        set_name = field_name_parts[1]
                    else:
                        set_name = "standard"
                elif field_name in unsupervised_clustering_metric_names:
                    set_name = "unsupervised"
                else:
                    continue

                if field_value:
                    metrics.setdefault(set_name, {})
                    metrics[set_name][field_name] = field_value

            for set_name, set_metrics in metrics.items():
                for field_name in optimised_metric_names:
                    set_metrics[field_name] = model_fields[field_name]

            # Save metrics
            model_likelihood_metrics.setdefault(model_type, {})
            model_likelihood_metrics[model_type][likelihood] = {
                field_name: model_fields[field_name]
                for field_name in optimised_metric_names
            }

            for set_name in metrics:
                set_method_likelihood_metrics[set_name].setdefault(
                    method, {}
                )
                set_method_likelihood_metrics[set_name][method][likelihood] = (
                    metrics[set_name])

        print("Plotting model metrics.")

        # Only optimised metrics

        # Clean up likelihood names

        models = set()
        likelihoods = set()

        for model in model_likelihood_metrics:
            models.add(model)
            for likelihood in model_likelihood_metrics[model]:
                likelihoods.add(likelihood)

        model_replacements = _replacements_for_cleaned_up_specifications(
            models,
            detail_separator=r"\((.+)\)",
            specification_separator="-"
        )

        likelihood_replacements = _replacements_for_cleaned_up_specifications(
            likelihoods,
            detail_separator=r"\((.+)\)",
            specification_separator="-"
        )

        # Rearrange data
        metrics_sets = []
        models = set()
        likelihoods = set()

        for model, likelihood_metrics in model_likelihood_metrics.items():

            model = model_replacements[model]
            models.add(model)

            for likelihood, metrics in likelihood_metrics.items():

                likelihood = likelihood_replacements[likelihood]
                likelihoods.add(likelihood)

                metrics_set = copy.deepcopy(metrics)
                metrics_set["model"] = model
                metrics_set["likelihood"] = likelihood

                metrics_sets.append(metrics_set)

        # Determine order for methods and likelihoods

        likelihood_order = sorted(
            likelihoods,
            key=_create_specifications_sorter(
                order=LIKELIHOOD_DISRIBUTION_ORDER,
                detail_separator=r"\((.+)\)",
                specification_separator="-"
            )
        )

        model_order = sorted(
            models,
            key=_create_specifications_sorter(
                order=MODEL_TYPE_ORDER,
                detail_separator=r"\((.+)\)",
                specification_separator="-"
            )
        )

        for optimised_metric_name in optimised_metric_names:

            optimised_metric_symbol = optimised_metric_symbols[
                optimised_metric_name
            ]

            figure, figure_name = plot_model_metrics(
                metrics_sets,
                key=optimised_metric_name,
                primary_differentiator_key="model",
                primary_differentiator_order=model_order,
                secondary_differentiator_key="likelihood",
                secondary_differentiator_order=likelihood_order,
                label=optimised_metric_symbol,
                name=[
                    data_set_path.replace(os.sep, "-"),
                    optimised_metric_name
                ]
            )
            figures.save_figure(
                figure=figure,
                name=figure_name,
                options=export_options,
                directory=cross_analysis_directory
            )

        # Optimised metrics and clustering metrics
        for set_name, method_likelihood_metrics in (
                set_method_likelihood_metrics.items()):
            if not method_likelihood_metrics:
                continue

            # Clean up method and likelihood names
            methods = set()
            likelihoods = set()

            for method in method_likelihood_metrics:
                methods.add(method)
                for likelihood in method_likelihood_metrics[method]:
                    likelihoods.add(likelihood)

            method_replacements = _replacements_for_cleaned_up_specifications(
                methods,
                detail_separator=r"\((.+)\)",
                specification_separator="-"
            )

            likelihood_replacements = (
                _replacements_for_cleaned_up_specifications(
                    likelihoods,
                    detail_separator=r"\((.+)\)",
                    specification_separator="-"
                )
            )

            # Rearrange data
            metrics_sets = []
            methods = set()
            likelihoods = set()

            for method, likelihood_metrics in (
                    method_likelihood_metrics.items()):
                method = method_replacements[method]
                methods.add(method)

                for likelihood, metrics in likelihood_metrics.items():

                    likelihood = likelihood_replacements[likelihood]
                    likelihoods.add(likelihood)

                    metrics_set = copy.deepcopy(metrics)
                    metrics_set["method"] = method
                    metrics_set["likelihood"] = likelihood

                    metrics_sets.append(metrics_set)

            # Determine order for methods and likelihoods
            likelihood_order = sorted(
                likelihoods,
                key=_create_specifications_sorter(
                    order=LIKELIHOOD_DISRIBUTION_ORDER,
                    detail_separator=r"\((.+)\)",
                    specification_separator="-"
                )
            )
            method_order = sorted(
                methods,
                key=_create_specifications_sorter(
                    order=MODEL_TYPE_ORDER,
                    detail_separator=r"\((.+)\)",
                    specification_separator="-"
                )
            )

            special_cases = {}
            for method in method_order:
                for other_method in method_order:
                    if other_method == method:
                        continue
                    elif other_method.startswith(method):
                        special_cases[method] = {
                            "errorbar_colour": "darken"
                        }

            # Baselines
            if set_other_method_metrics:
                other_method_metrics = set_other_method_metrics.get(
                    set_name, None)
            else:
                other_method_metrics = None

            # Clustering metrics
            if set_name == "unsupervised":
                clustering_metric_names = unsupervised_clustering_metric_names
            else:
                clustering_metric_names = supervised_clustering_metric_names

            for optimised_metric_name, clustering_metric_name in (
                    product(optimised_metric_names, clustering_metric_names)):

                clustering_metric_symbol = CLUSTERING_METRICS[
                    clustering_metric_name
                ]["symbol"]
                optimised_metric_symbol = optimised_metric_symbols[
                    optimised_metric_name
                ]

                figure, figure_name = plot_model_metric_sets(
                    metrics_sets,
                    x_key=optimised_metric_name,
                    y_key=clustering_metric_name,
                    primary_differentiator_key="likelihood",
                    primary_differentiator_order=likelihood_order,
                    secondary_differentiator_key="method",
                    secondary_differentiator_order=method_order,
                    special_cases=special_cases,
                    other_method_metrics=other_method_metrics,
                    x_label=optimised_metric_symbol,
                    y_label=clustering_metric_symbol,
                    name=[
                        data_set_path.replace(os.sep, "-"),
                        set_name,
                        clustering_metric_name,
                        optimised_metric_name
                    ]
                )
                figures.save_figure(
                    figure=figure,
                    name=figure_name,
                    options=export_options,
                    directory=cross_analysis_directory
                )

        print()

    if log_summary:
        log_string = "\n".join(log_string_parts)
        with open(log_path, "w") as log_file:
            log_file.write(log_string)

    return 0


def _metrics_sets_in_analyses_directory(analyses_directory,
                                        data_set_included_strings=None,
                                        data_set_excluded_strings=None,
                                        model_included_strings=None,
                                        model_excluded_strings=None):

    metrics_filename = METRICS_BASENAME + ZIPPED_PICKLE_EXTENSION

    metrics_set = {}

    for path, _, filenames in os.walk(analyses_directory):

        data_set_model = path.replace(analyses_directory, "")
        data_set_model_parts = data_set_model.split(os.sep)
        data_set = os.sep.join(data_set_model_parts[:3])
        model = os.sep.join(data_set_model_parts[3:])

        # Verify data set match

        data_set_match = _match_string(
            data_set,
            data_set_included_strings,
            data_set_excluded_strings
        )

        if not data_set_match:
            continue

        # Verify model match

        model_match = _match_string(
            model,
            model_included_strings,
            model_excluded_strings
        )

        if not model_match:
            continue

        # Verify metrics found

        if metrics_filename in filenames:

            model_parts = model.split(os.sep)

            model = os.sep.join(model_parts[:3])

            if len(model_parts) == 4:
                run = "default"
                version = model_parts[3]
            elif len(model_parts) == 5:
                run = model_parts[3]
                version = model_parts[4]

            if data_set not in metrics_set:
                metrics_set[data_set] = {}

            if model not in metrics_set[data_set]:
                metrics_set[data_set][model] = {}

            if run not in metrics_set[data_set][model]:
                metrics_set[data_set][model][run] = {}

            metrics_path = os.path.join(path, metrics_filename)

            with gzip.open(metrics_path, "r") as metrics_file:
                metrics_data = pickle.load(metrics_file)

            predictions = {}

            for filename in filenames:
                if (filename.startswith(PREDICTION_BASENAME)
                        and filename.endswith(ZIPPED_PICKLE_EXTENSION)):

                    prediction_name = filename.replace(
                        ZIPPED_PICKLE_EXTENSION, "").replace(
                        PREDICTION_BASENAME, "").replace("-", "")

                    prediction_path = os.path.join(path, filename)

                    with gzip.open(prediction_path, "r") as prediction_file:
                        prediction_data = pickle.load(
                            prediction_file)

                    predictions[prediction_name] = prediction_data

            if predictions:
                metrics_data["predictions"] = predictions

            metrics_set[data_set][model][run][version] = metrics_data

    return metrics_set


def _metrics_for_other_methods(data_set_directory,
                               other_methods=None,
                               prediction_included_strings=None,
                               prediction_excluded_strings=None):

    if other_methods is None:
        other_methods = []
    elif not isinstance(other_methods, list):
        other_methods = [other_methods]

    other_method_metrics = {}

    for other_method in other_methods:

        other_method_data_set_directory = data_set_directory
        other_method_prediction_basename = PREDICTION_BASENAME

        other_method_parts = other_method.split("-")
        other_method = other_method_parts.pop(0)

        other_method = proper_string(
            normalise_string(other_method),
            OTHER_METHOD_NAMES
        )
        directory_name = normalise_string(other_method)

        other_method_specifications = OTHER_METHOD_SPECIFICATIONS.get(
            other_method, None
        )

        if other_method_specifications:

            replacement_directory_name = other_method_specifications.get(
                "directory name", None
            )
            training_set_name = other_method_specifications.get(
                "training set name", None
            )
            evaluation_set_name = other_method_specifications.get(
                "evaluation set name", None
            )

            if replacement_directory_name:
                directory_name = replacement_directory_name

            if training_set_name == "full":
                other_method_data_set_directory = re.sub(
                    pattern=r"/split-[\w\d.]+?/",
                    repl="/no_split/",
                    string=other_method_data_set_directory
                )

            if evaluation_set_name != "test":
                other_method_prediction_basename = (
                    other_method_prediction_basename.replace(
                        "test",
                        evaluation_set_name,
                        1
                    )
                )

        if other_method_parts:
            directory_name = "-".join([directory_name] + other_method_parts)

        method_directory = os.path.join(
            other_method_data_set_directory,
            directory_name
        )

        if not os.path.exists(method_directory):
            continue

        for path, directory_names, filenames in os.walk(method_directory):
            for filename in filenames:
                if (filename.startswith(other_method_prediction_basename)
                        and filename.endswith(ZIPPED_PICKLE_EXTENSION)):

                    prediction_match = _match_string(
                        filename,
                        prediction_included_strings,
                        prediction_excluded_strings
                    )

                    if not prediction_match:
                        continue

                    prediction_path = os.path.join(path, filename)

                    with gzip.open(prediction_path, "r") as prediction_file:
                        prediction = pickle.load(prediction_file)

                    method = prediction.get("prediction method", None)
                    clustering_metric_values = prediction.get(
                        "clustering metric values", [])

                    if method != other_method:
                        method = " + ".join([other_method, method])

                    if method == BASELINE_METHOD_TYPE:
                        method = BASELINE_METHOD_TYPE_ALIAS

                    for metric_name, metric_set in (
                            clustering_metric_values.items()):
                        for metric_set_name, metric_value in (
                                metric_set.items()):
                            if metric_value is None:
                                continue
                            elif metric_set_name.startswith("clusters"):

                                metric_details = CLUSTERING_METRICS.get(
                                    metric_name, dict())
                                metric_kind = metric_details.get("kind", None)

                                if metric_kind and metric_kind == "supervised":
                                    set_name = "standard"
                                    if metric_set_name.endswith("superset"):
                                        set_name = "superset"
                                elif (metric_kind
                                        and metric_kind == "unsupervised"):
                                    set_name = "unsupervised"
                                else:
                                    set_name = "unknown"

                                other_method_metrics.setdefault(
                                    set_name, {}
                                )
                                other_method_metrics[set_name].setdefault(
                                    method, {})
                                other_method_metrics[set_name][
                                    method].setdefault(metric_name, [])
                                other_method_metrics[set_name][method][
                                    metric_name].append(float(metric_value))

    return other_method_metrics


def _parse_metrics_for_runs_and_versions_of_model(
        runs,
        log_summary=False,
        prediction_included_strings=None,
        prediction_excluded_strings=None,
        epoch_cut_off=None):

    run_version_summary_metrics = {
        key: {} for key in ["default", "multiple"]
    }
    correlation_sets = {}

    if log_summary:
        log_string_parts = []

    for run_name, versions in sorted(runs.items()):

        if run_name == "default":
            run_title = "default run"
            run_key = run_name
        else:
            run_title = run_name.replace("_", " ", 1)
            run_key = "multiple"

        if len(runs) > 1:
            print(heading(capitalise_string(run_title)))

            if log_summary:
                log_string_parts.append(
                    heading(capitalise_string(run_title), plain=True)
                )

        version_epoch_summary_metrics = {}

        for version_name, metrics in versions.items():

            epochs = "0 epochs"
            version = "end of training"
            samples = []

            for version_field in version_name.split("-"):
                field_name, field_value = version_field.split(
                    "_", maxsplit=1)
                if field_name == "e" and field_value.isdigit():
                    number_of_epochs = int(field_value)
                    epochs = field_value + " epochs"
                elif field_name in ["mc", "iw"]:
                    if field_value.isdigit() and int(field_value) > 1:
                        samples += "{} {} samples".format(
                            field_value, field_name.upper())
                else:
                    version = version_field.replace("_", " ").replace(
                        "best model", "optimal parameters")

            if epoch_cut_off and number_of_epochs > epoch_cut_off:
                continue

            version_title = "; ".join([epochs, version] + samples)

            metrics_string_parts = []
            summary_metrics = {}

            # Time
            timestamp = metrics["timestamp"]
            metrics_string_parts.append(
                "Timestamp: {}".format(format_time(timestamp))
            )

            # Epochs
            n_epochs = metrics["number of epochs trained"]
            metrics_string_parts.append(
                "Epochs trained: {}".format(n_epochs)
            )
            summary_metrics["epochs"] = n_epochs
            metrics_string_parts.append("")

            # Evaluation
            evaluation = metrics["evaluation"]
            losses = [
                "log_likelihood",
                "lower_bound",
                "reconstruction_error",
                "kl_divergence",
                "kl_divergence_z",
                "kl_divergence_z1",
                "kl_divergence_z2",
                "kl_divergence_y"
            ]

            for loss in losses:
                if loss in evaluation:
                    metrics_string_parts.append(
                        "{}: {:-.6g}".format(
                            loss, evaluation[loss][-1]
                        )
                    )

            lower_bound = evaluation.get("lower_bound", [None])[-1]
            reconstruction_error = evaluation.get(
                "reconstruction_error", [None])[-1]
            kl_divergence_y = evaluation.get("kl_divergence_y", [None])[-1]
            kl_divergence_z = None

            if "kl_divergence" in evaluation:
                kl_divergence_z = evaluation["kl_divergence"][-1]
            elif "kl_divergence_z" in evaluation:
                kl_divergence_z = evaluation["kl_divergence_z"][-1]

            summary_metrics.update({
                "ELBO": lower_bound,
                "ENRE": reconstruction_error,
                "KL_z": kl_divergence_z,
                "KL_y": kl_divergence_y
            })

            # Accuracies
            accuracies = ["accuracy", "superset_accuracy"]

            for accuracy in accuracies:
                if accuracy in metrics and metrics[accuracy]:
                    metrics_string_parts.append("{}: {:6.2f} %".format(
                        accuracy, 100 * metrics[accuracy][-1]))

            metrics_string_parts.append("")

            # Statistics
            if isinstance(metrics["statistics"], list):
                statistics_sets = metrics["statistics"]
            else:
                statistics_sets = None

            reconstructed_statistics = None

            if statistics_sets:
                for statistics_set in statistics_sets:
                    if "reconstructed" in statistics_set["name"]:
                        reconstructed_statistics = statistics_set

            if reconstructed_statistics:
                metrics_string_parts.append(
                    format_summary_statistics(reconstructed_statistics)
                )
            metrics_string_parts.append("")

            # Predictions
            if "predictions" in metrics:
                for predictions in metrics["predictions"].values():

                    prediction_string_parts = []
                    method = predictions.get(
                        "prediction method",
                        None
                    )

                    if not method:
                        method = "model"

                    number_of_classes = predictions.get(
                        "number of classes",
                        None
                    )

                    clustering_string = "{} ({} classes)".format(
                        method, number_of_classes)

                    prediction_string_parts.append(clustering_string)

                    prediction_string = ", ".join(prediction_string_parts)

                    prediction_match = _match_string(
                        prediction_string,
                        prediction_included_strings,
                        prediction_excluded_strings
                    )

                    if not prediction_match:
                        continue

                    clustering_metric_values = predictions.get(
                        "clustering metric values", None
                    )

                    if clustering_metric_values:

                        metrics_string_parts.append(
                            prediction_string + ":")

                        for metric_name, set_metrics in (
                                clustering_metric_values.items()):
                            metrics_string_parts.append(
                                "    {}:".format(
                                    capitalise_string(metric_name)
                                )
                            )

                            for set_name, set_value in set_metrics.items():
                                if set_value:
                                    set_value = float(set_value)
                                    metrics_string_parts.append(
                                        "        {}: {:.6g}".format(
                                            set_name,
                                            set_value
                                        )
                                    )
                                else:
                                    continue

                                if "clusters" in set_name and set_value:

                                    metric_key = "; ".join([
                                        "clustering",
                                        prediction_string,
                                        metric_name
                                    ])
                                    if "superset" in set_name:
                                        metric_key += " (superset)"
                                    summary_metrics[metric_key] = set_value

                                    # Correlation sets

                                    if set_value == 0:
                                        continue

                                    correlation_set_name = "; ".join([
                                        prediction_string,
                                        metric_name,
                                        set_name
                                    ])

                                    correlation_sets.setdefault(
                                        correlation_set_name,
                                        {
                                            "ELBO": [],
                                            "clustering metric": []
                                        }
                                    )

                                    correlation_sets[correlation_set_name][
                                        "ELBO"].append(lower_bound)
                                    correlation_sets[correlation_set_name][
                                        "clustering metric"].append(set_value)

                        metrics_string_parts.append("")

            metrics_string = "\n".join(metrics_string_parts)

            if len(versions) > 1:
                print(subheading(capitalise_string(version_title)))

            print(metrics_string)

            if log_summary:
                if len(versions) > 1:
                    log_string_parts.append(subheading(
                        capitalise_string(version_title), plain=True))
                log_string_parts.append(metrics_string)

            # Summary metrics
            version_key = "; ".join([version] + samples)

            version_epoch_summary_metrics.setdefault(version_key, {})
            version_epoch_summary_metrics[version_key][number_of_epochs] = (
                summary_metrics)

        for version_key, epoch_summary_metrics in (
                version_epoch_summary_metrics.items()):

            run_version_summary_metrics[run_key].setdefault(
                version_key, {
                    "runs": 0,
                    "version": version_key
                }
            )
            run_version_summary_metrics[run_key][version_key]["runs"] += 1

            maximum_number_of_epochs = max(epoch_summary_metrics.keys())
            summary_metrics = epoch_summary_metrics[maximum_number_of_epochs]

            for metric_key, metric_value in summary_metrics.items():
                if run_key == "default":
                    run_version_summary_metrics[run_key][version_key][
                        metric_key] = metric_value
                else:
                    run_version_summary_metrics[run_key][
                        version_key].setdefault(metric_key, [])
                    run_version_summary_metrics[run_key][version_key][
                        metric_key].append(metric_value)

    results = {
        "summary_metrics_sets": [],
        "correlation_sets": correlation_sets
    }

    for run_key, version_summary_metrics in (
            run_version_summary_metrics.items()):
        for version_key, summary_metrics in version_summary_metrics.items():

            if run_key == "default":
                runs = "default run"
            else:
                runs = summary_metrics["runs"]
                if isinstance(runs, int):
                    runs = "{} runs".format(runs)
            summary_metrics["runs"] = runs

            clustering_field_names = []
            for field_name in summary_metrics:
                if field_name.startswith("clustering"):
                    clustering_field_names.append(field_name)
            clustering_metric_values = {}

            for field_name in clustering_field_names:
                metric_value = summary_metrics.pop(field_name, None)
                if metric_value:
                    field_name_parts = field_name.split("; ")
                    method = field_name_parts[1]
                    name = field_name_parts[2]
                    clustering_metric_values.setdefault(method, {})
                    clustering_metric_values[method][name] = metric_value

            if clustering_metric_values:
                original_summary_metrics = summary_metrics

                for method in clustering_metric_values:
                    summary_metrics = copy.deepcopy(
                        original_summary_metrics
                    )
                    summary_metrics.update(clustering_metric_values[method])
                    summary_metrics["clustering method"] = method
                    results["summary_metrics_sets"].append(summary_metrics)
            else:
                results["summary_metrics_sets"].append(
                    summary_metrics
                )

    if log_summary:
        results["log_string_parts"] = log_string_parts

    return results


def _match_string(string, included_strings=None, excluded_strings=None):

    match = True

    if included_strings:
        for search_string in included_strings:
            if search_string in string:
                match *= True
            else:
                match *= False

    if excluded_strings:
        for search_string in excluded_strings:
            if search_string not in string:
                match *= True
            else:
                match *= False

    return match


def _title_from_name(name, replacement_dictionaries=None):

    if replacement_dictionaries:
        if not isinstance(replacement_dictionaries, list):
            replacement_dictionaries = [replacement_dictionaries]

        for replacements in replacement_dictionaries:
            for pattern, replacement in replacements.items():
                if not isinstance(replacement, str):
                    replacement_function = replacement

                    match = re.search(pattern, name)

                    if match:
                        replacement = replacement_function(match)

                name = re.sub(pattern, replacement, name)

    name = name.replace("/", "; ")
    name = name.replace("-", "; ")
    name = name.replace("_", " ")

    return name


def _data_set_title_from_data_set_name(name):
    replacement_dictionaries = [
        DATA_SET_NAME_REPLACEMENTS,
        SPLIT_REPLACEMENTS,
        FEATURE_REPLACEMENTS,
        EXAMPLE_FEATURE_REPLACEMENTS,
        EXAMPLE_REPLACEMENTS
    ]
    return _title_from_name(name, replacement_dictionaries)


def _model_title_from_model_name(name):
    replacement_dictionaries = [
        MODEL_REORDER_REPLACEMENTS,
        MODEL_REPLACEMENTS,
        SECONDARY_MODEL_REPLACEMENTS,
        DISTRIBUTION_MODIFICATION_REPLACEMENTS,
        DISTRIBUTION_REPLACEMENTS,
        NETWORK_REPLACEMENTS,
        SAMPLE_REPLACEMENTS,
        MISCELLANEOUS_MODEL_REPLACEMENTS
    ]
    return _title_from_name(name, replacement_dictionaries)


def _clustering_method_title_from_clustering_method_name(name):
    replacement_dictionaries = [
        INBUILT_CLUSTERING_REPLACEMENTS,
        CLUSTERING_METHOD_REPLACEMENTS
    ]
    return _title_from_name(name, replacement_dictionaries)


def _generate_model_ids():

    numbers = list(map(str, range(10)))
    letters = list(ascii_uppercase)

    values = numbers + letters

    for value1, value2 in product(values, values):
        model_id = value1 + value2
        if model_id.isdigit():
            continue
        yield model_id


def _best_variant(*variants, additional_other_option=None):

    def variant_sort_key(variant):

        other = variant.get("other", None)

        if other:
            other_set = set(other.split("; "))
        else:
            other_set = set()

        if additional_other_option in other_set:
            additional_other_option_available = True
        else:
            additional_other_option_available = False

        version_rankings = {
            "EOT": 0,  # end of training
            "ES": 1,  # early stopping
            "OP": 2   # optimal parameters
        }

        version = variant.get("version", None)
        ranking = version_rankings.get(version, -1)

        epoch_number = variant.get("epoch_number", -1)

        if isinstance(epoch_number, list):
            epoch_number = statistics.mean(epoch_number)

        variant_sort_key = [
            additional_other_option_available,
            ranking,
            epoch_number
        ]

        return variant_sort_key

    sorted_variants = sorted(variants, key=variant_sort_key)
    best_variant = sorted_variants[-1]

    return best_variant


def _comparison_table_column_sorter(name):
    name = str(name)
    column_names = SORTED_COMPARISON_TABLE_COLUMN_NAMES

    n_columns = len(column_names)
    index_width = len(str(n_columns))

    indices = set()

    for column_index, column_name in enumerate(column_names):
        if name.startswith(column_name):
            indices.add(column_index)

    if len(indices) > 1:
        if name in SORTED_COMPARISON_TABLE_COLUMN_NAMES:
            index = SORTED_COMPARISON_TABLE_COLUMN_NAMES.index(name)
        else:
            index = indices.pop()
    elif len(indices) == 1:
        index = indices.pop()
    else:
        index = n_columns

    name = "{:{}d} {}".format(index, index_width, name)

    return name


def _replacements_for_cleaned_up_specifications(specification_sets=set(),
                                                detail_separator="",
                                                specification_separator=""):

    # Categorise specification variations
    specification_types = {}

    for specification_set in specification_sets:
        specifications = re.split(specification_separator, specification_set)

        for i, specification in enumerate(specifications):
            specification_parts = re.split(
                detail_separator, specification, maxsplit=1)
            specification_types.setdefault(i, {})
            specification_type = specification_parts[0]
            if len(specification_parts) > 1:
                specification_details = " ".join(specification_parts[1:])
            else:
                specification_details = None
            specification_types[i].setdefault(specification_type, set())
            specification_types[i][specification_type].add(
                specification_details)

    # Simplify specification variations
    replacements = {}

    for specification_set in specification_sets:
        specifications = re.split(specification_separator, specification_set)

        replacement_parts = []

        for i, specification in enumerate(specifications):
            specification_parts = re.split(
                detail_separator, specification, maxsplit=1)
            specification_type = specification_parts[0]

            if len(specification_types[i][specification_type]) <= 1:
                replacement = specification_type
            else:
                replacement = specification

            replacement_parts.append(replacement)

        replacements[specification_set] = specification_separator.join(
            replacement_parts)

    return replacements


def _create_specifications_sorter(order=None, detail_separator="",
                                  specification_separator=""):

    if order is None:
        order = []

    def specifications_sorter(specifications):
        specifications = re.split(specification_separator, specifications)

        key = []

        for specification in specifications:
            specification_parts = re.split(
                detail_separator, specification, maxsplit=1)
            specification_type = specification_parts[0]
            if specification_type in order:
                specification_ranking = order.index(specification_type)
            else:
                specification_ranking = -1
            key.append(specification_ranking)
            if len(specification_parts) > 1:
                key.extend(specification_parts[1:])

        return key

    return specifications_sorter
