#!/usr/bin/env python3

# ======================================================================== #
# 
# Copyright (c) 2017 - 2018 scVAE authors
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
import copy

import pickle
import gzip
import numpy
import pandas
import statistics

import re
import textwrap

from itertools import product
from string import ascii_uppercase
from math import inf
from scipy.stats import pearsonr

import argparse

from analysis import (
    formatStatistics, saveFigure,
    plotCorrelations, plotELBOHeatMap,
    plotModelMetrics, plotModelMetricSets,
    clustering_metrics
)
from auxiliary import (
    formatTime,
    normaliseString, properString, capitaliseString,
    title, subtitle, heading, subheading,
    prod
)
from miscellaneous.prediction import PREDICTION_METHOD_NAMES

metrics_basename = "test-metrics"
prediction_basename = "test-prediction"

zipped_pickle_extension = ".pkl.gz"
log_extension = ".log"

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

OTHER_METHOD_NAMES = {
    "k-means": ["k_means", "kmeans"],
    "Seurat": ["seurat"]
}

OTHER_METHOD_SPECIFICATIONS = {
    "Seurat": {
        "directory name": "Seurat",
        "training set name": "full",
        "evaluation set name": "full"
    },
}

def main(log_directory = None, results_directory = None,
    data_set_included_strings = [], 
    data_set_excluded_strings = [], 
    model_included_strings = [],
    model_excluded_strings = [],
    prediction_included_strings = [],
    prediction_excluded_strings = [],
    additional_other_option = None,
    epoch_cut_off = inf,
    other_methods = [],
    export_options = [],
    log_summary = False):
    
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
    
    if log_directory:
        log_directory = os.path.normpath(log_directory) + os.sep

    if results_directory:
        
        # Directory and filenames
        
        results_directory = os.path.normpath(results_directory) + os.sep
        
        cross_analysis_name_parts = []
        
        for search_strings_set in search_strings_sets:
            
            search_strings = search_strings_set["strings"]
            search_abbreviation = search_strings_set["abbreviation"]
            
            if search_strings:
                cross_analysis_name_parts.append("{}_{}".format(
                    search_abbreviation,
                    "_".join(search_strings)
                ))
        
        if additional_other_option:
            cross_analysis_name_parts.append("a_{}".format(
                additional_other_option))
        
        if epoch_cut_off and epoch_cut_off != inf:
            cross_analysis_name_parts.append("e_{}".format(epoch_cut_off))
        
        if cross_analysis_name_parts:
            cross_analysis_name = "-".join(cross_analysis_name_parts)
        else:
            cross_analysis_name = "all"
        
        cross_analysis_directory = os.path.join(
            results_directory,
            "cross_analysis",
            cross_analysis_name
        )
        
        if log_summary:
            log_filename = cross_analysis_name + log_extension
            log_path = os.path.join(cross_analysis_directory, log_filename)
        
        # Print filtering
        
        explanation_string_parts = []
        
        for search_strings_set in search_strings_sets:
            
            search_strings = search_strings_set["strings"]
            search_kind = search_strings_set["kind"]
            search_inclusiveness = search_strings_set["inclusiveness"]
            
            if search_strings:
                explanation_string_parts.append("{} {}s with: {}.".format(
                    "Including" if search_inclusiveness else "Excluding",
                    search_kind,
                    ", ".join(search_strings)
                ))
        
        if additional_other_option:
            explanation_string_parts.append(
                "Additional other option to use for models in model metrics "
                    "plot: {}.".format(additional_other_option)
            )
        
        if epoch_cut_off and epoch_cut_off != inf:
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
        
        metrics_sets = metricsSetsInResultsDirectory(
            results_directory,
            data_set_included_strings,
            data_set_excluded_strings,
            model_included_strings,
            model_excluded_strings
        )
        
        model_IDs = modelID()
        
        for data_set_path, models in metrics_sets.items():
            
            data_set_title = dataSetTitleFromDataSetName(data_set_path)
            
            print(title(data_set_title))
            
            if log_summary:
                log_string_parts.append(title(data_set_title, plain = True))
            
            summary_metrics_sets = {}
            correlation_sets = {}
            
            for model_name, runs in models.items():
                
                # Setup
                
                model_title = modelTitleFromModelName(model_name)
                model_ID = next(model_IDs)
                model_ID_string = "ID: {}\n".format(model_ID)
                
                print(subtitle(model_title))
                print(model_ID_string)
                
                if log_summary:
                    log_string_parts.append(
                        subtitle(model_title, plain = True)
                    )
                    log_string_parts.append(model_ID_string)
                
                # Parse metrics for runs and versions of model
                
                metrics_results = parseMetricsForRunsAndVersionsOfModel(
                    runs = runs,
                    log_summary = log_summary,
                    prediction_included_strings = prediction_included_strings,
                    prediction_excluded_strings = prediction_excluded_strings,
                    epoch_cut_off = epoch_cut_off
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
                        clustering_method_title = \
                            clusteringMethodTitleFromClusteringMethodName(
                            model_summary_metrics_set["clustering method"]
                        )
                    else:
                        clustering_method_title = "---"
                    
                    set_title = "; ".join([
                        model_title,
                        clustering_method_title,
                        model_summary_metrics_set["runs"],
                        model_summary_metrics_set["version"]
                    ])
                    
                    summary_metrics_set = {"ID": model_ID}
                    
                    for field_name, field_value in \
                        model_summary_metrics_set.items():
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
                            correlation_sets[set_name][metric_name] = \
                                metric_values
            
            if len(summary_metrics_sets) <= 1:
                continue
            
            # Correlations
            
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
                        subtitle("Metric correlations", plain = True))
                    log_string_parts.append(correlation_string + "\n")
                
                print("Plotting correlations.")
                figure, figure_name = plotCorrelations(
                    correlation_sets,
                    x_key = "ELBO",
                    y_key = "clustering metric",
                    x_label = r"$\mathcal{L}$",
                    y_label = "",
                    name = data_set_path.replace(os.sep, "-")
                )
                saveFigure(figure, figure_name, export_options,
                    cross_analysis_directory)
                print()
            
            # Comparisons
            
            if other_methods:
                set_other_method_metrics = metricsForOtherMethods(
                    data_set_directory=os.path.join(
                        results_directory,
                        data_set_path
                    ),
                    other_methods=other_methods,
                    prediction_included_strings=prediction_included_strings,
                    prediction_excluded_strings=prediction_excluded_strings
                )
            else:
                set_other_method_metrics = None
            
            ## Setup
            
            model_field_names = set()
            # model_field_names = model_spec_names + model_metric_names
            
            for model_title in summary_metrics_sets:
                model_title_parts = model_title.split("; ")
                summary_metrics_sets[model_title].update({
                    "type": model_title_parts.pop(0),
                    "likelihood": model_title_parts.pop(0),
                    "sizes": model_title_parts.pop(0),
                    "version": ABBREVIATIONS[model_title_parts.pop(-1)],
                    "runs": model_title_parts.pop(-1)
                        .replace("default run", "D")
                        .replace(" runs", ""),
                    "clustering method": model_title_parts.pop(-1),
                    "other": "; ".join(model_title_parts)
                })
                model_field_names.update(
                    summary_metrics_sets[model_title].keys()
                )
            
            model_field_names = sorted(list(model_field_names),
                key = comparisonTableColumnSorter)
            
            for summary_metrics_set in summary_metrics_sets.values():
                for field_name in model_field_names:
                    summary_metrics_set.setdefault(field_name, None)
            
            ## Network architecture
            
            network_architecture_ELBOs = {}
            network_architecture_versions = {}
            
            for model_fields in summary_metrics_sets.values():
                if model_fields["type"] == "VAE(G)" \
                    and model_fields["likelihood"] == "NB" \
                    and model_fields["other"] == "BN" \
                    and model_fields["runs"] == "D":
                    
                    ELBO = model_fields["ELBO"]
                    
                    architecture = model_fields["sizes"]
                    h, l = architecture.rsplit("×", maxsplit = 1)
                    
                    version = {
                        "version": model_fields["version"],
                        "epoch_number": model_fields["epochs"],
                    }
                    
                    network_architecture_ELBOs.setdefault(l, {})
                    network_architecture_versions.setdefault(l, {})
                    
                    if h not in network_architecture_ELBOs[l]:
                        network_architecture_ELBOs[l][h] = ELBO
                        network_architecture_versions[l][h] = version
                    else:
                        previous_version = network_architecture_versions[l][h]
                        best_version = bestVariants(version, previous_version)
                        if version == best_version:
                            network_architecture_versions[l][h] = version
                            network_architecture_ELBOs[l][h] = ELBO
            
            if network_architecture_ELBOs:
                network_architecture_ELBOs = pandas.DataFrame(
                    network_architecture_ELBOs
                )
                network_architecture_ELBOs = network_architecture_ELBOs\
                    .reindex(
                        columns = sorted(
                            network_architecture_ELBOs.columns,
                            key = lambda s: int(s)
                        )
                    )
                network_architecture_ELBOs = network_architecture_ELBOs\
                    .reindex(
                        index = sorted(
                            network_architecture_ELBOs.index,
                            key = lambda s: prod(map(int, s.split("×")))
                        )
                    )
                
                if network_architecture_ELBOs.size > 1:
                    figure, figure_name = plotELBOHeatMap(
                        network_architecture_ELBOs,
                        x_label = "Latent dimension",
                        y_label = "Number of hidden units",
                        z_symbol = "\mathcal{L}",
                        name = data_set_path.replace(os.sep, "-")
                    )
                    saveFigure(figure, figure_name, export_options,
                        cross_analysis_directory)
            
            ## Table
            
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
                key = lambda key_value_pair:
                    numpy.mean(key_value_pair[-1]["ELBO"]),
                reverse = True
            )
            
            for model_title, model_fields in comparisons.items():
                for field_name, field_value in model_fields.items():
                    
                    if isinstance(field_value, list) \
                        and len(field_value) == 1:
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
                            "Type `{}` not supported in comparison table."
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
                    "{}: {}".format(capitaliseString(field_name), field_value)
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
                        capitaliseString(field_name),
                        width = field_width,
                        placeholder = "…"
                    )
                elif field_name == field_name.lower():
                    field_name = capitaliseString(field_name)
                
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
                    key = lambda key_value_pair:
                        comparison_field_names.index(key_value_pair[0])
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
                for set_name, other_method_metrics \
                    in set_other_method_metrics.items():
                    
                    for method, method_metrics \
                        in other_method_metrics.items():
                        
                        for metric_name, metric_values \
                            in method_metrics.items():
                            
                            set_metric_name = metric_name
                            
                            if set_name == "superset":
                                set_metric_name += " (superset)"
                            
                            other_methods_metric_values.setdefault(method, {})
                            other_methods_metric_values[method].setdefault(
                                set_metric_name, metric_values
                            )
                
                for method, metric_values \
                    in other_methods_metric_values.items():
                    
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
                log_string_parts.append(subtitle("Comparison", plain = True))
                log_string_parts.append(comparison_table + "\n")
                log_string_parts.append(
                    common_comparison_fields_string + "\n"
                )
                if set_other_method_metrics:
                    log_string_parts.append(other_methods_string + "\n")
            
            ## Plot
            
            ### Setup
            
            #### Model filter based on the most frequent architecture
            
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
            
            #### Metric names
            
            optimised_metric_names = ["ELBO", "ENRE", "KL_z", "KL_y"]
            optimised_metric_symbols = {
                "ELBO": r"$\mathcal{L}$",
                "ENRE": r"$\log p(x|z)$",
                "KL_z": r"KL$_z(q||p)$",
                "KL_y": r"KL$_y(q||p)$",
            }
            
            supervised_clustering_metric_names = [
                n for n, d in CLUSTERING_METRICS.items()
                if d["kind"] == "supervised"
            ]
            unsupervised_clustering_metric_names = [
                n for n, d in CLUSTERING_METRICS.items()
                if d["kind"] == "unsupervised"
            ]
            
            ### Collect metrics from relevant models
            
            model_likelihood_metrics = {}
            set_method_likelihood_metrics = {
                "standard": {},
                "superset": {},
                "unsupervised": {}
            }
            method_likelihood_variants = {}
            method_likelihood_miscellaneous = {}
            
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
                                field_values.discard(
                                    additional_other_option
                                )
                            
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
                        clustering_method = clustering_method\
                            .replace(", ", "-")
                        method_parts.append(clustering_method)
                
                if method_parts:
                    method = "-".join(method_parts)
                else:
                    method = "---"
                
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
                    best_variant = bestVariants(
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
                        pattern = r" \((.+)\)",
                        string = field_name,
                        maxsplit = 1
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
                    set_method_likelihood_metrics[set_name][method]\
                        [likelihood] = metrics[set_name]
            
            print("Plotting model metrics.")
            
            # Only optimised metrics
            
            # Clean up likelihood names
            
            models = set()
            likelihoods = set()
            
            for model in model_likelihood_metrics:
                models.add(model)
                for likelihood in model_likelihood_metrics[model]:
                    likelihoods.add(likelihood)
            
            model_replacements = \
                replacementsForCleanedUpSpecifications(
                    models,
                    detail_separator = r"\((.+)\)", 
                    specification_separator = "-"
                )
            
            likelihood_replacements = \
                replacementsForCleanedUpSpecifications(
                    likelihoods,
                    detail_separator = r"\((.+)\)", 
                    specification_separator = "-"
                )
            
            # Rearrange data
            
            metrics_sets = []
            models = set()
            likelihoods = set()
            
            for model, likelihood_metrics in \
                model_likelihood_metrics.items():
                
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
                key = createSpecificationsSorter(
                    order = LIKELIHOOD_DISRIBUTION_ORDER,
                    detail_separator = r"\((.+)\)", 
                    specification_separator = "-"
                )
            )
            
            model_order = sorted(
                models,
                key = createSpecificationsSorter(
                    order = MODEL_TYPE_ORDER,
                    detail_separator = r"\((.+)\)", 
                    specification_separator = "-"
                )
            )
            
            for optimised_metric_name in optimised_metric_names:
                
                optimised_metric_symbol = optimised_metric_symbols[
                    optimised_metric_name
                ]
                
                figure, figure_name = plotModelMetrics(
                    metrics_sets,
                    key = optimised_metric_name,
                    primary_differentiator_key = "model",
                    primary_differentiator_order = model_order,
                    secondary_differentiator_key = "likelihood",
                    secondary_differentiator_order = likelihood_order,
                    label = optimised_metric_symbol,
                    name = [
                        data_set_path.replace(os.sep, "-"),
                        optimised_metric_name
                    ]
                )
                saveFigure(figure, figure_name, export_options,
                    cross_analysis_directory)
            
            # Optimised metrics and clustering metrics
            
            for set_name, method_likelihood_metrics in \
                set_method_likelihood_metrics.items():
                
                if not method_likelihood_metrics:
                    continue
                
                # Clean up method and likelihood names
                
                methods = set()
                likelihoods = set()
                
                for method in method_likelihood_metrics:
                    methods.add(method)
                    for likelihood in method_likelihood_metrics[method]:
                        likelihoods.add(likelihood)
                
                method_replacements = \
                    replacementsForCleanedUpSpecifications(
                        methods,
                        detail_separator = r"\((.+)\)", 
                        specification_separator = "-"
                    )
                
                likelihood_replacements = \
                    replacementsForCleanedUpSpecifications(
                        likelihoods,
                        detail_separator = r"\((.+)\)", 
                        specification_separator = "-"
                    )
                
                # Rearrange data
                
                metrics_sets = []
                methods = set()
                likelihoods = set()
                
                for method, likelihood_metrics in \
                    method_likelihood_metrics.items():
                    
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
                    key = createSpecificationsSorter(
                        order = LIKELIHOOD_DISRIBUTION_ORDER,
                        detail_separator = r"\((.+)\)", 
                        specification_separator = "-"
                    )
                )
                
                method_order = sorted(
                    methods,
                    key = createSpecificationsSorter(
                        order = MODEL_TYPE_ORDER,
                        detail_separator = r"\((.+)\)", 
                        specification_separator = "-"
                    )
                )
                
                # Special cases
                
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
                    clustering_metric_names = \
                        unsupervised_clustering_metric_names
                else:
                    clustering_metric_names = \
                        supervised_clustering_metric_names
                
                # Loop over metrics
                
                for optimised_metric_name, clustering_metric_name in \
                    product(optimised_metric_names, clustering_metric_names):
                    
                    # Figure
                    
                    clustering_metric_symbol = CLUSTERING_METRICS[
                        clustering_metric_name
                    ]["symbol"]
                    optimised_metric_symbol = optimised_metric_symbols[
                        optimised_metric_name
                    ]
                    
                    figure, figure_name = plotModelMetricSets(
                        metrics_sets,
                        x_key = optimised_metric_name,
                        y_key = clustering_metric_name,
                        primary_differentiator_key = "likelihood",
                        primary_differentiator_order =
                            likelihood_order,
                        secondary_differentiator_key = "method",
                        secondary_differentiator_order = method_order,
                        special_cases = special_cases,
                        other_method_metrics = other_method_metrics,
                        x_label = optimised_metric_symbol,
                        y_label = clustering_metric_symbol,
                        name = [
                            data_set_path.replace(os.sep, "-"),
                            set_name,
                            clustering_metric_name,
                            optimised_metric_name
                        ]
                    )
                    saveFigure(figure, figure_name, export_options,
                        cross_analysis_directory)
                    
            print()
        
        if log_summary:
            
            log_string = "\n".join(log_string_parts)
            
            with open(log_path, "w") as log_file:
                log_file.write(log_string)

def metricsSetsInResultsDirectory(results_directory,
    data_set_included_strings, data_set_excluded_strings,
    model_included_strings, model_excluded_strings):
    
    metrics_filename = metrics_basename + zipped_pickle_extension
    
    metrics_set = {}
    
    for path, _, filenames in os.walk(results_directory):
        
        data_set_model = path.replace(results_directory, "")
        data_set_model_parts = data_set_model.split(os.sep)
        data_set = os.sep.join(data_set_model_parts[:3])
        model = os.sep.join(data_set_model_parts[3:])
        
        # Verify data set match
        
        data_set_match = matchString(
            data_set,
            data_set_included_strings,
            data_set_excluded_strings
        )
        
        if not data_set_match:
            continue
        
        # Verify model match
        
        model_match = matchString(
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
            
            if not data_set in metrics_set:
                metrics_set[data_set] = {}
            
            if not model in metrics_set[data_set]:
                metrics_set[data_set][model] = {}
            
            if not run in metrics_set[data_set][model]:
                metrics_set[data_set][model][run] = {}
            
            metrics_path = os.path.join(path, metrics_filename)
            
            with gzip.open(metrics_path, "r") as metrics_file:
                metrics_data = pickle.load(metrics_file)
            
            predictions = {}
            
            for filename in filenames:
                if filename.startswith(prediction_basename) \
                    and filename.endswith(zipped_pickle_extension):
                    
                    prediction_name = filename\
                        .replace(zipped_pickle_extension, "")\
                        .replace(prediction_basename, "")\
                        .replace("-", "")
                    
                    prediction_path = os.path.join(path, filename)
                    
                    with gzip.open(prediction_path, "r") as \
                        prediction_file:
                        
                        prediction_data = pickle.load(
                            prediction_file)
                    
                    predictions[prediction_name] = prediction_data
            
            if predictions:
                metrics_data["predictions"] = predictions
            
            metrics_set[data_set][model][run][version] = \
                metrics_data
    
    return metrics_set

def metricsForOtherMethods(data_set_directory,
                           other_methods=[],
                           prediction_included_strings = None,
                           prediction_excluded_strings = None):
    
    if not isinstance(other_methods, list):
        other_methods = [other_methods]
    
    other_method_metrics = {}
    
    for other_method in other_methods:
        
        other_method_data_set_directory = data_set_directory
        other_method_prediction_basename = prediction_basename
        
        other_method = properString(
            normaliseString(other_method),
            OTHER_METHOD_NAMES
        )
        directory_name = normaliseString(other_method)
        
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
                other_method_prediction_basename \
                    = other_method_prediction_basename.replace(
                        "test",
                        evaluation_set_name,
                        1
                    )
        
        method_directory = os.path.join(
            other_method_data_set_directory,
            directory_name
        )
        
        if not os.path.exists(method_directory):
            continue
        
        for path, directory_names, filenames in os.walk(method_directory):
            for filename in filenames:
                if filename.startswith(other_method_prediction_basename) \
                    and filename.endswith(zipped_pickle_extension):
                    
                    prediction_match = matchString(
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
                    
                    for metric_name, metric_set \
                        in clustering_metric_values.items():
                        
                        for metric_set_name, metric_value in metric_set.items():
                            if metric_value is None:
                                continue
                            elif metric_set_name.startswith("clusters"):
                                
                                metric_details = clustering_metrics.get(
                                    metric_name, dict())
                                metric_kind = metric_details.get("kind", None)
                                
                                if metric_kind and metric_kind == "supervised":
                                    
                                    set_name = "standard"
                                    
                                    if metric_set_name.endswith("superset"):
                                        set_name = "superset"
                                
                                elif metric_kind \
                                    and metric_kind == "unsupervised":
                                    
                                    set_name = "unsupervised"
                                
                                else:
                                    set_name = "unknown"
                                
                                other_method_metrics.setdefault(
                                    set_name, {}
                                )
                                other_method_metrics[set_name]\
                                    .setdefault(method, {})
                                other_method_metrics[set_name][method]\
                                    .setdefault(metric_name, [])
                                other_method_metrics[set_name][method]\
                                    [metric_name].append(float(metric_value))
                        
                        # metric_value = metric_set.get("clusters", None)
                        # metric_value_superset = metric_set.get(
                        #     "clusters; superset", None)
    
    return other_method_metrics

def parseMetricsForRunsAndVersionsOfModel(
        runs,
        log_summary = False,
        prediction_included_strings = None,
        prediction_excluded_strings = None,
        epoch_cut_off = inf
    ):
    
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
            print(heading(capitaliseString(run_title)))
            
            if log_summary:
                log_string_parts.append(
                    heading(capitaliseString(run_title), plain = True)
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
                        samples_parts += "{} {} samples".format(
                            field_value, field_name.upper())
                else:
                    version = version_field.replace("_", " ")\
                        .replace("best model", "optimal parameters")
            
            if number_of_epochs > epoch_cut_off:
                continue

            version_title = "; ".join([epochs, version] + samples)
            
            metrics_string_parts = []
            summary_metrics = {}
            
            # Time
            
            timestamp = metrics["timestamp"]
            metrics_string_parts.append(
                "Timestamp: {}".format(formatTime(timestamp))
            )
            
            # Epochs
            
            E = metrics["number of epochs trained"]
            metrics_string_parts.append(
                "Epochs trained: {}".format(E)
            )
            summary_metrics["epochs"] = E
            
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
            
            ELBO = evaluation.get("lower_bound", [None])[-1]
            ENRE = evaluation.get("reconstruction_error", [None])[-1]
            KL_y = evaluation.get("kl_divergence_y", [None])[-1]
            KL_z = None
            
            if "kl_divergence" in evaluation:
                KL_z = evaluation["kl_divergence"][-1]
            elif "kl_divergence_z" in evaluation:
                KL_z = evaluation["kl_divergence_z"][-1]
            
            
            summary_metrics.update({
                "ELBO": ELBO,
                "ENRE": ENRE,
                "KL_z": KL_z,
                "KL_y": KL_y
            })
            
            # Accuracies
            
            accuracies = ["accuracy", "superset_accuracy"]
            
            for accuracy in accuracies:
                if accuracy in metrics \
                    and metrics[accuracy]:
                    metrics_string_parts.append("{}: {:6.2f} %".format(
                        accuracy, 100 * metrics[accuracy][-1]
                    ))
            
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
                    formatStatistics(reconstructed_statistics)
                )
            
            metrics_string_parts.append("")
            
            # Predictions
            
            if "predictions" in metrics:
                
                for predictions in metrics["predictions"].values():
                    
                    prediction_string_parts = []
                    
                    decomposition_method = predictions.get(
                        "decomposition method",
                        None
                    )
                    
                    if decomposition_method:
                        decomposition_dimensionality = predictions.get(
                            "decomposition dimensionality",
                            None
                        )
                        
                        decomposition_string = "{} ({} components)".format(
                            decomposition_method,
                            decomposition_dimensionality
                        )
                        
                        prediction_string_parts.append(decomposition_string)
                    
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
                    
                    prediction_match = matchString(
                        prediction_string,
                        prediction_included_strings,
                        prediction_excluded_strings
                    )
                    
                    if not prediction_match:
                        continue
                    
                    clustering_metric_values = predictions.get(
                        "clustering metric values", None
                    )
                    
                    if not clustering_metric_values:
                        
                        old_ARIs = {}
                        
                        for key, value in predictions.items():
                            if key.startswith("ARI") and value:
                                label = re.sub(
                                    r"ARI \((.+)\)",
                                    r"\1",
                                    key
                                )
                                old_ARIs[label] = value
                        
                        if old_ARIs:
                            clustering_metric_values = {
                                "adjusted Rand index": old_ARIs
                            }
                    
                    if clustering_metric_values:
                        
                        metrics_string_parts.append(
                            prediction_string + ":")
                        
                        for metric_name, set_metrics in \
                            clustering_metric_values.items():
                            
                            metrics_string_parts.append(
                                "    {}:".format(
                                    capitaliseString(metric_name)
                                )
                            )
                            
                            for set_name, set_value in \
                                set_metrics.items():
                                
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
                                    
                                    correlation_sets[correlation_set_name]\
                                        ["ELBO"].append(ELBO)
                                    correlation_sets[correlation_set_name]\
                                        ["clustering metric"]\
                                        .append(set_value)
                        
                        metrics_string_parts.append("")
            
            metrics_string = "\n".join(metrics_string_parts)
            
            if len(versions) > 1:
                print(subheading(capitaliseString(version_title)))
            
            print(metrics_string)
            
            if log_summary:
                if len(versions) > 1:
                    log_string_parts.append(
                        subheading(capitaliseString(version_title),
                            plain = True)
                    )
                log_string_parts.append(metrics_string)
            
            # Summary metrics
            
            version_key = "; ".join([version] + samples)
            
            version_epoch_summary_metrics.setdefault(version_key, {})
            version_epoch_summary_metrics[version_key][number_of_epochs] \
                = summary_metrics
        
        for version_key, epoch_summary_metrics in \
            version_epoch_summary_metrics.items():
            
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
                    run_version_summary_metrics[run_key][version_key]\
                        [metric_key] = metric_value
                else:
                    run_version_summary_metrics[run_key][version_key]\
                        .setdefault(metric_key, [])
                    run_version_summary_metrics[run_key][version_key]\
                        [metric_key].append(metric_value)
    
    results = {
        "summary_metrics_sets": [],
        "correlation_sets": correlation_sets
    }
    
    for run_key, version_summary_metrics in \
        run_version_summary_metrics.items():
        
        for version_key, summary_metrics in version_summary_metrics.items():
            
            # Runs
            
            if run_key == "default":
                runs = "default run"
            else:
                runs = summary_metrics["runs"]
                if isinstance(runs, int):
                    runs = "{} runs".format(runs)
            
            summary_metrics["runs"] = runs
            
            # Clustering
            
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

def matchString(string, included_strings, excluded_strings):
    
    match = True
    
    for search_string in included_strings:
        if search_string in string:
            match *= True
        else:
            match *= False
    
    for search_string in excluded_strings:
        if search_string not in string:
            match *= True
        else:
            match *= False
    
    return match

def titleFromName(name, replacement_dictionaries = None):
    
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

data_set_name_replacements = {
    "10x": "10x",
    "10x_20k": "10x (20k samples)",
    "10x_arc_lira": "10x ARC LIRA",
    "development": "Development",
    r"dimm_sc_10x_(\w+)": lambda match: "3′ ({})".format(match.group(1)),
    "gtex": "GTEx",
    r"mnist_(\w+)": lambda match: "MNIST ({})".format(match.group(1)),
    r"sample_?(sparse)?": lambda match: "Sample"
        if len(match.groups()) == 1
        else "Sample ({})".format(match.group(1)),
    "tcga_kallisto": "TCGA (Kallisto)"
}

split_replacements = {
    r"split-(\w+)_(0\.\d+)": lambda match:
        "{} split ({:.3g} %)".format(
            match.group(1),
            100 * float(match.group(2))
        )
}

feature_replacements = {
    "features_mapped": "feature mapping",
    r"keep_gini_indices_above_([\d.]+)": lambda match:
        "features with Gini index above {}".format(int(float(match.group(1)))),
    r"keep_highest_gini_indices_([\d.]+)": lambda match:
        " {} features with highest Gini indices".format(
            int(float(match.group(1)))),
    r"keep_variances_above_([\d.]+)": lambda match:
        "features with variance above {}".format(
            int(float(match.group(1)))),
    r"keep_highest_variances_([\d.]+)": lambda match:
        "{} most varying features".format(int(float(match.group(1))))
}

example_feaute_replacements = {
    "macosko": "Macosko",
    "remove_zeros": "examples with only zeros removed",
    r"remove_count_sum_above_([\d.]+)": lambda match:
        "examples with count sum above {} removed".format(
            int(float(match.group(1))))
}

example_replacements = {
    r"keep_(\w+)": lambda match: "{} examples".format(match.group(1)\
        .replace("_", ", ")),
    r"remove_(\w+)": lambda match: "{} examples removed".format(match.group(1)\
        .replace("_", ", ")),
    "excluded_classes": "excluded classes removed",
}

preprocessing_replacements = {
    "gini": "Gini indices",
    "idf": "IDF"
}

def dataSetTitleFromDataSetName(name):
    
    replacement_dictionaries = [
        data_set_name_replacements,
        split_replacements,
        feature_replacements,
        example_feaute_replacements,
        example_replacements,
        preprocessing_replacements
    ]
    
    return titleFromName(name, replacement_dictionaries)

reorder_replacements = {
    r"(-sum)(-l_\d+-h_[\d_]+)": lambda match: "".join(reversed(match.groups()))
}

model_replacements = {
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

secondary_model_replacements = {
    r"gaussian_mixture-c_(\d+)": lambda match: "GM({})".format(match.group(1)),
    r"-ia_(\w+)": lambda match: ", i: {}".format(match.group(1)),
    r"-ga_(\w+)": lambda match: ", g: {}".format(match.group(1))
}

distribution_modification_replacements = {
    "constrained_poisson": "CP",
    "zero_inflated_": "ZI",
    r"/(\w+)-k_(\d+)": lambda match: "/PC{}({})".format(match.group(1),
        match.group(2))
}

distribution_replacements = {
    "gaussian": "G",
    "bernoulli": "B",
    "poisson": "P",
    "negative_binomial": "NB",
    "lomax": "L",
    "pareto": "Pa",
}

network_replacements = {
    r"l_(\d+)-h_([\d_]+)": lambda match: "{}×{}".format(
        match.group(2).replace("_", "×"),
        match.group(1)
    )
}

sample_replacements = {
    r"-mc_(\d+)": lambda match: "" if int(match.group(1)) == 1 else
        "-{} MC samples".format(match.groups(1)),
    r"-iw_(\d+)": lambda match: "" if int(match.group(1)) == 1 else
        "-{} IW samples".format(match.groups(1))
}

miscellaneous_replacements = {
    "sum": "CS",
    "-kl-": "-",
    "bn": "BN",
    r"dropout_([\d._]+)": lambda match: "dropout: {}".format(
        match.group(1).replace("_", ", ")),
    r"wu_(\d+)": lambda match: "WU({})".format(match.group(1)),
    r"klw_([\d.]+)": lambda match: "KLW: {}".format(match.group(1))
}

def modelTitleFromModelName(name):
    
    replacement_dictionaries = [
        reorder_replacements,
        model_replacements,
        secondary_model_replacements,
        distribution_modification_replacements,
        distribution_replacements,
        network_replacements,
        sample_replacements,
        miscellaneous_replacements
    ]
    
    return titleFromName(name, replacement_dictionaries)

def clusteringMethodTitleFromClusteringMethodName(name):
    
    replacement_dictionaries = [
        {
            r"model \(\d+ classes\)": "M"
        },
        {
            r"(\w+) \((\d+) classes\)": r"\1(\2)",
            r"(\w+) \((\d+) components\)": r"\1(\2)",
            "k-means": "kM",
            "t-SNE": "tSNE"
        }
    ]
    
    return titleFromName(name, replacement_dictionaries)

def modelID():
    
    numbers = list(map(str, range(10)))
    letters = list(ascii_uppercase)
    
    values = numbers + letters
    
    for value1, value2 in product(values, values):
        model_id = value1 + value2
        if model_id.isdigit():
            continue
        yield model_id

def bestVariants(*variants, additional_other_option=None):
    
    def variantSortKey(variant):
        
        # Other
        
        other = variant.get("other", None)
        
        if other:
            other_set = set(other.split("; "))
        else:
            other_set = set()
        
        if additional_other_option in other_set:
            additional_other_option_available = True
        else:
            additional_other_option_available = False
        
        # Version
        
        version_rankings = {
            "EOT": 0, # end of training
            "ES": 1,  # early stopping
            "OP": 2   # optimal parameters
        }
        
        version = variant.get("version", None)
        ranking = version_rankings.get(version, -1)
        
        # Epoch number
        
        epoch_number = variant.get("epoch_number", -1)
        
        if isinstance(epoch_number, list):
            epoch_number = statistics.mean(epoch_number)
        
        # Sort key
        
        variant_sort_key = [
            additional_other_option_available,
            ranking,
            epoch_number
        ]
        
        return variant_sort_key
    
    sorted_variants = sorted(variants, key = variantSortKey)
    best_variant = sorted_variants[-1]
    
    return best_variant

def comparisonTableColumnSorter(name):
    name = str(name)
    
    column_names = SORTED_COMPARISON_TABLE_COLUMN_NAMES
    
    K = len(column_names)
    index_width = len(str(K))
    
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
        index = K
    
    name =  "{:{}d} {}".format(index, index_width, name)
    
    return name

def replacementsForCleanedUpSpecifications(
        specification_sets = set(),
        detail_separator = "", 
        specification_separator = ""
    ):
    
    # Categorise specification variations
    
    specification_types = {}
    
    for specification_set in specification_sets:
        specifications = re.split(specification_separator, specification_set)
        
        for i, specification in enumerate(specifications):
            specification_parts = re.split(detail_separator, specification,
                maxsplit = 1)
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
            specification_parts = re.split(detail_separator, specification,
                maxsplit = 1)
            specification_type = specification_parts[0]
            
            if len(specification_types[i][specification_type]) <= 1:
                replacement = specification_type
            else:
                replacement = specification
            
            replacement_parts.append(replacement)
        
        replacements[specification_set] = specification_separator.join(
            replacement_parts)
    
    return replacements

def createSpecificationsSorter(order = [], detail_separator = "",
    specification_separator = ""):
    
    def specificationsSorter(specifications):
        specifications = re.split(specification_separator, specifications)
        
        key = []
        
        for specification in specifications:
            specification_parts = re.split(detail_separator, specification,
                maxsplit = 1)
            specification_type = specification_parts[0]
            if specification_type in order:
                specification_ranking = order.index(specification_type)
            else:
                specification_ranking = -1
            key.append(specification_ranking)
            if len(specification_parts) > 1:
                key.extend(specification_parts[1:])
        
        return key
    
    return specificationsSorter

parser = argparse.ArgumentParser(
    description="Cross-analyse models.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--log-directory", "-L",
    type = str,
    help = "directory where models were logged"
)
parser.add_argument(
    "--results-directory", "-R",
    type = str,
    help = "directory where results were saved"
)
parser.add_argument(
    "--data-set-included-strings", "-d",
    type = str,
    nargs = "*",
    default = [],
    help = "strings to include in data set directories"
)
parser.add_argument(
    "--data-set-excluded-strings", "-D",
    type = str,
    nargs = "*",
    default = [],
    help = "strings to exclude in data set directories"
)
parser.add_argument(
    "--model-included-strings", "-m",
    type = str,
    nargs = "*",
    default = [],
    help = "strings to include in model directories"
)
parser.add_argument(
    "--model-excluded-strings", "-M",
    type = str,
    nargs = "*",
    default = [],
    help = "strings to exclude in model directories"
)
parser.add_argument(
    "--prediction-included-strings", "-p",
    type = str,
    nargs = "*",
    default = [],
    help = "strings to include in prediction methods"
)
parser.add_argument(
    "--prediction-excluded-strings", "-P",
    type = str,
    nargs = "*",
    default = [],
    help = "strings to exclude in prediction methods"
)
parser.add_argument(
    "--additional-other-option", "-a",
    type = str,
    nargs = "?",
    default = None,
    help = "additional other option for models in model metrics plots"
)
parser.add_argument(
    "--epoch-cut-off", "-e",    
    type = int,
    default = inf
)
parser.add_argument(
    "--other-methods", "-o",
    type = str,
    nargs = "*",
    default = [],
    help = "other methods to plot in model metrics plot, if available"
)
parser.add_argument(
    "--export-options",
    type = str,
    nargs = "?",
    default = [],
    help = "analyse model evolution for video"
)
parser.add_argument(
    "--log-summary", "-s",
    action = "store_true",
    help = "log summary (saved in results directory)"
)
parser.add_argument(
    "--skip-logging-summary", "-S",
    dest = "log_summary",
    action = "store_false",
    help = "do not log summary"
)
parser.set_defaults(log_summary = False)

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
