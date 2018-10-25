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
    plotCorrelations, plotELBOHeatMap
)
from auxiliary import (
    formatTime, capitaliseString,
    title, subtitle, heading, subheading,
    prod
)

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


def main(log_directory = None, results_directory = None,
    data_set_included_strings = [], 
    data_set_excluded_strings = [], 
    model_included_strings = [],
    model_excluded_strings = [],
    prediction_included_strings = [],
    prediction_excluded_strings = [],
    epoch_cut_off = inf,
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
                explanation_string_parts.append("{} {} with: {}.".format(
                    "Including" if search_inclusiveness else "Excluding",
                    search_kind,
                    ", ".join(search_strings)
                ))
        
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
        
        for data_set_name, models in metrics_sets.items():
            
            data_set_title = dataSetTitleFromDataSetName(data_set_name)
            
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
                    
                    for set_key, set_value in \
                        model_summary_metrics_set.items():
                            summary_metrics_set[set_key] = set_value
                    
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
                
                correlation_string_parts.append("")
                correlation_string_parts.append("Plotting correlations.")
                figure, figure_name = plotCorrelations(
                    correlation_sets,
                    x_key = "ELBO",
                    y_key = "clustering metric",
                    x_label = r"$\mathcal{L}$",
                    y_label = "",
                    name = data_set_name.replace(os.sep, "-")
                )
                saveFigure(figure, figure_name, export_options,
                    cross_analysis_directory)
                
                correlation_string = "\n".join(correlation_string_parts)
                
                print(subtitle("ELBO--ARI correlations"))
                print(correlation_string + "\n")
            
                if log_summary:
                    log_string_parts.append(subtitle("ELBO--ARI correlations", plain = True))
                    log_string_parts.append(correlation_string + "\n")
            
            # Comparisons
            
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
            
            for model_fields in summary_metrics_sets.values():
                if model_fields["type"] == "VAE(G)" \
                    and model_fields["likelihood"] == "NB" \
                    and model_fields["other"] == "BN" \
                    and model_fields["runs"] == "D":

                    epochs = model_fields["epochs"]
                    version = model_fields["version"]
                    architecture = model_fields["sizes"]
                    ELBO = model_fields["ELBO"]

                    h, l = architecture.rsplit("×", maxsplit = 1)

                    network_architecture_ELBOs.setdefault(l, {})

                    if h not in network_architecture_ELBOs[l]:
                        network_architecture_ELBOs[l][h] = ELBO
                    else:
                        current_ELBO = network_architecture_ELBOs[l][h]
                        if ELBO > current_ELBO:
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
                        name = data_set_name.replace(os.sep, "-")
                    )
                    saveFigure(figure, figure_name, export_options,
                        cross_analysis_directory)
            
            ## Table
            
            comparisons = copy.deepcopy(summary_metrics_sets)
            comparison_field_names = copy.deepcopy(model_field_names)
            
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
                        
                        field_values = numpy.array(field_value)
                        
                        mean = field_values.mean()
                        sd = field_values.std()
                        
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
            
            if log_summary:
                log_string_parts.append(subtitle("Comparison", plain = True))
                log_string_parts.append(comparison_table + "\n")
                log_string_parts.append(
                    common_comparison_fields_string + "\n"
                )
        
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
            
            if "lower_bound" in evaluation:
                ELBO = evaluation["lower_bound"][-1]
            else:
                ELBO = None
            
            summary_metrics["ELBO"] = ELBO
            
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
                    
                    method = predictions["prediction method"]
                    number_of_classes = predictions["number of classes"]
                    
                    if not method:
                        method = "model"
                    
                    prediction_string = "{} ({} classes)".format(
                        method, number_of_classes)
                    
                    prediction_match = matchString(
                        prediction_string,
                        prediction_included_strings,
                        prediction_excluded_strings
                    )
                    
                    if not prediction_match:
                        continue
                    
                    clustering_metrics = predictions.get(
                        "clustering metric values", None
                    )
                    
                    if not clustering_metrics:
                        
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
                            clustering_metrics = {
                                "adjusted Rand index": old_ARIs
                            }
                    
                    if clustering_metrics:
                        
                        metrics_string_parts.append(
                            prediction_string + ":")
                        
                        for metric_name, set_metrics in \
                            clustering_metrics.items():
                            
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
            
            clustering_metrics = {}
            
            for field_name in clustering_field_names:
                metric_value = summary_metrics.pop(field_name, None)
                if metric_value:
                    field_name_parts = field_name.split("; ")
                    method = field_name_parts[1]
                    name = field_name_parts[2]
                    clustering_metrics.setdefault(method, {})
                    clustering_metrics[method][name] = metric_value
            
            if clustering_metrics:
                original_summary_metrics = summary_metrics
                
                for method in clustering_metrics:
                    summary_metrics = copy.deepcopy(
                        original_summary_metrics
                    )
                    summary_metrics.update(clustering_metrics[method])
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
    "-kl": "",
    "bn": "BN",
    r"dropout_([\d._]+)": lambda match: "dropout: {}".format(
        match.group(1).replace("_", ", ")),
    r"wu_(\d+)": lambda match: "WU({})".format(match.group(1))
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
            "k-means": "kM"
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
    "--epoch-cut-off",
    type = int,
    default = inf
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
parser.add_argument(
    "--export-options",
    type = str,
    nargs = "?",
    default = [],
    help = "analyse model evolution for video"
)

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
