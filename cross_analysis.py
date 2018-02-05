#!/usr/bin/env python3

import os

import pickle
import gzip

import re

from itertools import product
from string import ascii_uppercase

import argparse

from analysis import formatStatistics
from auxiliary import formatTime, title, subtitle

test_metrics_basename = "test-metrics"
test_prediction_basename = "test-prediction"

zipped_pickle_extension = ".pkl.gz"
log_extension = ".log"

def main(log_directory = None, results_directory = None,
    data_set_include_search_strings = [], 
    model_include_search_strings = [],
    data_set_exclude_search_strings = [], 
    model_exclude_search_strings = [],
    log_summary = False):
    
    if log_directory:
        log_directory = os.path.normpath(log_directory) + os.sep

    if results_directory:
        
        results_directory = os.path.normpath(results_directory) + os.sep
        
        explanation_string_parts = []
        
        def appendExplanationForSearchStrings(search_strings, inclusive, kind):
            if search_strings:
                explanation_string_parts.append("{} {} with: {}.".format(
                    "Including" if inclusive else "Excluding",
                    kind,
                    ", ".join(search_strings)
                ))
        
        appendExplanationForSearchStrings(
            data_set_include_search_strings,
            inclusive = True,
            kind = "data sets"
        )
        appendExplanationForSearchStrings(
            data_set_exclude_search_strings,
            inclusive = False,
            kind = "data sets"
        )
        appendExplanationForSearchStrings(
            model_include_search_strings,
            inclusive = True,
            kind = "models"
        )
        appendExplanationForSearchStrings(
            model_exclude_search_strings,
            inclusive = False,
            kind = "models"
        )
        
        explanation_string = "\n".join(explanation_string_parts)
        
        print(explanation_string)
        
        print()
        
        if log_summary:
            
            log_filename_parts = []
            
            def appendSearchStrings(search_strings, symbol):
                if search_strings:
                    log_filename_parts.append("{}_{}".format(
                        symbol,
                        "_".join(search_strings)
                    ))
            
            appendSearchStrings(data_set_include_search_strings, "d")
            appendSearchStrings(data_set_exclude_search_strings, "D")
            appendSearchStrings(model_include_search_strings, "m")
            appendSearchStrings(model_exclude_search_strings, "M")
            
            if not log_filename_parts:
                log_filename_parts.append("all")
            
            log_filename = "-".join(log_filename_parts) + log_extension
            log_path = os.path.join(results_directory, log_filename)
            
            log_string_parts = [explanation_string + "\n"]
        
        test_metrics_set = testMetricsInResultsDirectory(
            results_directory,
            data_set_include_search_strings,
            data_set_exclude_search_strings,
            model_include_search_strings,
            model_exclude_search_strings
        )
        
        model_IDs = modelID()
        
        for data_set_name, models in test_metrics_set.items():
            
            data_set_title = titleFromDataSetName(data_set_name)
            
            print(title(data_set_title))
            
            if log_summary:
                log_string_parts.append(title(data_set_title, plain = True))
            
            comparisons = {}
            
            for model_name, test_metrics in models.items():
                
                model_title = titleFromModelName(model_name)
                
                metrics_string_parts = []
                
                # ID
                
                model_ID = next(model_IDs)
                metrics_string_parts.append(
                    "ID: {}".format(model_ID)
                )
                
                # Time
                
                timestamp = test_metrics["timestamp"]
                metrics_string_parts.append(
                    "Timestamp: {}".format(formatTime(timestamp))
                )
                
                # Epochs
                
                E = test_metrics["number of epochs trained"]
                metrics_string_parts.append("Epochs trained: {}".format(E))
                
                metrics_string_parts.append("")
                
                # Evaluation
                
                evaluation = test_metrics["evaluation"]
                
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
                            "{}: {:.5g}".format(loss, evaluation[loss][-1])
                        )
                
                if "lower_bound" in evaluation:
                    model_lower_bound = evaluation["lower_bound"][-1]
                else:
                    model_lower_bound = None
                
                # Accuracies
                
                accuracies = ["accuracy", "superset_accuracy"]
                
                for accuracy in accuracies:
                    if accuracy in test_metrics and test_metrics[accuracy]:
                        metrics_string_parts.append("{}: {:6.2f} %".format(
                            accuracy, 100 * test_metrics[accuracy][-1]))
                
                metrics_string_parts.append("")
                
                # Statistics
                
                if isinstance(test_metrics["statistics"], list):
                    statistics_sets = test_metrics["statistics"]
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
                
                model_ARIs = []
                
                if "predictions" in test_metrics:
                    
                    for prediction in test_metrics["predictions"].values():
                        
                        ARIs = {}
                        
                        for key, value in prediction.items():
                            if key.startswith("ARI") and value:
                                ARIs[key] = value
                        
                        method = prediction["prediction method"]
                        number_of_classes = prediction["number of classes"]
                        
                        if ARIs:
                            metrics_string_parts.append(
                                "{} ({} classes):".format(
                                    method, number_of_classes
                                )
                            )
                            
                            for ARI_name, ARI_value in ARIs.items():
                                metrics_string_parts.append(
                                    "    {}: {:.5g}".format(
                                        ARI_name, ARI_value
                                    )
                                )
                            
                            metrics_string_parts.append("")
                        
                        model_ARIs.extend(ARIs.values())
                
                comparisons[model_title] = {
                    "ID": model_ID,
                    "ELBO": model_lower_bound,
                    "ARI": model_ARIs
                }
                
                metrics_string = "\n".join(metrics_string_parts)
                
                print(subtitle(model_title))
                print(metrics_string)
                
                if log_summary:
                    log_string_parts.append(
                        subtitle(model_title, plain = True)
                    )
                    log_string_parts.append(metrics_string)
            
            if len(comparisons) <= 1:
                continue
            
            # Comparison
            
            model_spec_names = [
                "ID",
                "type",
                "distribution",
                "sizes",
                "other",
                "epochs"
            ]
            
            model_spec_short_names = {
                "ID": "#",
                "type": "T",
                "distribution": "LD",
                "sizes": "S",
                "other": "O",
                "epochs": "E"
            }
            
            model_metric_names = [
                "ELBO",
                "ARI"
            ]
            
            model_field_names = model_spec_names + model_metric_names
            
            for model_title in comparisons:
                model_title_parts = model_title.split("; ")
                comparisons[model_title].update({
                    "type": model_title_parts.pop(0),
                    "distribution": model_title_parts.pop(0),
                    "sizes": model_title_parts.pop(0),
                    "epochs": model_title_parts.pop(-1).replace(" epochs", ""),
                    "other": "; ".join(model_title_parts)
                })
            
            sorted_comparison_items = sorted(
                comparisons.items(),
                key = lambda key_value_pair: key_value_pair[-1]["ELBO"],
                reverse = True
            )
            
            for model_title, model_fields in comparisons.items():
                for field_name, field_value in model_fields.items():
                    
                    if isinstance(field_value, str):
                        continue
                    
                    elif not field_value:
                        string = ""
                    
                    elif isinstance(field_value, float):
                        string = "{:-.5g}".format(field_value)
                    
                    elif isinstance(field_value, int):
                        string = "{:d}".format(field_value)
                    
                    elif isinstance(field_value, list):
                        
                        minimum = min(field_value)
                        maximum = max(field_value)
                        
                        if minimum == maximum:
                            string = "{:4.2f}".format(maximum)
                        else:
                            string = "{:4.2f}--{:4.2f}".format(
                                minimum, maximum)
                    
                    else:
                        raise TypeError(
                            "Type `{}` not supported in comparison table."
                                .format(type(field_value))
                        )
                    
                    comparisons[model_title][field_name] = string
            
            comparison_table_rows = []
            table_column_spacing = "  "
            
            comparison_table_column_widths = {}
            
            for field_name in model_field_names:
                comparison_table_column_widths[field_name] = max(
                    [len(metrics[field_name]) for metrics in
                        comparisons.values()]
                )
            
            comparison_table_heading_parts = []
            
            for field_name in model_field_names:
                
                field_width = comparison_table_column_widths[field_name]
                
                if field_width == 0:
                    continue
                
                if field_name in model_spec_names:
                    if len(field_name) > field_width:
                        field_name = model_spec_short_names[field_name]
                    elif field_name == field_name.lower():
                        field_name = field_name.capitalize()
                
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
                        model_field_names.index(key_value_pair[0])
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

            if log_summary:
                log_string_parts.append(subtitle("Comparison", plain = True))
                log_string_parts.append(comparison_table + "\n")
        
        if log_summary:
            
            log_string = "\n".join(log_string_parts)
            
            with open(log_path, "w") as log_file:
                log_file.write(log_string)

def testMetricsInResultsDirectory(results_directory,
    data_set_include_search_strings, data_set_exclude_search_strings,
    model_include_search_strings, model_exclude_search_strings):
    
    test_metrics_filename = test_metrics_basename + zipped_pickle_extension
    
    test_metrics_set = {}
    
    for path, _, filenames in os.walk(results_directory):
        
        data_set_model = path.replace(results_directory, "")
        data_set_model_parts = data_set_model.split(os.sep)
        data_set = os.sep.join(data_set_model_parts[:3])
        model = os.sep.join(data_set_model_parts[3:])
        
        # Verify data set match
        
        data_set_match = True
        
        for data_set_search_string in data_set_include_search_strings:
            if data_set_search_string in data_set:
                data_set_match *= True
            else:
                data_set_match *= False
        
        for data_set_search_string in data_set_exclude_search_strings:
            if data_set_search_string not in data_set:
                data_set_match *= True
            else:
                data_set_match *= False
        
        if not data_set_match:
            continue
        
        # Verify model match
        
        model_match = True
        
        for model_search_string in model_include_search_strings:
            if model_search_string in model:
                model_match *= True
            else:
                model_match *= False
        
        for model_search_string in model_exclude_search_strings:
            if model_search_string not in model:
                model_match *= True
            else:
                model_match *= False
        
        if not model_match:
            continue
        
        # Verify metrics found
        
        if test_metrics_filename in filenames:
            
            if not data_set in test_metrics_set:
                test_metrics_set[data_set] = {}
            
            test_metrics_path = os.path.join(path, test_metrics_filename)
            
            with gzip.open(test_metrics_path, "r") as test_metrics_file:
                test_metrics_data = pickle.load(test_metrics_file)
            
            predictions = {}
            
            for filename in filenames:
                if filename.startswith(test_prediction_basename) \
                    and filename.endswith(zipped_pickle_extension):
                    
                    prediction_name = filename\
                        .replace(zipped_pickle_extension, "")\
                        .replace(test_prediction_basename, "")\
                        .replace("-", "")
                    
                    test_prediction_path = os.path.join(path, filename)
                    
                    with gzip.open(test_prediction_path, "r") as \
                        test_prediction_file:
                        
                        test_prediction_data = pickle.load(
                            test_prediction_file)
                    
                    predictions[prediction_name] = test_prediction_data
            
            if predictions:
                test_metrics_data["predictions"] = predictions
            
            test_metrics_set[data_set][model] = test_metrics_data
    
    return test_metrics_set

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

def titleFromDataSetName(name):
    
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

model_version_replacements = {
    r"e_(\d+)-?(\w+)?": lambda match: "{} epochs".format(match.group(1))
        if not match.group(2)
        else "{} epochs ({})".format(
            match.group(1),
            match.group(2)
        ),
    "best_model": "*",
    "early_stopping": "ES"
}

miscellaneous_replacements = {
    "sum": "CS",
    "-kl": "",
    "bn": "BN",
    r"dropout_([\d._]+)": lambda match: "dropout: {}".format(
        match.group(1).replace("_", ", ")),
    r"wu_(\d+)": lambda match: "WU({})".format(match.group(1))
}

def titleFromModelName(name):
    
    replacement_dictionaries = [
        reorder_replacements,
        model_replacements,
        secondary_model_replacements,
        distribution_modification_replacements,
        distribution_replacements,
        network_replacements,
        sample_replacements,
        model_version_replacements,
        miscellaneous_replacements
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
    "--data-set-include-search-strings", "-d",
    type = str,
    nargs = "*",
    default = [],
    help = "list of search strings to include in data set directories"
)
parser.add_argument(
    "--model-include-search-strings", "-m",
    type = str,
    nargs = "*",
    default = [],
    help = "list of search strings to include in model directories"
)
parser.add_argument(
    "--data-set-exclude-search-strings", "-D",
    type = str,
    nargs = "*",
    default = [],
    help = "list of search strings to exclude in data set directories"
)
parser.add_argument(
    "--model-exclude-search-strings", "-M",
    type = str,
    nargs = "*",
    default = [],
    help = "list of search strings to exclude in model directories"
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
