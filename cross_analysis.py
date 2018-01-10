#!/usr/bin/env python3

import os

import pickle
import gzip

import re

import argparse

from analysis import formatStatistics
from auxiliary import formatTime, title, subtitle

test_metrics_basename = "test-metrics"
test_prediction_basename = "test-prediction"

zipped_pickle_extension = ".pkl.gz"

def main(log_directory = None, results_directory = None,
    data_set_include_search_strings = [], 
    model_include_search_strings = [],
    data_set_exclude_search_strings = [], 
    model_exclude_search_strings = []):
    
    if log_directory:
        log_directory = os.path.normpath(log_directory) + os.sep

    if results_directory:
        
        results_directory = os.path.normpath(results_directory) + os.sep
        
        test_metrics_set = testMetricsInResultsDirectory(results_directory)
        
        for data_set_name, models in test_metrics_set.items():
            
            data_set_match = True
            
            for data_set_search_string in data_set_include_search_strings:
                if data_set_search_string in data_set_name:
                    data_set_match *= True
                else:
                    data_set_match *= False
            
            for data_set_search_string in data_set_exclude_search_strings:
                if data_set_search_string not in data_set_name:
                    data_set_match *= True
                else:
                    data_set_match *= False
            
            if not data_set_match:
                continue
            
            matched_model_names = []
            
            for model_name in models:
                
                model_match = True
                
                for model_search_string in model_include_search_strings:
                    if model_search_string in model_name:
                        model_match *= True
                    else:
                        model_match *= False
                
                for model_search_string in model_exclude_search_strings:
                    if model_search_string not in model_name:
                        model_match *= True
                    else:
                        model_match *= False
                
                if model_match:
                    matched_model_names.append(model_name)
            
            if not matched_model_names:
                continue
            
            data_set_title = titleFromDataSetName(data_set_name)
            
            print(title(data_set_title))
            
            comparison_table = {}
            
            for model_name in matched_model_names:
                
                model_title = titleFromModelName(model_name)
                test_metrics = models[model_name]
                
                metrics_string_parts = []
                
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
                
                model_ARI = None
                
                if "predictions" in test_metrics:
                    
                    ARI_min = 2
                    ARI_max = -1
                    any_ARIs = False
                    
                    for prediction in test_metrics["predictions"].values():
                        
                        ARIs = {}
                        
                        for key, value in prediction.items():
                            if key.startswith("ARI") and value:
                                ARIs[key] = value
                                ARI_min = min(ARI_min, value)
                                ARI_max = max(ARI_max, value)
                                any_ARIs = True
                        
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
                    
                    if any_ARIs:
                        model_ARI = {
                            "min": ARI_min,
                            "max": ARI_max
                        }
                
                comparison_table[model_title] = {
                    "lower bound": model_lower_bound,
                    "ARI": model_ARI
                }
                
                metrics_string = "\n".join(metrics_string_parts)
                
                print(subtitle(model_title))
                print(metrics_string)
            
            if len(comparison_table) <= 1:
                continue
            
            # Comparison
            
            comparison_table_rows = []
            table_column_spacing = "  "
            
            sorted_comparison_table_items = sorted(
                comparison_table.items(),
                key = lambda key_value_pair: key_value_pair[-1]["lower bound"],
                reverse = True
            )
            
            model_title_width = max(map(len, comparison_table))
            lower_bound_width = max(map(
                lambda ELBO: len("{:-.5g}".format(ELBO)),
                [metrics["lower bound"] for metrics
                    in comparison_table.values()]
            ))
            
            any_ARIs = any([
                bool(metrics["ARI"]) for metrics in comparison_table.values()
            ])
            
            comparison_table_heading_parts = [
                "{:{}}".format("Model", model_title_width),
                "{:{}}".format("ELBO", lower_bound_width)
            ]
            
            if any_ARIs:
                comparison_table_heading_parts.append("ARI       ")
            
            comparison_table_heading = table_column_spacing.join(
                comparison_table_heading_parts
            )
            comparison_table_toprule = "-" * len(comparison_table_heading)
            
            comparison_table_rows.append(comparison_table_heading)
            comparison_table_rows.append(comparison_table_toprule)
            
            for model_title, model_metrics in sorted_comparison_table_items:
                
                comparison_table_row_parts = [
                    "{:{}}".format(model_title, model_title_width),
                    "{:{}.5g}".format(model_metrics["lower bound"],
                        lower_bound_width)
                ]
                
                if model_metrics["ARI"]:
                    
                    ARI_min = model_metrics["ARI"]["min"]
                    ARI_max = model_metrics["ARI"]["max"]
                    
                    if ARI_min == ARI_max:
                        ARI_string = "      {:4.2f}".format(ARI_max)
                    else:
                        ARI_string = "{:4.2f}--{:4.2f}".format(
                            ARI_min, ARI_max)
                    
                    comparison_table_row_parts.append(ARI_string)
                    
                comparison_table_rows.append(
                    table_column_spacing.join(comparison_table_row_parts)
                )
            
            comparison_table = "\n".join(comparison_table_rows)
            
            print(subtitle("Comparison"))
            print(comparison_table + "\n")

def testMetricsInResultsDirectory(results_directory):
    
    test_metrics_filename = test_metrics_basename + zipped_pickle_extension
    
    test_metrics_set = {}
    
    for path, _, filenames in os.walk(results_directory):
        
        if test_metrics_filename in filenames:
            
            data_set_model = path.replace(results_directory, "")
            data_set_model_parts = data_set_model.split(os.sep)
            data_set = os.sep.join(data_set_model_parts[:3])
            model = os.sep.join(data_set_model_parts[3:])
            
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
    
    name = name.replace("/", " ▸ ")
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
    r"sample_?(sparse)?": lambda match: "Sample" \
        if len(match.groups()) == 1
        else "Sample ({})".format(match.group(1)),
    "tcga_kallisto": "TCGA (Kallisto)"
}

split_replacements = {
    r"split-(\w+)_(0\.\d+)": lambda match: \
        "{} split ({:.3g} %)".format(
            match.group(1),
            100 * float(match.group(2))
        )
}

feature_replacements = {
    "features_mapped": "feature mapping",
    r"keep_gini_indices_above_([\d.]+)": lambda match: \
        "features with Gini index above {}".format(int(float(match.group(1)))),
    r"keep_highest_gini_indices_([\d.]+)": lambda match: \
        " {} features with highest Gini indices".format(
            int(float(match.group(1)))),
    r"keep_variances_above_([\d.]+)": lambda match: \
        "features with variance above {}".format(
            int(float(match.group(1)))),
    r"keep_highest_variances_([\d.]+)": lambda match: \
        "{} most varying features".format(int(float(match.group(1))))
}

example_feaute_replacements = {
    "macosko": "Macosko",
    "remove_zeros": "examples with only zeros removed",
    r"remove_count_sum_above_([\d.]+)": lambda match: \
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

GMVAE_replacements = {
    r"GMVAE/gaussian_mixture-c_(\d+)-?p?_?(\w+)?": lambda match: \
        "GMVAE({})".format(match.group(1)) \
        if not match.group(2) \
        else "GMVAE({}; {})".format(*match.groups())
}

mixture_replacements = {
    r"gaussian_mixture-c_(\d+)": lambda match: "GM({})".format(match.group(1))
}

distribution_modification_replacements = {
    "constrained poisson": "CP",
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
    r"-mc_(\d+)": lambda match: "" if int(match.group(1)) == 1 else \
        "-{} MC samples".format(match.groups(1)),
    r"-iw_(\d+)": lambda match: "" if int(match.group(1)) == 1 else \
        "-{} IW samples".format(match.groups(1))
}

model_version_replacements = {
    r"e_(\d+)-?(\w+)?": lambda match: "{} epochs".format(match.group(1)) \
        if not match.group(2) \
        else match.group(2).replace("_", " ").replace(" model", "")
}

miscellaneous_replacements = {
    # "sum": "count sum",
    "-kl": "",
    "bn": "BN",
    r"dropout_([\d._]+)": lambda match: "dropout: {}".format(
        match.group(1).replace("_", ", ")),
    r"wu_(\d+)": lambda match: "WU: {}".format(match.group(1))
}

def titleFromModelName(name):
    
    replacement_dictionaries = [
        GMVAE_replacements,
        mixture_replacements,
        distribution_modification_replacements,
        distribution_replacements,
        network_replacements,
        sample_replacements,
        model_version_replacements,
        miscellaneous_replacements
    ]
    
    return titleFromName(name, replacement_dictionaries)

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

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
