#!/usr/bin/env python3

import os
import pickle
import gzip
import argparse

from analysis import formatStatistics
from auxiliary import formatTime, title, subtitle

test_metrics_filename = "test-metrics.pkl.gz"

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
        
        for data_set, models in test_metrics_set.items():
            
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
            
            title(data_set)
            
            for model, test_metrics in models.items():
                
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
                
                subtitle(model)
                
                # Time
                
                timestamp = test_metrics["timestamp"]
                
                print("timestamp: {}".format(formatTime(timestamp)))
                
                # Epochs
                
                E = test_metrics["number of epochs trained"]
                print("epochs_trained: {}".format(E))
                
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
                        print("{}: {:.5g}".format(loss, evaluation[loss][-1]))
                
                # Accuracies
                
                accuracies = ["accuracy", "superset_accuracy"]
                
                for accuracy in accuracies:
                    if accuracy in test_metrics and test_metrics[accuracy]:
                        print("{}: {:6.2f} %".format(
                            accuracy, 100 * test_metrics[accuracy][-1]))
                
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
                    print(formatStatistics(reconstructed_statistics))
                
                print()

def testMetricsInResultsDirectory(results_directory):
    
    test_metrics_set = {}
    
    for path, directories, files in os.walk(results_directory):
        if test_metrics_filename in files:
            
            data_set_model = path.replace(results_directory, "")
            data_set_model_parts = data_set_model.split(os.sep)
            data_set = os.sep.join(data_set_model_parts[:3])
            model = os.sep.join(data_set_model_parts[3:])
            
            if not data_set in test_metrics_set:
                test_metrics_set[data_set] = {}
            
            test_metrics_path = os.path.join(path, test_metrics_filename)
            
            with gzip.open(test_metrics_path, "r") as test_metrics_file:
                test_metrics_data = pickle.load(test_metrics_file)
                
            test_metrics_set[data_set][model] = test_metrics_data
    
    return test_metrics_set

parser = argparse.ArgumentParser(
    description='Cross-analyse models.',
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
