#!/usr/bin/env python3

import os
import pickle
import gzip
import argparse

test_metrics_filename = "test_metrics.pkl.gz"

def main(log_directory = None, results_directory = None):
    
    if log_directory:
        log_directory = os.path.normpath(log_directory) + os.sep
    
    if results_directory:
        results_directory = os.path.normpath(results_directory) + os.sep
        test_metrics_set = testMetricsInResultsDirectory(results_directory)
        
        for data_set, models in test_metrics_set.items():
            print(data_set)
            print()
            for model, test_metrics in models.items():
                print(model)
                
                E = test_metrics["number of epochs trained"]
                evaluation = test_metrics["evaluation"]
                
                print("Epochs trained: {}".format(E))
                
                losses = [
                    "log_likelihood",
                    "lower_bound",
                    "reconstruction_error",
                    "kl_divergence_z1",
                    "kl_divergence_z2",
                    "kl_divergence_y"
                ]
                
                for loss in losses:
                    if loss in evaluation:
                        print("{}: {:.5g}".format(loss, evaluation[loss][-1]))
                
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

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
