#!/usr/bin/env python3

import data
import analysis

from models import VariationalAutoEncoder, ImportanceWeightedVariationalAutoEncoder, SimpleNeuralNetwork

import os
import argparse
import json
import itertools

def main(data_set_name, data_directory = "data",
    log_directory = "log", results_directory = "results",
    feature_selection = None, preprocessing_methods = None,
    splitting_method = "default", splitting_fraction = 0.8,
    model_configurations_path = None, model_type = "VAE",
    latent_size = 50, hidden_sizes = [500],
    number_of_importance_weighted_samples = [5],
    number_of_monte_carlo_samples = [10],
    latent_distribution = "gaussian",
    number_of_latent_clusters = 1,
    reconstruction_distribution = "poisson",
    number_of_reconstruction_classes = 0, number_of_warm_up_epochs = 50,
    batch_normalisation = True, count_sum = True,
    number_of_epochs = 200, batch_size = 100, learning_rate = 1e-4,
    reset_training = False, analyse = True, analyse_data = False):
    
    # Load and split data
    
    data_set = data.DataSet(
        data_set_name,
        directory = data_directory,
        feature_selection = feature_selection,
        preprocessing_methods = preprocessing_methods
    )
    
    training_set, validation_set, test_set = data_set.split(
        splitting_method, splitting_fraction)
    
    print()
    
    # Set up log and results directories
    
    log_directory = data.directory(log_directory, data_set,
        splitting_method, splitting_fraction)
    results_directory = data.directory(results_directory, data_set,
        splitting_method, splitting_fraction)
    
    # Analyse data
    
    if analyse_data:
        analysis.analyseData(
            [data_set, training_set, validation_set, test_set],
            results_directory
        )
        print()
    
    # Set up model configurations
    
    print("Setting up model configurations.")
    
    model_configurations, configuration_errors = setUpModelConfigurations(
        model_configurations_path, model_type,
        latent_size, hidden_sizes,
        number_of_importance_weighted_samples,
        number_of_monte_carlo_samples,
        latent_distribution,
        number_of_latent_clusters,
        reconstruction_distribution,
        number_of_reconstruction_classes, number_of_warm_up_epochs,
        batch_normalisation, count_sum, number_of_epochs,
        batch_size, learning_rate
    )
    
    if configuration_errors:
        print("Invalid model configurations:")
        for errors in configuration_errors:
            for error in errors:
                print("    ", error)
    
    print()
    
    print("Looping over models.\n")
    
    if analyse:
        models_summaries = {}
    
    # Loop over models
    
    feature_size = data_set.number_of_features
    
    for model_configuration in model_configurations:
        
        model_type = model_configuration["model_type"]
        hidden_sizes = model_configuration["hidden_sizes"]
        latent_distribution = \
            model_configuration["latent_distribution"]
        reconstruction_distribution = \
            model_configuration["reconstruction_distribution"]
        number_of_reconstruction_classes = \
            model_configuration["number_of_reconstruction_classes"]
        batch_normalisation = model_configuration["batch_normalisation"]
        count_sum = model_configuration["count_sum"]
        number_of_epochs = model_configuration["number_of_epochs"]
        batch_size = model_configuration["batch_size"]
        learning_rate = model_configuration["learning_rate"]
        
        if latent_distribution == "gaussian mixture":
            number_of_latent_clusters = \
                model_configuration["number_of_latent_clusters"]
        
        if "AE" in model_type:
            latent_size = model_configuration["latent_size"]
            number_of_warm_up_epochs = \
                model_configuration["number_of_warm_up_epochs"]
            analytical_kl_term = latent_distribution == "gaussian"
        
        # Modeling
        
        if model_type == "VAE":
            
            number_of_monte_carlo_samples = model_configuration[
                "number_of_monte_carlo_samples"]
            
            model = VariationalAutoEncoder(
                feature_size, latent_size, hidden_sizes,
                number_of_monte_carlo_samples, analytical_kl_term,
                latent_distribution, number_of_latent_clusters,
                reconstruction_distribution,
                number_of_reconstruction_classes,
                batch_normalisation, count_sum, number_of_warm_up_epochs,
                log_directory = log_directory
            )
        
        elif model_type == "IWVAE":
            
            # Dictionary holding number of samples needed for the "monte carlo"
            # estimator and "importance weighting" during both "train" and
            # "test" time.
            numbers_of_samples = model_configuration["numbers_of_samples"]
            
            model = ImportanceWeightedVariationalAutoEncoder(
                feature_size, latent_size, hidden_sizes,
                numbers_of_samples, analytical_kl_term,
                latent_distribution, number_of_latent_clusters,
                reconstruction_distribution,
                number_of_reconstruction_classes,
                batch_normalisation, count_sum, number_of_warm_up_epochs,
                log_directory = log_directory
            )
        
        elif model_type == "SNN":
            
            model = SimpleNeuralNetwork(
                feature_size, hidden_sizes,
                reconstruction_distribution,
                number_of_reconstruction_classes,
                batch_normalisation, count_sum,
                log_directory = log_directory
            )
        
        else:
            return ValueError("Model type not found.")
        
        print(model.description)
        print()
        
        print(model.parameters)
        print()
        
        status = model.train(
            training_set, validation_set,
            number_of_epochs, batch_size, learning_rate,
            reset_training
        )
        
        if not status["completed"]:
            print(status["message"])
            error_path = os.path.join(model.log_directory, "error.log")
            with open(error_path, "w") as error_file:
                error_file.write(
                    "completed: {}\n".format(status["completed"]) + \
                    "message: {}\n".format(status["message"])
                )
            print()
            continue
        
        print()
        
        if "AE" in model.type:
            transformed_test_set, reconstructed_test_set, latent_test_set = \
                model.evaluate(test_set, batch_size)
        else:
            transformed_test_set, reconstructed_test_set = \
                model.evaluate(test_set, batch_size)
            latent_test_set = None
        
        print()
        
        # Analysis
        
        if analyse:
            
            analysis.analyseModel(model, results_directory)
            
            analysis.analyseResults(transformed_test_set, reconstructed_test_set,
                latent_test_set, model, results_directory)
            
            print()

def setUpModelConfigurations(model_configurations_path, model_type,
    latent_size, hidden_sizes,
    number_of_importance_weighted_samples, number_of_monte_carlo_samples,
    latent_distribution, number_of_latent_clusters,
    reconstruction_distribution, number_of_reconstruction_classes,
    number_of_warm_up_epochs, batch_normalisation, count_sum, number_of_epochs,
    batch_size, learning_rate):
    
    model_configurations = []
    configuration_errors = []
    
    if model_configurations_path:
        
        with open(model_configurations_path, "r") as configurations_file:
            configurations = json.load(configurations_file)
        
        model_types = configurations["model types"]
        network = configurations["network"]
        likelihood = configurations["likelihood"]
        training = configurations["training"]
        
        for model_type, type_configuration in model_types.items():
            
            configurations_product = itertools.product(
                network["structure of hidden layers"],
                network["count sum"],
                network["batch normalisation"],
                likelihood["latent distributions"],
                likelihood["reconstruction distributions"],
                likelihood["numbers of reconstruction classes"]
            )
            
            for hidden_sizes, count_sum, batch_normalisation, \
                latent_distribution, reconstruction_distribution, \
                number_of_reconstruction_classes in configurations_product:
                
                base_model_configuration = {
                        "model_type": model_type,
                        "hidden_sizes": hidden_sizes,
                        "latent_distribution":
                            latent_distribution,
                        "reconstruction_distribution":
                            reconstruction_distribution,
                        "number_of_reconstruction_classes":
                            number_of_reconstruction_classes,
                        "batch_normalisation": batch_normalisation,
                        "count_sum": count_sum,
                        "number_of_epochs": training["number of epochs"],
                        "batch_size": training["batch size"],
                        "learning_rate": training["learning rate"]
                }
                
                if "VAE" in model_type:
                    base_model_configuration["number_of_warm_up_epochs"] = \
                        type_configuration["number of warm-up epochs"]
                    if "IW" in model_type:
                        base_model_configuration["numbers_of_samples"] = \
                            type_configuration["numbers of samples"]
                    else:
                        base_model_configuration[
                            "number_of_monte_carlo_samples"] = \
                            type_configuration["number_of_monte_carlo_samples"]
                
                sub_model_configurations = []
                
                if latent_distribution == "gaussian mixture" and \
                    "AE" in model_type:
                    
                    sub_configurations_product = itertools.product(
                        likelihood["numbers of latent clusters"],
                        type_configuration["latent sizes"]
                    )
                    
                    for number_of_latent_clusters, latent_size in \
                        sub_configurations_product:
                        model_configuration = base_model_configuration.copy()
                        model_configuration["number_of_latent_clusters"] = \
                            number_of_latent_clusters
                        model_configuration["latent_size"] = latent_size
                        sub_model_configurations.append(model_configuration)
                
                elif latent_distribution == "gaussian mixture":
                    for number_of_latent_clusters in \
                        likelihood["numbers of latent clusters"]:
                        model_configuration = base_model_configuration.copy()
                        model_configuration["number_of_latent_clusters"] = \
                            number_of_latent_clusters
                        sub_model_configurations.append(model_configuration)
                
                elif "VAE" in model_type:
                    for latent_size in type_configuration["latent sizes"]:
                        model_configuration = base_model_configuration.copy()
                        model_configuration["latent_size"] = latent_size
                        sub_model_configurations.append(model_configuration)
                
                else:
                    sub_model_configurations.append(base_model_configuration)
                
                for model_configuration in sub_model_configurations:
                    
                    validity, errors = \
                        validateModelConfiguration(model_configuration)
                    
                    if validity:
                        model_configurations.append(model_configuration)
                    else:
                        configuration_errors.append(errors)

    else:
        model_configuration = {
            "model_type": model_type,
            "hidden_sizes": hidden_sizes,
            "latent_distribution": latent_distribution,
            "reconstruction_distribution": reconstruction_distribution,
            "number_of_reconstruction_classes":
                number_of_reconstruction_classes,
            "batch_normalisation": batch_normalisation,
            "count_sum": count_sum,
            "number_of_epochs": number_of_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
        
        if latent_distribution == "gaussian mixture":
            model_configuration["number_of_latent_clusters"] = \
                number_of_latent_clusters
        
        if "VAE" in model_type:
            model_configuration["latent_size"] = latent_size
            model_configuration["number_of_warm_up_epochs"] = \
                number_of_warm_up_epochs
            
            if "IW" in model_type:
                numbers_of_samples = {
                    "training": {
                        "importance weighting": number_of_importance_weighted_samples[0],
                        "monte carlo": number_of_monte_carlo_samples[0]
                    },
                    
                    "evaluation": {
                        "importance weighting": number_of_importance_weighted_samples[0],
                        "monte carlo": number_of_monte_carlo_samples[0]
                    }
                }
                
                if len(number_of_importance_weighted_samples) > 1:
                    numbers_of_samples["evaluation"]["importance weighting"] = \
                        number_of_importance_weighted_samples[1]
                
                if len(number_of_monte_carlo_samples) > 1:
                    numbers_of_samples["evaluation"]["monte carlo"] = \
                        number_of_monte_carlo_samples[1]
                
                model_configuration["numbers_of_samples"] = numbers_of_samples
            
            else:
                
                
                if len(number_of_monte_carlo_samples) > 1:
                    number_of_monte_carlo_samples = {
                        "training": number_of_monte_carlo_samples[0],
                        "evaluation": number_of_monte_carlo_samples[1]
                    }
                else:
                    number_of_monte_carlo_samples = {
                        "training": number_of_monte_carlo_samples[0],
                        "evaluation": number_of_monte_carlo_samples[0]
                    }
                
                model_configuration["number_of_monte_carlo_samples"] = \
                    number_of_monte_carlo_samples
        
        validity, errors = validateModelConfiguration(model_configuration)
        
        if validity:
            model_configurations.append(model_configuration)
        else:
            configuration_errors.append(errors)
    
    if len(configuration_errors) == 0:
        configuration_errors = None
    
    return model_configurations, configuration_errors

def validateModelConfiguration(model_configuration):
    
    validity = True
    errors = []
    
    reconstruction_distribution = \
        model_configuration["reconstruction_distribution"]
    number_of_reconstruction_classes = \
        model_configuration["number_of_reconstruction_classes"]
    
    # Likelihood
    
    likelihood_validity = True
    likelihood_error = ""
    
    if number_of_reconstruction_classes > 0:
        likelihood_error = "Reconstruction classification with"
        
        likelihood_error_list = []
        
        if reconstruction_distribution == "bernoulli":
            likelihood_error_list.append("the Bernoulli distribution")
            likelihood_validity = False
        
        if "zero-inflated" in reconstruction_distribution:
            likelihood_error_list.append("zero-inflated distributions")
            likelihood_validity = False
        
        if "negative binomial" in reconstruction_distribution:
            likelihood_error_list.append("any negative binomial distribution")
            likelihood_validity = False
        
        if "multinomial" in reconstruction_distribution:
            likelihood_error_list.append("the multinomial distribution")
            likelihood_validity = False
        
        number_of_distributions = len(likelihood_error_list)
        
        if number_of_distributions == 1:
            likelihood_error_distribution = likelihood_error_list[0]
        elif number_of_distributions == 2:
            likelihood_error_distribution = \
                " or ".join(likelihood_error_list)
        elif number_of_distributions >= 2:
            likelihood_error_distribution = \
                ", ".join(likelihood_error_list[:-1]) + ", or" \
                + likelihood_error_list[-1]
        
        if likelihood_validity:
            likelihood_error = ""
        else:
            likelihood_error += " " + likelihood_error_distribution + "."
    
    validity = validity and likelihood_validity
    errors.append(likelihood_error)
    
    # Return
    
    return validity, errors

parser = argparse.ArgumentParser(
    description='Model single-cell transcript counts using deep learning.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--data-set-name", "-n",
    type = str,
    default = "sample",
    help = "name of data set"
)
parser.add_argument(
    "--data-directory", "-D",
    type = str,
    default = "data",
    help = "directory where data is placed"
)
parser.add_argument(
    "--log-directory", "-L",
    type = str,
    default = "log",
    help = "directory where models are logged"
)
parser.add_argument(
    "--results-directory", "-R",
    type = str,
    default = "results",
    help = "directory where results are saved"
)
parser.add_argument(
    "--feature-selection", "-f",
    type = str,
    nargs = "?",
    default = None,
    help = "method for selecting features"
)
parser.add_argument(
    "--preprocessing-methods", "-p",
    type = str,
    nargs = "*",
    default = None,
    help = "methods for preprocessing data (applied in order)"
)
parser.add_argument(
    "--splitting-method", "-s",
    type = str,
    default = "random",
    help = "method for splitting data into training, validation, and test sets"
)
parser.add_argument(
    "--splitting-fraction",
    type = float,
    default = 0.9,
    help = "fraction to use when splitting data into training, validation, and test sets"
)
parser.add_argument(
    "--model-configurations", "-m",
    dest = "model_configurations_path",
    type = str,
    default = None,
    help = "file with model configurations"
)
parser.add_argument(
    "--model-type", "-M",
    type = str,
    default = "VAE",
    help = "type of model"
)
parser.add_argument(
    "--latent-size", "-l",
    type = int,
    default = 50,
    help = "size of latent space"
)
parser.add_argument(
    "--hidden-sizes", "-H",
    type = int,
    nargs = "+",
    default = [500],
    help = "sizes of hidden layers"
)
parser.add_argument(
    "--number-of-importance-weighted-samples",
    type = int,
    nargs = "+",
    default = [5],
    help = "the number of importance weighted samples (if two numbers given, the first will be used for training and the second for evaluation)"
)
parser.add_argument(
    "--number-of-monte-carlo-samples",
    type = int,
    nargs = "+",
    default = [10],
    help = "the number of Monte Carlo samples (if two numbers given, the first will be used for training and the second for evaluation)"
)
parser.add_argument(
    "--latent-distribution", "-q",
    type = str,
    default = "gaussian",
    help = "distribution for the latent variables"
)
parser.add_argument(
    "--number-of-latent-clusters",
    type = int,
    help = "the number of modes in the gaussian mixture"
)
parser.add_argument(
    "--reconstruction-distribution", "-r",
    type = str,
    default = "poisson",
    help = "distribution for the reconstructions"
)
parser.add_argument(
    "--number-of-reconstruction-classes", "-k",
    type = int,
    default = 0,
    help = "the maximum count for which to use classification"
)
parser.add_argument(
    "--number-of-epochs", "-e",
    type = int,
    default = 200,
    help = "number of epochs for which to train"
)
parser.add_argument(
    "--batch-size", "-i",
    type = int,
    default = 100,
    help = "batch size used when training"
)
parser.add_argument(
    "--learning-rate", "-S",
    type = float,
    default = 1e-4,
    help = "learning rate when training"
)
parser.add_argument(
    "--number-of-warm-up-epochs", "-w",
    type = int,
    default = 0,
    help = "number of epochs with a linear weight on the KL-term"
)
parser.add_argument(
    "--batch-normalisation", "-b",
    action = "store_true",
    help = "use batch normalisation"
)
parser.add_argument(
    "--no-batch-normalisation", "-B",
    dest = "batch_normalisation",
    action = "store_false",
    help = "do not use batch normalisation"
)
parser.set_defaults(batch_normalisation = True)
parser.add_argument(
    "--count-sum", "-c",
    action = "store_true",
    help = "use count sum"
)
parser.add_argument(
    "--no-count-sum", "-C",
    dest = "count_sum",
    action = "store_false",
    help = "do not use count sum"
)
parser.set_defaults(count_sum = True)
parser.add_argument(
    "--reset-training",
    action = "store_true",
    help = "reset already trained model"
)
parser.add_argument(
    "--analyse",
    action = "store_true",
    help = "perform analysis"
)
parser.add_argument(
    "--skip-analysis",
    dest = "analyse",
    action = "store_false",
    help = "skip analysis"
)
parser.set_defaults(analyse = True)
parser.add_argument(
    "--analyse-data",
    action = "store_true",
    help = "perform data analysis"
)
parser.add_argument(
    "--skip-data-analysis",
    dest = "analyse_data",
    action = "store_false",
    help = "skip data analysis"
)
parser.set_defaults(analyse_data = False)

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
