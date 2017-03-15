#!/usr/bin/env python3 -u

import data
import modeling
import analysis

import os
import argparse
import json
import itertools

def main(data_set_name, data_directory = "data",
    log_directory = "log", results_directory = "results",
    splitting_method = "random", splitting_fraction = 0.8,
    model_configurations_path = None, model_type = "VAE",
    latent_size = 50, hidden_sizes = [500],
    reconstruction_distribution = "poisson",
    number_of_reconstruction_classes = 0, number_of_warm_up_epochs = 50,
    batch_normalisation = True, count_sum = True,
    number_of_epochs = 200, batch_size = 100, learning_rate = 1e-4,
    reset_training = False, analyse = True, analyse_data = False):
    
    # Setup
    
    log_directory = os.path.join(log_directory, data_set_name)
    results_directory = os.path.join(results_directory, data_set_name)
    
    # Data
    
    data_set = data.DataSet(data_set_name, data_directory)
    
    print()
    
    training_set, validation_set, test_set = data_set.split(
        splitting_method, splitting_fraction)
    
    feature_size = data_set.number_of_features
    
    print()
    
    if analyse_data:
        analysis.analyseData(
            [data_set, training_set, validation_set, test_set],
            results_directory
        )
        print()
    
    # Loop over distribution
    
    print("Setting up model configurations.")
    
    model_configurations, configuration_errors = setUpModelConfigurations(
        model_configurations_path, model_type,
        latent_size, hidden_sizes, reconstruction_distribution,
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
    
    for model_configuration in model_configurations:
        
        model_type = model_configuration["model_type"]
        hidden_sizes = model_configuration["hidden_sizes"]
        reconstruction_distribution = \
            model_configuration["reconstruction_distribution"]
        number_of_reconstruction_classes = \
            model_configuration["number_of_reconstruction_classes"]
        batch_normalisation = model_configuration["batch_normalisation"]
        count_sum = model_configuration["count_sum"]
        number_of_epochs = model_configuration["number_of_epochs"]
        batch_size = model_configuration["batch_size"]
        learning_rate = model_configuration["learning_rate"]
        
        # Modeling
        
        if model_type == "VAE":
            latent_size = model_configuration["latent_size"]
            number_of_warm_up_epochs = \
                model_configuration["number_of_warm_up_epochs"]
            
            model = modeling.VariationalAutoEncoder(
                feature_size, latent_size, hidden_sizes,
                reconstruction_distribution,
                number_of_reconstruction_classes,
                batch_normalisation, count_sum, number_of_warm_up_epochs,
                log_directory = log_directory
            )
        
        print()
        
        model.train(training_set, validation_set,
            number_of_epochs, batch_size, learning_rate,
            reset_training)
        
        print()
        
        transformed_test_set, reconstructed_test_set, latent_set, \
            evaluation_test = model.evaluate(test_set, batch_size)
        
        print()
        
        # Analysis
        
        if analyse:
            
            learning_curves = analysis.analyseModel(model, results_directory)
            
            analysis.analyseResults(transformed_test_set, reconstructed_test_set,
                latent_set, evaluation_test, model, results_directory)
            
            models_summaries[model.name] = {
                "description": model.description,
                "configuration": model_configuration,
                "learning curves": learning_curves,
                "test evaluation": evaluation_test
            }
            
            print()
    
    analysis.analyseAllModels(models_summaries, results_directory)

def setUpModelConfigurations(model_configurations_path, model_type,
    latent_size, hidden_sizes, reconstruction_distribution,
    number_of_reconstruction_classes, number_of_warm_up_epochs,
    batch_normalisation, count_sum, number_of_epochs,
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
        
        configurations_product = itertools.product(
            network["structure of hidden layers"],
            network["count sum"],
            network["batch normalisation"],
            likelihood["reconstruction distributions"],
            likelihood["numbers of reconstruction classes"]
        )
        
        for model_type, type_configurations in model_types.items():
            for hidden_sizes, count_sum, batch_normalisation, \
                reconstruction_distribution, number_of_reconstruction_classes \
                in configurations_product:
                    
                    model_configuration = {
                            "model_type": model_type,
                            "hidden_sizes": hidden_sizes,
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
                    
                    if model_type == "VAE":
                        for latent_size in type_configurations["latent sizes"]:
                            model_configuration["latent_size"] = latent_size
                        model_configuration["number_of_warm_up_epochs"] = \
                            type_configurations["number of warm-up epochs"]
                    
                    validity, errors = \
                        validateModelConfiguration(model_configuration)
                    
                    if validity:
                        model_configurations.append(model_configuration)
                    else:
                        configuration_errors.append(errors)
        
    else:
        model_configuration = {
            "model_type": model_type,
            "latent_size": latent_size,
            "hidden_sizes": hidden_sizes,
            "reconstruction_distribution": reconstruction_distribution,
            "number_of_reconstruction_classes":
                number_of_reconstruction_classes,
            "number_of_warm_up_epochs": number_of_warm_up_epochs,
            "batch_normalisation": batch_normalisation,
            "count_sum": count_sum,
            "number_of_epochs": number_of_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
        
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
    errors = ""
    
    # Likelihood
    
    likelihood_validity = True
    likelihood_error = ""
    
    reconstruction_distribution = \
        model_configuration["reconstruction_distribution"]
    number_of_reconstruction_classes = \
        model_configuration["number_of_reconstruction_classes"]
    
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
    
    # Over-all validity
    
    validity = likelihood_validity
    errors = [likelihood_error]
    
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
    "--splitting-method", "-s",
    type = str,
    default = "random", 
    help = "method for splitting data into training, validation, and test sets"
)
parser.add_argument(
    "--splitting-fraction", "-f",
    type = float,
    default = 0.8,
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
    help = "perform analysis"
)
parser.add_argument(
    "--skip-analyse-data",
    dest = "analyse_data",
    action = "store_false",
    help = "skip analysis"
)
parser.set_defaults(analyse_data = False)

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
