#!/usr/bin/env python3

import data
import analysis

from models import (
    VariationalAutoEncoder,
    OriginalVariationalAutoEncoder,
    ImportanceWeightedVariationalAutoEncoder,
    SimpleNeuralNetwork,
    ClusterVariationalAutoEncoder,
    GaussianMixtureVariationalAutoEncoder,
    GaussianMixtureVariationalAutoEncoder_alternative
)

from auxiliary import title, subtitle

import os
import argparse
import json
import itertools
import random

def main(data_set_name, data_directory = "data",
    log_directory = "log", results_directory = "results",
    feature_selection = [], feature_parameter = None, example_filter = [],
    preprocessing_methods = [], noisy_preprocessing_methods = [],
    splitting_method = "default", splitting_fraction = 0.8,
    model_type = "VAE", latent_size = 50, hidden_sizes = [500],
    number_of_importance_samples = [5],
    number_of_monte_carlo_samples = [10],
    latent_distribution = "gaussian",
    number_of_latent_clusters = 1,
    parameterise_latent_posterior = False,
    reconstruction_distribution = "poisson",
    number_of_reconstruction_classes = 0,
    prior_probabilities_method = "uniform",
    number_of_warm_up_epochs = 50,
    proportion_of_free_KL_nats = 0.0,
    batch_normalisation = True,
    dropout_keep_probabilities = [],
    count_sum = True,
    number_of_epochs = 200, plot_for_every_n_epochs = None, 
    batch_size = 100, learning_rate = 1e-4,
    decomposition_methods = ["PCA"], highlight_feature_indices = [],
    reset_training = False, skip_modelling = False,
    analyse = True, evaluation_set_name = "test", analyse_data = False,
    analyses = ["default"], analysis_level = "normal",
    video = False):
    
    print()
    
    # Load and split data
    
    title("Loading and splitting data")
    
    data_set = data.DataSet(
        data_set_name,
        directory = data_directory,
        feature_selection = feature_selection,
        feature_parameter = feature_parameter,
        example_filter = example_filter,
        preprocessing_methods = preprocessing_methods,
        noisy_preprocessing_methods = noisy_preprocessing_methods
    )
    
    training_set, validation_set, test_set = data_set.split(
        splitting_method, splitting_fraction)
    
    all_data_sets = [data_set, training_set, validation_set, test_set]
    
    print()
    
    # Set up log and results directories
    
    log_directory = data.directory(log_directory, data_set,
        splitting_method, splitting_fraction)
    data_results_directory = data.directory(results_directory, data_set,
        splitting_method, splitting_fraction, preprocessing = False)
    results_directory = data.directory(results_directory, data_set,
        splitting_method, splitting_fraction)
    
    # Analyse data
    
    if analyse and analyse_data:
        subtitle("Analysing data")
        analysis.analyseData(
            all_data_sets,
            decomposition_methods, highlight_feature_indices,
            analyses, analysis_level,
            data_results_directory
        )
        print()
    
    # Modelling
    
    if skip_modelling:
        print("Modelling skipped.")
        return
    
    title("Modelling")
    
    if "VAE" in model_type:
        if latent_distribution == "gaussian":
            analytical_kl_term = True
        else:
            analytical_kl_term = False
    
    model_valid, model_errors = validateModelParameters(
        model_type, latent_distribution,
        reconstruction_distribution, number_of_reconstruction_classes,
        parameterise_latent_posterior
    )
    
    if not model_valid:
        for model_error in model_errors:
            print(model_error)
        print("")
        print("Modelling cancelled.")
    
    feature_size = data_set.number_of_features
    number_of_monte_carlo_samples = parseSampleLists(
        number_of_monte_carlo_samples)
    number_of_importance_samples = parseSampleLists(
        number_of_importance_samples)
    
    subtitle("Setting up model")
    
    if model_type == "VAE":
        model = VariationalAutoEncoder(
            feature_size = feature_size,
            latent_size = latent_size,
            hidden_sizes = hidden_sizes,
            number_of_monte_carlo_samples =number_of_monte_carlo_samples,
            analytical_kl_term = analytical_kl_term,
            latent_distribution = latent_distribution,
            number_of_latent_clusters = number_of_latent_clusters,
            parameterise_latent_posterior = parameterise_latent_posterior,
            reconstruction_distribution = reconstruction_distribution,
            number_of_reconstruction_classes = number_of_reconstruction_classes,
            batch_normalisation = batch_normalisation,
            dropout_keep_probabilities = dropout_keep_probabilities,
            count_sum = count_sum,
            number_of_warm_up_epochs = number_of_warm_up_epochs,
            log_directory = log_directory,
            results_directory = results_directory
        )
    
    elif model_type == "IWVAE":
        model = ImportanceWeightedVariationalAutoEncoder(
            feature_size = feature_size,
            latent_size = latent_size,
            hidden_sizes = hidden_sizes,
            number_of_monte_carlo_samples =number_of_monte_carlo_samples,
            number_of_importance_samples = number_of_importance_samples,
            analytical_kl_term = analytical_kl_term,
            latent_distribution = latent_distribution,
            number_of_latent_clusters = number_of_latent_clusters,
            parameterise_latent_posterior = parameterise_latent_posterior,
            reconstruction_distribution = reconstruction_distribution,
            number_of_reconstruction_classes = number_of_reconstruction_classes,
            batch_normalisation = batch_normalisation,
            dropout_keep_probabilities = dropout_keep_probabilities,
            count_sum = count_sum,
            number_of_warm_up_epochs = number_of_warm_up_epochs,
            log_directory = log_directory,
            results_directory = results_directory
        )

    elif model_type == "GMVAE_alt":
        if prior_probabilities_method == "uniform":
            prior_probabilities = None
        elif prior_probabilities_method == "infer":
            prior_probabilities = data_set.class_probabilities
        elif prior_probabilities_method == "literature":
            prior_probabilities = data_set.literature_probabilities
        else:
            prior_probabilities = None
        
        if not prior_probabilities:
            prior_probabilities_method = "uniform"
            prior_probabilities_values = None
        else:
            prior_probabilities_values = list(prior_probabilities.values())
        
        prior_probabilities = {
            "method": prior_probabilities_method,
            "values": prior_probabilities_values
        }
        
        model = GaussianMixtureVariationalAutoEncoder_alternative(
            feature_size = feature_size,
            latent_size = latent_size,
            hidden_sizes = hidden_sizes,
            number_of_monte_carlo_samples = number_of_monte_carlo_samples,
            number_of_importance_samples = number_of_importance_samples, 
            analytical_kl_term = analytical_kl_term,
            prior_probabilities = prior_probabilities,
            number_of_latent_clusters = number_of_latent_clusters,
            proportion_of_free_KL_nats = proportion_of_free_KL_nats,
            reconstruction_distribution = reconstruction_distribution,
            number_of_reconstruction_classes = number_of_reconstruction_classes,
            batch_normalisation = batch_normalisation,
            dropout_keep_probabilities = dropout_keep_probabilities,
            count_sum = count_sum,
            number_of_warm_up_epochs = number_of_warm_up_epochs,
            log_directory = log_directory,
            results_directory= results_directory
        )
    
    else:
        return ValueError("Model type not found.")
    
    print(model.description)
    print()
    
    print(model.parameters)
    print()
    
    # Training
    
    subtitle("Training model")
    
    status = model.train(
        training_set, validation_set,
        number_of_epochs, batch_size, learning_rate,
        reset_training, plot_for_every_n_epochs
    )
    
    if not status["completed"]:
        print(status["message"])
        return
    
    status_filename = "status"
    if "trained" in status:
        status_filename += "-" + status["trained"]
    status_path = os.path.join(
        model.log_directory,
        status_filename + ".log"
    )
    with open(status_path, "w") as status_file:
        for status_field, status_value in status.items():
            if status_value:
                status_file.write(
                    status_field + ": " + str(status_value) + "\n"
                )
    
    print()
    
    # Evaluating
    
    for data_subset in all_data_sets:
        if data_subset.kind == evaluation_set_name:
            evaluation_set = data_subset
    
    subtitle("Evaluating on {} set".format(evaluation_set.kind))
    
    if "VAE" in model.type:
        transformed_evaluation_set, reconstructed_evaluation_set,\
            likelihood_evaluation_set, latent_evaluation_sets = \
            model.evaluate(evaluation_set, batch_size)
        
        if analysis.betterModelExists(model):
            better_model_exist = True
            
            print()
            subtitle("Evaluating on {} set with best model parameters"\
                .format(evaluation_set.kind))
            
            best_model_transformed_evaluation_set, \
                best_model_reconstructed_evaluation_set, \
                best_model_likelihood_evaluation_set, \
                best_model_latent_evaluation_sets = \
                model.evaluate(evaluation_set, batch_size,
                    use_best_model = True)
        else:
            better_model_exist = False
        
        if model.stopped_early:
            print()
            subtitle("Evaluating on {} set with earlier stopped model"\
                .format(evaluation_set.kind))
            
            early_stopped_transformed_evaluation_set, \
                early_stopped_reconstructed_evaluation_set, \
                early_stopped_likelihood_evaluation_set, \
                early_stopped_latent_evaluation_sets = \
                model.evaluate(evaluation_set, batch_size,
                    use_early_stopping_model = True)
    else:
        transformed_evaluation_set, reconstructed_evaluation_set, \
            likelihood_evaluation_set = \
            model.evaluate(evaluation_set, batch_size)
        latent_evaluation_sets = None
        better_model_exist = False
    
    print()
    
    # Analysis
    
    title("Analyses")
    
    if analyse:
        
        subtitle("Analysing model")
        analysis.analyseModel(model, analyses, analysis_level,
            video, results_directory)

        subtitle("Analysing results for {} set".format(evaluation_set.kind))
        analysis.analyseResults(
            transformed_evaluation_set,
            reconstructed_evaluation_set,
            likelihood_evaluation_set,
            latent_evaluation_sets,
            model,
            decomposition_methods, highlight_feature_indices,
            analyses = analyses, analysis_level = analysis_level,
            results_directory = results_directory
        )
        
        if better_model_exist:
            subtitle("Analysing results for {} set".format(
                evaluation_set.kind) + " with best model parameters")
            analysis.analyseResults(
                best_model_transformed_evaluation_set,
                best_model_reconstructed_evaluation_set,
                best_model_likelihood_evaluation_set,
                best_model_latent_evaluation_sets,
                model,
                decomposition_methods, highlight_feature_indices,
                best_model = True,
                analyses = analyses, analysis_level = analysis_level,
                results_directory = results_directory
            )
        
        if model.stopped_early:
            subtitle("Analysing results for {} set".format(
                evaluation_set.kind) + " with earlier stopped model")
            analysis.analyseResults(
                early_stopped_transformed_evaluation_set,
                early_stopped_reconstructed_evaluation_set,
                early_stopped_likelihood_evaluation_set,
                early_stopped_latent_evaluation_sets,
                model,
                decomposition_methods, highlight_feature_indices,
                early_stopping = True,
                analyses = analyses, analysis_level = analysis_level,
                results_directory = results_directory
            )

def parseSampleLists(list_with_number_of_samples):
    
    if len(list_with_number_of_samples) == 2:
        number_of_samples = {
            "training": list_with_number_of_samples[0],
            "evaluation": list_with_number_of_samples[1]
        }
    
    elif len(list_with_number_of_samples) == 1:
        number_of_samples = {
            "training": list_with_number_of_samples[0],
            "evaluation": list_with_number_of_samples[0]
        }
    
    else:
        raise ValueError("List of number of samples can only contain " +
            "one or two numbers.")
    
    return number_of_samples

def validateModelParameters(model_type, latent_distribution,
    reconstruction_distribution, number_of_reconstruction_classes,
    parameterise_latent_posterior):
    
    validity = True
    errors = []
    
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
        
        # if "negative binomial" in reconstruction_distribution:
        #     likelihood_error_list.append("any negative binomial distribution")
        #     likelihood_validity = False
        
        if "constrained" in reconstruction_distribution:
            likelihood_error_list.append("the multinomial distribution")
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
    
    # Latent distribution
    
    if "VAE" in model_type:
        
        latent_distribution_validity = True
        latent_distribution_error = ""
    
        if model_type == "OVAE" and "mixture" in latent_distribution:
            latent_distribution_error = "Mixture latent distribution with " + \
                "original variational auto-encoder."
            latent_distribution_validity = False
        elif model_type in ["CVAE", "GMVAE", "GMVAE_alt"] and "mixture" not in latent_distribution:
            latent_distribution_error = "No mixture latent distribution with " + \
                "cluster variational auto-encoder."
            latent_distribution_validity = False
    
        validity = validity and latent_distribution_validity
        errors.append(latent_distribution_error)
    
    # Parameterisation of latent posterior for IWVAE
    if "VAE" in model_type:
        parameterise_validity = True
        parameterise_error = ""
        
        if not (model_type in ["VAE", "IWVAE"]
            and latent_distribution == "gaussian mixture") \
            and parameterise_latent_posterior:
            parameterise_error = "Cannot parameterise latent posterior parameters" \
                + " for " + model_type + " or " + latent_distribution \
                + " distribution."
            parameterise_validity = False
        
        validity = validity and parameterise_validity
        errors.append(parameterise_error)
    
    # Return
    
    return validity, errors

parser = argparse.ArgumentParser(
    description='Model single-cell transcript counts using deep learning.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--data-set-name", "-n",
    type = str,
    default = "mouse retina",
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
    "--feature-selection", "-F",
    type = str,
    nargs = "?",
    default = None,
    help = "method for selecting features"
)
parser.add_argument(
    "--feature-parameter",
    type = float,
    nargs = "?",
    default = None,
    help = "parameter for feature selection"
)
parser.add_argument(
    "--example-filter", "-E",
    type = str,
    nargs = "*",
    default = None,
    help = "method for filtering examples"
)
parser.add_argument(
    "--preprocessing-methods", "-p",
    type = str,
    nargs = "*",
    default = None,
    help = "methods for preprocessing data (applied in order)"
)
parser.add_argument(
    "--noisy-preprocessing-methods", "-N",
    type = str,
    nargs = "*",
    default = None,
    help = "methods for noisily preprocessing data at every epoch (applied in order)"
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
    "--model-type", "-m",
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
    default = [250, 250],
    help = "sizes of hidden layers"
)
parser.add_argument(
    "--number-of-importance-samples",
    type = int,
    nargs = "+",
    default = [1, 1000],
    help = "the number of importance weighted samples (if two numbers given, the first will be used for training and the second for evaluation)"
)
parser.add_argument(
    "--number-of-monte-carlo-samples",
    type = int,
    nargs = "+",
    default = [1],
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
    "--parameterise-latent-posterior",
    action = "store_true",
    help = "parameterise latent posterior parameters if possible"
)
parser.add_argument(
    "--do-not-parameterise-latent-posterior",
    dest = "parameterise_latent_posterior",
    action = "store_false",
    help = "do not parameterise latent posterior parameters"
)
parser.set_defaults(parameterise_latent_posterior = False)
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
    "--prior-probabilities-method",
    type = str,
    nargs = "?",
    default = "uniform",
    help = "method to set prior probabilities"
)
parser.add_argument(
    "--number-of-epochs", "-e",
    type = int,
    default = 200,
    help = "number of epochs for which to train"
)
parser.add_argument(
    "--plot-for-every-n-epochs",
    type = int,
    nargs = "?",
    help = "number of training epochs between each intermediate plot starting at the first"
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
    "--proportion-of-free-KL-nats",
    type = float,
    nargs = "?",
    default = 0.0,
    help = "Proportion of maximum KL_y divergence which has constant term and zero gradients, ´free bits´ method"
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
parser.add_argument(
    "--dropout-keep-probabilities", "-d",
    type = float,
    nargs = "*",
    default = [],
    help = "List of probabilities, p, of keeping connections when using dropout. Interval: ]0, 1[, where p in {0, 1, False} means no dropout."
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
parser.set_defaults(count_sum = False)
parser.add_argument(
    "--decomposition-methods",
    type = str,
    nargs = "*",
    default = ["PCA"],
    help = "methods use to decompose values"
)
parser.add_argument(
    "--highlight-feature-indices",
    type = int,
    nargs = "*",
    default = [],
    help = "feature indices to highlight in analyses"
)
parser.add_argument(
    "--reset-training",
    action = "store_true",
    help = "reset already trained model"
)
parser.add_argument(
    "--perform-modelling",
    dest = "skip_modelling",
    action = "store_false",
    help = "perform modelling"
)
parser.add_argument(
    "--skip-modelling",
    action = "store_true",
    help = "skip modelling"
)
parser.set_defaults(skip_modelling = False)
parser.add_argument(
    "--analyse",
    action = "store_true",
    help = "perform analysis"
)
parser.add_argument(
    "--skip-analyses",
    dest = "analyse",
    action = "store_false",
    help = "skip analysis"
)
parser.set_defaults(analyse = True)
parser.add_argument(
    "--evaluation-set-name",
    type = str,
    nargs = "?",
    default = "test",
    help = "parameter for feature selection"
)
parser.add_argument(
    "--analyse-data",
    action = "store_true",
    help = "perform data analysis"
)
parser.add_argument(
    "--skip-data-analyses",
    dest = "analyse_data",
    action = "store_false",
    help = "skip data analysis"
)
parser.set_defaults(analyse_data = False)
parser.add_argument(
    "--analyses",
    type = str,
    nargs = "+",
    default = ["default"],
    help = "analyses to perform, which can be specified individually or as groups: simple, default, complete"
)
parser.add_argument(
    "--analysis-level",
    type = str,
    default = "normal",
    help = "level to which analyses are performed: limited, normal (default), extensive"
)
parser.add_argument(
    "--video", "-v",
    action = "store_true",
    help = "analyse model evolution for video"
)
parser.add_argument(
    "--no-video", "-V",
    dest = "video",
    action = "store_false",
    help = "do not analyse model evolution for video"
)
parser.set_defaults(video = False)

if __name__ == '__main__':
    arguments = parser.parse_args()
    main(**vars(arguments))
