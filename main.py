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

import data
import analysis

from models import (
    VariationalAutoencoder,
    GaussianMixtureVariationalAutoencoder
)

from distributions import distributions, latent_distributions

from miscellaneous.decomposition import (
    DECOMPOSITION_METHOD_NAMES, DEFAULT_DECOMPOSITION_DIMENSIONALITY
)
from miscellaneous.prediction import (
    predict, PREDICTION_METHOD_NAMES, PREDICTION_METHOD_SPECIFICATIONS
)

from auxiliary import (
    title, subtitle, heading,
    normaliseString, properString, enumerateListOfStrings,
    checkRunID,
    betterModelExists, modelStoppedEarly,
    removeEmptyDirectories
)

import os
import argparse
import itertools
import random

import warnings

# TODO Remove when TensorFlow Probability library is updated to v0.6
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=PendingDeprecationWarning)

def main(input_file_or_name, data_directory = "data",
    log_directory = "log", results_directory = "results",
    temporary_log_directory = None,
    map_features = False, feature_selection = [], example_filter = [],
    preprocessing_methods = [], noisy_preprocessing_methods = [],
    split_data_set = True,
    splitting_method = "default", splitting_fraction = 0.9,
    model_type = "VAE", latent_size = 50, hidden_sizes = [500],
    number_of_importance_samples = [5],
    number_of_monte_carlo_samples = [10],
    inference_architecture = "MLP",
    latent_distribution = "gaussian",
    number_of_classes = None,
    parameterise_latent_posterior = False,
    generative_architecture = "MLP",
    reconstruction_distribution = "poisson",
    number_of_reconstruction_classes = 0,
    prior_probabilities_method = "uniform",
    number_of_warm_up_epochs = 0,
    kl_weight = 1,
    proportion_of_free_KL_nats = 0.0,
    batch_normalisation = True,
    dropout_keep_probabilities = [],
    count_sum = True,
    number_of_epochs = 200, plotting_interval_during_training = None, 
    batch_size = 100, learning_rate = 1e-4,
    run_id = None, new_run = False,
    prediction_method = None, prediction_training_set_name = "training",
    prediction_decomposition_method = None,
    prediction_decomposition_dimensionality = None,
    decomposition_methods = ["PCA"], highlight_feature_indices = [],
    reset_training = False, skip_modelling = False,
    model_versions = ["all"],
    analyse = True, evaluation_set_name = "test", analyse_data = False,
    analyses = ["default"], analysis_level = "normal", fast_analysis = False,
    export_options = []):
    
    # Setup
    
    model_versions = parseModelVersions(model_versions)
    
    ## Analyses
    
    if fast_analysis:
        analyse = True
        analyses = ["simple"]
        analysis_level = "limited"
    
    ## Distributions
    
    reconstruction_distribution = parseDistribution(
        reconstruction_distribution)
    latent_distribution = parseDistribution(latent_distribution)
    
    ## Model configuration validation
    
    if not skip_modelling:
        
        if run_id:
            run_id = checkRunID(run_id)
        
        model_valid, model_errors = validateModelParameters(
            model_type, latent_distribution,
            reconstruction_distribution, number_of_reconstruction_classes,
            parameterise_latent_posterior
        )
        
        if not model_valid:
            print("Model configuration is invalid:")
            for model_error in model_errors:
                print("    ", model_error)
            print()
            if analyse_data:
                print("Skipping modelling.")
                print("")
                skip_modelling = True
            else:
                print("Modelling cancelled.")
                return
    
    ## Binarisation
    
    binarise_values = False
    
    if reconstruction_distribution == "bernoulli":
        if noisy_preprocessing_methods:
            if noisy_preprocessing_methods[-1] != "binarise":
                noisy_preprocessing_methods.append("binarise")
                print("Appended binarisation method to noisy preprocessing,",
                    "because of the Bernoulli distribution.\n")
        else:
            binarise_values = True
    
    ## Data sets
    
    if not split_data_set or analyse_data or evaluation_set_name == "full" \
        or prediction_training_set_name == "full":
            full_data_set_needed = True
    else:
        full_data_set_needed = False
    
    # Data
    
    print(title("Data"))
    
    data_set = data.DataSet(
        input_file_or_name,
        directory = data_directory,
        map_features = map_features,
        feature_selection = feature_selection,
        example_filter = example_filter,
        preprocessing_methods = preprocessing_methods,
        binarise_values = binarise_values,
        noisy_preprocessing_methods = noisy_preprocessing_methods
    )
    
    if full_data_set_needed:
        data_set.load()
    
    if split_data_set:
        training_set, validation_set, test_set = data_set.split(
            splitting_method, splitting_fraction)
        all_data_sets = [data_set, training_set, validation_set, test_set]
    else:
        splitting_method = None
        training_set = data_set
        validation_set = None
        test_set = data_set
        all_data_sets = [data_set]
        evaluation_set_name = "full"
        prediction_training_set_name = "full"
    
    ## Setup of log and results directories
    
    log_directory = data.directory(log_directory, data_set,
        splitting_method, splitting_fraction)
    data_results_directory = data.directory(results_directory, data_set,
        splitting_method, splitting_fraction, preprocessing = False)
    results_directory = data.directory(results_directory, data_set,
        splitting_method, splitting_fraction)
    
    if temporary_log_directory:
        main_temporary_log_directory = temporary_log_directory
        temporary_log_directory = data.directory(temporary_log_directory,
            data_set, splitting_method, splitting_fraction)
    
    ## Data analysis
    
    if analyse and analyse_data:
        print(subtitle("Analysing data"))
        analysis.analyseData(
            data_sets = all_data_sets,
            decomposition_methods = decomposition_methods,
            highlight_feature_indices = highlight_feature_indices,
            analyses = analyses,
            analysis_level = analysis_level,
            export_options = export_options,
            results_directory = data_results_directory
        )
        print()
    
    ## Full data set clean up
    
    if not full_data_set_needed:
        data_set.clear()
    
    # Modelling
    
    if skip_modelling:
        print("Modelling skipped.")
        return
    
    print(title("Modelling"))
    
    # Set the number of features for the model
    feature_size = training_set.number_of_features
    
    # Parse numbers of samples
    number_of_monte_carlo_samples = parseSampleLists(
        number_of_monte_carlo_samples)
    number_of_importance_samples = parseSampleLists(
        number_of_importance_samples)
    
    # Use analytical KL term for single-Gaussian-VAE
    if "VAE" in model_type:
        if latent_distribution == "gaussian":
            analytical_kl_term = True
        else:
            analytical_kl_term = False
    
    # Change latent distribution to Gaussian mixture if not already set
    if model_type == "GMVAE" and latent_distribution != "gaussian mixture":
        latent_distribution = "gaussian mixture"
        print("The latent distribution was changed to",
            "a Gaussian-mixture model, because of the model chosen.\n")
    
    # Set the number of classes if not already set
    if not number_of_classes:
        if training_set.has_labels:
            number_of_classes = training_set.number_of_classes \
                - training_set.number_of_excluded_classes
        elif "mixture" in latent_distribution:
            raise ValueError(
                "For a mixture model and a data set without labels, "
                "the number of classes has to be set."
            )
        else:
            number_of_classes = 1
    
    print(subtitle("Model setup"))
    
    if model_type == "VAE":
        model = VariationalAutoencoder(
            feature_size = feature_size,
            latent_size = latent_size,
            hidden_sizes = hidden_sizes,
            number_of_monte_carlo_samples =number_of_monte_carlo_samples,
            number_of_importance_samples = number_of_importance_samples,
            analytical_kl_term = analytical_kl_term,
            inference_architecture = inference_architecture,
            latent_distribution = latent_distribution,
            number_of_latent_clusters = number_of_classes,
            parameterise_latent_posterior = parameterise_latent_posterior,
            generative_architecture = generative_architecture,
            reconstruction_distribution = reconstruction_distribution,
            number_of_reconstruction_classes = number_of_reconstruction_classes,
            batch_normalisation = batch_normalisation,
            dropout_keep_probabilities = dropout_keep_probabilities,
            count_sum = count_sum,
            number_of_warm_up_epochs = number_of_warm_up_epochs,
            kl_weight = kl_weight,
            log_directory = log_directory,
            results_directory = results_directory
        )

    elif model_type == "GMVAE":
        
        if prior_probabilities_method == "uniform":
            prior_probabilities = None
        elif prior_probabilities_method == "infer":
            prior_probabilities = training_set.class_probabilities
        elif prior_probabilities_method == "literature":
            prior_probabilities = training_set.literature_probabilities
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
        
        model = GaussianMixtureVariationalAutoencoder(
            feature_size = feature_size,
            latent_size = latent_size,
            hidden_sizes = hidden_sizes,
            number_of_monte_carlo_samples = number_of_monte_carlo_samples,
            number_of_importance_samples = number_of_importance_samples, 
            analytical_kl_term = analytical_kl_term,
            prior_probabilities = prior_probabilities,
            number_of_latent_clusters = number_of_classes,
            proportion_of_free_KL_nats = proportion_of_free_KL_nats,
            reconstruction_distribution = reconstruction_distribution,
            number_of_reconstruction_classes = number_of_reconstruction_classes,
            batch_normalisation = batch_normalisation,
            dropout_keep_probabilities = dropout_keep_probabilities,
            count_sum = count_sum,
            number_of_warm_up_epochs = number_of_warm_up_epochs,
            kl_weight = kl_weight,
            log_directory = log_directory,
            results_directory = results_directory
        )
    
    else:
        raise ValueError("Model type not found: `{}`.".format(model_type))
    
    print(model.description)
    print()
    
    print(model.parameters)
    print()
    
    ## Training
    
    print(subtitle("Model training"))
    
    status, run_id = model.train(
        training_set,
        validation_set,
        number_of_epochs = number_of_epochs,
        batch_size = batch_size,
        learning_rate = learning_rate,
        plotting_interval = plotting_interval_during_training,
        run_id = run_id,
        new_run = new_run,
        reset_training = reset_training,
        temporary_log_directory = temporary_log_directory
    )
    
    # Remove temporary directories created and emptied during training
    if temporary_log_directory and os.path.exists(main_temporary_log_directory):
        removeEmptyDirectories(main_temporary_log_directory)
    
    if not status["completed"]:
        print(status["message"])
        return
    
    status_filename = "status"
    if "epochs trained" in status:
        status_filename += "-" + status["epochs trained"]
    status_path = os.path.join(
        model.logDirectory(run_id = run_id),
        status_filename + ".log"
    )
    with open(status_path, "w") as status_file:
        for status_field, status_value in status.items():
            if status_value:
                status_file.write(
                    status_field + ": " + str(status_value) + "\n"
                )
    
    print()
    
    # Evaluation, prediction, and analysis
    
    ## Setup
    
    if analyse:
        if prediction_method:
            predict_labels_using_model = False
        elif "GM" in model.type:
            predict_labels_using_model = True
            prediction_method = "model"
        else:
            predict_labels_using_model = False
    else:
        predict_labels_using_model = False
    
    evaluation_title_parts = ["evaluation"]
    
    if analyse:
        if prediction_method:
            evaluation_title_parts.append("prediction")
        evaluation_title_parts.append("analysis")
    
    evaluation_title = enumerateListOfStrings(evaluation_title_parts)
    
    print(title(evaluation_title.capitalize()))
    
    ### Set selection
    
    for data_subset in all_data_sets:
        
        clear_subset = True
        
        if data_subset.kind == evaluation_set_name:
            evaluation_set = data_subset
            clear_subset = False
            
        if prediction_method \
            and data_subset.kind == prediction_training_set_name:
                prediction_training_set = data_subset
                clear_subset = False
        
        if clear_subset:
            data_subset.clear()
    
    ### Evaluation set
    
    evaluation_subset_indices = analysis.evaluationSubsetIndices(
        evaluation_set)
    
    print("Evaluation set: {} set.".format(evaluation_set.kind))
    
    ### Prediction method
    
    if prediction_method:
        
        prediction_method = properString(
            prediction_method,
            PREDICTION_METHOD_NAMES
        )
        
        prediction_method_specifications = PREDICTION_METHOD_SPECIFICATIONS\
            .get(prediction_method, {})
        prediction_method_inference = prediction_method_specifications.get(
            "inference", None)
        prediction_method_fixed_number_of_clusters \
            = prediction_method_specifications.get(
                "fixed number of clusters", None)
        prediction_method_cluster_kind = prediction_method_specifications.get(
            "cluster kind", None)
        
        if prediction_method_fixed_number_of_clusters:
            number_of_clusters = number_of_classes
        else:
            number_of_clusters = None
        
        if prediction_method_inference \
            and prediction_method_inference == "transductive":
            
            prediction_training_set = None
            prediction_training_set_name = None
        
        else:
            prediction_training_set_name = prediction_training_set.kind
        
        prediction_details = {
            "method": prediction_method,
            "number_of_classes": number_of_clusters,
            "training_set_name": prediction_training_set_name,
            "decomposition_method": prediction_decomposition_method,
            "decomposition_dimensionality":
                prediction_decomposition_dimensionality
        }
        
        print("Prediction method: {}.".format(prediction_method))
        
        if number_of_clusters:
            print("Number of clusters: {}.".format(number_of_clusters))
        
        if prediction_training_set:
            print("Prediction training set: {} set.".format(
                prediction_training_set.kind))
        
        prediction_id_parts = []
        
        if prediction_decomposition_method:
            
            prediction_decomposition_method = properString(
                prediction_decomposition_method,
                DECOMPOSITION_METHOD_NAMES
            )
            
            if not prediction_decomposition_dimensionality:
                prediction_decomposition_dimensionality \
                    = DEFAULT_DECOMPOSITION_DIMENSIONALITY
            
            prediction_id_parts += [
                prediction_decomposition_method,
                prediction_decomposition_dimensionality
            ]
            
            prediction_details.update({
                "decomposition_method": prediction_decomposition_method,
                "decomposition_dimensionality":
                    prediction_decomposition_dimensionality
            })
            
            print("Decomposition method before prediction: {}-d {}.".format(
                prediction_decomposition_dimensionality,
                prediction_decomposition_method
            ))
        
        prediction_id_parts.append(prediction_method)
        
        if number_of_clusters:
            prediction_id_parts.append(number_of_clusters)
        
        if prediction_training_set \
            and prediction_training_set.kind != "training":
                prediction_id_parts.append(prediction_training_set.kind)
        
        prediction_id = "_".join(map(
            lambda s: normaliseString(str(s)).replace("_", ""),
            prediction_id_parts
        ))
        prediction_details["id"] = prediction_id
    
    else:
        prediction_details = {}
    
    ### Model parameter sets
    
    model_parameter_set_names = []
    
    if "end_of_training" in model_versions:
        model_parameter_set_names.append("end of training")
    
    if "best_model" in model_versions \
        and betterModelExists(model, run_id = run_id):
            model_parameter_set_names.append("best model")
    
    if "early_stopping" in model_versions \
        and modelStoppedEarly(model, run_id = run_id):
            model_parameter_set_names.append("early stopping")
    
    print("Model parameter sets: {}.".format(enumerateListOfStrings(
        model_parameter_set_names)))
    
    print()
    
    ## Model analysis
    
    if analyse:
        
        print(subtitle("Model analysis"))
        analysis.analyseModel(
            model = model,
            run_id = run_id,
            analyses = analyses,
            analysis_level = analysis_level,
            export_options = export_options,
            results_directory = results_directory
        )
    
    ## Results evaluation, prediction, and analysis
    
    for model_parameter_set_name in model_parameter_set_names:
        
        if model_parameter_set_name == "best model":
            use_best_model = True
        else:
            use_best_model = False
        
        if model_parameter_set_name == "early stopping":
            use_early_stopping_model = True
        else:
            use_early_stopping_model = False
        
        model_parameter_set_name = model_parameter_set_name.capitalize()
        print(subtitle(model_parameter_set_name))
        
        # Evaluation
        
        model_parameter_set_name = model_parameter_set_name.replace(" ", "-")
        
        print(heading("{} evaluation".format(model_parameter_set_name)))
        
        if "VAE" in model.type:
            transformed_evaluation_set, reconstructed_evaluation_set,\
                latent_evaluation_sets = model.evaluate(
                    evaluation_set = evaluation_set,
                    evaluation_subset_indices = evaluation_subset_indices,
                    batch_size = batch_size,
                    predict_labels = predict_labels_using_model,
                    run_id = run_id,
                    use_best_model = use_best_model,
                    use_early_stopping_model = use_early_stopping_model
                )
        else:
            transformed_evaluation_set, reconstructed_evaluation_set = \
                model.evaluate(
                    evaluation_set = evaluation_set,
                    evaluation_subset_indices = evaluation_subset_indices,
                    batch_size = batch_size,
                    run_id = run_id,
                    use_best_model = use_best_model,
                    use_early_stopping_model = use_early_stopping_model
                )
            latent_evaluation_sets = None
        
        print()
        
        # Prediction
        
        if analyse and "VAE" in model.type and prediction_method \
            and not transformed_evaluation_set.has_predictions:
            
            print(heading("{} prediction".format(model_parameter_set_name)))
            
            latent_prediction_evaluation_set = latent_evaluation_sets["z"]
            
            if prediction_method_inference \
                and prediction_method_inference == "inductive":
                
                latent_prediction_training_sets = model.evaluate(
                    evaluation_set = prediction_training_set,
                    batch_size = batch_size,
                    run_id = run_id,
                    use_best_model = use_best_model,
                    use_early_stopping_model = use_early_stopping_model,
                    output_versions = "latent",
                    log_results = False
                )
                latent_prediction_training_set \
                    = latent_prediction_training_sets["z"]
                
                print()
            
            else:
                latent_prediction_training_set = None
            
            if prediction_decomposition_method:
                
                if latent_prediction_training_set:
                    latent_prediction_training_set, \
                        latent_prediction_evaluation_set \
                        = data.decomposeDataSubsets(
                            latent_prediction_training_set,
                            latent_prediction_evaluation_set,
                            method = prediction_decomposition_method,
                            number_of_components = 
                                prediction_decomposition_dimensionality,
                            random = True
                        )
                else:
                    latent_prediction_evaluation_set \
                        = data.decomposeDataSubsets(
                            latent_prediction_evaluation_set,
                            method = prediction_decomposition_method,
                            number_of_components = 
                                prediction_decomposition_dimensionality,
                            random = True
                        )
                
                print()
            
            cluster_ids, predicted_labels, predicted_superset_labels \
                = predict(
                    latent_prediction_training_set,
                    latent_prediction_evaluation_set,
                    prediction_method,
                    number_of_clusters
                )
            
            transformed_evaluation_set.updatePredictions(
                predicted_cluster_ids = cluster_ids,
                predicted_labels = predicted_labels,
                predicted_superset_labels = predicted_superset_labels
            )
            reconstructed_evaluation_set.updatePredictions(
                predicted_cluster_ids = cluster_ids,
                predicted_labels = predicted_labels,
                predicted_superset_labels = predicted_superset_labels
            )
            
            for variable in latent_evaluation_sets:
                latent_evaluation_sets[variable].updatePredictions(
                    predicted_cluster_ids = cluster_ids,
                    predicted_labels = predicted_labels,
                    predicted_superset_labels = predicted_superset_labels
            )
            
            print()
        
        # Analysis
        
        if analyse:
            
            print(heading("{} results analysis".format(model_parameter_set_name)))
            
            analysis.analyseResults(
                evaluation_set = transformed_evaluation_set,
                reconstructed_evaluation_set = reconstructed_evaluation_set,
                latent_evaluation_sets = latent_evaluation_sets,
                model = model,
                run_id = run_id,
                decomposition_methods = decomposition_methods,
                evaluation_subset_indices = evaluation_subset_indices,
                highlight_feature_indices = highlight_feature_indices,
                prediction_details = prediction_details,
                best_model = use_best_model,
                early_stopping = use_early_stopping_model,
                analyses = analyses, analysis_level = analysis_level,
                export_options = export_options,
                results_directory = results_directory
            )
        
        # Clean up
        
        if transformed_evaluation_set.version == "original":
            transformed_evaluation_set.resetPredictions()

def parseModelVersions(proposed_versions):
    
    version_alias_sets = {
        "end_of_training": ["eot", "end", "finish", "finished"],
        "best_model": ["bm", "best", "optimal", "optimal_parameters", "op"],
        "early_stopping": ["es", "early", "stop", "stopped"]
    }
    
    parsed_versions = []
    
    if not isinstance(proposed_versions, list):
        proposed_versions = [proposed_versions]
    
    if proposed_versions == ["all"]:
        parsed_versions = list(version_alias_sets.keys())
    
    else:
        for proposed_version in proposed_versions:
            
            normalised_proposed_version = normaliseString(proposed_version)
            parsed_version = None
            
            for version, version_aliases in version_alias_sets.items():
                if normalised_proposed_version == version \
                    or normalised_proposed_version in version_aliases:
                        parsed_version = version
                        break
            
            if parsed_version:
                parsed_versions.append(parsed_version)
            else:
                raise ValueError(
                    "`{}` is not a model version.".format(
                        proposed_version
                    )
                )
    
    return parsed_versions

def parseDistribution(distribution):
    distribution = normaliseString(distribution)
    distribution_names = list(distributions.keys())
    distribution_names += list(latent_distributions.keys())
    for distribution_name in distribution_names:
        if normaliseString(distribution_name) == distribution:
            return distribution_name
    raise ValueError("Distribution `{}` not found.".format(distribution))

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
        
        if "constrained" in reconstruction_distribution:
            likelihood_error_list.append("constrained distributions")
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
    
    if not likelihood_validity:
        errors.append(likelihood_error)
    
    # Parameterisation of latent posterior for VAE
    if "VAE" in model_type:
        parameterise_validity = True
        parameterise_error = ""
        
        if not (model_type in ["VAE"]
            and latent_distribution == "gaussian mixture") \
            and parameterise_latent_posterior:
            
            parameterise_error = "Cannot parameterise latent posterior " \
                + "parameters for " + model_type + " or " \
                + latent_distribution + " distribution."
            parameterise_validity = False
        
        validity = validity and parameterise_validity
        
        if not parameterise_validity:
            errors.append(parameterise_error)
    
    # Return
    
    return validity, errors

parser = argparse.ArgumentParser(
    description='Model single-cell transcript counts using deep learning.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--input", "-i",
    type = str,
    dest = "input_file_or_name",
    help = "input: data set name or path to input file"
)
parser.add_argument(
    "--data-directory", "-D",
    type = str,
    default = "data",
    help = "directory where data are placed"
)
parser.add_argument(
    "--log-directory", "-L",
    type = str,
    default = "log",
    help = "directory where models are stored"
)
parser.add_argument(
    "--results-directory", "-R",
    type = str,
    default = "results",
    help = "directory where results are saved"
)
parser.add_argument(
    "--temporary-log-directory", "-T",
    type = str,
    help = "directory for temporary storage"
)
parser.add_argument(
    "--map-features",
    action = "store_true",
    help = "map features using a feature mapping if available"
)
parser.add_argument(
    "--skip-mapping-features",
    dest = "map_features",
    action = "store_false",
    help = "do not map features using any feature mapping"
)
parser.set_defaults(map_features = False)
parser.add_argument(
    "--feature-selection", "-F",
    type = str,
    nargs = "*",
    default = None,
    help = "method for selecting features"
)
parser.add_argument(
    "--example-filter", "-E",
    type = str,
    nargs = "*",
    default = None,
    help = "method for filtering examples, optionally followed by parameters"
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
    "--split-data-set",
    action = "store_true",
    help = "split data set"
)
parser.add_argument(
    "--skip-splitting-data-set",
    dest = "split_data_set",
    action = "store_false",
    help = "do not split data set"
)
parser.set_defaults(split_data_set = True)
parser.add_argument(
    "--splitting-method",
    type = str,
    default = "default",
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
    default = [1],
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
    "--inference-architecture",
    type = str,
    default = "MLP",
    help = "architecture of the inference model"
)
parser.add_argument(
    "--latent-distribution", "-q",
    type = str,
    default = "gaussian",
    help = "distribution for the latent variables"
)
parser.add_argument(
    "--number-of-classes", "-K",
    type = int,
    help = "number of proposed clusters in data set"
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
    "--generative-architecture",
    type = str,
    default = "MLP",
    help = "architecture of the generative model"
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
    "--prior-probabilities-method",
    type = str,
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
    "--plotting-interval-during-training",
    type = int,
    nargs = "?",
    help = "number of training epochs between each intermediate plot starting at the first"
)
parser.add_argument(
    "--batch-size", "-M",
    type = int,
    default = 100,
    help = "batch size used when training"
)
parser.add_argument(
    "--learning-rate",
    type = float,
    default = 1e-4,
    help = "learning rate when training"
)
parser.add_argument(
    "--number-of-warm-up-epochs", "-w",
    type = int,
    default = 0,
    help = "number of epochs with a linear weight on the KL divergence"
)
parser.add_argument(
    "--kl-weight",
    type = float,
    default = 1,
    help = "weighting of KL divergence"
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
parser.set_defaults(batch_normalisation = True)
parser.add_argument(
    "--dropout-keep-probabilities", "-d",
    type = float,
    nargs = "*",
    default = [],
    help = "List of probabilities, p, of keeping connections when using dropout. Interval: ]0, 1[, where p in {0, 1, False} means no dropout."
)
parser.add_argument(
    "--count-sum", "-s",
    action = "store_true",
    help = "use count sum"
)
parser.add_argument(
    "--no-count-sum", "-S",
    dest = "count_sum",
    action = "store_false",
    help = "do not use count sum"
)
parser.set_defaults(count_sum = False)
parser.add_argument(
    "--run-id",
    type = str,
    nargs = "?",
    default = None,
    help = "ID for separate run of the model (can only contrain alphanumeric characters)"
)
parser.add_argument(
    "--new-run",
    action = "store_true",
    help = "train a model anew as a separate run with a generated run ID"
)
parser.add_argument(
    "--prediction-method", "-P",
    type = str,
    nargs = "?",
    default = None,
    help = "method for predicting labels"
)
parser.add_argument(
    "--prediction-training-set-name",
    type = str,
    default = "training",
    help = "name of the subset on which to train prediction method: training (default), validation, test, or full"
)
parser.add_argument(
    "--prediction-decomposition-method",
    type = str,
    nargs = "?",
    default = None,
    help = "method for decomposing values before predicting labels"
)
parser.add_argument(
    "--prediction-decomposition-dimensionality",
    type = int,
    nargs = "?",
    default = None,
    help = "dimensionality of decomposition of values before predicting labels"
)
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
    "--model-versions",
    type = str,
    nargs = "+",
    default = ["all"],
    help = "model versions to evaluate: end-of-training, best model, early-stopping"
)
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
    default = "test",
    help = "name of the subset to evaluate and analyse: training, validation, test (default), or full"
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
    "--fast-analysis", "-f",
    action = "store_true",
    help = "perform fast analysis (equivalent to: `--analyses simple --analysis-level limited`)"
)
parser.set_defaults(fast_analysis = False)
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
