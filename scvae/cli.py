# ======================================================================== #
#
# Copyright (c) 2017 - 2019 scVAE authors
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

import argparse
import os

import scvae
from scvae import analyses
from scvae.analyses.prediction import (
    PredictionSpecifications, predict_labels
)
from scvae.data import DataSet
from scvae.data.utilities import (
    build_directory_path, indices_for_evaluation_subset
)
from scvae.defaults import defaults
from scvae.models import (
    VariationalAutoencoder,
    GaussianMixtureVariationalAutoencoder
)
from scvae.models.utilities import (
    better_model_exists, model_stopped_early,
    parse_model_versions
)
from scvae.utilities import (
    title, subtitle, heading,
    normalise_string, enumerate_strings,
    remove_empty_directories
)


def analyse(data_set_file_or_name, data_format=None, data_directory=None,
            map_features=None, feature_selection=None, example_filter=None,
            preprocessing_methods=None, split_data_set=None,
            splitting_method=None, splitting_fraction=None,
            included_analyses=None, analysis_level=None,
            decomposition_methods=None, highlight_feature_indices=None,
            export_options=None, analyses_directory=None,
            **keyword_arguments):
    """Analyse data set."""

    if split_data_set is None:
        split_data_set = defaults["data"]["split_data_set"]
    if splitting_method is None:
        splitting_method = defaults["data"]["splitting_method"]
    if splitting_fraction is None:
        splitting_fraction = defaults["data"]["splitting_fraction"]
    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]

    print(title("Data"))

    data_set = DataSet(
        data_set_file_or_name,
        data_format=data_format,
        directory=data_directory,
        map_features=map_features,
        feature_selection=feature_selection,
        example_filter=example_filter,
        preprocessing_methods=preprocessing_methods,
    )
    data_set.load()

    if split_data_set:
        training_set, validation_set, test_set = data_set.split(
            method=splitting_method, fraction=splitting_fraction)
        all_data_sets = [data_set, training_set, validation_set, test_set]
    else:
        all_data_sets = [data_set]
        splitting_method = None
        splitting_fraction = None

    analyses_directory = build_directory_path(
        analyses_directory,
        data_set=data_set,
        splitting_method=splitting_method,
        splitting_fraction=splitting_fraction,
        preprocessing=False
    )

    print(subtitle("Analysing data"))

    analyses.analyse_data(
        data_sets=all_data_sets,
        decomposition_methods=decomposition_methods,
        highlight_feature_indices=highlight_feature_indices,
        included_analyses=included_analyses,
        analysis_level=analysis_level,
        export_options=export_options,
        analyses_directory=analyses_directory
    )

    return 0


def train(data_set_file_or_name, data_format=None, data_directory=None,
          map_features=None, feature_selection=None, example_filter=None,
          noisy_preprocessing_methods=None, preprocessing_methods=None,
          split_data_set=None, splitting_method=None, splitting_fraction=None,
          model_type=None, latent_size=None, hidden_sizes=None,
          number_of_importance_samples=None,
          number_of_monte_carlo_samples=None,
          inference_architecture=None, latent_distribution=None,
          number_of_classes=None, parameterise_latent_posterior=False,
          prior_probabilities_method=None,
          generative_architecture=None, reconstruction_distribution=None,
          number_of_reconstruction_classes=None, count_sum=None,
          proportion_of_free_nats_for_y_kl_divergence=None,
          minibatch_normalisation=None, batch_correction=None,
          dropout_keep_probabilities=None,
          number_of_warm_up_epochs=None, kl_weight=None,
          number_of_epochs=None, minibatch_size=None, learning_rate=None,
          run_id=None, new_run=False, reset_training=None,
          models_directory=None, caches_directory=None,
          analyses_directory=None, **keyword_arguments):
    """Train model on data set."""

    if split_data_set is None:
        split_data_set = defaults["data"]["split_data_set"]
    if splitting_method is None:
        splitting_method = defaults["data"]["splitting_method"]
    if splitting_fraction is None:
        splitting_fraction = defaults["data"]["splitting_fraction"]
    if models_directory is None:
        models_directory = defaults["models"]["directory"]

    print(title("Data"))

    binarise_values = False
    if reconstruction_distribution == "bernoulli":
        if noisy_preprocessing_methods:
            if noisy_preprocessing_methods[-1] != "binarise":
                noisy_preprocessing_methods.append("binarise")
        else:
            binarise_values = True

    data_set = DataSet(
        data_set_file_or_name,
        data_format=data_format,
        directory=data_directory,
        map_features=map_features,
        feature_selection=feature_selection,
        example_filter=example_filter,
        preprocessing_methods=preprocessing_methods,
        binarise_values=binarise_values,
        noisy_preprocessing_methods=noisy_preprocessing_methods
    )

    if split_data_set:
        training_set, validation_set, __ = data_set.split(
            method=splitting_method, fraction=splitting_fraction)
    else:
        data_set.load()
        splitting_method = None
        splitting_fraction = None
        training_set = data_set
        validation_set = None

    models_directory = build_directory_path(
        models_directory,
        data_set=data_set,
        splitting_method=splitting_method,
        splitting_fraction=splitting_fraction
    )

    if analyses_directory:
        analyses_directory = build_directory_path(
            analyses_directory,
            data_set=data_set,
            splitting_method=splitting_method,
            splitting_fraction=splitting_fraction
        )

    model_caches_directory = None
    if caches_directory:
        model_caches_directory = os.path.join(caches_directory, "log")
        model_caches_directory = build_directory_path(
            model_caches_directory,
            data_set=data_set,
            splitting_method=splitting_method,
            splitting_fraction=splitting_fraction
        )

    print(title("Model"))

    if number_of_classes is None:
        if training_set.has_labels:
            number_of_classes = (
                training_set.number_of_classes
                - training_set.number_of_excluded_classes)

    model = _setup_model(
        data_set=training_set,
        model_type=model_type,
        latent_size=latent_size,
        hidden_sizes=hidden_sizes,
        number_of_importance_samples=number_of_importance_samples,
        number_of_monte_carlo_samples=number_of_monte_carlo_samples,
        inference_architecture=inference_architecture,
        latent_distribution=latent_distribution,
        number_of_classes=number_of_classes,
        parameterise_latent_posterior=parameterise_latent_posterior,
        prior_probabilities_method=prior_probabilities_method,
        generative_architecture=generative_architecture,
        reconstruction_distribution=reconstruction_distribution,
        number_of_reconstruction_classes=number_of_reconstruction_classes,
        count_sum=count_sum,
        proportion_of_free_nats_for_y_kl_divergence=(
            proportion_of_free_nats_for_y_kl_divergence),
        minibatch_normalisation=minibatch_normalisation,
        batch_correction=batch_correction,
        dropout_keep_probabilities=dropout_keep_probabilities,
        number_of_warm_up_epochs=number_of_warm_up_epochs,
        kl_weight=kl_weight,
        models_directory=models_directory
    )

    print(model.description)
    print()

    print(model.parameters)
    print()

    print(subtitle("Training"))

    if analyses_directory:
        intermediate_analyser = analyses.analyse_intermediate_results
    else:
        intermediate_analyser = None

    model.train(
        training_set,
        validation_set,
        number_of_epochs=number_of_epochs,
        minibatch_size=minibatch_size,
        learning_rate=learning_rate,
        intermediate_analyser=intermediate_analyser,
        run_id=run_id,
        new_run=new_run,
        reset_training=reset_training,
        analyses_directory=analyses_directory,
        temporary_log_directory=model_caches_directory
    )

    # Remove temporary directories created and emptied during training
    if model_caches_directory and os.path.exists(caches_directory):
        remove_empty_directories(caches_directory)

    return 0


def evaluate(data_set_file_or_name, data_format=None, data_directory=None,
             map_features=None, feature_selection=None, example_filter=None,
             noisy_preprocessing_methods=None, preprocessing_methods=None,
             split_data_set=None, splitting_method=None,
             splitting_fraction=None,
             model_type=None, latent_size=None, hidden_sizes=None,
             number_of_importance_samples=None,
             number_of_monte_carlo_samples=None,
             inference_architecture=None, latent_distribution=None,
             number_of_classes=None, parameterise_latent_posterior=False,
             prior_probabilities_method=None,
             generative_architecture=None, reconstruction_distribution=None,
             number_of_reconstruction_classes=None, count_sum=None,
             proportion_of_free_nats_for_y_kl_divergence=None,
             minibatch_normalisation=None, batch_correction=None,
             dropout_keep_probabilities=None,
             number_of_warm_up_epochs=None, kl_weight=None,
             minibatch_size=None, run_id=None, models_directory=None,
             included_analyses=None, analysis_level=None,
             decomposition_methods=None, highlight_feature_indices=None,
             export_options=None, analyses_directory=None,
             evaluation_set_kind=None, sample_size=None,
             prediction_method=None, prediction_training_set_kind=None,
             model_versions=None, **keyword_arguments):
    """Evaluate model on data set."""

    if split_data_set is None:
        split_data_set = defaults["data"]["split_data_set"]
    if splitting_method is None:
        splitting_method = defaults["data"]["splitting_method"]
    if splitting_fraction is None:
        splitting_fraction = defaults["data"]["splitting_fraction"]
    if models_directory is None:
        models_directory = defaults["models"]["directory"]
    if evaluation_set_kind is None:
        evaluation_set_kind = defaults["evaluation"]["data_set_name"]
    if sample_size is None:
        sample_size = defaults["models"]["sample_size"]
    if prediction_method is None:
        prediction_method = defaults["evaluation"]["prediction_method"]
    if prediction_training_set_kind is None:
        prediction_training_set_kind = defaults["evaluation"][
            "prediction_training_set_kind"]
    if model_versions is None:
        model_versions = defaults["evaluation"]["model_versions"]
    if analyses_directory is None:
        analyses_directory = defaults["analyses"]["directory"]

    evaluation_set_kind = normalise_string(evaluation_set_kind)
    prediction_training_set_kind = normalise_string(
        prediction_training_set_kind)
    model_versions = parse_model_versions(model_versions)

    print(title("Data"))

    binarise_values = False
    if reconstruction_distribution == "bernoulli":
        if noisy_preprocessing_methods:
            if noisy_preprocessing_methods[-1] != "binarise":
                noisy_preprocessing_methods.append("binarise")
        else:
            binarise_values = True

    data_set = DataSet(
        data_set_file_or_name,
        data_format=data_format,
        directory=data_directory,
        map_features=map_features,
        feature_selection=feature_selection,
        example_filter=example_filter,
        preprocessing_methods=preprocessing_methods,
        binarise_values=binarise_values,
        noisy_preprocessing_methods=noisy_preprocessing_methods
    )

    if not split_data_set or evaluation_set_kind == "full":
        data_set.load()

    if split_data_set:
        training_set, validation_set, test_set = data_set.split(
            method=splitting_method, fraction=splitting_fraction)
        data_subsets = [data_set, training_set, validation_set, test_set]
        for data_subset in data_subsets:
            clear_data_subset = True
            if data_subset.kind == evaluation_set_kind:
                evaluation_set = data_subset
                clear_data_subset = False
            if data_subset.kind == prediction_training_set_kind:
                prediction_training_set = data_subset
                clear_data_subset = False
            if clear_data_subset:
                data_subset.clear()
    else:
        splitting_method = None
        splitting_fraction = None
        evaluation_set = data_set
        prediction_training_set = data_set

    evaluation_subset_indices = indices_for_evaluation_subset(
        evaluation_set)

    models_directory = build_directory_path(
        models_directory,
        data_set=evaluation_set,
        splitting_method=splitting_method,
        splitting_fraction=splitting_fraction
    )
    analyses_directory = build_directory_path(
        analyses_directory,
        data_set=evaluation_set,
        splitting_method=splitting_method,
        splitting_fraction=splitting_fraction
    )

    print(title("Model"))

    if number_of_classes is None:
        if evaluation_set.has_labels:
            number_of_classes = (
                evaluation_set.number_of_classes
                - evaluation_set.number_of_excluded_classes)

    model = _setup_model(
        data_set=evaluation_set,
        model_type=model_type,
        latent_size=latent_size,
        hidden_sizes=hidden_sizes,
        number_of_importance_samples=number_of_importance_samples,
        number_of_monte_carlo_samples=number_of_monte_carlo_samples,
        inference_architecture=inference_architecture,
        latent_distribution=latent_distribution,
        number_of_classes=number_of_classes,
        parameterise_latent_posterior=parameterise_latent_posterior,
        prior_probabilities_method=prior_probabilities_method,
        generative_architecture=generative_architecture,
        reconstruction_distribution=reconstruction_distribution,
        number_of_reconstruction_classes=number_of_reconstruction_classes,
        count_sum=count_sum,
        proportion_of_free_nats_for_y_kl_divergence=(
            proportion_of_free_nats_for_y_kl_divergence),
        minibatch_normalisation=minibatch_normalisation,
        batch_correction=batch_correction,
        dropout_keep_probabilities=dropout_keep_probabilities,
        number_of_warm_up_epochs=number_of_warm_up_epochs,
        kl_weight=kl_weight,
        models_directory=models_directory
    )

    if not model.has_been_trained(run_id=run_id):
        raise Exception("Cannot analyse model when it has not been trained.")

    if ("best_model" in model_versions
            and not better_model_exists(model, run_id=run_id)):
        model_versions.remove("best_model")

    if ("early_stopping" in model_versions
            and not model_stopped_early(model, run_id=run_id)):
        model_versions.remove("early_stopping")

    print(subtitle("Analysis"))

    analyses.analyse_model(
        model=model,
        run_id=run_id,
        included_analyses=included_analyses,
        analysis_level=analysis_level,
        export_options=export_options,
        analyses_directory=analyses_directory
    )

    print(title("Results"))

    print("Evaluation set: {} set.".format(evaluation_set.kind))
    print("Model version{}: {}.".format(
        "" if len(model_versions) == 1 else "s",
        enumerate_strings(
            [v.replace("_", " ") for v in model_versions], conjunction="and")))

    if prediction_method:
        prediction_specifications = PredictionSpecifications(
            method=prediction_method,
            number_of_clusters=number_of_classes,
            training_set_kind=prediction_training_set.kind
        )
        print("Prediction method: {}.".format(
            prediction_specifications.method))
        print("Number of clusters: {}.".format(
            prediction_specifications.number_of_clusters))
        print("Prediction training set: {} set.".format(
            prediction_specifications.training_set_kind))

    print()

    for model_version in model_versions:

        use_best_model = False
        use_early_stopping_model = False
        if model_version == "best_model":
            use_best_model = True
        elif model_version == "early_stopping":
            use_early_stopping_model = True

        print(subtitle(model_version.replace("_", " ").capitalize()))

        print(heading("{} evaluation".format(
            model_version.replace("_", "-").capitalize())))

        (
            transformed_evaluation_set,
            reconstructed_evaluation_set,
            latent_evaluation_sets
        ) = model.evaluate(
            evaluation_set=evaluation_set,
            evaluation_subset_indices=evaluation_subset_indices,
            minibatch_size=minibatch_size,
            run_id=run_id,
            use_best_model=use_best_model,
            use_early_stopping_model=use_early_stopping_model,
            output_versions="all"
        )
        print()

        if sample_size:
            print(heading("{} sampling".format(
                model_version.replace("_", "-").capitalize())))

            sample_reconstruction_set, __ = model.sample(
                sample_size=sample_size,
                minibatch_size=minibatch_size,
                run_id=run_id,
                use_best_model=use_best_model,
                use_early_stopping_model=use_early_stopping_model
            )
            print()
        else:
            sample_reconstruction_set = None

        if prediction_method:
            print(heading("{} prediction".format(
                model_version.replace("_", "-").capitalize())))

            latent_prediction_training_sets = model.evaluate(
                evaluation_set=prediction_training_set,
                minibatch_size=minibatch_size,
                run_id=run_id,
                use_best_model=use_best_model,
                use_early_stopping_model=use_early_stopping_model,
                output_versions="latent",
                log_results=False
            )
            print()

            cluster_ids, predicted_labels, predicted_superset_labels = (
                predict_labels(
                    training_set=latent_prediction_training_sets["z"],
                    evaluation_set=latent_evaluation_sets["z"],
                    specifications=prediction_specifications
                )
            )

            evaluation_set_versions = [
                transformed_evaluation_set, reconstructed_evaluation_set
            ] + list(latent_evaluation_sets.values())

            for evaluation_set_version in evaluation_set_versions:
                evaluation_set_version.update_predictions(
                    prediction_specifications=prediction_specifications,
                    predicted_cluster_ids=cluster_ids,
                    predicted_labels=predicted_labels,
                    predicted_superset_labels=predicted_superset_labels
                )
            print()

        print(heading("{} analysis".format(
            model_version.replace("_", "-").capitalize())))

        analyses.analyse_results(
            evaluation_set=transformed_evaluation_set,
            reconstructed_evaluation_set=reconstructed_evaluation_set,
            latent_evaluation_sets=latent_evaluation_sets,
            model=model,
            run_id=run_id,
            sample_reconstruction_set=sample_reconstruction_set,
            decomposition_methods=decomposition_methods,
            evaluation_subset_indices=evaluation_subset_indices,
            highlight_feature_indices=highlight_feature_indices,
            best_model=use_best_model,
            early_stopping=use_early_stopping_model,
            included_analyses=included_analyses,
            analysis_level=analysis_level,
            export_options=export_options,
            analyses_directory=analyses_directory
        )

    return 0


def cross_analyse(analyses_directory,
                  include_data_sets=None, exclude_data_sets=None,
                  include_models=None, exclude_models=None,
                  include_prediction_methods=None,
                  exclude_prediction_methods=None,
                  extra_model_specification_for_plots=None,
                  no_prediction_methods_for_gmvae_in_plots=False,
                  epoch_cut_off=None, other_methods=None,
                  export_options=None, log_summary=None,
                  **keyword_arguments):
    """Cross-analyse models and results for split data sets."""

    analyses.cross_analysis.cross_analyse(
        analyses_directory=analyses_directory,
        data_set_included_strings=include_data_sets,
        data_set_excluded_strings=exclude_data_sets,
        model_included_strings=include_models,
        model_excluded_strings=exclude_models,
        prediction_included_strings=include_prediction_methods,
        prediction_excluded_strings=exclude_prediction_methods,
        additional_other_option=extra_model_specification_for_plots,
        no_prediction_methods_for_gmvae_in_plots=(
            no_prediction_methods_for_gmvae_in_plots),
        epoch_cut_off=epoch_cut_off,
        other_methods=other_methods,
        export_options=export_options,
        log_summary=log_summary
    )

    return 0


def _setup_model(data_set, model_type=None,
                 latent_size=None, hidden_sizes=None,
                 number_of_importance_samples=None,
                 number_of_monte_carlo_samples=None,
                 inference_architecture=None, latent_distribution=None,
                 number_of_classes=None, parameterise_latent_posterior=False,
                 prior_probabilities_method=None,
                 generative_architecture=None,
                 reconstruction_distribution=None,
                 number_of_reconstruction_classes=None, count_sum=None,
                 proportion_of_free_nats_for_y_kl_divergence=None,
                 minibatch_normalisation=None, batch_correction=None,
                 dropout_keep_probabilities=None,
                 number_of_warm_up_epochs=None, kl_weight=None,
                 models_directory=None):

    if model_type is None:
        model_type = defaults["model"]["type"]
    if batch_correction is None:
        batch_correction = defaults["model"]["batch_correction"]

    feature_size = data_set.number_of_features
    number_of_batches = data_set.number_of_batches

    if not data_set.has_batches:
        batch_correction = False

    if normalise_string(model_type) == "vae":
        model = VariationalAutoencoder(
            feature_size=feature_size,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            number_of_monte_carlo_samples=number_of_monte_carlo_samples,
            number_of_importance_samples=number_of_importance_samples,
            inference_architecture=inference_architecture,
            latent_distribution=latent_distribution,
            number_of_latent_clusters=number_of_classes,
            parameterise_latent_posterior=parameterise_latent_posterior,
            generative_architecture=generative_architecture,
            reconstruction_distribution=reconstruction_distribution,
            number_of_reconstruction_classes=number_of_reconstruction_classes,
            minibatch_normalisation=minibatch_normalisation,
            batch_correction=batch_correction,
            number_of_batches=number_of_batches,
            dropout_keep_probabilities=dropout_keep_probabilities,
            count_sum=count_sum,
            number_of_warm_up_epochs=number_of_warm_up_epochs,
            kl_weight=kl_weight,
            log_directory=models_directory
        )

    elif normalise_string(model_type) == "gmvae":
        prior_probabilities_method_for_model = prior_probabilities_method
        if prior_probabilities_method == "uniform":
            prior_probabilities = None
        elif prior_probabilities_method == "infer":
            prior_probabilities_method_for_model = "custom"
            prior_probabilities = data_set.class_probabilities
        else:
            prior_probabilities = None

        model = GaussianMixtureVariationalAutoencoder(
            feature_size=feature_size,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            number_of_monte_carlo_samples=number_of_monte_carlo_samples,
            number_of_importance_samples=number_of_importance_samples,
            prior_probabilities_method=prior_probabilities_method_for_model,
            prior_probabilities=prior_probabilities,
            latent_distribution=latent_distribution,
            number_of_latent_clusters=number_of_classes,
            proportion_of_free_nats_for_y_kl_divergence=(
                proportion_of_free_nats_for_y_kl_divergence),
            reconstruction_distribution=reconstruction_distribution,
            number_of_reconstruction_classes=number_of_reconstruction_classes,
            minibatch_normalisation=minibatch_normalisation,
            batch_correction=batch_correction,
            number_of_batches=number_of_batches,
            dropout_keep_probabilities=dropout_keep_probabilities,
            count_sum=count_sum,
            number_of_warm_up_epochs=number_of_warm_up_epochs,
            kl_weight=kl_weight,
            log_directory=models_directory
        )

    else:
        raise ValueError("Model type not found: `{}`.".format(model_type))

    return model


def _parse_default(default):
    if not isinstance(default, bool) and default != 0 and not default:
        default = None
    return default


def main():
    parser = argparse.ArgumentParser(
        prog=scvae.__name__,
        description=scvae.__description__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--version", "-V",
        action="version",
        version='%(prog)s {version}'.format(version=scvae.__version__))
    subparsers = parser.add_subparsers(help="commands", dest="command")
    subparsers.required = True

    data_set_subparsers = []
    model_subparsers = []
    training_subparsers = []
    evaluation_subparsers = []
    analysis_subparsers = []

    parser_analyse = subparsers.add_parser(
        name="analyse",
        description="Analyse single-cell transcript counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_analyse.set_defaults(func=analyse)
    data_set_subparsers.append(parser_analyse)
    analysis_subparsers.append(parser_analyse)

    parser_train = subparsers.add_parser(
        name="train",
        description="Train model on single-cell transcript counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_train.set_defaults(func=train)
    data_set_subparsers.append(parser_train)
    model_subparsers.append(parser_train)
    training_subparsers.append(parser_train)

    parser_evaluate = subparsers.add_parser(
        name="evaluate",
        description="Evaluate model on single-cell transcript counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_evaluate.set_defaults(func=evaluate)
    data_set_subparsers.append(parser_evaluate)
    model_subparsers.append(parser_evaluate)
    evaluation_subparsers.append(parser_evaluate)
    analysis_subparsers.append(parser_evaluate)

    parser_cross_analyse = subparsers.add_parser(
        name="cross-analyse",
        description="Cross-analyse models and results on withheld data sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_cross_analyse.set_defaults(func=cross_analyse)

    for subparser in data_set_subparsers:
        subparser.add_argument(
            dest="data_set_file_or_name",
            help="data set name or path to data set file"
        )
        subparser.add_argument(
            "--format", "-f",
            dest="data_format",
            metavar="FORMAT",
            default=_parse_default(defaults["data"]["format"]),
            help="format of the data set"
        )
        subparser.add_argument(
            "--data-directory", "-D",
            metavar="DIRECTORY",
            default=_parse_default(defaults["data"]["directory"]),
            help="directory where data are placed or copied"
        )
        subparser.add_argument(
            "--map-features",
            action="store_true",
            default=_parse_default(defaults["data"]["map_features"]),
            help="map features using a feature mapping, if available"
        )
        subparser.add_argument(
            "--feature-selection", "-F",
            metavar="SELECTION",
            nargs="+",
            default=_parse_default(defaults["data"]["feature_selection"]),
            help="method for selecting features"
        )
        subparser.add_argument(
            "--example-filter", "-E",
            metavar="FILTER",
            nargs="+",
            default=_parse_default(defaults["data"]["example_filter"]),
            help=(
                "method for filtering examples, optionally followed by "
                "parameters"
            )
        )
        subparser.add_argument(
            "--preprocessing-methods", "-p",
            metavar="METHOD",
            nargs="+",
            default=_parse_default(defaults["data"]["preprocessing_methods"]),
            help="methods for preprocessing data (applied in order)"
        )
        subparser.add_argument(
            "--split-data-set",
            action="store_true",
            default=_parse_default(defaults["data"]["split_data_set"]),
            help="split data set into training, validation, and test sets"
        )
        subparser.add_argument(
            "--splitting-method",
            metavar="METHOD",
            default=_parse_default(defaults["data"]["splitting_method"]),
            help=(
                "method for splitting data into training, validation, and "
                "test sets"
            )
        )
        subparser.add_argument(
            "--splitting-fraction",
            metavar="FRACTION",
            type=float,
            default=_parse_default(defaults["data"]["splitting_fraction"]),
            help=(
                "fraction to use when splitting data into training, "
                "validation, and test sets"
            )
        )

    for subparser in model_subparsers:
        subparser.add_argument(
            "--model-type", "-m",
            metavar="TYPE",
            default=_parse_default(defaults["models"]["type"]),
            help="type of model; either VAE or GMVAE"
        )
        subparser.add_argument(
            "--latent-size", "-l",
            metavar="SIZE",
            type=int,
            default=_parse_default(defaults["models"]["latent_size"]),
            help="size of latent space"
        )
        subparser.add_argument(
            "--hidden-sizes", "-H",
            metavar="SIZE",
            type=int,
            nargs="+",
            default=_parse_default(defaults["models"]["hidden_sizes"]),
            help="sizes of hidden layers"
        )
        subparser.add_argument(
            "--number-of-importance-samples",
            metavar="NUMBER",
            type=int,
            nargs="+",
            default=_parse_default(defaults["models"]["number_of_samples"]),
            help=(
                "the number of importance weighted samples (if two numbers "
                "are given, the first will be used for training and the "
                "second for evaluation)"
            )
        )
        subparser.add_argument(
            "--number-of-monte-carlo-samples",
            metavar="NUMBER",
            type=int,
            nargs="+",
            default=_parse_default(defaults["models"]["number_of_samples"]),
            help=(
                "the number of Monte Carlo samples (if two numbers are given, "
                "the first will be used for training and the second for "
                "evaluation)"
            )
        )
        subparser.add_argument(
            "--inference-architecture",
            metavar="KIND",
            default=_parse_default(defaults["models"][
                "inference_architecture"]),
            help="architecture of the inference model"
        )
        subparser.add_argument(
            "--latent-distribution", "-q",
            metavar="DISTRIBUTION",
            help=(
                "distribution for the latent variable(s); defaults depends on "
                "model type")
        )
        subparser.add_argument(
            "--number-of-classes", "-K",
            metavar="NUMBER",
            type=int,
            help="number of proposed clusters in data set"
        )
        subparser.add_argument(
            "--parameterise-latent-posterior",
            action="store_true",
            default=_parse_default(defaults["models"][
                "parameterise_latent_posterior"]),
            help="parameterise latent posterior parameters, if possible"
        )
        subparser.add_argument(
            "--generative-architecture",
            metavar="KIND",
            default=_parse_default(defaults["models"][
                "generative_architecture"]),
            help="architecture of the generative model"
        )
        subparser.add_argument(
            "--reconstruction-distribution", "-r",
            metavar="DISTRIBUTION",
            default=_parse_default(defaults["models"][
                "reconstruction_distribution"]),
            help="distribution for the reconstructions"
        )
        subparser.add_argument(
            "--number-of-reconstruction-classes", "-k",
            metavar="NUMBER",
            type=int,
            default=_parse_default(defaults["models"][
                "number_of_reconstruction_classes"]),
            help="the maximum count for which to use classification"
        )
        subparser.add_argument(
            "--prior-probabilities-method",
            metavar="METHOD",
            default=_parse_default(defaults["models"][
                "prior_probabilities_method"]),
            help="method to set prior probabilities"
        )

        subparser.add_argument(
            "--number-of-warm-up-epochs", "-w",
            metavar="NUMBER",
            type=int,
            default=_parse_default(defaults["models"][
                "number_of_warm_up_epochs"]),
            help=(
                "number of initial epochs with a linear weight on the "
                "KL divergence")
        )
        subparser.add_argument(
            "--kl-weight",
            metavar="WEIGHT",
            type=float,
            default=_parse_default(defaults["models"]["kl_weight"]),
            help="weighting of KL divergence"
        )
        subparser.add_argument(
            "--proportion-of-free-nats-for-y-kl-divergence",
            metavar="PROPORTION",
            type=float,
            default=_parse_default(defaults["models"][
                "proportion_of_free_nats_for_y_kl_divergence"]),
            help=(
                "proportion of maximum y KL divergence, which has constant "
                "term and zero gradients, for the GMVAE (free-bits method)"
            )
        )
        subparser.add_argument(
            "--minibatch-normalisation", "-b",
            action="store_true",
            default=_parse_default(defaults["models"][
                "minibatch_normalisation"]),
            help="use batch normalisation for minibatches in models"
        )
        subparser.add_argument(
            "--batch-correction", "--bc",
            action="store_true",
            default=_parse_default(defaults["models"][
                "batch_correction"]),
            help="use batch correction in models"
        )
        subparser.add_argument(
            "--dropout-keep-probabilities",
            metavar="PROBABILITY",
            type=float,
            nargs="+",
            default=_parse_default(defaults["models"][
                "dropout_keep_probabilities"]),
            help=(
                "list of probabilities, p, of keeping connections when using "
                "dropout (interval: ]0, 1[, where p in {0, 1, False} means no "
                "dropout)"
            )
        )
        subparser.add_argument(
            "--count-sum",
            action="store_true",
            default=_parse_default(defaults["models"]["count_sum"]),
            help="use count sum"
        )
        subparser.add_argument(
            "--minibatch-size", "-B",
            metavar="SIZE",
            type=int,
            default=_parse_default(defaults["models"]["minibatch_size"]),
            help="minibatch size for stochastic optimisation algorithm"
        )
        subparser.add_argument(
            "--run-id",
            metavar="ID",
            type=str,
            default=_parse_default(defaults["models"]["run_id"]),
            help=(
                "ID for separate run of the model (can only contain "
                "alphanumeric characters)"
            )
        )
        subparser.add_argument(
            "--models-directory", "-M",
            metavar="DIRECTORY",
            default=_parse_default(defaults["models"]["directory"]),
            help="directory where models are stored"
        )

    for subparser in training_subparsers:
        subparser.add_argument(
            "--number-of-epochs", "-e",
            metavar="NUMBER",
            type=int,
            default=_parse_default(defaults["models"]["number_of_epochs"]),
            help="number of epochs for which to train"
        )
        subparser.add_argument(
            "--learning-rate",
            metavar="RATE",
            type=float,
            default=_parse_default(defaults["models"]["learning_rate"]),
            help="learning rate when training"
        )
        subparser.add_argument(
            "--new-run",
            action="store_true",
            default=_parse_default(defaults["models"]["new_run"]),
            help="train a model anew as a separate run with a generated run ID"
        )
        subparser.add_argument(
            "--reset-training",
            action="store_true",
            default=_parse_default(defaults["models"]["reset_training"]),
            help="reset already trained model"
        )
        subparser.add_argument(
            "--caches-directory", "-C",
            metavar="DIRECTORY",
            help="directory for temporary storage"
        )
        subparser.add_argument(
            "--analyses-directory", "-A",
            metavar="DIRECTORY",
            default=None,
            help="directory where analyses are saved"
        )

    for subparser in analysis_subparsers:
        subparser.add_argument(
            "--included-analyses",
            metavar="ANALYSIS",
            nargs="+",
            default=_parse_default(defaults["analyses"]["included_analyses"]),
            help=(
                "analyses to perform, which can be specified individually or "
                "as groups: simple, standard, all"
            )
        )
        subparser.add_argument(
            "--analysis-level",
            metavar="LEVEL",
            default=_parse_default(defaults["analyses"]["analysis_level"]),
            help=(
                "level to which analyses are performed: "
                "limited, normal, extensive"
            )
        )
        subparser.add_argument(
            "--decomposition-methods",
            metavar="METHOD",
            nargs="+",
            default=_parse_default(
                defaults["analyses"]["decomposition_method"]),
            help="methods use to decompose values"
        )
        subparser.add_argument(
            "--highlight-feature-indices",
            metavar="INDEX",
            type=int,
            nargs="+",
            default=_parse_default(
                defaults["analyses"]["highlight_feature_indices"]),
            help="feature indices to highlight in analyses"
        )
        subparser.add_argument(
            "--export-options",
            metavar="OPTION",
            nargs="+",
            default=_parse_default(defaults["analyses"]["export_options"]),
            help="export options for analyses"
        )
        subparser.add_argument(
            "--analyses-directory", "-A",
            metavar="DIRECTORY",
            default=_parse_default(defaults["analyses"]["directory"]),
            help="directory where analyses are saved"
        )

    for subparser in evaluation_subparsers:
        subparser.add_argument(
            "--evaluation-set-kind",
            metavar="KIND",
            default=_parse_default(defaults["evaluation"]["data_set_kind"]),
            help=(
                "kind of subset to evaluate and analyse: "
                "training, validation, test (default), or full"
            )
        )
        subparser.add_argument(
            "--sample-size",
            metavar="SIZE",
            type=int,
            default=_parse_default(defaults["models"]["sample_size"]),
            help="sample size for sampling model"
        )
        subparser.add_argument(
            "--prediction-method", "-P",
            metavar="METHOD",
            default=_parse_default(
                defaults["evaluation"]["prediction_method"]),
            help="method for predicting labels"
        )
        subparser.add_argument(
            "--prediction-training-set-kind",
            metavar="KIND",
            default=_parse_default(
                defaults["evaluation"]["prediction_training_set_kind"]),
            help=(
                "kind of subset to evaluate and analyse: "
                "training, validation, test (default), or full"
            )
        )
        subparser.add_argument(
            "--model-versions",
            metavar="VERSION",
            nargs="+",
            default=_parse_default(defaults["evaluation"]["model_versions"]),
            help=(
                "model versions to evaluate: end-of-training, best-model, "
                "early-stopping"
            )
        )

    parser_cross_analyse.add_argument(
        "analyses_directory",
        metavar="ANALYSES_DIRECTORY",
        help="directory where analyses were saved"
    )
    parser_cross_analyse.add_argument(
        "--include-data-sets", "-d",
        metavar="TEXT",
        nargs="+",
        help="only include data set that match TEXT"
    )
    parser_cross_analyse.add_argument(
        "--exclude-data-sets", "-D",
        metavar="TEXT",
        nargs="+",
        help="exclude data sets that match TEXT"
    )
    parser_cross_analyse.add_argument(
        "--include-models", "-m",
        metavar="TEXT",
        nargs="+",
        help="only include models that match TEXT"
    )
    parser_cross_analyse.add_argument(
        "--exclude-models", "-M",
        metavar="TEXT",
        nargs="+",
        help="exclude models that match TEXT"
    )
    parser_cross_analyse.add_argument(
        "--include-prediction-methods", "-p",
        metavar="TEXT",
        nargs="+",
        help="only include prediction methods that match TEXT"
    )
    parser_cross_analyse.add_argument(
        "--exclude-prediction-methods", "-P",
        metavar="TEXT",
        nargs="+",
        help="exclude prediction methods that match TEXT"
    )
    parser_cross_analyse.add_argument(
        "--extra-model-specification-for-plots", "-a",
        metavar="SPECIFICATION",
        help=(
            "extra model specification required in model metrics plots"
        )
    )
    parser_cross_analyse.add_argument(
        "--no-prediction-methods-for-gmvae-in-plots",
        action="store_true",
        default=False,
        help=(
            "do not include prediction methods for GMVAE in model metrics "
            "plots"
        )
    )
    parser_cross_analyse.add_argument(
        "--epoch-cut-off", "-e",
        metavar="EPOCH_NUMBER",
        type=int,
        help="exclude models trained for longer than this number of epochs"
    )
    parser_cross_analyse.add_argument(
        "--other-methods", "-o",
        metavar="METHOD",
        nargs="+",
        help="other methods to plot in model metrics plot, if available"
    )
    parser_cross_analyse.add_argument(
        "--export-options",
        metavar="OPTION",
        nargs="+",
        default=_parse_default(defaults["analyses"]["export_options"]),
        help="export options for cross-analyses"
    )
    parser_cross_analyse.add_argument(
        "--log-summary", "-s",
        action="store_true",
        default=_parse_default(defaults["cross_analysis"]["log_summary"]),
        help="log summary (saved in ANALYSES_DIRECTORY)"
    )

    arguments = parser.parse_args()
    status = arguments.func(**vars(arguments))
    return status
