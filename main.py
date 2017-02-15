#!/usr/bin/env python3

import data
import modeling
import analysis

import argparse

def main(data_directory, log_directory, results_directory,
    (splitting_method, splitting_fraction),
    latent_size, hidden_sizes, reconstruction_distribution,
    number_of_reconstruction_classes,
    number_of_warm_up_epochs,
    number_of_epochs, batch_size, learning_rate,
    reset_training):
    
    # Data
    
    data_set = data.DataSet(data_set_name, data_directory)
    
    training_set, validation_set, test_set = data_set.split(
        splitting_method, splitting_fraction)
    
    feature_size = data_set.feature_size
    
    # Modeling
    
    model = modeling.VaritationalAutoEncoder(
        feature_size, latent_size, hidden_sizes,
        reconstruction_distribution, number_of_reconstruction_classes,
        log_directory
    )
    
    model.train(training_set, validation_set,
        number_of_epochs, batch_size, learning_rate,
        reset_training)
    
    # Analysis
    
    analysis.analyseModel(model, model.name, results_directory)
    
    reconstructed_test_set, latent_set, test_metrics = model.evaluate(test_set)
    
    analysis.analyseResults(test_set, reconstructed_test_set, latent_set,
        model.name, results_directory)
