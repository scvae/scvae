import os
import sys
import shutil

import time

import re

from math import floor

import urllib.request

from tensorboard.backend.event_processing import event_multiplexer
import numpy

# Time

def formatTime(t):
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t))

def formatDuration(seconds):
    if seconds < 0.001:
        return "<1 ms"
    elif seconds < 1:
        return "{:.0f} ms".format(1000 * seconds)
    elif seconds < 60:
        return "{:.3g} s".format(seconds)
    elif seconds < 60 * 60:
        minutes = floor(seconds / 60)
        seconds = seconds % 60
        if round(seconds) == 60:
            seconds = 0
            minutes += 1
        return "{:.0f}m {:.0f}s".format(minutes, seconds)
    else:
        hours = floor(seconds / 60 / 60)
        minutes = floor((seconds / 60) % 60)
        seconds = seconds % 60
        if round(seconds) == 60:
            seconds = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            hours += 1
        return "{:.0f}h {:.0f}m {:.0f}s".format(hours, minutes, seconds)

# Strings

def normaliseString(s):
    
    s = s.lower()
    
    replacements = {
        "_": [" ", "-", "/"],
        "": ["(", ")", ",", "$"]
    }
    
    for replacement, characters in replacements.items():
        pattern = "[" + re.escape("".join(characters)) + "]"
        s = re.sub(pattern, replacement, s)
    
    return s

def properString(s, translation):
    
    s = normaliseString(s)
    
    for proper_string, normalised_strings in translation.items():
        if s in normalised_strings:
            return proper_string

def enumerateListOfStrings(list_of_strings):
    if len(list_of_strings) == 1:
        enumerated_string = list_of_strings[0]
    elif len(list_of_strings) == 2:
        enumerated_string = " and ".join(list_of_strings)
    elif len(list_of_strings) >= 3:
        enumerated_string = "{}, and {}".format(
            ", ".join(list_of_strings[:-1]),
            list_of_strings[-1]
        )
    return enumerated_string

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Loading function for TensorBoard summaries

def loadNumberOfEpochsTrained(model, early_stopping = False, best_model = False):
    
    # Setup
    
    ## Data set kind
    data_set_kind = "training"
    
    ## Loss depending on model type
    if model.type == "SNN":
        loss = "log_likelihood"
    elif "AE" in model.type:
        loss = "lower_bound"
    
    ## Log directory
    if early_stopping:
        log_directory = model.early_stopping_log_directory
    elif best_model:
        log_directory = model.best_model_log_directory
    else:
        log_directory = model.log_directory
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        log_directory)
    multiplexer.Reload()
    
    # Loading
    
    # Losses for every epochs
    scalars = multiplexer.Scalars(data_set_kind, "losses/" + loss)
    
    # First estimate of number of epochs
    E_1 = len(scalars)
    
    # Second estimate of number of epochs
    E_2 = 0
    
    for scalar in scalars:
        E_2 = max(E_2, scalar.step)
    
    assert E_1 == E_2
    
    return E_1

def loadLearningCurves(model, data_set_kinds = "all", early_stopping = False,
    best_model = False, log_directory = None):
    
    # Setup
    
    learning_curve_sets = {}
    
    ## Data set kinds
    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "evaluation"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]
    
    ## Log directory
    if not log_directory:
        log_directory = model.logDirectory(
            early_stopping = early_stopping,
            best_model = best_model
        )
    
    ## Losses depending on model type
    if model.type == "SNN":
        losses = ["log_likelihood"]
    elif model.type == "CVAE":
        losses = ["lower_bound", "reconstruction_error",
            "kl_divergence_z1", "kl_divergence_z2", "kl_divergence_y"]
    elif model.type in ["GMVAE", "GMVAE_alt"]:
        losses = ["lower_bound", "reconstruction_error",
            "kl_divergence_z", "kl_divergence_y"]
    elif "AE" in model.type:
        losses = ["lower_bound", "reconstruction_error", "kl_divergence"]
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        log_directory)
    multiplexer.Reload()
    
    # Loading
    
    for data_set_kind in data_set_kinds:
        
        learning_curve_set = {}
        
        for loss in losses:
            
            scalars = multiplexer.Scalars(data_set_kind, "losses/" + loss)
            
            learning_curve = numpy.empty(len(scalars))
            
            if len(scalars) == 1:
                learning_curve[0] = scalars[0].value
            else:
                for scalar in scalars:
                    learning_curve[scalar.step - 1] = scalar.value
            
            learning_curve_set[loss] = learning_curve
        
        learning_curve_sets[data_set_kind] = learning_curve_set
    
    if len(data_set_kinds) == 1:
        learning_curve_sets = learning_curve_sets[data_set_kinds[0]]
    
    return learning_curve_sets

def loadAccuracies(model, data_set_kinds = "all", superset = False,
    early_stopping = False, best_model = False):
    
    # Setup
    
    accuracies = {}
    
    ## Data set kinds
    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "evaluation"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]
    
    ## Log directory
    if early_stopping:
        log_directory = model.early_stopping_log_directory
    elif best_model:
        log_directory = model.best_model_log_directory
    else:
        log_directory = model.log_directory
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        log_directory)
    multiplexer.Reload()
    
    ## Tag
    
    accuracy_tag = "accuracy"
    
    if superset:
        accuracy_tag = "superset_" + accuracy_tag
    
    # Loading
    
    errors = 0
    
    for data_set_kind in data_set_kinds:
        
        try:
            scalars = multiplexer.Scalars(data_set_kind, accuracy_tag)
        except KeyError:
            accuracies[data_set_kind] = None
            errors += 1
            continue
        
        accuracies_kind = numpy.empty(len(scalars))
        
        if len(scalars) == 1:
            accuracies_kind[0] = scalars[0].value
        else:
            for scalar in scalars:
                accuracies_kind[scalar.step - 1] = scalar.value
        
        accuracies[data_set_kind] = accuracies_kind
    
    if len(data_set_kinds) == 1:
        accuracies = accuracies[data_set_kinds[0]]
    
    if errors == len(data_set_kinds):
        accuracies = None
    
    return accuracies

def loadCentroids(model, data_set_kinds = "all", early_stopping = False,
    best_model = False):
    
    # Setup
    
    centroids_sets = {}
    
    ## Data set kinds
    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "evaluation"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]
    
    ## Exit if SNN model
    if model.type == "SNN":
        return None
    
    ## Log directory
    if early_stopping:
        log_directory = model.early_stopping_log_directory
    elif best_model:
        log_directory = model.best_model_log_directory
    else:
        log_directory = model.log_directory
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        log_directory)
    multiplexer.Reload()
    
    # Loading
    
    for data_set_kind in data_set_kinds:
        
        centroids_set = {}
        
        for distribution in ["prior", "posterior"]:
            
            try:
                scalars = multiplexer.Scalars(data_set_kind,
                    distribution + "/cluster_0/probability")
            except KeyError:
                centroids_set[distribution] = None
                continue
            
            # Number of epochs
            E = len(scalars)
            
            # Number of clusters
            if "mixture" in model.latent_distribution[distribution]["name"]:
                K = model.number_of_latent_clusters
            else:
                K = 1
            
            # Number of latent dimensions
            L = model.latent_size
            
            # Initialise
            z_probabilities = numpy.empty((E, K))
            z_means = numpy.empty((E, K, L))
            z_variances = numpy.empty((E, K, L))
            z_covariance_matrices = numpy.empty((E, K, L, L))
            
            # Looping
            for k in range(K):
                
                probability_scalars = multiplexer.Scalars(data_set_kind,
                    distribution + "/cluster_{}/probability".format(k))
                
                if len(probability_scalars) == 1:
                    z_probabilities[0][k] = probability_scalars[0].value
                else:
                    for scalar in probability_scalars:
                        z_probabilities[scalar.step - 1][k] = scalar.value
                
                for l in range(L):
                    
                    mean_scalars = multiplexer.Scalars(data_set_kind,
                        distribution + "/cluster_{}/mean/dimension_{}".format(
                            k, l))
                    
                    if len(mean_scalars) == 1:
                        z_means[0][k][l] = mean_scalars[0].value
                    else:
                        for scalar in mean_scalars:
                            z_means[scalar.step - 1][k][l] = scalar.value
                    
                    variance_scalars = multiplexer.Scalars(data_set_kind,
                        distribution + 
                            "/cluster_{}/variance/dimension_{}"
                                .format(k, l))
                    
                    if len(variance_scalars) == 1:
                        z_variances[0][k][l] = \
                            variance_scalars[0].value
                    else:
                        for scalar in variance_scalars:
                            z_variances[scalar.step - 1][k][l] = \
                                scalar.value
                
                for e in range(E):
                    z_covariance_matrices[e, k] = numpy.diag(z_variances[e, k])
            
            if data_set_kind == "evaluation":
                z_probabilities = z_probabilities[0]
                z_means = z_means[0]
                z_covariance_matrices = z_covariance_matrices[0]
            
            centroids_set[distribution] = {
                "probabilities": z_probabilities,
                "means": z_means,
                "covariance_matrices": z_covariance_matrices
            }
        
        centroids_sets[data_set_kind] = centroids_set
    
    if len(data_set_kinds) == 1:
        centroids_sets = centroids_sets[data_set_kinds[0]]
    
    return centroids_sets

def loadKLDivergences(model, early_stopping = False, best_model = False):
    
    # Setup
    
    ## Log directory
    if early_stopping:
        log_directory = model.early_stopping_log_directory
    elif best_model:
        log_directory = model.best_model_log_directory
    else:
        log_directory = model.log_directory
    
    ## TensorBoard class with summaries
    multiplexer = event_multiplexer.EventMultiplexer().AddRunsFromDirectory(
        log_directory)
    multiplexer.Reload()
    
    N_epochs = len(multiplexer.Scalars("training",
        "kl_divergence_neurons/0"))
    
    if "mixture" in model.latent_distribution_name:
        latent_size = 1
    else:
        latent_size = model.latent_size

    KL_neurons = numpy.empty([N_epochs, latent_size])
    
    # Loading
    
    for i in range(latent_size):
        scalars = multiplexer.Scalars("training",
            "kl_divergence_neurons/{}".format(i))
        
        for scalar in scalars:
            KL_neurons[scalar.step - 1][i] = scalar.value
    
    return KL_neurons

def betterModelExists(model):
    E_current = loadNumberOfEpochsTrained(model, best_model = False)
    E_best = loadNumberOfEpochsTrained(model, best_model = True)
    return E_best < E_current

def modelStoppedEarly(model):
    stopped_early, _ = model.earlyStoppingStatus()
    return stopped_early

# IO

def copyFile(URL, path):
    shutil.copyfile(URL, path)

def removeEmptyDirectories(source_directory):
    for directory_path, _, _ in os.walk(source_directory, topdown = False):
        if directory_path == source_directory:
            break
        try:
            os.rmdir(directory_path)
        except OSError as os_error:
            pass
            # print(os_error)

def downloadFile(URL, path):
    urllib.request.urlretrieve(URL, path, download_report_hook)

def download_report_hook(block_num, block_size, total_size):
    bytes_read = block_num * block_size
    if total_size > 0:
        percent = bytes_read / total_size * 100
        sys.stderr.write("\r{:3.0f}%.".format(percent))
        if bytes_read >= total_size:
            sys.stderr.write("\n")
    else:
        sys.stderr.write("\r{:d} bytes.".format(bytes_read))

# Shell output

RESETFORMAT = "\033[0m"
BOLD = "\033[1m"

def bold(string):
    """Convert to bold type."""
    return BOLD + string + RESETFORMAT

def underline(string, character="-"):
    """Convert string to header marks"""
    return character * len(string)

def heading(string, underline_symbol = "-", plain = False):
    string = "{}\n{}\n".format(string, underline(string, underline_symbol))
    if not plain:
        string = bold(string)
    return string

def title(string, plain = False):
    underline_symbol = "◼︎"
    return heading(string, underline_symbol, plain)

def subtitle(string, plain = False):
    underline_symbol = "≡"
    return heading(string, underline_symbol, plain)

def chapter(string, plain = False):
    underline_symbol = "="
    return heading(string, underline_symbol, plain)
