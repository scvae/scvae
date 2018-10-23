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

import os
import re
import sys
import shutil
import time
from collections import namedtuple
from functools import reduce
from math import floor
from operator import mul

import urllib.request

import numpy
import tensorflow

# Math

def prod(iterable):
    return reduce(mul, iterable, 1)

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

def properString(original_string, translation, normalise = True):
    
    if normalise:
        transformed_string = normaliseString(original_string)
    else:
        transformed_string = original_string
    
    for proper_string, related_strings in translation.items():
        if transformed_string in related_strings:
            return proper_string
    
    return original_string

def capitaliseString(original_string):
    string_parts = re.split(
        pattern=r"(\s)",
        string=original_string,
        maxsplit=1
    )
    if len(string_parts) == 3:
        first_word, split_character, rest_of_original_string = string_parts
        if re.match(pattern=r"[A-Z]", string=first_word):
            capitalised_first_word = first_word
        else:
            capitalised_first_word = first_word.capitalize()
        capitalised_string = capitalised_first_word + split_character \
            + rest_of_original_string
    else:
        if re.match(pattern=r"[A-Z]", string=original_string):
            capitalised_string = original_string
        else:
            capitalised_string = original_string.capitalize()
    return capitalised_string

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

# Functions for models
# TODO Move auxiliary model functions to `models/auxiliary.py`
# Note that doing this with the current setup causes an import loop.

def loadNumberOfEpochsTrained(model, run_id = None, early_stopping = False,
        best_model = False):
    
    # Setup
    
    ## Data set kind
    data_set_kind = "training"
    
    ## Loss depending on model type
    if "VAE" in model.type:
        loss = "lower_bound"
    else:
        loss = "log_likelihood"
    
    loss_prefix = "losses/"
    loss = loss_prefix + loss
    
    ## Log directory
    log_directory = model.logDirectory(
        run_id = run_id,
        early_stopping = early_stopping,
        best_model = best_model
    )
    
    # Loading
    
    # Losses for every epochs
    scalar_sets = summary_reader(log_directory, data_set_kind, loss)
    
    if scalar_sets and data_set_kind in scalar_sets:
        data_set_scalars = scalar_sets[data_set_kind]
    else:
        data_set_scalars = None
    
    if data_set_scalars and loss in data_set_scalars:
        scalars = data_set_scalars[loss]
    else:
        scalars = None
    
    if scalars:
        
        # First estimate of number of epochs
        E_1 = len(scalars)
        
        # Second estimate of number of epochs
        E_2 = max([scalar.step for scalar in scalars])
        
        assert E_1 == E_2
        
        E = E_1
    
    else:
        E = None
    
    return E

def loadLearningCurves(model, data_set_kinds = "all", run_id = None,
    early_stopping = False, best_model = False, log_directory = None):
    
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
            run_id = run_id,
            early_stopping = early_stopping,
            best_model = best_model
        )
    
    ## Losses depending on model type
    if model.type == "GMVAE":
        losses = [
            "lower_bound",
            "reconstruction_error",
            "kl_divergence_z",
            "kl_divergence_y"
    ]
    elif "VAE" == model.type:
        losses = ["lower_bound", "reconstruction_error", "kl_divergence"]
    else:
        losses = ["log_likelihood"]
    
    loss_prefix = "losses/"
    loss_searches = list(map(lambda s: loss_prefix + s, losses))
    
    # Loading
    
    scalar_sets = summary_reader(log_directory, data_set_kinds, loss_searches)
    
    # Organising
    
    for data_set_kind in data_set_kinds:
        
        learning_curve_set = {}
        
        for loss in losses:
            
            loss_tag = loss_prefix + loss
            
            if scalar_sets and data_set_kind in scalar_sets:
                data_set_scalars = scalar_sets[data_set_kind]
            else:
                data_set_scalars = None
            
            if data_set_scalars and loss_tag in data_set_scalars:
                scalars = data_set_scalars[loss_tag]
            else:
                scalars = None
            
            if scalars:
                
                learning_curve = numpy.empty(len(scalars))
                
                if len(scalars) == 1:
                    learning_curve[0] = scalars[0].value
                else:
                    for scalar in scalars:
                        learning_curve[scalar.step - 1] = scalar.value
            
            else:
                learning_curve = None
            
            learning_curve_set[loss] = learning_curve
        
        learning_curve_sets[data_set_kind] = learning_curve_set
    
    if len(data_set_kinds) == 1:
        learning_curve_sets = learning_curve_sets[data_set_kinds[0]]
    
    return learning_curve_sets

def loadAccuracies(model, data_set_kinds = "all", superset = False,
    run_id = None, early_stopping = False, best_model = False):
    
    # Setup
    
    accuracies = {}
    
    ## Data set kinds
    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "evaluation"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]
    
    ## Log directory
    log_directory = model.logDirectory(
        run_id = run_id,
        early_stopping = early_stopping,
        best_model = best_model
    )
    
    ## Tag
    
    accuracy_tag = "accuracy"
    
    if superset:
        accuracy_tag = "superset_" + accuracy_tag
    
    # Loading
    
    scalar_sets = summary_reader(
        log_directory=log_directory,
        data_set_kinds=data_set_kinds,
        tag_searches=[accuracy_tag]
    )
    
    # Organising
    
    empty_scalar_sets = 0
    
    for data_set_kind in data_set_kinds:
        
        if scalar_sets and data_set_kind in scalar_sets:
            data_set_scalars = scalar_sets[data_set_kind]
        else:
            data_set_scalars = None
        
        if data_set_scalars and accuracy_tag in data_set_scalars:
            scalars = data_set_scalars[accuracy_tag]
        else:
            scalars = None
        
        if scalars:
            
            data_set_accuracies = numpy.empty(len(scalars))
            
            if len(scalars) == 1:
                data_set_accuracies[0] = scalars[0].value
            else:
                for scalar in scalars:
                    data_set_accuracies[scalar.step - 1] = scalar.value
            
            accuracies[data_set_kind] = data_set_accuracies
        
        else:
            accuracies[data_set_kind] = None
            empty_scalar_sets += 1
    
    if len(data_set_kinds) == 1:
        accuracies = accuracies[data_set_kinds[0]]
    
    if empty_scalar_sets == len(data_set_kinds):
        accuracies = None
    
    return accuracies

def loadCentroids(model, data_set_kinds = "all", run_id = None,
    early_stopping = False, best_model = False):
    
    # Setup
    
    ## Data set kinds
    if data_set_kinds == "all":
        data_set_kinds = ["training", "validation", "evaluation"]
    elif not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]
    
    ## Exit if not VAE model
    if "VAE" not in model.type:
        return None
    
    ## Log directory
    log_directory = model.logDirectory(
        run_id = run_id,
        early_stopping = early_stopping,
        best_model = best_model
    )
    
    ## Tag search
    
    centroid_tag = "cluster"
    
    # Loading
    
    scalar_sets = summary_reader(
        log_directory=log_directory,
        data_set_kinds=data_set_kinds,
        tag_searches=[centroid_tag]
    )
    
    # Organising
    
    centroids_sets = {}
    
    for data_set_kind in data_set_kinds:
        
        centroids_set = {}
        
        for distribution in ["prior", "posterior"]:
            
            cluster_tag = distribution + "/cluster_0/probability"
            
            if scalar_sets and data_set_kind in scalar_sets:
                data_set_scalars = scalar_sets[data_set_kind]
            else:
                data_set_scalars = None
            
            if data_set_scalars and cluster_tag in data_set_scalars:
                scalars = data_set_scalars[cluster_tag]
            else:
                scalars = None
            
            if not scalars:
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
                
                probability_scalars = data_set_scalars\
                    [distribution + "/cluster_{}/probability".format(k)]
                
                if len(probability_scalars) == 1:
                    z_probabilities[0][k] = probability_scalars[0].value
                else:
                    for scalar in probability_scalars:
                        z_probabilities[scalar.step - 1][k] = scalar.value
                
                for l in range(L):
                    
                    mean_scalars = data_set_scalars\
                        [distribution + "/cluster_{}/mean/dimension_{}"
                            .format(k, l)]
                    
                    if len(mean_scalars) == 1:
                        z_means[0][k][l] = mean_scalars[0].value
                    else:
                        for scalar in mean_scalars:
                            z_means[scalar.step - 1][k][l] = scalar.value
                    
                    variance_scalars = data_set_scalars\
                        [distribution + "/cluster_{}/variance/dimension_{}"
                            .format(k, l)]
                    
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

def loadKLDivergences(model, run_id = None, early_stopping = False,
    best_model = False):
    
    # Setup
    
    ## Data set kind
    data_set_kind = "training"
    
    ## Log directory
    log_directory = model.logDirectory(
        run_id = run_id,
        early_stopping = early_stopping,
        best_model = best_model
    )
    
    ## Tag search
    kl_divergence_neurons_tag_prefix = "kl_divergence_neurons/"
    
    # Loading
    
    scalar_sets = summary_reader(
        log_directory=log_directory,
        data_set_kinds=data_set_kind,
        tag_searches=[kl_divergence_neurons_tag_prefix]
    )
    
    if scalar_sets and data_set_kind in scalar_sets:
        data_set_scalars = scalar_sets[data_set_kind]
    else:
        data_set_scalars = None
    
    # Additional setup
    
    kl_divergence_neuron_0_tag = kl_divergence_neurons_tag_prefix + "0"
    
    if data_set_scalars and kl_divergence_neuron_0_tag in data_set_scalars:
        scalars = data_set_scalars[kl_divergence_neuron_0_tag]
    else:
        scalars = None
    
    if scalars:
        
        N_epochs = len(scalars)
        
        if "mixture" in model.latent_distribution_name:
            latent_size = 1
        else:
            latent_size = model.latent_size
        
        KL_neurons = numpy.empty([N_epochs, latent_size])
        
        # Organising
        
        for i in range(latent_size):
            kl_divergence_neuron_i_tag = kl_divergence_neurons_tag_prefix \
                + str(i)
            
            if kl_divergence_neuron_i_tag in data_set_scalars:
                scalars = data_set_scalars[kl_divergence_neuron_i_tag]
            else:
                scalars = None
            
            if scalars:
                for scalar in scalars:
                    KL_neurons[scalar.step - 1, i] = scalar.value
            else:
                KL_neurons[:, i] = numpy.full(N_epochs, numpy.nan)
    
    else:
        KL_neurons = None
    
    return KL_neurons

ScalarEvent = namedtuple('ScalarEvent', ['wall_time', 'step', 'value'])

def summary_reader(log_directory, data_set_kinds, tag_searches):

    if not isinstance(data_set_kinds, list):
        data_set_kinds = [data_set_kinds]

    if not isinstance(tag_searches, list):
        tag_searches = [tag_searches]

    if os.path.exists(log_directory):

        scalars = {}

        for data_set_kind in data_set_kinds:
            data_set_log_directory = os.path.join(log_directory, data_set_kind)

            if os.path.exists(data_set_log_directory):

                data_set_scalars = {}

                for filename in sorted(os.listdir(data_set_log_directory)):
                    if filename.startswith("event"):
                        events_path = os.path.join(
                            data_set_log_directory,
                            filename
                        )
                        events = tensorflow.train.summary_iterator(
                            events_path)
                        for event in events:
                            for value in event.summary.value:
                                for tag_search in tag_searches:
                                    if tag_search in value.tag:
                                        tag = value.tag
                                        scalar = ScalarEvent(
                                            wall_time=event.wall_time,
                                            step=event.step,
                                            value=value.simple_value
                                        )
                                        if tag not in data_set_scalars:
                                            data_set_scalars[tag] = []
                                        data_set_scalars[tag].append(scalar)
            else:
                data_set_scalars = None

            scalars[data_set_kind] = data_set_scalars
    
    else:
        scalars = None

    return scalars

def betterModelExists(model, run_id = None):
    E_current = loadNumberOfEpochsTrained(model, run_id = run_id,
        best_model = False)
    E_best = loadNumberOfEpochsTrained(model, run_id = run_id,
        best_model = True)
    if E_best:
        better_model_exists = E_best < E_current
    else:
        better_model_exists = False
    return better_model_exists

def modelStoppedEarly(model, run_id = None):
    stopped_early, _ = model.earlyStoppingStatus(run_id = run_id)
    return stopped_early

def checkRunID(run_id):
    if run_id is not None:
        run_id = str(run_id)
        if not re.fullmatch(r"[\w]+", run_id):
            raise ValueError(
                "`run_id` can only contain letters, numbers, and "
                "underscores ('_')."
            )
    else:
        raise TypeError("The run ID has not been set.")
    return run_id

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
    underline_symbol = "═"
    return heading(string, underline_symbol, plain)

def subtitle(string, plain = False):
    underline_symbol = "─"
    return heading(string, underline_symbol, plain)

def subheading(string, plain = False):
    underline_symbol = "╌"
    return heading(string, underline_symbol, plain)
