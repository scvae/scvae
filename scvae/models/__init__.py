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

__all__ = [
    "VariationalAutoencoder",
    "GaussianMixtureVariationalAutoencoder"
]

import importlib
import os

import tensorflow

from scvae.utilities import suppress_stdout
with suppress_stdout():
    importlib.import_module("tensorflow.contrib.layers")

from scvae.models.variational_autoencoder import (
    VariationalAutoencoder)  # noqa: E402
from scvae.models.gaussian_mixture_variational_autoencoder import (
    GaussianMixtureVariationalAutoencoder)  # noqa: E402

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
