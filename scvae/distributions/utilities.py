# ======================================================================== #
#
# Copyright (c) 2017 - 2020 scVAE authors
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

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from scvae.distributions.exponentially_modified_normal import (
    ExponentiallyModifiedNormal)
from scvae.distributions.lomax import Lomax
from scvae.distributions.multivariate_normal import MultivariateNormalTriL
from scvae.distributions.zero_inflated import ZeroInflated
from scvae.utilities import normalise_string

DISTRIBUTIONS = {
    "gaussian": {
        "parameters": {
            "mu": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity,
                "initial value": tf.zeros
            },
            "log_sigma": {
                "support": [-3, 3],
                "activation function": tf.identity,
                "initial value": tf.zeros
            }
        },
        "class": lambda theta: tfp.distributions.Normal(
            loc=theta["mu"],
            scale=tf.exp(theta["log_sigma"])
        )
    },

    "softplus gaussian": {
        "parameters": {
            "mean": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity,
                "initial value": tf.zeros
            },
            "variance": {
                "support": [0, np.inf],
                "activation function": tf.nn.softplus,
                "initial value": tf.ones
            }
        },
        "class": lambda theta: tfp.distributions.Normal(
            loc=theta["mean"],
            scale=tf.sqrt(theta["variance"])
        )
    },

    "multivariate gaussian": {
        "parameters": {
            "locations": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity,
                "initial value": tf.zeros
            },
            "scales": {
                "support": [0, np.inf],
                "activation function": tf.nn.softplus,
                "initial value": tf.ones,
                "size function": lambda m: int(m * (m + 1) / 2)
            }
        },
        "class": lambda theta: MultivariateNormalTriL(
            loc=theta["locations"],
            scale_tril=tfp.distributions.fill_triangular(theta["scales"])
        )
    },

    "gaussian mixture": {
        "parameters": {
            "logits": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity,
                "initial value": tf.ones
            },
            "mus": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity,
                "initial value": lambda x: tf.random_normal(x, stddev=1)
            },
            "log_sigmas": {
                "support": [-3, 3],
                "activation function": tf.identity,
                "initial value": tf.zeros
            }
        },
        "class": lambda theta: tfp.distributions.Mixture(
            cat=tfp.distributions.Categorical(logits=theta["logits"]),
            components=[
                tfp.distributions.MultivariateNormalDiag(
                    loc=m,
                    scale_diag=tf.exp(s)
                )
                for m, s in zip(theta["mus"], theta["log_sigmas"])
            ]
        )
    },

    "log-normal": {
        "parameters": {
            "mean": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity
            },
            "variance": {
                "support": [0, np.inf],
                "activation function": tf.nn.softplus
            }
        },
        "class": lambda theta: tfp.distributions.LogNormal(
            loc=theta["mean"],
            scale=tf.sqrt(theta["variance"])
        )
    },

    "exponentially_modified_gaussian": {
        "parameters": {
            "location": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity
            },
            "scale": {
                "support": [0, np.inf],
                "activation function": tf.nn.softplus
            },
            "rate": {
                "support": [0, np.inf],
                "activation function": tf.nn.softplus
            }
        },
        "class": lambda theta: ExponentiallyModifiedNormal(
            loc=theta["location"],
            scale=theta["scale"],
            rate=theta["rate"],
            validate_args=True,
            allow_nan_stats=False
        )
    },

    "gamma": {
        "parameters": {
            "concentration": {
                "support": [0, np.inf],
                "activation function": tf.nn.softplus
            },
            "rate": {
                "support": [0, np.inf],
                "activation function": tf.nn.softplus
            }
        },
        "class": lambda theta: tfp.distributions.Gamma(
            concentration=theta["concentration"],
            rate=theta["rate"]
        )
    },

    "categorical": {
        "parameters": {
            "logits": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity
            }
        },
        "class": lambda theta:
            tfp.distributions.Categorical(logits=theta["logits"]),
    },

    "bernoulli": {
        "parameters": {
            "logits": {
                "support": [-np.inf, np.inf],
                "activation function": tf.identity
            }
        },
        "class": lambda theta: tfp.distributions.Bernoulli(
            logits=theta["logits"]
        )
    },

    "poisson": {
        "parameters": {
            "log_lambda": {
                "support": [-10, 10],
                "activation function": tf.identity
            }
        },
        "class": lambda theta: tfp.distributions.Poisson(
            rate=tf.exp(theta["log_lambda"])
        )
    },

    "constrained poisson": {
        "parameters": {
            "lambda": {
                "support": [0, 1],
                "activation function": tf.nn.softmax
            }
        },
        "class": lambda theta, N: tfp.distributions.Poisson(
            rate=theta["lambda"] * N
        )
    },

    "lomax": {
        "parameters": {
            "log_concentration": {
                "support": [-10, 10],
                "activation function": tf.identity
            },
            "log_scale": {
                "support": [-10, 10],
                "activation function": tf.identity
            }
        },
        "class": lambda theta: Lomax(
            concentration=tf.exp(theta["log_concentration"]),
            scale=tf.exp(theta["log_scale"])
        )
    },

    "zero-inflated poisson": {
        "parameters": {
            "pi": {
                "support": [0, 1],
                "activation function": tf.sigmoid
            },
            "log_lambda": {
                "support": [-10, 10],
                "activation function": tf.identity
            }
        },
        "class": lambda theta: ZeroInflated(
            dist=tfp.distributions.Poisson(
                rate=tf.exp(theta["log_lambda"])
            ),
            pi=theta["pi"]
        )
    },

    "negative binomial": {
        "parameters": {
            "p": {
                "support": [0, 1],
                "activation function": tf.sigmoid
            },
            "log_r": {
                "support": [-10, 10],
                "activation function": tf.identity
            }
        },
        "class": lambda theta: tfp.distributions.NegativeBinomial(
            total_count=tf.exp(theta["log_r"]),
            probs=theta["p"]
        )
    },

    "zero-inflated negative binomial": {
        "parameters": {
            "pi": {
                "support": [0, 1],
                "activation function": tf.sigmoid
            },
            "p": {
                "support": [0, 1],
                "activation function": tf.sigmoid
            },
            "log_r": {
                "support": [-10, 10],
                "activation function": tf.identity
            }
        },
        "class": lambda theta: ZeroInflated(
            dist=tfp.distributions.NegativeBinomial(
                total_count=tf.exp(theta["log_r"]),
                probs=theta["p"]
            ),
            pi=theta["pi"]
        )
    }
}
DISTRIBUTIONS["modified gaussian"] = DISTRIBUTIONS["softplus gaussian"]

LATENT_DISTRIBUTIONS = {
    "gaussian": {
        "prior": {
            "name": "gaussian",
            "parameters": {
                "mu": 0.0,
                "log_sigma": 0.0
            }
        },
        "posterior": {
            "name": "gaussian",
            "parameters": {}
        }
    },
    "unit-variance gaussian": {
        "prior": {
            "name": "gaussian",
            "parameters": {
                "mu": 0.0,
                "log_sigma": 0.0
            }
        },
        "posterior": {
            "name": "gaussian",
            "parameters": {
                "log_sigma": 0.0
            }
        }
    }
}

GAUSSIAN_MIXTURE_DISTRIBUTIONS = {
    "gaussian mixture": {
        "z prior": "softplus gaussian",
        "z posterior": "softplus gaussian"
    },
    "full-covariance gaussian mixture": {
        "z prior": "multivariate gaussian",
        "z posterior": "multivariate gaussian"
    },
    "legacy gaussian mixture": {
        "z prior": "modified gaussian",
        "z posterior": "modified gaussian"
    }
}


def parse_distribution(distribution, model_type=None):

    distribution = normalise_string(distribution)

    if model_type is None:
        kind = "reconstruction"
        distributions = DISTRIBUTIONS

    elif isinstance(model_type, str):
        kind = "latent"
        if model_type == "VAE":
            distributions = LATENT_DISTRIBUTIONS
        elif model_type == "GMVAE":
            distributions = GAUSSIAN_MIXTURE_DISTRIBUTIONS
        else:
            raise ValueError("Model type not found.")

    else:
        raise TypeError("`model_type` should be a string.")

    distribution_names = list(distributions.keys())
    parsed_distribution_name = None

    for distribution_name in distribution_names:
        if normalise_string(distribution_name) == distribution:
            parsed_distribution_name = distribution_name

    if parsed_distribution_name is None:
        raise ValueError(
            "{} distribution `{}` not supported{}.".format(
                kind.capitalize(), distribution,
                " for {}".format(model_type) if model_type else ""))

    return parsed_distribution_name
