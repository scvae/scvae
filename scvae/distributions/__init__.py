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

import tensorflow as tf
from numpy import inf

from tensorflow_probability import distributions as tensorflow_distributions

from tensorflow.python.ops.nn import relu, softmax, softplus
from tensorflow import sigmoid, identity

from distributions.zero_inflated import ZeroInflated
from distributions.categorised import Categorised
from distributions.exponentially_modified_normal import (
    ExponentiallyModifiedNormal
)
from distributions.lomax import Lomax

distributions = {
    "gaussian": {
        "parameters": {
            "mu": {
                "support": [-inf, inf],
                "activation function": identity,
                "initial value": tf.zeros
            },
            "log_sigma": {
                "support": [-3, 3],
                "activation function": identity,
                "initial value": tf.zeros
            }
        },
        "class": lambda theta: tensorflow_distributions.Normal(
            loc = theta["mu"], 
            scale = tf.exp(theta["log_sigma"])
        )
    },

    "modified gaussian": {
        "parameters": {
            "mean": {
                "support": [-inf, inf],
                "activation function": identity,
                "initial value": tf.zeros
            },
            "variance": {
                "support": [0, inf],
                "activation function": softplus,
                "initial value": tf.ones
            }
        },
        "class": lambda theta: tensorflow_distributions.Normal(
            loc = theta["mean"], 
            scale = tf.sqrt(theta["variance"])
        )
    },

    "gaussian mixture": {
        "parameters": {
            "logits": {
                "support": [-inf, inf],
                "activation function": identity,
                "initial value": tf.ones
            },
            "mus": {
                "support": [-inf, inf],
                "activation function": identity,
                "initial value": lambda x: tf.random_normal(x, stddev = 1)
            },
            "log_sigmas": {
                "support": [-3, 3],
                "activation function": identity,
                "initial value": tf.zeros
            }
        },
        "class": lambda theta: tensorflow_distributions.Mixture(
            cat = tensorflow_distributions.Categorical(logits = theta["logits"]), 
            components = [tensorflow_distributions.MultivariateNormalDiag(
                loc = m, scale_diag = tf.exp(s)) for m, s in 
                zip(theta["mus"], theta["log_sigmas"])]
        )
    },

    "log-normal": {
        "parameters": {
            "mean": {
                "support": [-inf, inf],
                "activation function": identity
            },
            "variance": {
                "support": [0, inf],
                "activation function": softplus
            }
        },
        "class": lambda theta: tensorflow_distributions.LogNormal(
            loc = theta["mean"], 
            scale = tf.sqrt(theta["variance"])
        )
    },

    "exponentially_modified_gaussian": {
        "parameters": {
            "location": {
                "support": [-inf, inf],
                "activation function": identity
            },
            "scale": {
                "support": [0, inf],
                "activation function": softplus
            },
            "rate": {
                "support": [0, inf],
                "activation function": softplus
            }
        },
        "class": lambda theta: ExponentiallyModifiedNormal(
            loc = theta["location"], 
            scale = theta["scale"],
            rate = theta["rate"],
            validate_args=True,
            allow_nan_stats=False
        )
    },

    "gamma": {
        "parameters": {
            "concentration": {
                "support": [0, inf],
                "activation function": softplus
            },
            "rate": {
                "support": [0, inf],
                "activation function": softplus
            }
        },
        "class": lambda theta: tensorflow_distributions.Gamma(
            concentration = theta["concentration"], 
            rate = theta["rate"]
        )
    },

    "categorical": {
        "parameters": {
            "logits": {
                "support": [-inf, inf],
                "activation function": identity
            }
        },
        "class": lambda theta: 
            tensorflow_distributions.Categorical(logits = theta["logits"]), 
    },

    "bernoulli": {
        "parameters": {
            "logits": {
                "support": [-inf, inf],
                "activation function": identity
            }
        },
        "class": lambda theta: tensorflow_distributions.Bernoulli(
            logits = theta["logits"]
        )
    },
    
    "poisson": {
        "parameters": {
            "log_lambda": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: tensorflow_distributions.Poisson(
            rate = tf.exp(theta["log_lambda"])
        )
    },

    "constrained poisson": {
        "parameters": {
            "lambda": {
                "support": [0, 1],
                "activation function": softmax
            }
        },
        "class": lambda theta, N: tensorflow_distributions.Poisson(
            rate = theta["lambda"] * N
        )
    },

    "lomax": {
        "parameters": {
            "log_concentration": {
                "support": [-10, 10],
                "activation function": identity
            },
            "log_scale": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: Lomax(
            concentration = tf.exp(theta["log_concentration"]),
            scale = tf.exp(theta["log_scale"])
        )
    },

    "zero-inflated poisson": {
        "parameters": {
            "pi": {
                "support": [0, 1],
                "activation function": sigmoid
            },
            "log_lambda": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: ZeroInflated(
            tensorflow_distributions.Poisson(
                rate = tf.exp(theta["log_lambda"])
            ),
            pi = theta["pi"]
        )
    },
    
    "negative binomial": {
        "parameters": {
            "p": {
                "support": [0, 1],
                "activation function": sigmoid
            },
            "log_r": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: tensorflow_distributions.NegativeBinomial(
            total_count = tf.exp(theta["log_r"]),
            probs = theta["p"]
        )
    },
    
    "zero-inflated negative binomial": {
        "parameters": {
            "pi": {
                "support": [0, 1],
                "activation function": sigmoid
            },
            "p": {
                "support": [0, 1],
                "activation function": sigmoid
            },
            "log_r": {
                "support": [-10, 10],
                "activation function": identity
            }
        },
        "class": lambda theta: ZeroInflated(
            tensorflow_distributions.NegativeBinomial(
                total_count = tf.exp(theta["log_r"]),
                probs = theta["p"]
            ),
            pi = theta["pi"]
        )
    }
}

latent_distributions = {
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
    },
    "gaussian mixture": {
        "prior": {
            "name": "gaussian mixture",
            "parameters": {}
        },
        "posterior": {
            "name": "gaussian mixture", 
            "parameters": {}
        }
    },
    "half gaussian mixture": {
        "prior": {
            "name": "gaussian mixture",
            "parameters": {}
        },
        "posterior": {
            "name": "gaussian", 
            "parameters": {}
        }
    },
    "fixed gaussian mixture": {
        "prior": {
            "name": "gaussian mixture",
            "parameters": {}
        },
        "posterior": {
            "name": "gaussian", 
            "parameters": {}
        }
    }
}

model_inference_graph = {
    "explicit gaussian mixture": {
        "posteriors": {
            "q_z_given_x_y": {
                "name": "gaussian", 
                "parameters": {},
                "conditioning": ["encoder", "q_y_given_x"]
            },
            "q_y_given_x": {
                "name": "categorical",
                "parameters": {},
                "conditioning": ["encoder"]
            }
        },
        "priors": {
            "p_z_given_y": {
                "name": "gaussian",
                "parameters": {},
                "conditioning": ["decoder"]
            }
        }
    },
}
