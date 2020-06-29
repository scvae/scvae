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

__all__ = [
    "decompose",
    "DECOMPOSITION_METHOD_NAMES",
    "DECOMPOSITION_METHOD_LABEL"
]

from scvae.analyses.decomposition.decomposition import (
    decompose,
    DECOMPOSITION_METHOD_NAMES, DECOMPOSITION_METHOD_LABEL
)
