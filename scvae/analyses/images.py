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

import os

import numpy
import PIL
import scipy.sparse

from scvae.analyses.figures import saving

IMAGE_EXTENSION = ".png"
DEFAULT_NUMBER_OF_RANDOM_EXAMPLES_FOR_COMBINED_IMAGES = 100


def combine_images_from_data_set(data_set, indices=None,
                                 number_of_random_examples=None, name=None):

    image_name = saving.build_figure_name("random_image_examples", name)
    random_state = numpy.random.RandomState(13)

    if indices is not None:
        n_examples = len(indices)
        if number_of_random_examples is not None:
            n_examples = min(n_examples, number_of_random_examples)
            indices = random_state.permutation(indices)[:n_examples]
    else:
        if number_of_random_examples is not None:
            n_examples = number_of_random_examples
        else:
            n_examples = DEFAULT_NUMBER_OF_RANDOM_EXAMPLES_FOR_COMBINED_IMAGES
        indices = random_state.permutation(
            data_set.number_of_examples)[:n_examples]

    if n_examples == 1:
        image_name = saving.build_figure_name("image_example", name)
    else:
        image_name = saving.build_figure_name("image_examples", name)

    width, height = data_set.feature_dimensions

    examples = data_set.values[indices]
    if scipy.sparse.issparse(examples):
        examples = examples.A
    examples = examples.reshape(n_examples, width, height)

    column = int(numpy.ceil(numpy.sqrt(n_examples)))
    row = int(numpy.ceil(n_examples / column))

    image = numpy.zeros((row * width, column * height))

    for m in range(n_examples):
        c = int(m % column)
        r = int(numpy.floor(m / column))
        rows = slice(r*width, (r+1)*width)
        columns = slice(c*height, (c+1)*height)
        image[rows, columns] = examples[m]

    return image, image_name


def save_image(image, name, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    minimum = image.min()
    maximum = image.max()
    if 0 < minimum and minimum < 1 and 0 < maximum and maximum < 1:
        rescaled_image = 255 * image
    else:
        rescaled_image = (255 / (maximum - minimum) * (image - minimum))

    image = PIL.Image.fromarray(rescaled_image.astype(numpy.uint8))

    name += IMAGE_EXTENSION
    image_path = os.path.join(directory, name)
    image.save(image_path)
