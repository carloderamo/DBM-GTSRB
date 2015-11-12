# pylearn2 tutorial example: make_dataset.py by Ian Goodfellow
# See README before reading this file
#
#
# This script creates a preprocessed version of a dataset using pylearn2.
# It's not necessary to save preprocessed versions of your dataset to
# disk but this is an instructive example, because later we can show
# how to load your custom dataset in a yaml file.
#
# This is also a common use case because often you will want to preprocess
# your data once and then train several models on the preprocessed data.

PATCH_SHAPE = (32, 32)
PATCH_STRIDE = (0, 0)
NVIS =  PATCH_SHAPE[0] * PATCH_SHAPE[1] * 3

SHUFFLE = 0

import os.path
import numpy
from copy import deepcopy

# We'll need the serial module to save the dataset
from pylearn2.utils import serial

# Our raw dataset will be the CIFAR10 image dataset
from pylearn2.datasets import cifar10

# We'll need the preprocessing module to preprocess the dataset
from pylearn2.datasets import preprocessing

# Our raw training set is 32x32 color images
train = cifar10.CIFAR10(which_set="train")
test = cifar10.CIFAR10(which_set="test")

train_save_path = os.path.join('datasets', 'train.pkl')
test_save_path = os.path.join('datasets', 'test.pkl')
serial.save(train_save_path, train)
serial.save(test_save_path, test)
# We'd like to do several operations on them, so we'll set up a pipeline to
# do so.
pipeline = preprocessing.Pipeline()

# First we want to pull out small patches of the images, since it's easier
# to train an RBM on these
pipeline.items.append(
    preprocessing.ExtractGridPatches(patch_shape=PATCH_SHAPE, patch_stride=PATCH_STRIDE)
)

# Next we contrast normalize the patches. The default arguments use the
# same "regularization" parameters as those used in Adam Coates, Honglak
# Lee, and Andrew Ng's paper "An Analysis of Single-Layer Networks in
# Unsupervised Feature Learning"
pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))

# Finally we whiten the data using ZCA. Again, the default parameters to
# ZCA are set to the same values as those used in the previously mentioned
# paper.
pipeline.items.append(preprocessing.ZCA())

# Here we apply the preprocessing pipeline to the dataset. The can_fit
# argument indicates that data-driven preprocessing steps (such as the ZCA
# step in this example) are allowed to fit themselves to this dataset.
# Later we might want to run the same pipeline on the test set with the
# can_fit flag set to False, in order to make sure that the same whitening
# matrix was used on both datasets.
train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
test.apply_preprocessor(preprocessor=pipeline, can_fit=False)

# Shuffle after preprocessing. This is done to shuffle also the patches
if SHUFFLE: 
    for dataset in [train, test]:
        indices = numpy.arange(dataset.X.shape[0])
        rng = numpy.random.RandomState()   # if given an int argument will give reproducible results
        rng.shuffle(indices)
        # shuffle both the arrays consistently
        i = 0
        temp_X = deepcopy(dataset.X)
        temp_y = deepcopy(dataset.y)
        for idx in indices:
            dataset.X[i] = temp_X[idx]
            dataset.y[i] = temp_y[idx]
            i += 1

train_path = os.path.join('datasets', 'preprocessed_train.pkl')
test_path = os.path.join('datasets', 'preprocessed_test.pkl')
serial.save(train_path, train)
serial.save(test_path, test)
