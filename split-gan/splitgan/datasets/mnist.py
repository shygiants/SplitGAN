""" MNIST dataset """

import os

import tensorflow as tf

from utils import get_parse_fn

_FILE_PATTERN = 'mnist_%s.tfrecord'


def _preprocess(images):
    images = tf.image.convert_image_dtype(images, tf.float32)
    images -= 0.5
    images *= 2

    images = tf.image.grayscale_to_rgb(images)
    images = tf.image.resize_images(images, [32, 32])
    return images


def dataset_fn(split_name, dataset_dir, file_pattern=None):
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    filename = os.path.join(dataset_dir, file_pattern % split_name)

    dataset = tf.contrib.data.TFRecordDataset(filename)
    dataset = dataset.map(get_parse_fn([28, 28, 1], 1, preprocess=_preprocess))

    return dataset
