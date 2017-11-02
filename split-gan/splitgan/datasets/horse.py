""" Horse dataset """

import os

import tensorflow as tf

from utils import get_parse_fn

_FILE_PATTERN = 'horse2zebra_%sA.tfrecord'


def _preprocess(images):
    images = tf.image.convert_image_dtype(images, tf.float32)
    images -= 0.5
    images *= 2

    return images


def dataset_fn(split_name, dataset_dir, file_pattern=None):
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    filename = os.path.join(dataset_dir, file_pattern % split_name)

    if not tf.gfile.Exists(filename) and split_name == 'valid':
        filename = os.path.join(dataset_dir, file_pattern % 'test')

    dataset = tf.contrib.data.TFRecordDataset(filename)
    dataset = dataset.map(get_parse_fn([256, 256, 3],
                                       3,
                                       preprocess=_preprocess,
                                       labeled=False))

    return dataset
