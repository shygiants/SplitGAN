""" Fashion Synthesis dataset """

import os

import tensorflow as tf

from utils import get_parse_fn

_FILE_PATTERN = 'fashion_synth_%s.tfrecord'

image_size = 128


def _preprocess(images):
    images = tf.image.convert_image_dtype(images, tf.float32)
    images -= 0.5
    images *= 2

    return images


def dataset_fn(split_name, dataset_dir, file_pattern=None):
    if not file_pattern:
        file_pattern = _FILE_PATTERN
    filename = os.path.join(dataset_dir, file_pattern % split_name)

    dataset = tf.contrib.data.TFRecordDataset(filename)
    dataset = dataset.map(get_parse_fn([image_size, image_size, 3],
                                       3,
                                       preprocess=_preprocess,
                                       paired=True,
                                       labeled=False))

    return dataset
