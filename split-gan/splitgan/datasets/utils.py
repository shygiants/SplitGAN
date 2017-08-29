""" Utils for datasets """

import tensorflow as tf


def get_parse_fn(image_shape, channels, preprocess=None, paired=False, labeled=True):
    def parse_fn(record):
        keys_to_features = {
            'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        }

        if labeled:
            keys_to_features.update({
                'image/class/label': tf.FixedLenFeature(
                    [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64))
            })
        if not paired:
            keys_to_features.update({
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value='')
            })
        else:
            keys_to_features.update({
                'image/encoded/1': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/encoded/2': tf.FixedLenFeature((), tf.string, default_value='')
            })

        parsed = tf.parse_single_example(record, keys_to_features)

        image_format = parsed['image/format']
        if not paired:
            image_encoded = [parsed['image/encoded']]
        else:
            image_encoded = [parsed['image/encoded/1'],
                             parsed['image/encoded/2']]

        def decode(image):
            image = tf.cond(tf.equal(image_format, 'raw'),
                            true_fn=lambda: tf.decode_raw(image, tf.uint8),
                            false_fn=lambda: tf.image.decode_image(image, channels=channels))
            image = tf.reshape(image, image_shape)

            if preprocess is not None:
                image = preprocess(image)

            return image

        image = map(decode, image_encoded)

        label = tf.cast(parsed['image/class/label'], tf.int32) if labeled else -1

        if not paired:
            return {'image': image[0]}, label
        else:
            return {'image/1': image[0], 'image/2': image[1]}, label

    return parse_fn


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
        values: A string.

    Returns:
        A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """Returns a TF-Feature of floats.

    Args:
        values: A scalar of list of values.

    Returns:
        A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
    }))


def pair_image_to_tfexample(image_1, image_2, image_format):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded/1': bytes_feature(image_1),
        'image/encoded/2': bytes_feature(image_2),
        'image/format': bytes_feature(image_format),
    }))
