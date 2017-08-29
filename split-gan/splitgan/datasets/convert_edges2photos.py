""" Convert edges2photos dataset into TFRecords """

import argparse
import os
import sys

import tensorflow as tf
from utils import pair_image_to_tfexample

_datasets = [
    'edges2shoes',
    'edges2handbags',
]


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB PNG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)

        image = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        images = tf.split(image, 2, axis=1)
        self._images = map(tf.image.encode_jpeg, images)

    def read_image_dims(self, sess, image_data):
        image = self.decode_png(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_png(self, sess, image_data):
        edge, photo = sess.run(
            self._images, feed_dict={self._decode_jpeg_data: image_data})

        return edge, photo


def _convert_dataset(image_dir, output_filename):
    print 'Converting the {} split.'.format(image_dir.split('/')[-1])
    if tf.gfile.Exists(output_filename):
        print 'Dataset file already exists. Exiting without re-creating them.'
        return

    filenames = tf.gfile.ListDirectory(image_dir)
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session() as sess:
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for filename in filenames:
                    # Read the filename:
                    image_data = tf.gfile.FastGFile(
                        os.path.join(image_dir, filename), 'r').read()

                    edge, photo = image_reader.decode_png(sess, image_data)
                    example = pair_image_to_tfexample(edge, photo, 'jpeg')
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def run(dataset_dir):
    if not tf.gfile.Exists(dataset_dir):
        raise IOError('Dataset directory does not exist')

    def _get_output_filename(dataset, split_name):
        return os.path.join(dataset_dir, '{}_{}.tfrecord'.format(dataset, split_name))

    for dataset in _datasets:
        train_dir = os.path.join(dataset_dir, dataset, 'train')
        val_dir = os.path.join(dataset_dir, dataset, 'val')

        train_output_file = _get_output_filename(dataset, 'train')
        val_output_file = _get_output_filename(dataset, 'valid')

        _convert_dataset(train_dir, train_output_file)
        _convert_dataset(val_dir, val_output_file)

    print '\nFinished converting the Edges2Photos dataset!'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ###############
    # Directories #
    ###############
    parser.add_argument('--dataset-dir',
                        required=True,
                        type=str,
                        help='The directory where the dataset files are stored.')

    parser.add_argument('--verbosity',
                        choices=[
                            'DEBUG',
                            'ERROR',
                            'FATAL',
                            'INFO',
                            'WARN'
                        ],
                        default='INFO',
                        help='Set logging verbosity')

    parse_args, unknown = parser.parse_known_args()

    # Set python level verbosity
    tf.logging.set_verbosity(parse_args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[parse_args.verbosity] / 10)
    del parse_args.verbosity

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    run(**parse_args.__dict__)
