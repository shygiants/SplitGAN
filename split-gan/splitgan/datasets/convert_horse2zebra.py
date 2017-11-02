""" Convert horse2zebra dataset into TFRecords """

import argparse
import os
import sys

import tensorflow as tf
from utils import image_to_tfexample

_datasets = [
    'horse2zebra',
]


def _convert_dataset(image_dir, output_filename):
    print 'Converting the {} split.'.format(image_dir.split('/')[-1])
    if tf.gfile.Exists(output_filename):
        print 'Dataset file already exists. Exiting without re-creating them.'
        return

    filenames = tf.gfile.ListDirectory(image_dir)
    with tf.Graph().as_default():
        with tf.Session():
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for filename in filenames:
                    # Read the filename:
                    image_data = tf.gfile.FastGFile(
                        os.path.join(image_dir, filename), 'r').read()

                    example = image_to_tfexample(image_data, 'jpeg')
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def run(dataset_dir):
    if not tf.gfile.Exists(dataset_dir):
        raise IOError('Dataset directory does not exist')

    def _get_output_filename(dataset, split_name):
        return os.path.join(dataset_dir, '{}_{}.tfrecord'.format(dataset, split_name))

    for dataset in _datasets:
        train_a_dir = os.path.join(dataset_dir, dataset, 'trainA')
        train_b_dir = os.path.join(dataset_dir, dataset, 'trainB')
        test_a_dir = os.path.join(dataset_dir, dataset, 'testA')
        test_b_dir = os.path.join(dataset_dir, dataset, 'testB')

        train_a_output_file = _get_output_filename(dataset, 'trainA')
        train_b_output_file = _get_output_filename(dataset, 'trainB')
        test_a_output_file = _get_output_filename(dataset, 'testA')
        test_b_output_file = _get_output_filename(dataset, 'testB')

        _convert_dataset(train_a_dir, train_a_output_file)
        _convert_dataset(train_b_dir, train_b_output_file)
        _convert_dataset(test_a_dir, test_a_output_file)
        _convert_dataset(test_b_dir, test_b_output_file)

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
