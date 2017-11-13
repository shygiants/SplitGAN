""" Convert Fashion Synthesis dataset into TFRecords """

import argparse
import os
import sys
import h5py

import tensorflow as tf
import numpy as np
from scipy import io
from utils import pair_image_to_tfexample


class ImageEncoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that encodes RGB JPEG data.
        self._encode_jpeg_data = tf.placeholder(dtype=tf.uint8)

        image = tf.image.encode_jpeg(self._encode_jpeg_data, quality=100)
        self._image = image

    def encode_jpeg(self, sess, image_data):
        return sess.run(self._image, feed_dict={self._encode_jpeg_data: image_data})


def _convert_dataset(dataset_file, indices, output_filename, session_config):
    print 'Converting the {} split.'.format(output_filename.split('/')[-1])
    if tf.gfile.Exists(output_filename):
        print 'Dataset file already exists. Exiting without re-creating them.'
        return

    keys = dataset_file.keys()
    labels = dataset_file[keys[0]]
    labels = np.transpose(labels, axes=[0, 3, 2, 1])
    labels = labels[indices]
    images = dataset_file[keys[1]]
    images = np.transpose(images, axes=[0, 3, 2, 1])
    images = images[indices]
    mean_images = dataset_file[keys[2]]
    mean_images = np.transpose(mean_images, axes=[2, 1, 0])

    with tf.Graph().as_default():
        image_encoder = ImageEncoder()
        with tf.Session(config=session_config) as sess:
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for image, label in zip(images, labels):
                    image += mean_images

                    image = (255.0 / image.max() * (image - image.min())).astype(np.uint8)
                    label = (255.0 / label.max() * (label - label.min())).astype(np.uint8)
                    label = np.tile(label, (1, 1, 3))

                    image = image_encoder.encode_jpeg(sess, image)
                    label = image_encoder.encode_jpeg(sess, label)

                    example = pair_image_to_tfexample(label, image, 'jpeg')
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def run(dataset_dir, gpu):
    if not tf.gfile.Exists(dataset_dir):
        raise IOError('Dataset directory does not exist')

    session_config = None
    if gpu is not None:
        session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=gpu
            )
        )

    def _get_output_filename(split_name):
        return os.path.join(dataset_dir, 'fashion_synth_{}.tfrecord'.format(split_name))

    dataset_file = h5py.File(os.path.join(dataset_dir, 'fashion-synthesis', 'Img', 'G2.h5'), 'r')

    ind_file = io.loadmat(os.path.join(dataset_dir, 'fashion-synthesis', 'Eval', 'ind.mat'))

    def _get_ind_var(key):
        return np.squeeze(ind_file[key]) - 1

    train_output_file = _get_output_filename('train')
    val_output_file = _get_output_filename('valid')

    _convert_dataset(dataset_file, _get_ind_var('train_ind'), train_output_file, session_config)
    _convert_dataset(dataset_file, _get_ind_var('test_ind'), val_output_file, session_config)

    print '\nFinished converting the Fashion Synthesis dataset!'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ###############
    # Directories #
    ###############
    parser.add_argument('--dataset-dir',
                        required=True,
                        type=str,
                        help='The directory where the dataset files are stored.')

    ##############
    # Run Config #
    ##############
    parser.add_argument('--gpu',
                        type=str,
                        help='GPU ids for training.',
                        default=None)
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
