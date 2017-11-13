""" A factory pattern class which returns dataset """

import mnist
import mnist_m
import edges2shoes
import fashion_synth
import horse
import horse_gray
import zebra

datasets_map = {
    'mnist': mnist,
    'mnist_m': mnist_m,
    'edges2shoes': edges2shoes,
    'fashion_synth': fashion_synth,
    'horse': horse,
    'horse_gray': horse_gray,
    'zebra': zebra,
}


def get_dataset(dataset_name, split_name, dataset_dir, file_pattern=None):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)

    return datasets_map[dataset_name].dataset_fn(split_name, dataset_dir, file_pattern=file_pattern)


def get_image_size(dataset_name):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)

    return datasets_map[dataset_name].image_size
