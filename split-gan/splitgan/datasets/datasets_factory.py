""" A factory pattern class which returns dataset """

import mnist
import mnist_m
import edges2shoes
import horse
import zebra

datasets_map = {
    'mnist': mnist,
    'mnist_m': mnist_m,
    'edges2shoes': edges2shoes,
    'horse': horse,
    'zebra': zebra,
}


def get_dataset(dataset_name, split_name, dataset_dir, file_pattern=None):
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)

    return datasets_map[dataset_name].dataset_fn(split_name, dataset_dir, file_pattern=file_pattern)
