""" A factory pattern class which returns model_fn """

import discogan
import splitgan
import cyclegan

models_map = {
    'discogan': discogan.model_fn,
    'splitgan': splitgan.model_fn,
    'cyclegan': cyclegan.model_fn,
}


def get_model(model_name):
    if model_name not in models_map:
        raise ValueError('Name of model unknown %s' % model_name)
    return models_map[model_name]
