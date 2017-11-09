""" Main task """

import argparse
import functools
import os

import tensorflow as tf
from datasets import datasets_factory
from models import models_factory


def run(job_dir,
        model_name,
        use_joint_discr,
        paired_dataset,
        domain_a,
        domain_b,
        dataset_dir,
        train_batch_size,
        eval_batch_size,
        pool_size,
        train_steps,
        alpha1,
        alpha2,
        beta1,
        beta2,
        lambda1,
        lambda2,
        weight_decay,
        num_layers,
        depth,
        num_blocks,
        split_rate,
        gpu,
        skip,
        eval_only,
        random_seed):

    def input_fn(dataset, batch_size):
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels

    # Define datasets
    print paired_dataset

    def get_dataset(split_name):
        if paired_dataset is not None:
            dataset = datasets_factory.get_dataset(paired_dataset, split_name, dataset_dir)
            dataset = dataset.map(
                lambda ft, lbl: ({'x_a': ft['image/2'], 'x_b': ft['image/1']}, lbl))
        else:
            dataset_a = datasets_factory.get_dataset(domain_a, split_name, dataset_dir)
            dataset_b = datasets_factory.get_dataset(domain_b, split_name, dataset_dir)

            dataset = tf.contrib.data.Dataset.zip((dataset_a, dataset_b))
            dataset = dataset.map(
                lambda a, b: ({'x_a': a[0]['image'], 'x_b': b[0]['image']}, {'label_a': a[1], 'label_b': b[1]}))

        return dataset

    dataset_train = get_dataset('train')
    dataset_valid = get_dataset('valid').take(eval_batch_size)

    # Hyperparameters
    params = {
        'pool_size': pool_size,
        'alpha1': alpha1,
        'alpha2': alpha2,
        'beta1': beta1,
        'beta2': beta2,
        'lambda1': lambda1,
        'lambda2': lambda2,
        'weight_decay': weight_decay,
        'num_layers': num_layers,
        'depth': depth,
        'num_blocks': num_blocks,
        'split_rate': split_rate,
        'use_joint_discr': use_joint_discr,
    }
    job_dir = os.path.join(job_dir,
                           str(depth),
                           str(num_layers),
                           str(split_rate),
                           str(alpha1),
                           str(alpha2),
                           str(beta1),
                           str(beta2),
                           str(lambda1),
                           str(lambda2),
                           'rs{}'.format(random_seed))
    if skip and tf.gfile.Exists(job_dir):
        print 'Training is already done. Skip it. {}'.format(job_dir)
        return

    session_config = None
    if gpu is not None:
        session_config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                visible_device_list=gpu
            )
        )

    # Define models
    model_fn = models_factory.get_model(model_name)
    estimator = tf.estimator.Estimator(
        model_fn,
        model_dir=job_dir,
        config=tf.estimator.RunConfig().replace(session_config=session_config,
                                                save_summary_steps=1000,
                                                tf_random_seed=random_seed),
        params=params)

    eval_steps = 5000
    for _ in range(train_steps/eval_steps):
        if not eval_only:
            # Run train
            estimator.train(functools.partial(input_fn,
                                              dataset=dataset_train,
                                              batch_size=train_batch_size),
                            steps=eval_steps)
        estimator.evaluate(functools.partial(input_fn,
                                             dataset=dataset_valid,
                                             batch_size=eval_batch_size),
                           steps=1,
                           hooks=[tf.contrib.training.SummaryAtEndHook(log_dir=os.path.join(job_dir, 'eval'))])
        if eval_only:
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    ###############
    # Directories #
    ###############
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help="""
                                GCS or local dir for checkpoints, exports, and
                                summaries. Use an existing directory to load a
                                trained model, or a new directory to retrain""")
    parser.add_argument('--dataset-dir',
                        required=True,
                        type=str,
                        help='The directory where the dataset files are stored.')

    #########
    # Model #
    #########
    parser.add_argument('--model-name',
                        type=str,
                        help='The name of the model to use.',
                        choices=models_factory.models_map.keys(),
                        default='splitgan')
    parser.add_argument('--use-joint-discr',
                        type=str2bool,
                        default=True,
                        help='Whether to use joint discriminator')

    ############
    # Datasets #
    ############
    parser.add_argument('--paired-dataset',
                        type=str,
                        help='The name of the paired dataset.',
                        default=None)
    parser.add_argument('--domain-a',
                        type=str,
                        help='The name of the domain A.',
                        default='mnist_m')
    parser.add_argument('--domain-b',
                        type=str,
                        help='The name of the domain B.',
                        default='mnist')

    ############
    # Training #
    ############
    parser.add_argument('--train-steps',
                        type=int,
                        help='Maximum number of training steps to perform.')

    ###################
    # Hyperparameters #
    ###################
    parser.add_argument('--train-batch-size',
                        type=int,
                        default=64,
                        help='Batch size for training steps')
    parser.add_argument('--eval-batch-size',
                        type=int,
                        default=20,
                        help='Batch size for eval steps')
    parser.add_argument('--pool-size',
                        type=int,
                        default=50,
                        help='Pool size for history of generated images')
    parser.add_argument('--alpha1',
                        type=float,
                        default=0.0002,
                        help='Learning rate for discriminator A.')
    parser.add_argument('--alpha2',
                        type=float,
                        default=0.0002,
                        help='Learning rate for discriminator B.')
    parser.add_argument('--beta1',
                        type=float,
                        default=0.0002,
                        help='Learning rate for generator AB and reconstructing A.')
    parser.add_argument('--beta2',
                        type=float,
                        default=0.0002,
                        help='Learning rate for generator BA and reconstructing B.')
    parser.add_argument('--lambda1',
                        type=float,
                        default=10.,
                        help='Weight for ABA cycle loss')
    parser.add_argument('--lambda2',
                        type=float,
                        default=10.,
                        help='Weight for BAB cycle loss')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.00001,
                        help='Weight decay rate for regularization')
    parser.add_argument('--num-layers',
                        type=int,
                        default=4,
                        help='Number of layers')
    parser.add_argument('--depth',
                        type=int,
                        default=32,
                        help='Initial depth of ConvNets')
    parser.add_argument('--num-blocks',
                        type=int,
                        default=9,
                        help='Number of ResNet blocks')
    parser.add_argument('--split-rate',
                        type=int,
                        default=0,
                        help='Split rate of z_a')

    ##############
    # Run Config #
    ##############
    parser.add_argument('--gpu',
                        type=str,
                        help='GPU ids for training.',
                        default=None)
    parser.add_argument('--skip',
                        type=str2bool,
                        default=False,
                        help='Whether to skip if the directory already exists')
    parser.add_argument('--eval-only',
                        type=str2bool,
                        default=False,
                        help='Run evaluation only')
    parser.add_argument('--random-seed',
                        type=int,
                        default=1,
                        help='Random seed')
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
