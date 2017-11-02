""" CycleGAN implementation """

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from utils import encoder, decoder, transformer, discriminator, \
    normalize_images, run_train_ops_stepwise, image_pool


def model_fn(features, labels, mode, params):
    x_a = features['x_a']
    x_b = features['x_b']

    # Hyperparameters
    pool_size = params['pool_size']
    weight_decay = params['weight_decay']
    num_layers = params['num_layers']
    depth = params['depth']
    num_blocks = params['num_blocks']
    alpha1 = params['alpha1']
    alpha2 = params['alpha2']
    beta1 = params['beta1']
    beta2 = params['beta2']
    lambda1 = params['lambda1']
    lambda2 = params['lambda2']

    latent_depth = depth * 2 ** (num_layers - 1)

    with tf.variable_scope('CycleGAN', values=[x_a, x_b]):
        add_arg_scope(tf.layers.conv2d)
        with arg_scope([tf.layers.conv2d],
                       kernel_initializer=tf.glorot_normal_initializer(),
                       bias_initializer=tf.glorot_uniform_initializer()):
            def generator_ab(inputs_a, reuse=None):
                with tf.variable_scope('Generator_AB', values=[inputs_a], reuse=reuse):
                    z_a = encoder(inputs_a, num_layers, initial_depth=depth, scope='Encoder_AB')

                    ####################
                    # Transformer part #
                    ####################
                    z_a = transformer(z_a, latent_depth, num_blocks=num_blocks, scope='Transformer_AB')

                    outputs_ab = decoder(z_a, num_layers, initial_depth=depth, scope='Decoder_AB')

                    return outputs_ab

            def generator_ba(inputs_b, reuse=None):
                with tf.variable_scope('Generator_BA', values=[inputs_b], reuse=reuse):
                    z_b = encoder(inputs_b, num_layers, initial_depth=depth, scope='Encoder_BA')

                    ####################
                    # Transformer part #
                    ####################
                    z_b = transformer(z_b, latent_depth, num_blocks=num_blocks, scope='Transformer_BA')

                    outputs_ba = decoder(z_b, num_layers, initial_depth=depth, scope='Decoder_BA')

                    return outputs_ba

            global_step = tf.train.get_or_create_global_step()

            ##################
            # Generator part #
            ##################
            if mode == Modes.TRAIN or mode == Modes.EVAL:
                x_ab = generator_ab(x_a)
                x_ba = generator_ba(x_b)

                images_a = [x_a, x_ab]
                images_b = [x_b, x_ba]

                x_aba = generator_ba(x_ab, reuse=True)
                x_bab = generator_ab(x_ba, reuse=True)

                images_a.append(x_aba)
                images_b.append(x_bab)

                if mode == Modes.TRAIN:
                    # TODO: History of generated images
                    fake_b, pool_b = image_pool(x_ab, pool_size, scope='image_pool_B')
                    fake_a, pool_a = image_pool(x_ba, pool_size, scope='image_pool_A')
                else:
                    fake_b = x_ab
                    fake_a = x_ba

                ######################
                # Discriminator part #
                ######################
                logits_a_real, probs_a_real = discriminator(x_a, num_layers + 1, initial_depth=depth,
                                                            scope='Discriminator_A')
                logits_b_real, probs_b_real = discriminator(x_b, num_layers + 1, initial_depth=depth,
                                                            scope='Discriminator_B')
                logits_b_fake_d, probs_b_fake_d = discriminator(fake_b, num_layers + 1, initial_depth=depth,
                                                                scope='Discriminator_B', reuse=True)
                logits_a_fake_d, probs_a_fake_d = discriminator(fake_a, num_layers + 1, initial_depth=depth,
                                                                scope='Discriminator_A', reuse=True)
                logits_b_fake, probs_b_fake = discriminator(x_ab, num_layers + 1, initial_depth=depth,
                                                            scope='Discriminator_B', reuse=True)
                logits_a_fake, probs_a_fake = discriminator(x_ba, num_layers + 1, initial_depth=depth,
                                                            scope='Discriminator_A', reuse=True)
                with tf.name_scope('logits'):
                    tf.summary.histogram('logits_a_real', logits_a_real)
                    tf.summary.histogram('logits_b_real', logits_b_real)
                    tf.summary.histogram('logits_a_fake', logits_a_fake)
                    tf.summary.histogram('logits_b_fake', logits_b_fake)
                    tf.summary.histogram('logits_a_fake_d', logits_a_fake_d)
                    tf.summary.histogram('logits_b_fake_d', logits_b_fake_d)

    if mode == Modes.TRAIN or mode == Modes.EVAL:
        ##########
        # Losses #
        ##########
        t_vars = tf.trainable_variables()

        def search_fn(keyword):
            return lambda var: keyword in var.name

        d_a_vars = filter(search_fn('Discriminator_A'), t_vars)
        d_b_vars = filter(search_fn('Discriminator_B'), t_vars)
        g_vars = filter(search_fn('Generator'), t_vars)

        # Discriminator losses
        l_d_a_real = tf.reduce_mean(tf.squared_difference(logits_a_real, 1.))
        l_d_a_fake = tf.reduce_mean(tf.square(logits_a_fake_d))
        l_d_b_real = tf.reduce_mean(tf.squared_difference(logits_b_real, 1.))
        l_d_b_fake = tf.reduce_mean(tf.square(logits_b_fake_d))

        l_d_a = (l_d_a_real + l_d_a_fake) * .5
        l_d_b = (l_d_b_real + l_d_b_fake) * .5

        # Generator losses
        l_g_ab_gan = tf.reduce_mean(tf.squared_difference(logits_b_fake, 1.))
        l_g_ba_gan = tf.reduce_mean(tf.squared_difference(logits_a_fake, 1.))

        l_const_a = tf.reduce_mean(tf.losses.absolute_difference(x_a, x_aba))
        l_const_b = tf.reduce_mean(tf.losses.absolute_difference(x_b, x_bab))

        loss = l_const_a + l_const_b

        cyclic_loss = lambda1 * l_const_a + lambda2 * l_const_b

        l_g_ab = l_g_ab_gan + cyclic_loss
        l_g_ba = l_g_ba_gan + cyclic_loss
        l_g = l_g_ab_gan + l_g_ba_gan + cyclic_loss

        with tf.name_scope('losses'):
            tf.summary.scalar('L_D_A_Real', l_d_a_real)
            tf.summary.scalar('L_D_B_Real', l_d_b_real)
            tf.summary.scalar('L_D_A_Fake', l_d_a_fake)
            tf.summary.scalar('L_D_B_Fake', l_d_b_fake)
            tf.summary.scalar('L_D_A', l_d_a)
            tf.summary.scalar('L_D_B', l_d_b)
            tf.summary.scalar('L_G_AB_GAN', l_g_ab_gan)
            tf.summary.scalar('L_G_BA_GAN', l_g_ba_gan)
            tf.summary.scalar('L_Const_A', l_const_a)
            tf.summary.scalar('L_Const_B', l_const_b)
            tf.summary.scalar('L_G_AB', l_g_ab)
            tf.summary.scalar('L_G_BA', l_g_ba)

    if mode == Modes.TRAIN:
        def get_train_op(learning_rate, loss, var_list):
            start_decay_step = 100000
            decay_steps = 100000
            starter_learning_rate = learning_rate
            end_learning_rate = 0.0

            learning_rate = tf.where(
                tf.greater_equal(global_step, start_decay_step),
                tf.train.polynomial_decay(starter_learning_rate, global_step - start_decay_step,
                                          decay_steps, end_learning_rate,
                                          power=1.0),
                starter_learning_rate)

            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.5,
                beta2=0.999
            )
            grads = optimizer.compute_gradients(loss, var_list=var_list)
            tf.contrib.training.add_gradients_summaries(grads)
            return optimizer.apply_gradients(grads, global_step=global_step), learning_rate

        train_op_d_a, alpha1 = get_train_op(alpha1, l_d_a, d_a_vars)
        train_op_d_b, alpha2 = get_train_op(alpha2, l_d_b, d_b_vars)
        train_op_g, beta1 = get_train_op(beta1, l_g, g_vars)

        train_ops = [train_op_d_a, train_op_d_b, train_op_g]
        train_op = tf.group(*train_ops)

        with tf.name_scope('hyperparameters'):
            tf.summary.scalar('alpha1', alpha1)
            tf.summary.scalar('alpha2', alpha2)
            tf.summary.scalar('beta1', beta1)
            tf.summary.scalar('beta2', beta2)
            tf.summary.scalar('lambda1', lambda1)
            tf.summary.scalar('lambda2', lambda2)

    # Image summaries
    if mode == Modes.TRAIN or mode == Modes.EVAL:
        images_a_concat = tf.concat(
            axis=2,
            values=map(normalize_images, images_a))
        images_b_concat = tf.concat(
            axis=2,
            values=map(normalize_images, images_b))

        tf.summary.image('ImagesA', images_a_concat, max_outputs=20)
        tf.summary.image('ImagesB', images_b_concat, max_outputs=20)

        if mode == Modes.TRAIN:
            pool_a = normalize_images(pool_a)
            pool_b = normalize_images(pool_b)
            tf.summary.image('PoolA', pool_a, max_outputs=pool_size)
            tf.summary.image('PoolB', pool_b, max_outputs=pool_size)

    if mode == Modes.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    if mode == Modes.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    if mode == Modes.PREDICT:
        predictions = {
            'x_ab': x_ab,
            'x_ba': x_ba,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
