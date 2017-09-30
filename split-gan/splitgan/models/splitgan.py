""" SplitGAN implementation """

import math

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from utils import encoder, decoder, discriminator, \
    normalize_images, run_train_ops_stepwise, transformer, instance_norm, downsample, joint_discriminator


def model_fn(features, labels, mode, params):
    x_a = features['x_a']
    x_b = features['x_b']

    # Hyperparameters
    weight_decay = params['weight_decay']
    num_layers = params['num_layers']
    depth = params['depth']
    num_blocks = params['num_blocks']
    split_rate = params['split_rate']
    alpha1 = params['alpha1']
    alpha2 = params['alpha2']
    beta1 = params['beta1']
    beta2 = params['beta2']
    lambda1 = params['lambda1']
    lambda2 = params['lambda2']
    gamma = params['gamma']
    use_avg_pool = params['use_avg_pool']
    use_joint_discr = params['use_joint_discr']

    latent_depth = depth * 2 ** (num_layers - 1)
    log_depth = int(math.log(latent_depth, 2))
    log_depth_b = log_depth - 1 - split_rate
    depth_b = 2 ** log_depth_b
    depth_a_b = latent_depth - depth_b

    num_downsample = 2
    depth_a_b_pooled = depth_a_b * 2 ** num_downsample

    with tf.variable_scope('SplitGAN', values=[x_a, x_b]):
        add_arg_scope(tf.layers.conv2d)
        # TODO: Save random seeds of initializers
        with arg_scope([tf.layers.conv2d],
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
            def generator_ab(inputs_a, reuse=None):
                with tf.variable_scope('Generator_AB', values=[inputs_a], reuse=reuse):
                    z_a = encoder(inputs_a, num_layers, initial_depth=depth, scope='Encoder_A')

                    # z is split into c_b, z_a-b
                    c_b, z_a_b = tf.split(z_a,
                                          num_or_size_splits=[depth_b, depth_a_b],
                                          axis=3)

                    if use_avg_pool:
                        z_a_b = downsample(z_a_b, num_downsample, depth_a_b)
                        z_a_b = tf.reduce_mean(z_a_b, axis=[1, 2], keep_dims=True)

                    outputs_ab = decoder(c_b, num_layers, scope='Decoder_B',
                                         initial_depth=depth / 2 ** (1 + split_rate))

                return outputs_ab, z_a_b

            def generator_ba(inputs_b, z_a_b, reuse=None):
                with tf.variable_scope('Generator_BA', values=[inputs_b], reuse=reuse):
                    z_b = encoder(inputs_b, num_layers, scope='Encoder_B',
                                  initial_depth=depth / 2 ** (1 + split_rate))

                    if use_avg_pool:
                        height = tf.shape(z_b)[1]
                        z_a_b = tf.tile(z_a_b, [1, height, height, 1])

                    # Concat z_b and z_a-b
                    z_a_b = tf.layers.conv2d(z_a_b,
                                     depth_b,
                                     1,
                                     strides=(1, 1),
                                     padding='SAME',
                                     use_bias=False)
                    z_a_b = instance_norm(z_a_b)
                    z_a_b = tf.nn.relu(z_a_b)

                    c_a = z_b + gamma * z_a_b

                    ####################
                    # Transformer part #
                    ####################
                    c_a = transformer(c_a, depth_b, num_blocks=num_blocks,
                                      scope='Transformer_A', reuse=reuse)

                    outputs_ba = decoder(c_a, num_layers, initial_depth=depth, scope='Decoder_A')

                return outputs_ba

            global_step = tf.train.get_or_create_global_step()

            ##################
            # Generator part #
            ##################
            x_ab, z_a_b = generator_ab(x_a)
            x_ba = generator_ba(x_b, z_a_b)

            images_a = [x_a, x_ab]
            images_b = [x_b, x_ba]

            if mode == Modes.TRAIN or mode == Modes.EVAL:
                x_aba = generator_ba(x_ab, z_a_b, reuse=True)
                x_bab, z_a_b_fake = generator_ab(x_ba, reuse=True)

                images_a.append(x_aba)
                images_b.append(x_bab)

                ######################
                # Discriminator part #
                ######################
                def joint(x, z):
                    if use_joint_discr:
                        height = tf.shape(x)[1]
                        z = tf.tile(z, [1, height, height, 1])
                        return tf.concat([x, z], 3, name='joint')
                    else:
                        return x

                logits_a_real, probs_a_real = joint_discriminator(x_a,
                                                                  z_a_b,
                                                                  num_layers - 1,
                                                                  initial_depth_x=2*depth,
                                                                  initial_depth_z=2*depth_a_b_pooled,
                                                                  scope='Discriminator_A')
                logits_b_real, probs_b_real = discriminator(x_b,
                                                            num_layers + 1,
                                                            initial_depth=2*depth,
                                                            scope='Discriminator_B')
                logits_b_fake, probs_b_fake = discriminator(x_ab,
                                                            num_layers + 1,
                                                            initial_depth=2 * depth,
                                                            scope='Discriminator_B',
                                                            reuse=True)
                logits_a_fake, probs_a_fake = joint_discriminator(x_ba,
                                                                  z_a_b_fake,
                                                                  num_layers - 1,
                                                                  initial_depth_x=2 * depth,
                                                                  initial_depth_z=2 * depth_a_b_pooled,
                                                                  scope='Discriminator_A',
                                                                  reuse=True)

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
        l_d_a_fake = tf.reduce_mean(tf.square(logits_a_fake))
        l_d_b_real = tf.reduce_mean(tf.squared_difference(logits_b_real, 1.))
        l_d_b_fake = tf.reduce_mean(tf.square(logits_b_fake))

        l_d_a = (l_d_a_real + l_d_a_fake) * .5
        l_d_b = (l_d_b_real + l_d_b_fake) * .5

        # Generator losses
        l_g_ab_gan = tf.reduce_mean(tf.squared_difference(logits_b_fake, 1.))
        l_g_ba_gan = tf.reduce_mean(tf.squared_difference(logits_a_fake, 1.))

        l_const_a = tf.reduce_mean(tf.losses.absolute_difference(x_a, x_aba))
        l_const_b = tf.reduce_mean(tf.losses.absolute_difference(x_b, x_bab))
        loss = l_const_a + l_const_b

        l_g_a = l_g_ab_gan + lambda1 * l_const_a
        l_g_b = l_g_ba_gan + lambda2 * l_const_b

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
            tf.summary.scalar('L_G_A', l_g_a)
            tf.summary.scalar('L_G_B', l_g_b)

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
        train_op_g_a, beta1 = get_train_op(beta1, l_g_a, g_vars)
        train_op_g_b, beta2 = get_train_op(beta2, l_g_b, g_vars)

        train_ops = [train_op_d_a, train_op_d_b, train_op_g_a, train_op_g_b]
        train_op = run_train_ops_stepwise(train_ops, global_step)

        with tf.name_scope('hyperparameters'):
            tf.summary.scalar('alpha1', alpha1)
            tf.summary.scalar('alpha2', alpha2)
            tf.summary.scalar('beta1', beta1)
            tf.summary.scalar('beta2', beta2)
            tf.summary.scalar('lambda1', lambda1)
            tf.summary.scalar('lambda2', lambda2)

    # Image summaries
    images_a_concat = tf.concat(
        axis=2,
        values=map(normalize_images, images_a))
    images_b_concat = tf.concat(
        axis=2,
        values=map(normalize_images, images_b))

    tf.summary.image('ImagesA', images_a_concat, max_outputs=10)
    tf.summary.image('ImagesB', images_b_concat, max_outputs=10)

    if mode == Modes.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    if mode == Modes.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)
    if mode == Modes.PREDICT:
        predictions = {
            'x_ab': x_ab,
            'x_ba': x_ba
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
