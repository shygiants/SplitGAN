""" DiscoGAN implementation """

import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from utils import encoder, decoder, discriminator, normalize_images, run_train_ops_stepwise


def model_fn(features, labels, mode, params):
    x_a = features['x_a']
    x_b = features['x_b']

    # Hyperparameters
    weight_decay = params['weight_decay']
    num_layers = params['num_layers']
    learning_rate = params['learning_rate']
    recon_weight = params['recon_weight']
    recon_start = params['recon_start']

    with tf.variable_scope('DiscoGAN', values=[x_a, x_b]):
        add_arg_scope(tf.layers.conv2d)
        add_arg_scope(tf.layers.batch_normalization)
        # TODO: Save random seeds of initializers
        with arg_scope([tf.layers.conv2d],
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)):
            with arg_scope([tf.layers.batch_normalization],
                           training=mode == Modes.TRAIN):
                def generator_ab(inputs_a, reuse=None):
                    with tf.variable_scope('Generator_AB', values=[inputs_a], reuse=reuse):
                        z = encoder(inputs_a, num_layers, scope='Encoder_A')
                        outputs_ab = decoder(z, num_layers, scope='Decoder_B')

                    return outputs_ab

                def generator_ba(inputs_b, reuse=None):
                    with tf.variable_scope('Generator_BA', values=[inputs_b], reuse=reuse):
                        z = encoder(inputs_b, num_layers, scope='Encoder_B')
                        outputs_ba = decoder(z, num_layers, scope='Decoder_A')

                    return outputs_ba

                global_step = tf.train.get_or_create_global_step()

                ##################
                # Generator part #
                ##################
                x_ab = generator_ab(x_a)
                x_ba = generator_ba(x_b)

                images_a = [x_a, x_ab]
                images_b = [x_b, x_ba]

                if mode == Modes.TRAIN or mode == Modes.EVAL:
                    x_aba = generator_ba(x_ab, reuse=True)
                    x_bab = generator_ab(x_ba, reuse=True)

                    images_a.append(x_aba)
                    images_b.append(x_bab)

                    ######################
                    # Discriminator part #
                    ######################
                    logits_a_real, probs_a_real, fms_a_real = discriminator(x_a, 4, scope='Discriminator_A')
                    logits_b_real, probs_b_real, fms_b_real = discriminator(x_b, 4, scope='Discriminator_B')
                    logits_b_fake, probs_b_fake, fms_b_fake = discriminator(x_ab, 4, scope='Discriminator_B', reuse=True)
                    logits_a_fake, probs_a_fake, fms_a_fake = discriminator(x_ba, 4, scope='Discriminator_A', reuse=True)

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

        l_d_a_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            tf.ones_like(logits_a_real), logits_a_real))
        l_d_a_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(logits_a_fake), logits_a_fake))
        l_d_b_real = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            tf.ones_like(logits_b_real), logits_b_real))
        l_d_b_fake = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(logits_b_fake), logits_b_fake))

        l_d_a = (l_d_a_real + l_d_a_fake) * .5
        l_d_b = (l_d_b_real + l_d_b_fake) * .5

        l_g_ab_gan = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            tf.ones_like(logits_b_fake), logits_b_fake))
        l_g_ba_gan = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
            tf.ones_like(logits_a_fake), logits_a_fake))

        def fm_loss(fms_real, fms_fake):
            losses = []
            for fm_real, fm_fake in zip(fms_real, fms_fake):
                l2 = tf.squared_difference(tf.reduce_mean(fm_real, axis=0),
                                           tf.reduce_mean(fm_fake, axis=0))
                losses.append(tf.reduce_mean(l2))

            return tf.add_n(losses)

        l_g_ab_fm = fm_loss(fms_b_real, fms_b_fake)
        l_g_ba_fm = fm_loss(fms_a_real, fms_a_fake)

        l_const_a = tf.reduce_mean(tf.losses.mean_squared_error(
            x_a, x_aba))
        l_const_b = tf.reduce_mean(tf.losses.mean_squared_error(
            x_b, x_bab))
        loss = l_const_a + l_const_b

        rate = tf.cond(tf.less(global_step, recon_start),
                       true_fn=lambda: recon_weight / 50.,
                       false_fn=lambda: recon_weight)
        l_g_a = (1. - rate) * (.1 * l_g_ab_gan + .9 * l_g_ab_fm) + rate * l_const_a
        l_g_b = (1. - rate) * (.1 * l_g_ba_gan + .9 * l_g_ba_fm) + rate * l_const_b

        with tf.name_scope('losses'):
            tf.summary.scalar('L_D_A_Real', l_d_a_real)
            tf.summary.scalar('L_D_B_Real', l_d_b_real)
            tf.summary.scalar('L_D_A_Fake', l_d_a_fake)
            tf.summary.scalar('L_D_B_Fake', l_d_b_fake)
            tf.summary.scalar('L_D_A', l_d_a)
            tf.summary.scalar('L_D_B', l_d_b)
            tf.summary.scalar('L_G_AB_GAN', l_g_ab_gan)
            tf.summary.scalar('L_G_BA_GAN', l_g_ba_gan)
            tf.summary.scalar('L_G_AB_FM', l_g_ab_fm)
            tf.summary.scalar('L_G_BA_FM', l_g_ba_fm)
            tf.summary.scalar('L_Const_A', l_const_a)
            tf.summary.scalar('L_Const_B', l_const_b)
            tf.summary.scalar('L_G_A', l_g_a)
            tf.summary.scalar('L_G_B', l_g_b)

        train_op_d_a = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.999
        ).minimize(l_d_a, global_step=global_step, var_list=d_a_vars)
        train_op_d_b = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.999
        ).minimize(l_d_b, global_step=global_step, var_list=d_b_vars)

        train_op_g_a = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.999
        ).minimize(l_g_a, global_step=global_step, var_list=g_vars)
        train_op_g_b = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.5,
            beta2=0.999
        ).minimize(l_g_b, global_step=global_step, var_list=g_vars)

        train_ops = [train_op_d_a, train_op_d_b, train_op_g_a, train_op_g_b]
        train_op = run_train_ops_stepwise(train_ops, global_step)

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
