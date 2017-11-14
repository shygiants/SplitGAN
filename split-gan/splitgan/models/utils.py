""" Utils for models """

import tensorflow as tf
import operator


def reflection_pad(inputs, padding):
    pad_2d = [padding] * 2
    inputs = tf.pad(inputs, [[0, 0], pad_2d, pad_2d, [0, 0]], mode='REFLECT')
    return inputs


def gated_split(inputs, kernel_size=3, dense=False, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Gated_Split', [inputs], reuse=reuse):
        if dense:
            depth = inputs.get_shape()[1]
            T = tf.layers.dense(inputs,
                                depth,
                                use_bias=True)
        else:
            depth = inputs.get_shape()[3]
            T = tf.layers.conv2d(inputs,
                                 depth,
                                 kernel_size,
                                 strides=(1, 1),
                                 padding='SAME',
                                 use_bias=True)
        T = tf.nn.sigmoid(T)
        tf.summary.histogram('T', T)
        return inputs * T, inputs * (1. - T)


def resize_deconv(inputs, filters, kernel_size, strides=(1, 1), use_bias=True, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Resize_Deconv', [inputs], reuse=reuse):
        shape = tf.shape(inputs)
        inputs = tf.image.resize_images(inputs, [strides[0] * shape[1], strides[1] * shape[2]],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        inputs = reflection_pad(inputs, 1)
        inputs = tf.layers.conv2d(inputs,
                                  filters,
                                  kernel_size,
                                  strides=(1, 1),
                                  padding='VALID',
                                  use_bias=use_bias)

    return inputs


def deconv(inputs, filters, kernel_size, strides=(1, 1), use_bias=True, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Deconv', [inputs], reuse=reuse):
        inputs = tf.layers.conv2d_transpose(inputs,
                                            filters,
                                            kernel_size,
                                            strides=strides,
                                            padding='SAME',
                                            use_bias=use_bias)
        return inputs


def encoder(inputs, num_layers, kernel_size=3, initial_depth=32, dense_dim=None, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Encoder', [inputs], reuse=reuse):
        with tf.variable_scope('Conv2d_0_{}'.format(initial_depth), values=[inputs]):
            inputs = reflection_pad(inputs, 3)
            inputs = tf.layers.conv2d(inputs,
                                      initial_depth,
                                      7,
                                      strides=(1, 1),
                                      padding='VALID',
                                      use_bias=True)
            inputs = tf.contrib.layers.instance_norm(inputs)
            inputs = tf.nn.relu(inputs)

        for n in range(1, num_layers):
            depth = initial_depth * 2 ** n
            depth_used = min(depth, 64 * 8)
            with tf.variable_scope('Conv2d_{}_{}'.format(n, depth_used), values=[inputs]):
                inputs = tf.layers.conv2d(inputs,
                                          depth_used,
                                          kernel_size,
                                          strides=(2, 2),
                                          padding='SAME',
                                          use_bias=True)
                inputs = tf.contrib.layers.instance_norm(inputs)
                inputs = tf.nn.relu(inputs)

        if dense_dim is not None:
            with tf.variable_scope('Dense', values=[inputs]):
                # inputs = tf.layers.flatten(inputs)
                # TODO: Remove hard coded shape
                inputs = tf.reshape(inputs, [-1, 4*4*512])
                inputs = tf.layers.dense(inputs, dense_dim, use_bias=True)
                inputs = tf.nn.relu(inputs)

    return inputs


def downsample(inputs, num_layers, initial_depth, kernel_size=3, activation=tf.nn.relu, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Downsample', [inputs], reuse=reuse):
        for n in range(num_layers):
            depth = initial_depth * 2 ** (n + 1)
            depth_used = min(depth, 64 * 8)
            with tf.variable_scope('Conv2d_{}_{}'.format(n, depth_used), values=[inputs]):
                inputs = tf.layers.conv2d(inputs,
                                          depth_used,
                                          kernel_size,
                                          strides=(2, 2),
                                          padding='SAME',
                                          use_bias=True)
                inputs = tf.contrib.layers.instance_norm(inputs)
                inputs = activation(inputs)

    return inputs


def decoder(inputs, num_layers, kernel_size=3, initial_depth=32, dense_dim=None, reshape=None, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Decoder', [inputs], reuse=reuse):
        if dense_dim is not None:
            with tf.variable_scope('Dense', values=[inputs]):
                flatten_reshape = reduce(operator.mul, reshape, 1)
                inputs = tf.layers.dense(inputs, flatten_reshape, use_bias=True)
                inputs = tf.nn.relu(inputs)
                inputs = tf.reshape(inputs, [-1] + reshape)

        for n in range(num_layers - 1):
            depth = initial_depth * 2**(num_layers - 2 - n)
            depth_used = min(depth, 64 * 8)
            with tf.variable_scope('Deconv2d_{}_{}'.format(n, depth_used), values=[inputs]):
                inputs = resize_deconv(inputs,
                                       depth_used,
                                       kernel_size,
                                       strides=(2, 2),
                                       use_bias=True)
                inputs = tf.contrib.layers.instance_norm(inputs)
                inputs = tf.nn.relu(inputs)

        inputs = reflection_pad(inputs, 3)
        inputs = tf.layers.conv2d(inputs,
                                  3,
                                  7,
                                  strides=(1, 1),
                                  padding='VALID',
                                  use_bias=True,
                                  name='Deconv2d_{}_{}'.format(num_layers - 1, 3))
        inputs = tf.nn.tanh(inputs)

    return inputs


def resnet_block(inputs, num_features, scope=None, reuse=None):
    shortcut = inputs
    with tf.variable_scope(scope, 'ResNet_Block', [inputs], reuse=reuse):
        # Layer 1
        inputs = reflection_pad(inputs, 1)
        inputs = tf.layers.conv2d(inputs,
                                  num_features,
                                  3,
                                  strides=(1, 1),
                                  padding='VALID',
                                  use_bias=True,
                                  name='Conv2d_0_{}'.format(num_features))
        inputs = tf.contrib.layers.instance_norm(inputs)
        inputs = tf.nn.relu(inputs)

        # Layer 2
        inputs = reflection_pad(inputs, 1)
        inputs = tf.layers.conv2d(inputs,
                                  num_features,
                                  3,
                                  strides=(1, 1),
                                  padding='VALID',
                                  use_bias=True,
                                  name='Conv2d_1_{}'.format(num_features))
        inputs = tf.contrib.layers.instance_norm(inputs)

    return inputs + shortcut


def transformer(inputs, num_features, num_blocks=6, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Transformer', [inputs], reuse=reuse):
        for i in range(num_blocks):
            inputs = resnet_block(inputs, num_features, scope='ResNet_Block_{}'.format(i))
    return inputs


def discriminator(inputs,
                  num_layers,
                  kernel_size=4,
                  initial_depth=64,
                  down_sample=True,
                  use_logit=True,
                  use_info=False,
                  dense_dim=None,
                  scope=None,
                  reuse=None):
    with tf.variable_scope(scope, 'Discriminator', [inputs], reuse=reuse):
        for n in range(num_layers):
            depth = initial_depth * 2 ** n
            depth_used = min(depth, 64 * 8)
            with tf.variable_scope('Conv2d_{}_{}'.format(n, depth_used), values=[inputs]):
                inputs = tf.layers.conv2d(inputs,
                                          depth_used,
                                          kernel_size,
                                          strides=(2, 2) if n != num_layers - 1 else (1, 1),
                                          padding='SAME',
                                          use_bias=True)
                if n != 0:
                    inputs = tf.contrib.layers.instance_norm(inputs)
                inputs = tf.nn.leaky_relu(inputs)

        if use_info:
            if dense_dim is not None:
                # info = tf.layers.flatten(inputs)
                # TODO: Remove hard coded shape
                info = tf.reshape(inputs, [-1, 2 * 2 * 512])
                info = tf.layers.dense(info, dense_dim, use_bias=True)
                inputs = tf.nn.leaky_relu(inputs)
            else:
                info = inputs
                info = downsample(info, 2, info.get_shape()[3], activation=tf.nn.leaky_relu)
                info = tf.reduce_mean(info, axis=[1, 2], keep_dims=True)

        if use_logit:
            logits = tf.layers.conv2d(inputs,
                                      1,
                                      kernel_size,
                                      strides=(1, 1),
                                      padding='SAME',
                                      use_bias=True,
                                      name='Logits')
            probs = tf.nn.sigmoid(logits, name='Probs')
            if use_info:
                return logits, probs, info
            else:
                return logits, probs
        else:
            if use_info:
                return inputs, info
            else:
                return inputs


def joint_discriminator(x, z, num_layers, kernel_size=4, initial_depth_x=64, initial_depth_z=64, scope=None, reuse=None):
    with tf.variable_scope(scope, 'JointDiscriminator', [x, z], reuse=reuse):
        x_discr = discriminator(x,
                                num_layers,
                                kernel_size=kernel_size,
                                initial_depth=initial_depth_x,
                                use_logit=False,
                                scope='X_Discriminator')

        z_discr = discriminator(z,
                                2,
                                kernel_size=1,
                                initial_depth=initial_depth_z,
                                down_sample=False,
                                use_logit=False,
                                scope='Z_Discriminator')
        height = tf.shape(x_discr)[1]
        z_discr = tf.tile(z_discr, [1, height, height, 1])
        concat = tf.concat([x_discr, z_discr], 3)

        return discriminator(concat,
                             2,
                             kernel_size=1,
                             initial_depth=initial_depth_x * 2 ** num_layers + initial_depth_z * 2,
                             scope='Joint_Discriminator')


def normalize_images(images):
    images -= tf.reduce_min(images)
    return images / tf.reduce_max(images)


def run_train_ops_stepwise(train_ops, global_step):
    num_train_ops = len(train_ops)
    train_op_idx = tf.truncatemod(global_step, num_train_ops)

    pred_fn_pairs = map(lambda (i, op): (tf.equal(train_op_idx, i), lambda: op),
                        enumerate(train_ops))

    train_op = tf.case(pred_fn_pairs, lambda: tf.no_op(), exclusive=True)

    return train_op


def image_pool(inputs, pool_size, image_size, scope=None):
    images_shape = [pool_size, image_size, image_size, 3]
    with tf.variable_scope(scope, 'image_pool', [inputs]):
        # TODO: Remove hard coded image shape
        images = tf.get_variable('images', images_shape,
                                 initializer=tf.zeros_initializer(dtype=tf.float32),
                                 trainable=False)
        num_images = tf.get_variable('num_images', (), dtype=tf.int32,
                                     initializer=tf.zeros_initializer(dtype=tf.int32),
                                     trainable=False)
        push = tf.less(num_images, pool_size, name='push')

        def push_n_identity():
            push_images = tf.scatter_update(images, [num_images], [tf.squeeze(inputs, axis=0)], name='push_images')
            with tf.control_dependencies([push_images]):
                increment = tf.assign_add(num_images, 1, name='increment')
                with tf.control_dependencies([increment]):
                    return tf.identity(inputs)

        def sample():
            r = tf.random_uniform((), minval=0., maxval=1., dtype=tf.float32)
            identity = tf.greater(r, 0.5, name='identity')

            def pop_n_push():
                rand_idx = tf.random_uniform((), minval=0, maxval=pool_size, dtype=tf.int32)
                to_return = images[rand_idx]
                push_images = tf.scatter_update(images, [rand_idx], [tf.squeeze(inputs, axis=0)], name='push_images')
                with tf.control_dependencies([push_images]):
                    return tf.identity(to_return)

            return tf.cond(identity, lambda: inputs, pop_n_push)

        return tf.reshape(tf.cond(push,
                                  true_fn=push_n_identity,
                                  false_fn=sample),
                          [-1, image_size, image_size, 3]), images
