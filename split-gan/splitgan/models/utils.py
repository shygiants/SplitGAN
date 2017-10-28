""" Utils for models """

import tensorflow as tf


def _weights(name, shape, mean=0.0, stddev=0.02):
    return tf.get_variable(name, shape,
                           initializer=tf.random_normal_initializer(
                               mean=mean, stddev=stddev, dtype=tf.float32))


def _biases(name, shape, constant=0.0):
    return tf.get_variable(name, shape,
                           initializer=tf.constant_initializer(constant))


def instance_norm(inputs, scope=None):
    with tf.variable_scope(scope, 'instance_norm', [inputs]):
        depth = inputs.get_shape()[3]
        scale = _weights('scale', [depth], mean=1.0)
        offset = _biases('offset', [depth])
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (inputs - mean) * inv
    return scale * normalized + offset


def reflection_pad(inputs, padding):
    pad_2d = [padding] * 2
    inputs = tf.pad(inputs, [[0, 0], pad_2d, pad_2d, [0, 0]], mode='REFLECT')
    return inputs


def leaky_relu_fn(negative_slope):
    def leaky_relu(x):
        return tf.maximum(negative_slope * x, x, name='lrelu')
    return leaky_relu


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


def encoder(inputs, num_layers, kernel_size=3, initial_depth=32, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Encoder', [inputs], reuse=reuse):
        with tf.variable_scope('Conv2d_0_{}'.format(initial_depth), values=[inputs]):
            inputs = reflection_pad(inputs, 3)
            inputs = tf.layers.conv2d(inputs,
                                      initial_depth,
                                      7,
                                      strides=(1, 1),
                                      padding='VALID',
                                      use_bias=False)
            inputs = instance_norm(inputs)
            inputs = tf.nn.relu(inputs)

        for n in range(1, num_layers):
            depth = initial_depth * 2 ** n
            with tf.variable_scope('Conv2d_{}_{}'.format(n, depth), values=[inputs]):
                inputs = tf.layers.conv2d(inputs,
                                          depth,
                                          kernel_size,
                                          strides=(2, 2),
                                          padding='SAME',
                                          use_bias=False)
                inputs = instance_norm(inputs)
                inputs = tf.nn.relu(inputs)

    return inputs


def decoder(inputs, num_layers, kernel_size=3, initial_depth=32, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Decoder', [inputs], reuse=reuse):
        for n in range(num_layers - 1):
            depth = initial_depth * 2**(num_layers - 2 - n)
            with tf.variable_scope('Deconv2d_{}_{}'.format(n, depth), values=[inputs]):
                inputs = deconv(inputs,
                                depth,
                                kernel_size,
                                strides=(2, 2),
                                use_bias=False)
                inputs = instance_norm(inputs)
                inputs = tf.nn.relu(inputs)

        inputs = reflection_pad(inputs, 3)
        inputs = tf.layers.conv2d(inputs,
                                  3,
                                  7,
                                  strides=(1, 1),
                                  padding='VALID',
                                  use_bias=False,
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
                                  name='Conv2d_0_{}'.format(num_features))
        inputs = instance_norm(inputs)
        inputs = tf.nn.relu(inputs)

        # Layer 2
        inputs = reflection_pad(inputs, 1)
        inputs = tf.layers.conv2d(inputs,
                                  num_features,
                                  3,
                                  strides=(1, 1),
                                  padding='VALID',
                                  name='Conv2d_1_{}'.format(num_features))
        inputs = instance_norm(inputs)

    return inputs + shortcut


def transformer(inputs, num_features, num_blocks=6, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Transformer', [inputs], reuse=reuse):
        for i in range(num_blocks):
            inputs = resnet_block(inputs, num_features, scope='ResNet_Block_{}'.format(i))
    return inputs


def discriminator(inputs, num_layers, kernel_size=4, initial_depth=64, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Discriminator', [inputs], reuse=reuse):
        lrelu = leaky_relu_fn(0.2)
        for n in range(num_layers):
            depth = initial_depth * 2 ** n
            with tf.variable_scope('Conv2d_{}_{}'.format(n, depth), values=[inputs]):
                inputs = tf.layers.conv2d(inputs,
                                          depth,
                                          kernel_size,
                                          strides=(2, 2),
                                          padding='SAME',
                                          use_bias=False)
                if n != 0:
                    inputs = instance_norm(inputs)
                inputs = lrelu(inputs)

        logits = tf.layers.conv2d(inputs,
                                  1,
                                  kernel_size,
                                  strides=(1, 1),
                                  padding='SAME',
                                  use_bias=True,
                                  name='Logits')
        probs = tf.nn.sigmoid(logits, name='Probs')

    return logits, probs


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
