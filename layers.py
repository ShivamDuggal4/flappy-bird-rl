import tensorflow as tf


def variable_summaries(var, var_name):
  """Attach summaries to a Tensor (for TensorBoard visualization) given va and var_name"""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean_' + var_name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev_'+var_name, stddev)
    tf.summary.scalar('max_'+var_name, tf.reduce_max(var))
    tf.summary.scalar('min_'+var_name, tf.reduce_min(var))
    tf.summary.histogram('histogram_'+var_name, var)

def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


def create_conv(layer_num, prev, shape, padding='SAME', stride=1, relu=False,
                 max_pooling=False, mp_ksize=[1, 2, 2, 1], mp_strides=[1, 2, 2, 1]):
    """
        Create a convolutional layer with relu and/mor max pooling(Optional)
    """
    strides = [1,stride,stride,1]
    _init = tf.random_uniform_initializer(-1,1)
    conv_w = tf.get_variable("conv_w_"+layer_num,shape=shape,initializer = _init)
    variable_summaries(conv_w,"conv_w_"+layer_num)
    conv_b = tf.get_variable("conv_b_"+layer_num,initializer=tf.ones(shape[-1])*0.01,dtype=tf.float32)
    variable_summaries(conv_b,"conv_b_"+layer_num)
    conv   = tf.nn.conv2d(prev, conv_w, strides=strides, padding=padding) + conv_b

    conv_batch_norm = tf.contrib.layers.batch_norm(conv,decay=0.9,epsilon=1e-5,scale=True,updates_collections=None)
    
    if relu:
        conv = tf.nn.relu(conv_batch_norm)

    if max_pooling:
        conv = tf.nn.max_pool(conv, ksize=mp_ksize, strides=mp_strides, padding='SAME')

    return conv, conv_w, conv_b


def fc(layer_num,prev, input_size, output_size, relu=False, sigmoid=False, no_bias=False,
        softmax=False):
    """
        Create fully connecter layer with relu(Optional)
    """
    shape = (input_size,output_size)
    _init = tf.random_uniform_initializer(-1,1)
    fc_w = tf.get_variable("fc_w"+layer_num,shape=shape,initializer = _init)
    fc_b = tf.get_variable("fc_b_" + layer_num,initializer=tf.ones(shape[-1])*0.01,dtype=tf.float32)
    pre_activation = tf.matmul(prev, fc_w)
    activation = None

    if not no_bias:
        pre_activation = pre_activation + fc_b
    if relu:
        activation = tf.nn.relu(pre_activation)
    if sigmoid:
        activation = tf.nn.sigmoid(pre_activation)
    if softmax:
        activation = tf.nn.softmax(pre_activation)

    if activation is None:
        activation = pre_activation

    return activation