import tensorflow as tf
import tensorflow.contrib.slim as slim

from flags import FLAGS

import numpy as np

par_dict = np.load('./dataset/vgg.npy').item()

def deconv_layer(inputs, up_scale, out_channels, method='transpose', scope=''):
    with tf.variable_scope(scope):
        if method == 'transpose':
            net = slim.conv2d_transpose(inputs, out_channels, (up_scale, up_scale),
                                        (up_scale, up_scale), activation_fn=None, padding='VALID')
        elif method == 'transpose+conv':
            net = slim.conv2d_transpose(inputs, out_channels, (up_scale, up_scale),
                                        (up_scale, up_scale), activation_fn=None, padding='VALID')
            net = slim.conv2d(net, out_channels, (3, 3), (1, 1))
        elif method == 'transpose+conv+relu':
            net = slim.conv2d_transpose(inputs, out_channels, (up_scale, up_scale),
                                        (up_scale, up_scale), padding='VALID')
            net = slim.conv2d(net, out_channels, (3, 3), (1, 1))
        elif method == 'bilinear':
            h = tf.shape(inputs)[-3] * up_scale
            w = tf.shape(inputs)[-2] * up_scale
            net = tf.image.resize_images(inputs, [h, w])
        else:
            raise Exception('Unrecognized Deconvolution Method: %s' % method)
        return net

def conv_from_npy(inputs, scope):
    with tf.variable_scope(scope):
        kernel = tf.get_variable(name='kernel', initializer=tf.constant(par_dict[scope][0]))
        bias = tf.get_variable(name='bias', initializer=tf.constant(par_dict[scope][1]))

        conv_output = tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME')
        conv_bias = tf.nn.relu(tf.nn.bias_add(conv_output, bias))
    return conv_bias

def residual_bottleneck(inputs, out_channel=256, scope=''):
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope(scope):
        net = slim.stack(inputs, slim.conv2d, [(out_channel // 2, [1, 1]), (out_channel // 2, [3, 3]),
                                               (out_channel, [1, 1])], scope='conv')
        if depth_in != out_channel:
            inputs = slim.conv2d(inputs, out_channel, (1, 1), scope='res'.format(scope))

        net += inputs
        return net

def hourglass_bottleneck(inputs, out_channel=256, scope=''):
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    with tf.variable_scope(scope):
        with tf.variable_scope('residual_branch'):
            net_1 = slim.stack(inputs, slim.conv2d, [(out_channel // 2, [1, 1]), (out_channel // 2, [3, 3]),
                                                     (out_channel, [1, 1])], scope='conv')

        with tf.variable_scope('structural_branch'):
            net = slim.max_pool2d(inputs, [2, 2], 2, scope='maxpool_2x2')
            branch_1 = slim.conv2d(net, out_channel // 2, [3, 3], scope='conv2d_3x3_1')
            branch_2 = slim.conv2d(branch_1, out_channel // 4, [3, 3], scope='conv2d_3x3_2')
            branch_3 = slim.conv2d(branch_2, out_channel // 4, [3, 3], scope='conv2d_3x3_3')
            net = tf.concat(axis=3, values=[branch_1, branch_2, branch_3])

            net_2 = slim.conv2d_transpose(net, out_channel, (2, 2), (2, 2), activation_fn=None, padding='VALID')

        with tf.variable_scope('identity_mapping_branch'):
            if depth_in != out_channel:
                net_3 = slim.conv2d(inputs, out_channel, (1, 1), scope='res'.format(scope))
            else:
                net_3 = inputs

        net = net_1 + net_2 + net_3
        return net

def multi_scale_learning_module_inception(inputs, out_channel=256, scope=''):
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    min_channel = out_channel // 4
    with tf.variable_scope(scope):
        with tf.variable_scope('branch_0'):
            branch_0 = slim.conv2d(inputs, out_channel, [1, 1], scope='conv2d_1x1')
        with tf.variable_scope('branch_1'):
            branch_1 = slim.conv2d(inputs, out_channel // 2, [1, 1], scope='conv2d_1x1')
            branch_1 = slim.conv2d(branch_1, out_channel, [3, 3], scope='conv2d_3x3')
        with tf.variable_scope('branch_2'):
            filters = tf.get_variable(name='dilated_weight', shape=[3, 3, depth_in, out_channel],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.09))
            branch_2 = tf.nn.atrous_conv2d(inputs, filters=filters, rate=2, padding='SAME', name='dilated_conv2d_3x3')
        with tf.variable_scope('branch_3'):
            branch_3 = slim.avg_pool2d(inputs, [2, 2], 2, scope='avgpool_2x2')
            #branch_3 = slim.conv2d(branch_3, min_channel, [1, 1], scope='conv2d_1x1')
            h = tf.shape(inputs)[-3]
            w = tf.shape(inputs)[-2]
            branch_3 = tf.image.resize_images(branch_3, [h, w])

        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

        net = slim.conv2d(net, out_channel, [1, 1], scope='conv2d_1x1')

        if depth_in != out_channel:
            inputs = slim.conv2d(inputs, out_channel, (1, 1), scope='res'.format(scope))

        net += inputs

    return net

def hourglass_module(inputs, depth=4, deconv='transpose'):
    with tf.variable_scope('depth_{}'.format(depth)):
        # buttom up layer
        net = slim.max_pool2d(inputs, (2, 2), scope='pool')
        net = slim.stack(net, hourglass_bottleneck, [256, 256, 256], scope='bottom_up')

        #connecting layers
        if depth > 0:
            net = hourglass_module(net, depth=depth - 1, deconv=deconv)
        else:
            net = hourglass_bottleneck(net, out_channel=256, scope='connecting_layer')

        #top down layers
        net = hourglass_bottleneck(net, out_channel=256, scope='top_down')
        net = deconv_layer(net, 2, 256, method=deconv, scope='deconv_layer')

        #residual layers
        net += slim.stack(inputs, residual_bottleneck, [256, 256, 256], scope='res_block')

        return net

def stacked_hourglass_network(inputs, n_stack=4, deconv='transpose', regression_channels=19):
    with slim.arg_scope(arg_scope()):
        with tf.variable_scope('init_prediction_module'):
            #feature extraction(preprocessing)
            with tf.variable_scope('preprocessing_module'):
                # D1
                net = slim.conv2d(inputs, 64, (7, 7), 2, scope='conv1')
                net = residual_bottleneck(net, out_channel=128, scope='res_conv_1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                #D2 feature extraction(multi_scale feature learning)
                net = slim.stack(net, multi_scale_learning_module_inception, [128, 128, 256], scope='res_conv_2')

            multi_scale_feature = net

            #hourglass module
            hg_out = [None] * n_stack
            hg_conv_out = [None] * n_stack
            hg_conv_drop_out = [None] * n_stack
            hg_loss_out = [None] * n_stack
            stack_inputs = [None] * n_stack
            with tf.variable_scope('hourglass_module'):
                with tf.variable_scope('stack_00'):
                    hg_out[0] = hourglass_module(net, depth=4, deconv=deconv)
                    hg_conv_out[0] = slim.conv2d(hg_out[0], 256, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
                                                 normalizer_fn=slim.batch_norm, scope='hg_conv')
                    hg_conv_drop_out[0] = slim.conv2d(hg_conv_out[0], 256, kernel_size=1, stride=1,
                                                      activation_fn=tf.nn.relu,
                                                      normalizer_fn=None, scope='hg_conv_drop')
                    hg_conv_drop_out[0] = tf.layers.dropout(hg_conv_drop_out[0], rate=0.1, training=True)
                    hg_loss_out[0] = slim.conv2d(hg_conv_out[0], regression_channels, 1, 1, activation_fn=tf.nn.relu,
                                                 normalizer_fn=None, scope='hg_conv_loss')
                    add_feats = slim.conv2d(hg_loss_out[0], 256, 1, 1, activation_fn=tf.nn.relu, normalizer_fn=None,
                                            scope='hg_conv_add')
                    stack_inputs[0] = tf.add(hg_conv_drop_out[0], add_feats)
                for i in range(1, n_stack - 1):
                    with tf.variable_scope('stack_%02d' % i):
                        hg_out[i] = hourglass_module(stack_inputs[i - 1], depth=4, deconv=deconv)
                        hg_conv_out[i] = slim.conv2d(hg_out[i], 256, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
                                                     normalizer_fn=slim.batch_norm, scope='hg_conv')
                        hg_conv_drop_out[i] = slim.conv2d(hg_conv_out[i], 256, kernel_size=1, stride=1,
                                                          activation_fn=tf.nn.relu,
                                                          normalizer_fn=None, scope='hg_conv_drop')
                        hg_conv_drop_out[i] = tf.layers.dropout(hg_conv_drop_out[i], rate=0.1, training=True)
                        hg_loss_out[i] = slim.conv2d(hg_conv_out[i], regression_channels, 1, 1,
                                                     activation_fn=tf.nn.relu,
                                                     normalizer_fn=None, scope='hg_conv_loss')
                        add_feats = slim.conv2d(hg_loss_out[i], 256, 1, 1, activation_fn=tf.nn.relu, normalizer_fn=None,
                                                scope='hg_conv_add')
                        stack_inputs[i] = tf.add_n([hg_conv_drop_out[i], add_feats, stack_inputs[i - 1]])

                with tf.variable_scope('stack_%02d' % (n_stack - 1)):
                    hg_out[n_stack - 1] = hourglass_module(stack_inputs[n_stack - 2], depth=4, deconv=deconv)
                    hg_conv_out[n_stack - 1] = slim.conv2d(hg_out[n_stack - 1], 256, kernel_size=1, stride=1,
                                                           activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                                           scope='hg_conv')
                    hg_conv_drop_out[n_stack - 1] = tf.layers.dropout(hg_conv_out[n_stack - 1], rate=0.1, training=True)
                    hg_loss_out[n_stack - 1] = slim.conv2d(hg_conv_drop_out[n_stack - 1], regression_channels, 1, 1,
                                                           activation_fn=tf.nn.relu, normalizer_fn=None,
                                                           scope='hg_conv_loss')
                attention_feature = hg_conv_drop_out[n_stack - 1]
            with tf.variable_scope('postprocessing_module'):
                # top down layers
                net = deconv_layer(hg_conv_drop_out[n_stack - 1], up_scale=4, out_channels=regression_channels, method=deconv)
                net = slim.conv2d(net, regression_channels, 1, scope='conv_final')

                #regression output
                net = slim.conv2d(net, regression_channels, 1, activation_fn=None, normalizer_fn=None,
                                  scope='conv_output')

        init_prediction = net

        with tf.variable_scope('attention_refinement_module'):
            net = inputs
            with tf.variable_scope('conv_1'):
                net = conv_from_npy(net, 'conv1_1')
                net = conv_from_npy(net, 'conv1_2')
                net = slim.max_pool2d(net, [2, 2], scope='pool')
            with tf.variable_scope('conv_2'):
                net = conv_from_npy(net, 'conv2_1')
                net = conv_from_npy(net, 'conv2_2')
                net = slim.max_pool2d(net, [2, 2], scope='pool')
            with tf.variable_scope('conv_3'):
                '''
                attention_feature_64x64 = attention_feature
                summarized_feature = slim.conv2d(attention_feature_64x64, 1, 1, activation_fn=tf.nn.relu,
                                                 normalizer_fn=None, scope='summarized_conv')
                attention_mask_64x64 = tf.nn.softmax(summarized_feature)
                net = tf.multiply(net, attention_mask_64x64)
                '''
                net = conv_from_npy(net, 'conv3_1')
                net = conv_from_npy(net, 'conv3_2')
                net = conv_from_npy(net, 'conv3_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool')
            with tf.variable_scope('conv_4'):
                '''
                attention_feature_32x32 = slim.max_pool2d(attention_feature, [2, 2], scope='pool')
                summarized_feature = slim.conv2d(attention_feature_32x32, 1, 1, activation_fn=tf.nn.relu,
                                                 normalizer_fn=None, scope='summarized_conv')
                attention_mask_32x32 = tf.nn.softmax(summarized_feature)
                net = tf.multiply(net, attention_mask_32x32)
                '''
                net = conv_from_npy(net, 'conv4_1')
                net = conv_from_npy(net, 'conv4_2')
                net = conv_from_npy(net, 'conv4_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool')
            with tf.variable_scope('conv_5'):
                '''
                attention_feature_16x16 = slim.max_pool2d(attention_feature_32x32, [2, 2], scope='pool')
                summarized_feature = slim.conv2d(attention_feature_16x16, 1, 1, activation_fn=tf.nn.relu,
                                                 normalizer_fn=None, scope='summarized_conv')
                attention_mask_16x16 = tf.nn.softmax(summarized_feature)
                net = tf.multiply(net, attention_mask_16x16)
                '''
                net = conv_from_npy(net, 'conv5_1')
                net = conv_from_npy(net, 'conv5_2')
                net = conv_from_npy(net, 'conv5_3')
                net = slim.max_pool2d(net, [2, 2], scope='pool')
            with tf.variable_scope('fully_connected'):
                net = tf.layers.flatten(net)
                net = slim.stack(net, slim.fully_connected, [4096, 2048], scope='fully_connected')
                net = slim.fully_connected(net, FLAGS.landmarkNum * 2, activation_fn=None, scope='final_output')

        ref_prediction = net
        # with tf.variable_scope('discriminator_module', reuse=tf.AUTO_REUSE):
        #     inputs_fake = tf.concat([inputs, net], 3)
        #     inputs_real = tf.concat([inputs, heatmap], 3)
        #     discriminator_output_fake = discriminator(inputs=inputs_fake, n_stack=FLAGS.num_stack)
        #     discriminator_output_real = discriminator(inputs=inputs_real, n_stack=FLAGS.num_stack)

    return hg_loss_out, init_prediction, ref_prediction#, discriminator_output_fake, discriminator_output_real

def discriminator(inputs, n_stack=FLAGS.num_stack, deconv='transpose', regression_channels=68):
    with slim.arg_scope(arg_scope()):
            # D1
        net = slim.conv2d(inputs, 64, (7, 7), 2, scope='conv1_dis')
        net = residual_bottleneck(net, out_channel=128, scope='res_conv1_dis')
        net = slim.max_pool2d(net, [2, 2], scope='pool1_dis')

            # D2 feature extraction(multi_scale feature learning)
        net = slim.stack(net, residual_bottleneck, [128, 128, 256], scope='res_conv2_dis')
            # net = slim.stack(net, multi_scale_learning_module_inception, [128, 128, 256], scope='res_conv_dis')
        with tf.variable_scope('discri_module'):
                # hourglass module_1stack
            with tf.variable_scope('stack_dis'):
                net = hourglass_module(net, depth=1, deconv=deconv)
            net = deconv_layer(net, up_scale=4, out_channels=regression_channels, method=deconv)
            net = slim.conv2d(net, regression_channels, 1)

                # regression output
            net = slim.conv2d(net, regression_channels, 1, activation_fn=None, normalizer_fn=None)

    return net

    #return hg_loss_out, init_prediction#, ref_prediction

def arg_scope(is_training=True,
              weight_decay=0.00004,
              stddev=0.09,
              batch_norm_decay=0.9997,
              batch_norm_epsilon=0.001):
  batch_norm_params = {
      'center': True,
      'scale': True,
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
  }
  if is_training is not None:
    batch_norm_params['is_training'] = is_training

  # Set weight_decay for weights in Conv layers.
  weights_init = tf.truncated_normal_initializer(stddev=stddev)
  regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=weights_init,
                      weights_regularizer=regularizer,
                      activation_fn=tf.nn.relu):
      with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
          with slim.arg_scope([slim.batch_norm], **batch_norm_params):
              with slim.arg_scope([slim.max_pool2d], padding='SAME') as sc:
                  return sc
