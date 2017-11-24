import sys
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v2

class DoubleLayer_Concat_Model:
  def __init__(self, num_classes = 2):
    self.conv1_depth = 2048
    self.num_classes = num_classes

  def forward(self, feat_map0, feat_map1, is_training=None, scope=None, reuse=None):
    with tf.variable_scope(scope, 'Linear', reuse=reuse) as sc:
      end_points_collection = sc.original_name_scope + '_end_points'
      with arg_scope([slim.conv2d], outputs_collections = end_points_collection):
        if is_training is not None:
          bn_scope = arg_scope([layers.batch_norm], is_training=is_training)
        else:
          bn_scope = arg_scope([])
        with bn_scope:
          net = tf.concat([feat_map0, feat_map1], axis=-1, name="mix_feature")
          net = slim.conv2d(net, self.num_classes, [1, 1], stride=1, activation_fn=None, normalizer_fn=None, scope='last_conv')
          net = tf.squeeze(net, [1, 2], name='logit')
    end_points = utils.convert_collection_to_dict(end_points_collection)
    return net, end_points

class CNN_Model:
  def __init__(self):
    pass

  def forward(self, inputs, reuse=None, is_training=None):
    net, endpts = resnet_v2.resnet_v2_50(inputs=inputs, reuse=reuse, is_training=is_training)
    return net, endpts

class Shared_Double_Net:
  def __init__(self, num_classes = 2):
    self.num_classes = num_classes
    self.cnn_model = CNN_Model()
    self.concat_model = DoubleLayer_Concat_Model(num_classes)

  def forward(self, image1, image2, is_training=None, scope=None, reuse=None):
    feat_map1, endpts1 = self.cnn_model.forward(inputs=image1, is_training=is_training)
    feat_map2, endpts2 = self.cnn_model.forward(inputs=image2, is_training=is_training, reuse=True)
    logit, endpts = self.concat_model.forward(feat_map1, feat_map2, is_training=is_training, scope='Linear')
    return logit, endpts1, endpts2, endpts
  def arg_scope(self, weight_decay=0.0005):
    return resnet_utils.resnet_arg_scope(weight_decay=weight_decay)
