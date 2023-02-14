# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements Small CNN model in keras using tensorflow backend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, ZeroPadding2D
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.data.experimental import AutoShardPolicy

from utils.resnet50 import conv_block, identity_block

from tensorflow.keras.models import load_model

import math

from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.tf_export import keras_export


from utils import  utils
import os
from os import path

BASE_WEIGHTS_PATH = 'https://storage.googleapis.com/keras-applications/'

WEIGHTS_HASHES = {
    'b0': ('902e53a9f72be733fc0bcb005b3ebbac',
           '50bc09e76180e00e4465e1a485ddc09d'),
    'b1': ('1d254153d4ab51201f1646940f018540',
           '74c4e6b3e1f6a1eea24c589628592432'),
    'b2': ('b15cce36ff4dcbd00b6dd88e7857a6ad',
           '111f8e2ac8aa800a7a99e3239f7bfb39'),
    'b3': ('ffd1fdc53d0ce67064dc6a9c7960ede0',
           'af6d107764bb5b1abb91932881670226'),
    'b4': ('18c95ad55216b8f92d7e70b3a046e2fc',
           'ebc24e6d6c33eaebbd558eafbeedf1ba'),
    'b5': ('ace28f2a6363774853a83a0b21b9421a',
           '38879255a25d3c92d5e44e04ae6cec6f'),
    'b6': ('165f6e37dce68623721b423839de8be5',
           '9ecce42647a20130c1f39a5d4cb75743'),
    'b7': ('8c03f828fec3ef71311cd463b6759d99',
           'cbcfe4450ddf6f3ad90b1b398090fe4a'),
}

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

layers = VersionAwareLayers()

class EfficientNetGrow(object):
  """Small convnet that matches sklearn api.

  Implements model from
  https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
  Adapts for inputs of variable size, expects data to be 4d tensor, with
  # of obserations as first dimension and other dimensions to correspond to
  length width and # of channels in image.
  """

  def __init__(self,
               random_state=1,
               epochs=200, # was 200
               batch_size=64,
               solver='Adam',
               learning_rate=0.0001,
               lr_decay=0.,
               augmentation=0):
    # params
    self.solver = solver
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.lr_decay = lr_decay
    # data
    self.encode_map = None
    self.decode_map = None
    self.model = None
    self._feature_model = None
    self.random_state = random_state
    self.n_classes = None

    self.data_augmentation = augmentation != 0

    self.id = None
    self.mode = "cifar"
    self.init_kernels = None

    self.image_path_mode = False

    self.model_path_test = "EfficientNetB0.h5"
    self.model_path = "EfficientNetB0_200epoch.h5"


  def EfficientNet(self,
          width_coefficient,
          depth_coefficient,
          default_size,
          dropout_rate=0.2,
          drop_connect_rate=0.2,
          depth_divisor=8,
          activation='swish',
          blocks_args='default',
          model_name='efficientnet',
          include_top=True,
          weights=None,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          classifier_activation='softmax'):
    """Instantiates the EfficientNet architecture using given scaling coefficients.
    Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
        https://arxiv.org/abs/1905.11946) (ICML 2019)
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Arguments:
      width_coefficient: float, scaling coefficient for network width.
      depth_coefficient: float, scaling coefficient for network depth.
      default_size: integer, default input image size.
      dropout_rate: float, dropout rate before final classifier layer.
      drop_connect_rate: float, dropout rate at skip connections.
      depth_divisor: integer, a unit of network width.
      activation: activation function.
      blocks_args: list of dicts, parameters to construct block modules.
      model_name: string, model name.
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
          (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False.
          It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.
    Returns:
      A `keras.Model` instance.
    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """
    if blocks_args == 'default':
      blocks_args = DEFAULT_BLOCKS_ARGS

    if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
      raise ValueError('The `weights` argument should be either '
                       '`None` (random initialization), `imagenet` '
                       '(pre-training on ImageNet), '
                       'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
      raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                       ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
      input_shape,
      default_size=default_size,
      min_size=32,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

    if input_tensor is None:
      img_input = layers.Input(shape=input_shape)
    else:
      if not backend.is_keras_tensor(input_tensor):
        img_input = layers.Input(tensor=input_tensor, shape=input_shape)
      else:
        img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def round_filters(filters, divisor=depth_divisor):
      """Round number of filters based on depth multiplier."""
      filters *= width_coefficient
      new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
      # Make sure that round down does not go down by more than 10%.
      if new_filters < 0.9 * filters:
        new_filters += divisor
      return int(new_filters)

    def round_repeats(repeats):
      """Round number of repeats based on depth multiplier."""
      return int(math.ceil(depth_coefficient * repeats))

    # Build stem
    x = img_input
    x = layers.Rescaling(1. / 255.)(x)
    x = layers.Normalization(axis=bn_axis)(x)

    x = layers.ZeroPadding2D(
      padding=imagenet_utils.correct_pad(x, 3),
      name='stem_conv_pad')(x)
    x = layers.Conv2D(
      round_filters(32),
      3,
      strides=2,
      padding='valid',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
      assert args['repeats'] > 0
      # Update block input and output filters based on depth multiplier.
      args['filters_in'] = round_filters(args['filters_in'])
      args['filters_out'] = round_filters(args['filters_out'])

      for j in range(round_repeats(args.pop('repeats'))):
        # The first block needs to take care of stride and filter size increase.
        if j > 0:
          args['strides'] = 1
          args['filters_in'] = args['filters_out']
        x = self.block(
          x,
          activation,
          drop_connect_rate * b / blocks,
          name='block{}{}_'.format(i + 1, chr(j + 97)),
          **args)
        b += 1

    # Build top
    x = layers.Conv2D(
      round_filters(1280),
      1,
      padding='same',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name='top_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)
    if include_top:
      x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
      if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='top_dropout')(x)
      imagenet_utils.validate_activation(classifier_activation, weights)
      x = layers.Dense(
        classes,
        activation=classifier_activation,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        name='predictions')(x)
    else:
      if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
      elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
      inputs = layer_utils.get_source_inputs(input_tensor)
    else:
      inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    # Load weights.
    if weights == 'imagenet':
      if include_top:
        file_suffix = '.h5'
        file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
      else:
        file_suffix = '_notop.h5'
        file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
      file_name = model_name + file_suffix
      weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHTS_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
      model.load_weights(weights_path)
    elif weights is not None:
      model.load_weights(weights)
    return model

  def block(self,
            inputs,
            activation='swish',
            drop_rate=0.,
            name='',
            filters_in=32,
            filters_out=16,
            kernel_size=3,
            strides=1,
            expand_ratio=1,
            se_ratio=0.,
            id_skip=True):
    """An inverted residual block.
    Arguments:
        inputs: input tensor.
        activation: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
    Returns:
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
      x = layers.Conv2D(
        filters,
        1,
        padding='same',
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'expand_conv')(
        inputs)
      x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
      x = layers.Activation(activation, name=name + 'expand_activation')(x)
    else:
      x = inputs

    # Depthwise Convolution
    if strides == 2:
      x = layers.ZeroPadding2D(
        padding=imagenet_utils.correct_pad(x, kernel_size),
        name=name + 'dwconv_pad')(x)
      conv_pad = 'valid'
    else:
      conv_pad = 'same'
    x = layers.DepthwiseConv2D(
      kernel_size,
      strides=strides,
      padding=conv_pad,
      use_bias=False,
      depthwise_initializer=CONV_KERNEL_INITIALIZER,
      name=name + 'dwconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = layers.Activation(activation, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
      filters_se = max(1, int(filters_in * se_ratio))
      se = layers.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
      se = layers.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
      se = layers.Conv2D(
        filters_se,
        1,
        padding='same',
        activation=activation,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'se_reduce')(
        se)
      se = layers.Conv2D(
        filters,
        1,
        padding='same',
        activation='sigmoid',
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + 'se_expand')(se)
      x = layers.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = layers.Conv2D(
      filters_out,
      1,
      padding='same',
      use_bias=False,
      kernel_initializer=CONV_KERNEL_INITIALIZER,
      name=name + 'project_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if id_skip and strides == 1 and filters_in == filters_out:
      if drop_rate > 0:
        x = layers.Dropout(
          drop_rate, noise_shape=(None, 1, 1, 1), name=name + 'drop')(x)
      x = layers.add([x, inputs], name=name + 'add')
    return x

  def EfficientNetB0(self,
                     include_top=True,
                     weights=None,
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                     classifier_activation='softmax',
                     **kwargs):
    return self.EfficientNet(
      1.0,
      1.0,
      224,
      0.2,
      model_name='efficientnetb0',
      include_top=include_top,
      weights=weights,
      input_tensor=input_tensor,
      input_shape=input_shape,
      pooling=pooling,
      classes=classes,
      classifier_activation=classifier_activation,
      **kwargs)

  """ Template starts """

  def build_model(self, input_shape, y, n_cell, n_layer_per_cell, n_kernel_1st_layer, _gpus=[0], image_path_mode=False):
    """
        Define a resnet model based on cell/decimation, layer in between, and number of kernels of the 1st layer.
        Between each each decimation /conv_block, it's a cell with n_layer/conv_block, each with same amount of kernels
        The default decimation is 3*3 maxpooling, TODO: later we can change this as well
        After decimation, the next cell will have 2x kernels each layer.
        Caution: input should not apply decimation to a level output is smaller than 1*1
        :param X: input np array
        :param n_cell: how many decimation
        :param n_layer_per_cell:
        :param n_kernel_1st_layer:
        :return:
        """
    # assumes that data axis order is same as the backend

    y_mat = self.create_y_mat(y)
    # fixing cell = 3, n_kernel_1st_layer = 64

    self.image_path_mode = image_path_mode

    # self._feature_model = self.EfficientNetB0(include_top=False, input_shape=input_shape, classes=self.n_classes, weights=None)

    if len(_gpus)==1:
      self.model = self.EfficientNetB0(input_shape=input_shape, classes=self.n_classes, weights=None)
    else:
      # with tf.device('/cpu:0'):
      #   self.ori_model = self.resnet_v1(input_shape=input_shape, depth=n_layer_per_cell, kernels=n_kernel_1st_layer, cell=n_cell, num_classes=self.n_classes)
      # self.model = keras.utils.multi_gpu_model(self.ori_model, gpus=_gpus)
      """multi-gpu-model deprecated"""
      strategy = tf.distribute.MirroredStrategy()
      print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
      with strategy.scope():
        self.model = self.EfficientNetB0(input_shape=input_shape, classes=self.n_classes, weights=None)

    # n_kernel = n_kernel_1st_layer
    try:
      optimizer = getattr(keras.optimizers, self.solver)
    except:
      raise NotImplementedError('optimizer not implemented in keras')
    # All optimizers with the exception of nadam take decay as named arg
    # try:
    #   opt = optimizer(lr=self.learning_rate, decay=self.lr_decay)
    # except:
    #   opt = optimizer(lr=self.learning_rate, schedule_decay=self.lr_decay)

    # self.model.compile(loss='categorical_crossentropy',
    #               optimizer=opt,
    #               metrics=['accuracy'])
    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='categorical_crossentropy'),
        metrics=['accuracy'])

    self.model.summary()
    # Save initial weights so that model can be retrained with same
    # initialization
    self.initial_weights = copy.deepcopy(self.model.get_weights())

    self.id = "EfficientNetB0_B{}".format(self.batch_size)
    if self.data_augmentation:
        self.id += "DataAug"

  def create_y_mat(self, y):
    y_encode = self.encode_y(y)
    y_encode = np.reshape(y_encode, (len(y_encode), 1))
    y_mat = keras.utils.to_categorical(y_encode, self.n_classes)
    return y_mat

  # Add handling for classes that do not start counting from 0
  def encode_y(self, y):
    if self.encode_map is None:
      self.classes_ = sorted(list(set(y)))
      self.n_classes = len(self.classes_)
      print("Building models with {} classes".format(self.n_classes))
      self.encode_map = dict(zip(self.classes_, range(len(self.classes_))))
      self.decode_map = dict(zip(range(len(self.classes_)), self.classes_))
    mapper = lambda x: self.encode_map[x]
    transformed_y = np.array(list(map(mapper, y)))
    return transformed_y

  def decode_y(self, y):
    mapper = lambda x: self.decode_map[x]
    transformed_y = np.array(list(map(mapper, y)))
    return transformed_y

  def lr_schedule(self, epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """

    lr = self.learning_rate
    lr = lr * self.batch_size / 64  # linear scale to batch sizes

    if epoch > 23:
      lr *= 0.5e-3
    elif epoch > 20:
      lr *= 1e-3
    elif epoch > 15:
      lr *= 1e-2
    elif epoch > 10:
      lr *= 1e-1
    print('Epoch: {} Learning rate: {}'.format(epoch, lr))
    return lr

  def fit(self, X_train, y_train, X_val, y_val, sample_weight=None):

    y_mat = self.create_y_mat(y_train)
    y_val_mat = self.create_y_mat(y_val)

    if utils.TEST:
        if path.exists(self.model_path):
            self.load_model(self.model_path)
            return None
        if path.exists(self.model_path_test):
            self.load_model(self.model_path_test)
            return None

    # We don't want incremental fit so reset learning rate and weights
    K.set_value(self.model.optimizer.lr, self.learning_rate)
    self.model.set_weights(self.initial_weights)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=50)
    # reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=30, factor=0.1, verbose=1)
    # reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=20, factor=0.1, verbose=1)

    lr_scheduler = keras.callbacks.LearningRateScheduler(self.lr_schedule)
      
    #
    # reduceLR = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                                              cooldown=0,
    #                                              patience=5,
    #                                              verbose=1,
    #                                              min_lr=0.5e-6)
  
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=20, factor=0.1, verbose=1)

    callback_list = [
                      # reduceLR,
                      lr_scheduler,
                      # es
                     ]

    # filepath = self.id + "_TrainingSize_{}_LRSchedule.h5".format(X_train.shape[0])

    if self.image_path_mode:

      print("Training Actually started")
      t_start = time.time()
      history = self.model.fit(X_train,
                               # steps_per_epoch=X_train.samples // self.batch_size,
                                         validation_data=X_val,
                                         epochs=self.epochs, verbose=1, shuffle=True,
                                         callbacks=callback_list)
      t_end = time.time()
      print("Training Time: {}".format(t_end - t_start))

    elif self.data_augmentation:
      datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=15,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
      # Compute quantities required for featurewise normalization
      # (std, mean, and principal components if ZCA whitening is applied).
      datagen.fit(X_train)
      # Fit the model on the batches generated by datagen.flow().

      print("Training Actually started")
      t_start = time.time()

      history = self.model.fit_generator(datagen.flow(X_train, y_mat, batch_size=self.batch_size),
                                         validation_data=(X_val, y_val_mat),
                                         epochs=self.epochs, verbose=1, workers=8,
                                         steps_per_epoch=len(X_train) / self.batch_size,
                                         callbacks=callback_list)
      t_end = time.time()
      print("Training Time: {}".format(t_end - t_start))
    else:
      print("Training Actually started")
      t_start = time.time()
      history = self.model.fit(
        X_train,
        y_mat,
        batch_size=self.batch_size,
        epochs=self.epochs,
        # shuffle=True,
        shuffle="batch", # for hd5 input
        sample_weight=sample_weight,
        verbose=1,
        validation_data=(X_val, y_val_mat),
        callbacks=callback_list)

      t_end = time.time()
      print("Training Time: {}".format(t_end - t_start))

    # self.save(filepath)
    if utils.TEST:
        self.save(self.model_path_test)
    else:
        self.save(self.model_path)

    return history

  def predict(self, X_val):
    # with tf.device("cpu:0"):
    predicted = self.model.predict(X_val, batch_size=self.batch_size, verbose=1)
    return predicted

  def score(self, X_val, val_y, _gpus=[0]):
    y_mat = self.create_y_mat(val_y)
    # if len(_gpus) > 1:
    #   val_acc = self.ori_model.evaluate(X_val, y_mat, verbose=0)[1]
    # else:
    #   val_acc = self.model.evaluate(X_val, y_mat, verbose=0)[1]
    if self.image_path_mode:
      val_acc = self.model.evaluate(X_val, batch_size=self.batch_size, verbose=1)[1]
    else:
      val_acc = self.model.evaluate(X_val, y_mat, batch_size=self.batch_size, verbose=1)[1]
    return val_acc

  def decision_function(self, X):
    return self.predict(X)

  def transform(self, X):
    # X = tf.convert_to_tensor(X, dtype=tf.float32)
    extractor = keras.Model(inputs=self.model.inputs,
                            outputs=self.model.get_layer('avg_pool').output)
    extractor.summary()
    # print(X.shape)
    # features = np.array(extractor(X))
    with tf.device("cpu:0"):
        features = extractor.predict(X, batch_size=self.batch_size, verbose=1)
    # features = self._feature_model.predict(X, batch_size=self.batch_size, verbose=1)
    # print(features.shape)
    return features

  # def transform(self, X):
  #   # X = tf.convert_to_tensor(X, dtype=tf.float32)
  #   # extractor = keras.Model(inputs=self.model.inputs,
  #   #                         outputs=self.model.get_layer('dense_1').input)
  #   # self._feature_model.summary()
  #   # print(X.shape)
  #   # features = np.array(extractor(X))
  #   features = self._feature_model.predict(X, batch_size=self.batch_size, verbose=1)
  #   # print(features.shape)
  #   return features
  # def transform(self, X):
  #   model = self.model
  #   inp = [model.input]
  #   activations = []
  #
  #   # Get activations of the first dense layer.
  #   output = [layer.output for layer in model.layers if
  #             layer.name == 'dense1'][0]
  #   func = K.function(inp + [K.learning_phase()], [output])
  #   for i in range(int(X.shape[0]/self.batch_size) + 1):
  #     minibatch = X[i * self.batch_size
  #                   : min(X.shape[0], (i+1) * self.batch_size)]
  #     list_inputs = [minibatch, 0.]
  #     # Learning phase. 0 = Test mode (no dropout or batch normalization)
  #     layer_output = func(list_inputs)[0]
  #     activations.append(layer_output)
  #   output = np.vstack(tuple(activations))
  #   return output

  def get_params(self, deep = False):
    params = {}
    params['solver'] = self.solver
    params['epochs'] = self.epochs
    params['batch_size'] = self.batch_size
    params['learning_rate'] = self.learning_rate
    params['weight_decay'] = self.lr_decay
    if deep:
      return copy.deepcopy(params)
    return copy.copy(params)

  def set_params(self, **parameters):
    for parameter, value in parameters.items():
      setattr(self, parameter, value)
    return self

  def save(self, filepath):
    print("Saving Model...")
    self.model.save(filepath)

  def save_and_delete(self, filepath):
    print("Saving Model...")
    self.model.save(filepath)
    del self.model
    keras.backend.clear_session()

  def load_model(self, filepath):
    print("Loading Model...")
    self.model = load_model(filepath)