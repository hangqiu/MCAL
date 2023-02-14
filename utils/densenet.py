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

import keras
import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential

from keras.models import Model
from keras.layers import Input, concatenate, ZeroPadding2D, Concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization

import os
import numpy as np
import tensorflow as tf
from keras.models import load_model

from utils.custom_layers import Scale

from keras.preprocessing.image import ImageDataGenerator
import time

from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

class DenseNetGrow(object):
  """Small convnet that matches sklearn api.

  Implements model from
  https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
  Adapts for inputs of variable size, expects data to be 4d tensor, with
  # of obserations as first dimension and other dimensions to correspond to
  length width and # of channels in image.
  """

  def __init__(self,
               random_state=1,
               epochs=200,
               batch_size=32,
               solver='adam',
               learning_rate=0.001,
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
    self.random_state = random_state
    self.n_classes = None

    self.data_augmentation = augmentation != 0

    self.id = None

    self.cell = None
    self.kernel = None
    self.gpus = None

  def build_model(self, X, y=None, cell=4, layer=32, kernel=64, _gpus=[0]):
    """use layer param as growth rate"""
    self.cell = cell
    self.kernel = kernel
    self.gpus = _gpus
    input_shape = X.shape[1:]
    if y is not None:
      _ = self.create_y_mat(y)
    
    # blocks = [6, 12, 24, 16] # DenseNet 121
    dropout_rate = None
    blocks = [6, 12, 24, 16]
    if cell==3:
      # blocks = [6, 6, 6]
      blocks = [12, 24, 12]
      dropout_rate = 0.2



    input_image = Input(shape=input_shape)
    # base_model = DenseNet121(include_top=False, weights=None,input_shape=input_shape,pooling='avg')
    base_model = self.DenseNetModel(blocks, 
                                    include_top=False, 
                                    weights=None,
                                    input_shape=input_shape,
                                    pooling='avg',
                                    nb_filters=kernel,
                                    growth_rate=layer,
                                    dropout_rate=dropout_rate)
    x = base_model(input_image)
    outputs = Dense(self.n_classes,activation='softmax',kernel_initializer='he_normal')(x)
    if len(_gpus)==1:
      self.model = Model(inputs=input_image, outputs=outputs)
    else:
      with tf.device('/cpu:0'):
        self.ori_model = Model(inputs=input_image, outputs=outputs)
      self.model = keras.utils.multi_gpu_model(self.ori_model, gpus=_gpus)

    
    try:
      optimizer = getattr(keras.optimizers, self.solver)
    except:
      raise NotImplementedError('optimizer not implemented in keras')
    # All optimizers with the exception of nadam take decay as named arg
    try:
      opt = optimizer(lr=self.learning_rate, decay=self.lr_decay)
    except:
      opt = optimizer(lr=self.learning_rate, schedule_decay=self.lr_decay)

    self.model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    self.model.summary()
    # Save initial weights so that model can be retrained with same
    # initialization
    self.initial_weights = copy.deepcopy(self.model.get_weights())

    self.id = "DenseNetC{}K{}G{}B{}".format(cell, kernel, kernel, self.batch_size)
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
    lr = lr * self.batch_size / 32 # linear scale to batch sizes
    lr /= 8
    if epoch > 180: # was 180
      lr *= 0.5e-3 # was 0.5e-3
    elif epoch > 160: # was 160
      lr *= 1e-3 # was 1e-3
    elif epoch > 120: # was 120
      lr *= 1e-2 # was 1e-2
    elif epoch > 80: # was 80
      lr *= 1e-1 # was 1e-1
    print('Epoch: {} Learning rate: {}'.format(epoch, lr))
    return lr

  def fit(self, X_train, y_train, X_val, y_val, sample_weight=None):
    y_mat = self.create_y_mat(y_train)
    y_val_mat = self.create_y_mat(y_val)

    if self.model is None:
      self.build_model(X_train, y_train, cell=self.cell, kernel=self.kernel, _gpus=self.gpus)

    # We don't want incremental fit so reset learning rate and weights
    K.set_value(self.model.optimizer.lr, self.learning_rate)
    self.model.set_weights(self.initial_weights)

    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=20, factor=0.1, verbose=1)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=50)
    lr_scheduler = keras.callbacks.LearningRateScheduler(self.lr_schedule)
    
    callback_list = [
                      # reduceLR,
                      lr_scheduler,
                      # es,
                     ]

    if self.data_augmentation:
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
        width_shift_range=5./32., # was 0.1
        # randomly shift images vertically
        height_shift_range=5./32., # was 0.1
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
                                         validation_data=(X_val, y_val_mat), shuffle=True,
                                         epochs=self.epochs, verbose=1, workers=8,
                                         steps_per_epoch=len(X_train) / self.batch_size,
                                         callbacks=callback_list)
      t_end = time.time()
      print("Training Time: {}".format(t_end - t_start))
    else:
      t_start = time.time()
      history = self.model.fit(
          X_train,
          y_mat,
          batch_size=self.batch_size,
          epochs=self.epochs,
          shuffle=True,
          sample_weight=sample_weight,
          verbose=1,
          validation_data=(X_val, y_val_mat),
          callbacks=callback_list)
      t_end = time.time()
      print("Training Time: {}".format(t_end - t_start))


    # self.save_and_delete()
    return history

  def predict(self, X_val):
    predicted = self.model.predict(X_val)
    return predicted

  def score(self, X_val, val_y):
    # if self.model == None:
    #   self.build_model(X_val, val_y, cell=self.cell, kernel=self.kernel, _gpus=[0])
    #   self.load_model()
    y_mat = self.create_y_mat(val_y)
    # if len(self.gpus) > 1:
    #   val_acc = self.ori_model.evaluate(X_val, y_mat, verbose=0)[1]
    # else:
    val_acc = self.model.evaluate(X_val, y_mat, verbose=0)[1]
    # self.delete()
    return val_acc

  def decision_function(self, X):
    return self.predict(X)

  def transform(self, X):
    model = self.model
    inp = [model.input]
    activations = []

    # Get activations of the first dense layer.
    output = [layer.output for layer in model.layers if
              layer.name == 'dense1'][0]
    func = K.function(inp + [K.learning_phase()], [output])
    for i in range(int(X.shape[0]/self.batch_size) + 1):
      minibatch = X[i * self.batch_size
                    : min(X.shape[0], (i+1) * self.batch_size)]
      list_inputs = [minibatch, 0.]
      # Learning phase. 0 = Test mode (no dropout or batch normalization)
      layer_output = func(list_inputs)[0]
      activations.append(layer_output)
    output = np.vstack(tuple(activations))
    return output

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

  def save(self, filepath=None):
    print("Saving Model...")
    if filepath is None:
      filepath = self.id + ".h5"
    self.model.save(filepath)

  def save_and_delete(self, filepath=None):
    print("Saving Model...")
    if filepath is None:
      filepath = self.id + ".h5"
    self.model.save(filepath)
    self.delete()
  
  def delete(self):
    del self.model
    self.model = None
    keras.backend.clear_session()

  def load_model(self, filepath=None):
    if filepath is None:
      filepath = self.id + ".h5"
    print("Loading Model...")
    self.model = load_model(filepath)

  def DenseNetModel(self,
             blocks,
             include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             nb_filters=64,
             growth_rate=32,
             dropout_rate=None,
             **kwargs):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `'channels_last'` data format)
            or `(3, 224, 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    conv_size = 3
    if input_shape[0] != 32:
      conv_size = 7

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(nb_filters, conv_size, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    for i in range(len(blocks)):
      x = dense_block(x, blocks[i], name='conv{}'.format(i+2), growth_rate=growth_rate, dropout_rate=dropout_rate)
      if i != len(blocks)-1:
        x = transition_block(x, 0.5, name='pool{}'.format(i+2), dropout_rate=dropout_rate)

    x = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = Activation('relu', name='relu')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = keras.utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='9d60b8095a5708f2dcce2bca79d332c7')
            elif blocks == [6, 12, 32, 32]:
                weights_path = keras.utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='d699b8f76981ab1b30698df4c175e90b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = keras.utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = keras.utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='30ee3e1110167f948a6b9946edeeb738')
            elif blocks == [6, 12, 32, 32]:
                weights_path = keras.utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = keras.utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model



# The following is from keras application densenet implementation


BASE_WEIGTHS_PATH = (
    'https://github.com/keras-team/keras-applications/'
    'releases/download/densenet/')
DENSENET121_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGTHS_PATH +
    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')

def dense_block(x, blocks, name, growth_rate=32, dropout_rate=None):
  """A dense block.
  # Arguments
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.
  # Returns
      output tensor for the block.
  """
  for i in range(blocks):
      x = conv_block(x, growth_rate, name=name + '_block' + str(i + 1), dropout_rate=dropout_rate)
  return x


def transition_block(x, reduction, name, dropout_rate=None):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name, dropout_rate=None):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    if dropout_rate:
        x1 = Dropout(dropout_rate)(x1)

    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    
    if dropout_rate:
        x1 = Dropout(dropout_rate)(x1)

    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def preprocess_input(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.
    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format,
                                           mode='torch', **kwargs)

                                