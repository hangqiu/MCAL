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




class ResNetGrow(object):
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
               batch_size=32,
               solver='Adam',
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
    self._feature_model = None
    self.random_state = random_state
    self.n_classes = None

    self.data_augmentation = augmentation != 0

    self.id = None
    self.mode = "cifar"
    self.init_kernels = None

    self.image_path_mode = False

  def resnet_layer(self,
                   inputs,
                   num_filters=16,
                   kernel_size=3,
                   strides=1,
                   activation='relu',
                   batch_normalization=True,
                   conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
      x = conv(x)
      if batch_normalization:
        x = BatchNormalization()(x)
      if activation is not None:
        x = Activation(activation)(x)
    else:
      if batch_normalization:
        x = BatchNormalization()(x)
      if activation is not None:
        x = Activation(activation)(x)
      x = conv(x)
    return x

  def resnet_v1(self, input_shape, depth, kernels=16, cell=3, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    # if (depth - 2) % 6 != 0:
    #   raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = kernels
    self.init_kernels = kernels
    # num_res_blocks = int((depth - 2) / 6)
    num_res_blocks = depth

    inputs = Input(shape=input_shape)
    # imagenet input v.s. cifar input
    kernel_size = 3
    strides = 1
    if input_shape[0]==224:
      kernel_size = 7
      strides = 2
      self.mode = "imagenet"
      # self.epochs = 50 # was 50

    x = self.resnet_layer(inputs=inputs, num_filters=num_filters, kernel_size=kernel_size, strides=strides)
    # Instantiate the stack of residual units
    for stack in range(cell):
      for res_block in range(num_res_blocks):
        strides = 1
        if stack > 0 and res_block == 0:  # first layer but not first stack
          strides = 2  # downsample
        y = self.resnet_layer(inputs=x,
                         num_filters=num_filters,
                         strides=strides)
        y = self.resnet_layer(inputs=y,
                         num_filters=num_filters,
                         activation=None)
        if stack > 0 and res_block == 0:  # first layer but not first stack
          # linear projection residual shortcut connection to match
          # changed dims
          x = self.resnet_layer(inputs=x,
                           num_filters=num_filters,
                           kernel_size=1,
                           strides=strides,
                           activation=None,
                           batch_normalization=False)
        x = keras.layers.add([x, y])
        x = Activation('relu')(x)
      num_filters *= 2


    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    # y = GlobalAveragePooling2D()(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten(name='feature')(x)
    # y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

  def resnet_v2(self, input_shape, depth, cell=3, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    # if (depth - 2) % 9 != 0:
    #   raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    # num_res_blocks = int((depth - 2) / 9)
    num_res_blocks = depth

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = self.resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(cell):
      for res_block in range(num_res_blocks):
        activation = 'relu'
        batch_normalization = True
        strides = 1
        if stage == 0:
          num_filters_out = num_filters_in * 4
          if res_block == 0:  # first layer and first stage
            activation = None
            batch_normalization = False
        else:
          num_filters_out = num_filters_in * 2
          if res_block == 0:  # first layer but not first stage
            strides = 2  # downsample

        # bottleneck residual unit
        y = self.resnet_layer(inputs=x,
                         num_filters=num_filters_in,
                         kernel_size=1,
                         strides=strides,
                         activation=activation,
                         batch_normalization=batch_normalization,
                         conv_first=False)
        y = self.resnet_layer(inputs=y,
                         num_filters=num_filters_in,
                         conv_first=False)
        y = self.resnet_layer(inputs=y,
                         num_filters=num_filters_out,
                         kernel_size=1,
                         conv_first=False)
        if res_block == 0:
          # linear projection residual shortcut connection to match
          # changed dims
          x = self.resnet_layer(inputs=x,
                           num_filters=num_filters_out,
                           kernel_size=1,
                           strides=strides,
                           activation=None,
                           batch_normalization=False)
        x = keras.layers.add([x, y])

      num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # y = GlobalAveragePooling2D()(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

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

    # self._feature_model = self.resnet_v1_conv_feature(input_shape=input_shape, depth=n_layer_per_cell, kernels=n_kernel_1st_layer, cell=n_cell, num_classes=self.n_classes)

    if len(_gpus)==1:
      self.model = self.resnet_v1(input_shape=input_shape, depth=n_layer_per_cell, kernels=n_kernel_1st_layer, cell=n_cell, num_classes=self.n_classes)
    else:
      # with tf.device('/cpu:0'):
      #   self.ori_model = self.resnet_v1(input_shape=input_shape, depth=n_layer_per_cell, kernels=n_kernel_1st_layer, cell=n_cell, num_classes=self.n_classes)
      # self.model = keras.utils.multi_gpu_model(self.ori_model, gpus=_gpus)
      """multi-gpu-model deprecated"""
      strategy = tf.distribute.MirroredStrategy()
      print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
      with strategy.scope():
        self.model = self.resnet_v1(input_shape=input_shape, depth=n_layer_per_cell, kernels=n_kernel_1st_layer, cell=n_cell, num_classes=self.n_classes)




    # n_kernel = n_kernel_1st_layer
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

    self.id = "ResNetC{}L{}K{}B{}".format(n_cell, n_layer_per_cell, n_kernel_1st_layer, self.batch_size)
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
    lr = lr * self.batch_size / 32 # linear scale to batch sizes
    # lr = lr * 10 / self.n_classes
    # lr = lr * 16 / self.init_kernels
    if self.init_kernels == 64:
        lr /= 10

    if epoch > 180:
      lr *= 0.5e-3
    elif epoch > 160:
      lr *= 1e-3
    elif epoch > 120:
      lr *= 1e-2
    elif epoch > 80:
      lr *= 1e-1
    print('Epoch: {} Learning rate: {}'.format(epoch, lr))
    return lr

  def lr_schedule_imagenet(self, epoch):
    lr = self.learning_rate
    lr = lr * self.batch_size / 32 # linear scale to batch sizes
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

    # We don't want incremental fit so reset learning rate and weights
    K.set_value(self.model.optimizer.lr, self.learning_rate)
    self.model.set_weights(self.initial_weights)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=50)
    # reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=30, factor=0.1, verbose=1)
    # reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=20, factor=0.1, verbose=1)

    lr_scheduler = keras.callbacks.LearningRateScheduler(self.lr_schedule)
    if self.mode == "imagenet":
      lr_scheduler = keras.callbacks.LearningRateScheduler(self.lr_schedule_imagenet)
      
    #
    # reduceLR = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                                              cooldown=0,
    #                                              patience=5,
    #                                              verbose=1,
    #                                              min_lr=0.5e-6)
  
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=20, factor=0.1, verbose=1)

    callback_list = [
                      reduceLR,
                      lr_scheduler,
                      # es
                     ]

    # filepath = self.id + "_TrainingSize_{}_LRSchedule.h5".format(X_train.shape[0])

    if self.image_path_mode:

      print("Training Actually started")
      t_start = time.time()
      history = self.model.fit(X_train,  validation_data=X_val,
                                         epochs=self.epochs, verbose=1, workers=8,
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

    return history

  def predict(self, X_val):
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

  # def transform(self, X):
  #   model = self.model
  #   inp = [model.input]
  #   activations = []
  #
  #   # Get activations of the first dense layer.
  #   output = [layer.output for layer in model.layers if
  #             layer.name == 'feature'][0]
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

  def transform(self, X):
    # X = tf.convert_to_tensor(X, dtype=tf.float32)
    extractor = keras.Model(inputs=self.model.inputs,
                            outputs=self.model.get_layer('feature').output)
    extractor.summary()
    # print(X.shape)
    # features = np.array(extractor(X))
    features = extractor.predict(X, verbose=1)
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
  #   features = self._feature_model.predict(X, batch_size=self.batch_size)
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

  # def resnet_v1_conv_feature(self, input_shape, depth, kernels=16, cell=3, num_classes=10):
  #   """ResNet Version 1 Model builder [a]
  #
  #   Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
  #   Last ReLU is after the shortcut connection.
  #   At the beginning of each stage, the feature map size is halved (downsampled)
  #   by a convolutional layer with strides=2, while the number of filters is
  #   doubled. Within each stage, the layers have the same number filters and the
  #   same number of filters.
  #   Features maps sizes:
  #   stage 0: 32x32, 16
  #   stage 1: 16x16, 32
  #   stage 2:  8x8,  64
  #   The Number of parameters is approx the same as Table 6 of [a]:
  #   ResNet20 0.27M
  #   ResNet32 0.46M
  #   ResNet44 0.66M
  #   ResNet56 0.85M
  #   ResNet110 1.7M
  #
  #   # Arguments
  #       input_shape (tensor): shape of input image tensor
  #       depth (int): number of core convolutional layers
  #       num_classes (int): number of classes (CIFAR10 has 10)
  #
  #   # Returns
  #       model (Model): Keras model instance
  #   """
  #   # if (depth - 2) % 6 != 0:
  #   #   raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
  #   # Start model definition.
  #   num_filters = kernels
  #   self.init_kernels = kernels
  #   # num_res_blocks = int((depth - 2) / 6)
  #   num_res_blocks = depth
  #
  #   inputs = Input(shape=input_shape)
  #   # imagenet input v.s. cifar input
  #   kernel_size = 3
  #   strides = 1
  #   if input_shape[0]==224:
  #     kernel_size = 7
  #     strides = 2
  #     self.mode = "imagenet"
  #     # self.epochs = 50 # was 50
  #
  #   x = self.resnet_layer(inputs=inputs, num_filters=num_filters, kernel_size=kernel_size, strides=strides)
  #   # Instantiate the stack of residual units
  #   for stack in range(cell):
  #     for res_block in range(num_res_blocks):
  #       strides = 1
  #       if stack > 0 and res_block == 0:  # first layer but not first stack
  #         strides = 2  # downsample
  #       y = self.resnet_layer(inputs=x,
  #                        num_filters=num_filters,
  #                        strides=strides)
  #       y = self.resnet_layer(inputs=y,
  #                        num_filters=num_filters,
  #                        activation=None)
  #       if stack > 0 and res_block == 0:  # first layer but not first stack
  #         # linear projection residual shortcut connection to match
  #         # changed dims
  #         x = self.resnet_layer(inputs=x,
  #                          num_filters=num_filters,
  #                          kernel_size=1,
  #                          strides=strides,
  #                          activation=None,
  #                          batch_normalization=False)
  #       x = keras.layers.add([x, y])
  #       x = Activation('relu')(x)
  #     num_filters *= 2
  #
  #
  #   # Add classifier on top.
  #   # v1 does not use BN after last shortcut connection-ReLU
  #   # outputs = GlobalAveragePooling2D()(x)
  #   x = AveragePooling2D(pool_size=8)(x)
  #   outputs = Flatten()(x)
  #
  #   # Instantiate model.
  #   model = Model(inputs=inputs, outputs=outputs)
  #   return model
