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
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator





class PlainGrow(object):
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
               solver='Adam',
               learning_rate=0.001,
               lr_decay=0.,
               augmentation=0):
    # params
    self.solver = solver
    self.epochs = epochs
    self.batch_size = batch_size
    # self.learning_rate = learning_rate  * self.batch_size / 32 # linear scale to batch sizes
    self.learning_rate = learning_rate  # linear scale to batch sizes
    self.lr_decay = lr_decay
    # data
    self.encode_map = None
    self.decode_map = None
    self.model = None
    self.random_state = random_state
    self.n_classes = None

    self.data_augmentation = augmentation != 0

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
    # lr = lr * self.batch_size / 32 # linear scale to batch sizes
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

  def PlainModel(self, input_shape, n_kernel, n_cell, n_layer_per_cell):
      model = Sequential()
      model.add(Conv2D(n_kernel, (3, 3), padding='same', activation='relu', input_shape=input_shape))

      for c in range(n_cell):
        l_start = 0
        if c == 0:
          l_start = 1
        for l in range(l_start, n_layer_per_cell):
          model.add(Conv2D(n_kernel, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        n_kernel *= 2
      model.add(GlobalAveragePooling2D())
      model.add(Flatten(name='feature'))
      # model.add(Dense(1024))
      # model.add(Activation('relu'))
      # model.add(Dropout(0.5))
      model.add(Dense(self.n_classes))
      model.add(Activation('softmax'))
      return model
  
  def build_model(self, X, y, n_cell, n_layer_per_cell, n_kernel_1st_layer, _gpus):
    # assumes that data axis order is same as the backend
    """
    Define a plain model based on decmiation, layer in between, and number of kernels of the 1st layer.
    Between each each decimation, it's a cell with n_layer, each with same amount of kernels
    The default decimation is 2*2 maxpooling, TODO: later we can change this as well
    After decimation, the next cell will have 2x kernels each layer.
    Caution: input should not apply decimation to a level output is smaller than 1*1
    :param X: input np array
    :param n_cell: how many decimation
    :param n_layer_per_cell:
    :param n_kernel_1st_layer:
    :return:
    """
    input_shape = X.shape[1:]
    np.random.seed(self.random_state)
    # tf.set_random_seed(self.random_state)
    tf.random.set_seed(self.random_state)
    # define how many classes by encoding y
    y_mat = self.create_y_mat(y)

    n_kernel = n_kernel_1st_layer

    if len(_gpus)==1:
      self.model = self.PlainModel(input_shape, n_kernel, n_cell, n_layer_per_cell)
    else:
      with tf.device('/cpu:0'):
        self.ori_model = self.PlainModel(input_shape, n_kernel, n_cell, n_layer_per_cell)
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

  def fit(self, X_train, y_train, X_val, y_val, sample_weight=None):
    y_mat = self.create_y_mat(y_train)
    y_val_mat = self.create_y_mat(y_val)

    # We don't want incremental fit so reset learning rate and weights
    K.set_value(self.model.optimizer.lr, self.learning_rate)
    self.model.set_weights(self.initial_weights)

    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=50)
    reduceLR = keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=20, factor=0.1, verbose=1)
    lr_scheduler = keras.callbacks.LearningRateScheduler(self.lr_schedule)
    callback_list = [reduceLR,
                     lr_scheduler
                    #  es
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
      history = self.model.fit(
          X_train,
          y_mat,
          batch_size=self.batch_size,
          epochs=self.epochs,
          shuffle="batch",
          sample_weight=sample_weight,
          verbose=1,
          validation_data=(X_val, y_val_mat),
          callbacks=callback_list)
    return history

  def predict(self, X_val):
    predicted = self.model.predict(X_val, batch_size=self.batch_size, verbose=1)
    return predicted

  def score(self, X_val, val_y, _gpus=[0]):
    y_mat = self.create_y_mat(val_y)
    if len(_gpus) > 1:
      val_acc = self.ori_model.evaluate(X_val, y_mat, batch_size=self.batch_size, verbose=0)[1]
    else:
      val_acc = self.model.evaluate(X_val, y_mat, batch_size=self.batch_size, verbose=0)[1]
    return val_acc

  def decision_function(self, X):
    return self.predict(X)

  def transform(self, X):
    # X = tf.convert_to_tensor(X, dtype=tf.float32)
    extractor = keras.Model(inputs=self.model.inputs,
                            outputs=self.model.get_layer('feature').output)
    extractor.summary()
    # print(X.shape)
    # features = np.array(extractor(X))
    features = extractor.predict(X, batch_size=self.batch_size, verbose=1)
    # features = self._feature_model.predict(X, batch_size=self.batch_size, verbose=1)
    # print(features.shape)
    return features
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
