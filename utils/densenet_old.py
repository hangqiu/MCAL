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
from keras.layers import Input, concatenate, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

import numpy as np
import tensorflow as tf
from keras.models import load_model

from utils.custom_layers import Scale

from keras.preprocessing.image import ImageDataGenerator
import time

class DenseNet(object):
  """Small convnet that matches sklearn api.

  Implements model from
  https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
  Adapts for inputs of variable size, expects data to be 4d tensor, with
  # of obserations as first dimension and other dimensions to correspond to
  length width and # of channels in image.
  """

  def __init__(self,
               random_state=1,
               epochs=2,
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

  def build_model(self, X, y=None, cell=4, layer=None, kernel=64, _gpus=[0]):
    self.cell = cell
    self.kernel = kernel
    input_shape = X.shape[1:]
    if y is not None:
      y_mat = self.create_y_mat(y)
    

    if len(_gpus)==1:
      self.model = self.DenseNetGrow(input_shape=input_shape, 
                            classes=self.n_classes,
                            nb_dense_block=cell,
                            nb_filter=kernel, 
                            growth_rate=16, # fix for now
                            )

    else:
      with tf.device('/cpu:0'):
        self.ori_model = self.DenseNetGrow(input_shape=input_shape, 
                            classes=self.n_classes,
                            nb_dense_block=cell,
                            nb_filter=kernel, 
                            growth_rate=16, # fix for now
                            )
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
    if epoch > 100:
      lr *= 0.5e-3
    elif epoch > 90:
      lr *= 1e-3
    elif epoch > 80:
      lr *= 1e-2
    elif epoch > 50:
      lr *= 1e-1
    print('Epoch: {} Learning rate: {}'.format(epoch, lr))
    return lr

  def fit(self, X_train, y_train, X_val, y_val, sample_weight=None):
    y_mat = self.create_y_mat(y_train)
    y_val_mat = self.create_y_mat(y_val)

    if self.model is None:
      self.build_model(X_train)

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
        rotation_range=0,
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


    self.save_and_delete()
    return history

  def predict(self, X_val):
    predicted = self.model.predict(X_val)
    return predicted

  def score(self, X_val, val_y, _gpus=[0]):
    if self.model == None:
      self.build_model(X_val, val_y, cell=self.cell, kernel=self.kernel, _gpus=[0])
      self.load_model()
    y_mat = self.create_y_mat(val_y)
    # if len(_gpus) > 1:
    #   val_acc = self.ori_model.evaluate(X_val, y_mat, verbose=0)[1]
    # else:
    val_acc = self.model.evaluate(X_val, y_mat, verbose=0)[1]
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
    del self.model
    self.model = None
    keras.backend.clear_session()

  def load_model(self, filepath=None):
    if filepath is None:
      filepath = self.id + ".h5"
    print("Loading Model...")
    self.model = load_model(filepath)


  def DenseNetGrow(self, input_shape=(32,32,3), 
                nb_dense_block=4, 
                growth_rate=32, 
                nb_filter=64, 
                classes=10,
                reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, 
                weights_path=None):
      '''Instantiate the DenseNet 121 architecture,
          # Arguments
              nb_dense_block: number of dense blocks to add to end
              growth_rate: number of filters to add per dense block
              nb_filter: initial number of filters
              reduction: reduction factor of transition blocks.
              dropout_rate: dropout rate
              weight_decay: weight decay factor
              classes: optional number of classes to classify images
              weights_path: path to pre-trained weights
          # Returns
              A Keras model instance.
      '''
      eps = 1.1e-5

      # compute compression factor
      compression = 1.0 - reduction

      # Handle Dimension Ordering for different backends
      global concat_axis
      # if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=input_shape, name='data')
      # else:
      #   concat_axis = 1
      #   img_input = Input(shape=(3, 32, 32), name='data')

      # From architecture for ImageNet (Table 1 in the paper)
      # nb_filter = 64
      nb_layers = [6,12,24,16] # For DenseNet-121

      # Initial convolution
      init_conv_size = 3
      if input_shape[0] != 32:
        init_conv_size = 7
      x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
      x =  Conv2D(nb_filter, init_conv_size, init_conv_size, subsample=(2, 2), name='conv1', bias=False)(x)
      x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
      x = Scale(axis=concat_axis, name='conv1_scale')(x)
      x = Activation('relu', name='relu1')(x)
      x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
      x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

      # Add dense blocks
      for block_idx in range(nb_dense_block - 1):
          stage = block_idx+2
          x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

          # Add transition_block
          x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
          nb_filter = int(nb_filter * compression)

      final_stage = stage + 1
      x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

      x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
      x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
      x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
      x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

      x = Dense(classes, name='fc6')(x)
      x = Activation('softmax', name='prob')(x)

      model = Model(img_input, x, name='densenet')

      if weights_path is not None:
        model.load_weights(weights_path)

      return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x =  Conv2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x =  Conv2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x =  Conv2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter