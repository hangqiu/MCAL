import warnings
warnings.filterwarnings('ignore')
import glob
import numpy as np 
import argparse 
import importlib
import os 
from os import path
import datetime

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf ; print("You have imported tensorflow version :", tf.__version__)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

import keras 
print("Keras version : ",keras.__version__)
from tensorflow.keras.optimizers import Adam , Adagrad, Adadelta, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar100 , cifar10
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0

batch_size = 512

data_dir = '../../dataset/imagenet/train/'

datagen = ImageDataGenerator(rescale=1./255.,  validation_split=0.05 )    
print("Creating training set....")
train_it = datagen.flow_from_directory(data_dir, target_size=(224, 224), color_mode="rgb", batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="training")
print("Creating validation set....")
val_it = datagen.flow_from_directory(data_dir, target_size=(224, 224), color_mode="rgb", batch_size=batch_size, class_mode="categorical", shuffle=True, seed=42, subset="validation")


class PrintLR(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
            model.save("b0_210203_model_continued.h5")
            print('\nLearning rate for epoch {} is {}'.format(epoch + 1,model.optimizer.lr.numpy()))
            print("Loss :", logs['loss'])
            print("Accuracy :",logs['categorical_accuracy'])
            print()
                                                          
checkpoint_dir = './training_checkpoints_210202_b0'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")


anne = ReduceLROnPlateau(monitor='categorical_accuracy', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [ 
    PrintLR(),
    anne,
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_best_only=True , verbose=1)
]
                                      

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
    print('Loading models....')
    if path.exists("model_b0_21130_full_2.h5"):
        model = tf.keras.models.load_model("model_b0_21130_full_2.h5")
        print("Loading old model"); print()
    else:
        model = EfficientNetB0(weights=None)

    print("Compiling Model....")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True , name='categorical_crossentropy'),
        metrics=[tf.keras.metrics.CategoricalAccuracy()] )
    print("Training started....")
    train_hist = model.fit_generator(
        train_it,  epochs=35, verbose=1, callbacks= callbacks,steps_per_epoch=train_it.samples // batch_size,
        validation_data= val_it,  validation_freq=1,validation_steps=val_it.samples // batch_size, shuffle=True)

    model.save("b0_210203_model_continued.h5")

    print(); print(); print("Loss vs epochs data - \n", train_hist.history["loss"] )
    print()
    print("Accuracy vs epochs data - \n", train_hist.history["categorical_accuracy"] )
    print()
