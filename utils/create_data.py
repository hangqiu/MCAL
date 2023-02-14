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

"""Make datasets and save specified directory.

Downloads datasets using scikit datasets and can also parse csv file
to save into pickle format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import os
import pickle
from io import StringIO
# import StringIO
import tarfile
# import urllib2
from urllib.request import urlopen

import keras.backend as K
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from keras.datasets import fashion_mnist

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
import sklearn.datasets.rcv1
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from absl import app
from absl import flags
from tensorflow.io import gfile

from PIL import Image

import time
import h5py
import math
import scipy.io as sio

from utils.image_dataset_loader import image_dataset_from_directory


flags.DEFINE_string('save_dir', '/tmp/data',
                    'Where to save outputs')
flags.DEFINE_string('datasets', '',
                    'Which datasets to download, comma separated.')
FLAGS = flags.FLAGS


class Dataset(object):

  def __init__(self, X, y):
    self.data = X
    self.target = y


def get_csv_data(filename):
  """Parse csv and return Dataset object with data and targets.

  Create pickle data from csv, assumes the first column contains the targets
  Args:
    filename: complete path of the csv file
  Returns:
    Dataset object
  """
  f = gfile.GFile(filename, 'r')
  mat = []
  for l in f:
    row = l.strip()
    row = row.replace('"', '')
    row = row.split(',')
    row = [float(x) for x in row]
    mat.append(row)
  mat = np.array(mat)
  y = mat[:, 0]
  X = mat[:, 1:]
  data = Dataset(X, y)
  return data


def get_wikipedia_talk_data():
  """Get wikipedia talk dataset.

  See here for more information about the dataset:
  https://figshare.com/articles/Wikipedia_Detox_Data/4054689
  Downloads annotated comments and annotations.
  """

  ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634'
  ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637'

  def download_file(url):
    # req = urllib2.Request(url)
    # response = urllib2.urlopen(req)
    response = urlopen(url)
    return response

  # Process comments
  comments = pd.read_table(
      download_file(ANNOTATED_COMMENTS_URL), index_col=0, sep='\t')
  # remove newline and tab tokens
  comments['comment'] = comments['comment'].apply(
      lambda x: x.replace('NEWLINE_TOKEN', ' '))
  comments['comment'] = comments['comment'].apply(
      lambda x: x.replace('TAB_TOKEN', ' '))

  # Process labels
  annotations = pd.read_table(download_file(ANNOTATIONS_URL), sep='\t')
  # labels a comment as an atack if the majority of annoatators did so
  labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

  # Perform data preprocessing, should probably tune these hyperparameters
  vect = CountVectorizer(max_features=30000, ngram_range=(1, 2))
  tfidf = TfidfTransformer(norm='l2')
  X = tfidf.fit_transform(vect.fit_transform(comments['comment']))
  y = np.array(labels)
  data = Dataset(X, y)
  return data


def get_keras_data(dataname):
  """Get datasets using keras API and return as a Dataset object."""
  if dataname == 'cifar10_keras':
    train, test = cifar10.load_data()
  elif dataname == 'cifar100_coarse_keras':
    train, test = cifar100.load_data('coarse')
  elif dataname == 'cifar100_keras':
    train, test = cifar100.load_data()
  elif dataname == 'mnist_keras':
    train, test = mnist.load_data()
  elif dataname == 'fashion_keras':
    train, test = fashion_mnist.load_data()
  else:
    raise NotImplementedError('dataset not supported')

  X = np.concatenate((train[0], test[0]))
  y = np.concatenate((train[1], test[1]))

  if dataname == 'mnist_keras' or dataname == 'fashion_keras':
    # Add extra dimension for channel
    num_rows = X.shape[1]
    num_cols = X.shape[2]
    X = X.reshape(X.shape[0], 1, num_rows, num_cols)
    if K.image_data_format() == 'channels_last':
      X = X.transpose(0, 2, 3, 1)

  y = y.flatten()
  data = Dataset(X, y)
  return data


# TODO(lishal): remove regular cifar10 dataset and only use dataset downloaded
# from keras to maintain image dims to create tensor for tf models
# Requires adding handling in run_experiment.py for handling of different
# training methods that require either 2d or tensor data.
def get_cifar10():
  """Get CIFAR-10 dataset from source dir.

  Slightly redundant with keras function to get cifar10 but this returns
  in flat format instead of keras numpy image tensor.
  """
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  def download_file(url):
    # req = urllib2.Request(url)
    # response = urllib2.urlopen(req)
    response = urlopen(url)
    return response
  response = download_file(url)
  tmpfile = BytesIO()
  while True:
    # Download a piece of the file from the connection
    s = response.read(16384)
    # Once the entire file has been downloaded, tarfile returns b''
    # (the empty bytes) which is a falsey value
    if not s:
      break
    # Otherwise, write the piece of the file to the temporary file.
    tmpfile.write(s)
  response.close()

  tmpfile.seek(0)
  tar_dir = tarfile.open(mode='r:gz', fileobj=tmpfile)
  X = None
  y = None
  for member in tar_dir.getnames():
    if '_batch' in member:
      filestream = tar_dir.extractfile(member).read()
      batch = pickle.load(StringIO.StringIO(filestream))
      if X is None:
        X = np.array(batch['data'], dtype=np.uint8)
        y = np.array(batch['labels'])
      else:
        X = np.concatenate((X, np.array(batch['data'], dtype=np.uint8)))
        y = np.concatenate((y, np.array(batch['labels'])))
  data = Dataset(X, y)
  return data


def resize_and_crop(img, target_sqaure_size):
  target_sqaure_size = float(target_sqaure_size)
  img_size = img.size
  resize_size = None
  crop_box = None
  if img_size[0] < img_size[1]:
    resize_size = (int(target_sqaure_size), math.ceil(img_size[1] / (img_size[0] / target_sqaure_size)))
    diff = (resize_size[1] - target_sqaure_size) / 2
    crop_box = (0, int(diff), int(target_sqaure_size), int(diff + target_sqaure_size))
  else:
    resize_size = (math.ceil(img_size[0] / (img_size[1] / target_sqaure_size)), int(target_sqaure_size))
    diff = (resize_size[0] - target_sqaure_size) / 2
    crop_box = (int(diff), 0, int(diff+target_sqaure_size), int(target_sqaure_size))
  
  return img.resize(resize_size).crop(crop_box)


def get_imagenet_v2():
    datadir = '../dataset/ImageNet/raw_data/train'
    dataset, labels = image_dataset_from_directory(datadir, image_size=(224, 224), batch_size=256, label_mode="categorical")
    return dataset, labels

def get_imagenet():
  debug = False
  datadir = '../dataset/ImageNet/raw_data/'
  
  wnids = list(map(lambda x: x.strip(), open(datadir+'synset_labels.txt').readlines()))
  base_base_path = datadir + "train/"
  # for i in range(len(wnids)):
  i=0
  data_X = None
  data_y = None
  data_batch_X = None
  data_batch_y = None
  data_batch = 1 if debug else 20
  t_start = time.time()

  target_size = 224 # square

  file_path = "/home/krchinta/research/dataset/tf-data/imagenet.h5"
  

  for wnid in sorted(os.listdir(base_base_path)):
      # wnid = wnids[i]
      print ("{}: {} / 1000".format(wnid, i + 1))
      base_path = datadir + "train/{0}/".format(wnid)
      images = os.listdir(base_path)
      # for j in range(500):
      im_batch = 10
      im_count = 0
      batch_X = None
      batch_y = None
      X = None
      y = None
      for im in images:
          path = datadir + "train/{0}/{1}".format(wnid, im)
          img = Image.open(path).convert('RGB')
          img = resize_and_crop(img, target_size)
          image = np.asarray(img, dtype=np.uint8)
          image = np.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
          if X is None:
              X = image
              y = np.array([i])
          else:
              # print(X.shape)
              # print(image.shape)
              X = np.concatenate((X, image))
              y = np.concatenate((y, np.array([i])))
          im_count += 1
          if im_count % im_batch == 0:
              if batch_X is None:
                  batch_X = X
                  batch_y = y
              else:
                  batch_X = np.concatenate((batch_X, X))
                  batch_y = np.concatenate((batch_y, y))
              X = None
              y = None

            # # assess the time
            # t_end = time.time()
            # print("batch_time: {} sec".format(t_end-t_start))
            # t_start = t_end

      # assess the time
      t_end = time.time()
      print("avg image time: {} sec".format((t_end-t_start)/len(images)))
      t_start = t_end

      t_concat = time.time()
      if data_batch_X is None:
          data_batch_X = batch_X
          data_batch_y = batch_y
      else:
          data_batch_X = np.concatenate((data_batch_X, batch_X))
          data_batch_y = np.concatenate((data_batch_y, batch_y))
      
      print("Total image processed: {}, concat time: {} sec".format(data_batch_X.shape[0], time.time()-t_concat))
      i += 1

      if i % data_batch == 0:
          t_concat = time.time()
          # if data_X is None:
          #     data_X = data_batch_X
          #     data_y = data_batch_y
          # else:
          #     data_X = np.concatenate((data_X, data_batch_X))
          #     data_y = np.concatenate((data_y, data_batch_y))
          if not os.path.exists(file_path):
              print("Creating imagenet.h5")
              print(data_batch_y.shape)
              hf = h5py.File(file_path,'w')
              hf.create_dataset("data", data=data_batch_X, maxshape=(None, target_size, target_size, 3))
              hf.create_dataset("label", data=data_batch_y, maxshape=(None, ))
              hf.close()
          else:
              hf = h5py.File(file_path, 'a')
              hf["data"].resize(hf["data"].shape[0] + data_batch_X.shape[0], axis=0)
              hf["data"][-data_batch_X.shape[0]:] = data_batch_X
              hf["label"].resize(hf["label"].shape[0] + data_batch_y.shape[0], axis=0)
              hf["label"][-data_batch_y.shape[0]:] = data_batch_y
              hf.close()

          data_batch_X = None
          data_batch_y = None

          print("Total classes processed: {}, concat time: {} sec".format(i, time.time()-t_concat))

      if debug and i > 5: 
          break
  hf = h5py.File(file_path,'r')
  data_X = hf.get("data")
  data_y = hf.get("label")
  data = Dataset(data_X, data_y)
  return data

def get_tiny_imagenet():

  # url = 'http://www.image-net.org/image/tiny/tiny-imagenet-200.zip'
  datadir = '/home/krchinta/research/dataset/tiny-imagenet-200/'
  X = None
  y = None
  wnids = list(map(lambda x: x.strip(), open(datadir+'wnids.txt').readlines()))
  for i in range(len(wnids)):
      wnid = wnids[i]
      print ("{}: {} / {}".format(wnid, i + 1, len(wnids)))
      for j in range(500):
          path = datadir + "train/{0}/images/{0}_{1}.JPEG".format(wnid, j)
          image = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
          image = np.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
          if X is None:
              X = image
              y = np.array([i])
          else:
              X = np.concatenate((X, image))
              y = np.concatenate((y, np.array([i])))
  data = Dataset(X, y)
  return data

def get_svhn():
  datadir = '/home/krchinta/research/dataset/svhn'
  train_images = sio.loadmat(datadir+'/train_32x32.mat')
  train_images_x = np.transpose(train_images['X'], (3, 0, 1, 2))
  train_images_y = train_images['y']
  test_images = sio.loadmat(datadir+'/test_32x32.mat')
  test_images_x = np.transpose(test_images['X'], (3, 0, 1, 2))
  test_images_y = test_images['y']
  extra_images = sio.loadmat(datadir+'/extra_32x32.mat')
  extra_images_x = np.transpose(extra_images['X'], (3, 0, 1, 2))
  extra_images_y = extra_images['y']
  print(train_images_x.shape)
  print(test_images_x.shape)
  print(extra_images_x.shape)
  print(train_images_y.shape)
  print(test_images_y.shape)
  print(extra_images_y.shape)
  images = np.concatenate([train_images_x, test_images_x])
  images = np.concatenate([images, extra_images_x])
  labels = np.concatenate([train_images_y, test_images_y])
  labels = np.concatenate([labels, extra_images_y])
  
  # replace label "10" with label "0"
  labels[labels == 10] = 0

  print(images.shape)
  print(labels.shape)

  data = Dataset(images, labels)
  return data

def get_mldata(dataset):
  # Use scikit to grab datasets and save them save_dir.
  save_dir = FLAGS.save_dir
  filename = os.path.join(save_dir, dataset[1]+'.pkl')

  if not gfile.Exists(save_dir):
    gfile.MkDir(save_dir)
  if not gfile.Exists(filename):
    if dataset[0][-3:] == 'csv':
      data = get_csv_data(dataset[0])
    elif dataset[0] == 'breast_cancer':
      data = load_breast_cancer()
    elif dataset[0] == 'iris':
      data = load_iris()
    elif dataset[0] == 'newsgroup':
      # Removing header information to make sure that no newsgroup identifying
      # information is included in data
      data = fetch_20newsgroups_vectorized(subset='all', remove=('headers'))
      tfidf = TfidfTransformer(norm='l2')
      X = tfidf.fit_transform(data.data)
      data.data = X
    elif dataset[0] == 'rcv1':
      sklearn.datasets.rcv1.URL = (
        'http://www.ai.mit.edu/projects/jmlr/papers/'
        'volume5/lewis04a/a13-vector-files/lyrl2004_vectors')
      sklearn.datasets.rcv1.URL_topics = (
        'http://www.ai.mit.edu/projects/jmlr/papers/'
        'volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz')
      data = sklearn.datasets.fetch_rcv1(
          data_home='/tmp')
    elif dataset[0] == 'wikipedia_attack':
      data = get_wikipedia_talk_data()
    elif dataset[0] == 'cifar10':
      data = get_cifar10()
    elif dataset[0] == 'tinyimagenet':
      data = get_tiny_imagenet()
    elif dataset[0] == 'imagenet':
      # data = get_imagenet()
      data, labels = get_imagenet_v2()
    elif 'keras' in dataset[0]:
      data = get_keras_data(dataset[0])
    elif 'svhn' in dataset[0]:
      data = get_svhn()
    else:
      try:
        data = fetch_mldata(dataset[0])
      except:
        raise Exception('ERROR: failed to fetch data from mldata.org')
    X = data.data
    y = data.target
    if X.shape[0] != y.shape[0]:
      X = np.transpose(X)
    assert X.shape[0] == y.shape[0]

    if not dataset[0] == 'imagenet':
      data = {'data': X, 'target': y}
      pickle.dump(data, gfile.GFile(filename, 'w'))


def main(argv):
  del argv  # Unused.
  # First entry of tuple is mldata.org name, second is the name that we'll use
  # to reference the data.
  datasets = [('mnist (original)', 'mnist'), ('australian', 'australian'),
              ('heart', 'heart'), ('breast_cancer', 'breast_cancer'),
              ('iris', 'iris'), ('vehicle', 'vehicle'), ('wine', 'wine'),
              ('waveform ida', 'waveform'), ('german ida', 'german'),
              ('splice ida', 'splice'), ('ringnorm ida', 'ringnorm'),
              ('twonorm ida', 'twonorm'), ('diabetes_scale', 'diabetes'),
              ('mushrooms', 'mushrooms'), ('letter', 'letter'), ('dna', 'dna'),
              ('banana-ida', 'banana'), ('letter', 'letter'), ('dna', 'dna'),
              ('newsgroup', 'newsgroup'), ('cifar10', 'cifar10'),
              ('cifar10_keras', 'cifar10_keras'),
              ('cifar100_keras', 'cifar100_keras'),
              ('cifar100_coarse_keras', 'cifar100_coarse_keras'),
              ('mnist_keras', 'mnist_keras'),
              ('wikipedia_attack', 'wikipedia_attack'),
              ('rcv1', 'rcv1'),
              ('fashion_keras','fashion_keras'),
              ('tinyimagenet','tinyimagenet'),
              ('imagenet','imagenet'),
              ('svhn','svhn')]

  if FLAGS.datasets:
    subset = FLAGS.datasets.split(',')
    datasets = [d for d in datasets if d[1] in subset]

  for d in datasets:
    print(d[1])
    get_mldata(d)


if __name__ == '__main__':
  app.run(main)
