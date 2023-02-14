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

"""Utility functions for run_experiment.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import pickle
import sys
import h5py

from keras.utils.io_utils import HDF5Matrix

import numpy as np
import scipy
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from tensorflow.io import gfile

from utils import image_dataset_loader
from utils.kernel_block_solver import BlockKernelSolver
from utils.small_cnn import SmallCNN
from utils.allconv import AllConv
from utils.densenet import DenseNetGrow
from utils.resnet import ResNet
from utils.vgg import VGG
from utils.inception import Inception
from utils.mobilenet import MobileNetV1
from utils.plain_grow import PlainGrow
from utils.resnet_grow import ResNetGrow
from utils.efficientnet_grow import EfficientNetGrow

from tensorflow.python.keras.preprocessing import dataset_utils


data_splits = [9./10, 1./20, 1./20]
# data_splits = [98./100, 1./100, 1./100]
# from mpi4py import MPI

# TEST = True
TEST = False
TEST_SAMPLER=True


class Logger(object):
  """Logging object to write to file and stdout."""

  def __init__(self, filename):
    self.terminal = sys.stdout
    self.log = gfile.GFile(filename, "w")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    self.terminal.flush()

  def flush_file(self):
    self.log.flush()


def create_checker_unbalanced(split, n, grid_size):
  """Creates a dataset with two classes that occupy one color of checkboard.

  Args:
    split: splits to use for class imbalance.
    n: number of datapoints to sample.
    grid_size: checkerboard size.
  Returns:
    X: 2d features.
    y: binary class.
  """
  y = np.zeros(0)
  X = np.zeros((0, 2))
  for i in range(grid_size):
    for j in range(grid_size):
      label = 0
      n_0 = int(n/(grid_size*grid_size) * split[0] * 2)
      if (i-j) % 2 == 0:
        label = 1
        n_0 = int(n/(grid_size*grid_size) * split[1] * 2)
      x_1 = np.random.uniform(i, i+1, n_0)
      x_2 = np.random.uniform(j, j+1, n_0)
      x = np.vstack((x_1, x_2))
      x = x.T
      X = np.concatenate((X, x))
      y_0 = label * np.ones(n_0)
      y = np.concatenate((y, y_0))
  return X, y


def flatten_X(X):
  shape = X.shape
  flat_X = X
  if len(shape) > 2:
    flat_X = np.reshape(X, (shape[0], np.product(shape[1:])))
  return flat_X

def load_imagenet_is_a_file(is_a_fp):
    # load the is-a file as a dictionary
    is_a = open(is_a_fp,"r")
    parent_child = dict()
    for mapping in is_a:
        fields = mapping.split(' ')
        parent = fields[0].strip()
        child = fields[1].strip()
        # print(parent)
        # print(child)
        parent_child[child] = parent
    return parent_child

def load_imagenet_sorted_classes(class_file):
    classes = []
    f = open(class_file,"r")
    for c in f:
        classes.append(c)
    return classes

def find_superclass_at_level(c, parent_child, level):
    c = c.strip()
    intermediate_parent = [c]
    while intermediate_parent[-1] in parent_child:
        intermediate_parent.append(parent_child[intermediate_parent[-1]])
    
    # there is at least 2 levels, since every class of the 1000 belongs to n00001740 (entity)
    superclass_level = max(level,-len(intermediate_parent)) 
    return intermediate_parent[superclass_level]

def remap_imagenet_superclassindex(y):
    # print(set(y))
    dir = '../dataset/ImageNet/raw_data/'
    classes = load_imagenet_sorted_classes(dir + "classes.txt")
    parent_child = load_imagenet_is_a_file(dir + "wordnet.is_a.txt")
    lvl = -5
    superclasses = []
    mapping = dict()
    for i in range(1000):
        c = classes[i]
        superclass = find_superclass_at_level(c, parent_child, lvl)
        if superclass not in superclasses:
            superclasses.append(superclass)
        mapping[i] = superclasses.index(superclass)
    # remap
    for j in range(len(y)):
        y[j] = mapping[y[j]]
    print("remapped classes: {}".format(set(y)))
    return y

def get_mldata(data_dir, name):
  """Loads data from data_dir.

  Looks for the file in data_dir.
  Assumes that data is in pickle format with dictionary fields data and target.


  Args:
    data_dir: directory to look in
    name: dataset name, assumes data is saved in the save_dir with filename
      <name>.pkl
  Returns:
    data and targets
  Raises:
    NameError: dataset not found in data folder.
  """
  dataname = name
  print("Loading {}...".format(dataname))
  if dataname == "checkerboard":
    X, y = create_checker_unbalanced(split=[1./5, 4./5], n=10000, grid_size=4)
  # elif dataname == "imagenet":
  #   filename = os.path.join(data_dir, dataname + ".h5")
  #
  #   if not gfile.exists(filename):
  #     raise NameError("ERROR: dataset {} not available".format(filename))
  #
  #   # rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
  #   # f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
  #   hf = h5py.File(filename, 'r')
  #   print("Getting data...")
  #   # X = hf.get("data")[:]
  #   X = hf["data"]
  #   # X = HDF5Matrix(filename, 'data')
  #   print("Getting labels...")
  #   # y = hf.get("label")[:]
  #   y = hf["label"][:]
  #   # y = HDF5Matrix(filename, 'label')
  #   print("Remapping...")
  #   y = remap_imagenet_superclassindex(y)
  #   # X = X / 255.
  #   y = y.flatten()
  else:
    filename = os.path.join(data_dir, dataname + ".pkl")
    if not gfile.exists(filename):
      raise NameError("ERROR: dataset {} not available".format(filename))
    data = pickle.load(gfile.GFile(filename, "rb"))
    X = data["data"]
    y = data["target"]
    if "keras" in dataname or "svhn" in dataname:
      X = X / 255.
      y = y.flatten()
  return X, y


def filter_data(X, y, keep=None):
  """Filters data by class indicated in keep.

  Args:
    X: train data
    y: train targets
    keep: defaults to None which will keep everything, otherwise takes a list
      of classes to keep

  Returns:
    filtered data and targets
  """
  if keep is None:
    return X, y
  keep_ind = [i for i in range(len(y)) if y[i] in keep]
  return X[keep_ind], y[keep_ind]


def get_class_counts(y_full, y):
  """Gets the count of all classes in a sample.

  Args:
    y_full: full target vector containing all classes
    y: sample vector for which to perform the count
  Returns:
    count of classes for the sample vector y, the class order for count will
    be the same as long as same y_full is fed in
  """
  classes = np.unique(y_full)
  classes = np.sort(classes)
  unique, counts = np.unique(y, return_counts=True)
  complete_counts = []
  for c in classes:
    if c not in unique:
      complete_counts.append(0)
    else:
      index = np.where(unique == c)[0][0]
      complete_counts.append(counts[index])
  return np.array(complete_counts)


def flip_label(y, percent_random):
  """Flips a percentage of labels for one class to the other.

  Randomly sample a percent of points and randomly label the sampled points as
  one of the other classes.
  Does not introduce bias.

  Args:
    y: labels of all datapoints
    percent_random: percent of datapoints to corrupt the labels

  Returns:
    new labels with noisy labels for indicated percent of data
  """
  classes = np.unique(y)
  y_orig = copy.copy(y)
  indices = list(range(y_orig.shape[0]))
  np.random.shuffle(indices)
  sample = indices[0:int(len(indices) * 1.0 * percent_random)]
  fake_labels = []
  for s in sample:
    label = y[s]
    class_ind = np.where(classes == label)[0][0]
    other_classes = np.delete(classes, class_ind)
    np.random.shuffle(other_classes)
    fake_label = other_classes[0]
    assert fake_label != label
    fake_labels.append(fake_label)
  y[sample] = np.array(fake_labels)
  assert all(y[indices[len(sample):]] == y_orig[indices[len(sample):]])
  return y


def get_model(method, seed=13, augmentation=0, batch=32, epochs=200):
  """Construct sklearn model using either logistic regression or linear svm.

  Wraps grid search on regularization parameter over either logistic regression
  or svm, returns constructed model

  Args:
    method: string indicating scikit method to use, currently accepts logistic
      and linear svm.
    seed: int or rng to use for random state fed to scikit method

  Returns:
    scikit learn model
  """
  # TODO(lishal): extend to include any scikit model that implements
  #   a decision function.
  # TODO(lishal): for kernel methods, currently using default value for gamma
  # but should probably tune.

  if method == "logistic":
    model = LogisticRegression(random_state=seed, multi_class="multinomial",
                               solver="lbfgs", max_iter=200)
    params = {"C": [10.0**(i) for i in range(-4, 5)]}
  elif method == "logistic_ovr":
    model = LogisticRegression(random_state=seed)
    params = {"C": [10.0**(i) for i in range(-5, 4)]}
  elif method == "linear_svm":
    model = LinearSVC(random_state=seed)
    params = {"C": [10.0**(i) for i in range(-4, 5)]}
  elif method == "kernel_svm":
    model = SVC(random_state=seed)
    params = {"C": [10.0**(i) for i in range(-4, 5)]}
  elif method == "kernel_ls":
    model = BlockKernelSolver(random_state=seed)
    params = {"C": [10.0**(i) for i in range(-6, 1)]}
  elif method == "small_cnn":
    # Model does not work with weighted_expert or simulate_batch
    model = SmallCNN(random_state=seed)
    return model
  elif method == "allconv":
    # Model does not work with weighted_expert or simulate_batch
    model = AllConv(random_state=seed)
    return model
  elif method == "densenet":
    # Model does not work with weighted_expert or simulate_batch
    model = DenseNetGrow(random_state=seed, augmentation=augmentation, batch_size=batch)
    return model
  elif method == "resnet":
    # Model does not work with weighted_expert or simulate_batch
    model = ResNet(random_state=seed)
    return model
  elif method == "vgg":
    # Model does not work with weighted_expert or simulate_batch
    model = VGG(random_state=seed)
    return model
  elif method == "inception":
    # Model does not work with weighted_expert or simulate_batch
    model = Inception(random_state=seed)
    return model
  elif method == "mobilenet":
    # Model does not work with weighted_expert or simulate_batch
    model = MobileNetV1(random_state=seed)
    return model
  elif method == "plain_grow":
    # Model does not work with weighted_expert or simulate_batch
    model = PlainGrow(random_state=seed, augmentation=augmentation, batch_size=batch, epochs=epochs)
    return model
  elif method == "resnet_grow":
    # Model does not work with weighted_expert or simulate_batch
    model = ResNetGrow(random_state=seed, augmentation=augmentation, batch_size=batch, epochs=epochs)
    return model
  elif method == "efficient_grow":
    # Model does not work with weighted_expert or simulate_batch
    model = EfficientNetGrow(random_state=seed, augmentation=augmentation, batch_size=batch, epochs=epochs)
    return model
  # elif method == "autokeras":
  #   # Model does not work with weighted_expert or simulate_batch
  #   model = AutoKeras(random_state=seed)
  #   return model



  else:
    raise NotImplementedError("ERROR: " + method + " not implemented")

  model = GridSearchCV(model, params, cv=3)
  return model


def calculate_entropy(batch_size, y_s):
  """Calculates KL div between training targets and targets selected by AL.

  Args:
    batch_size: batch size of datapoints selected by AL
    y_s: vector of datapoints selected by AL.  Assumes that the order of the
      data is the order in which points were labeled by AL.  Also assumes
      that in the offline setting y_s will eventually overlap completely with
      original training targets.
  Returns:
    entropy between actual distribution of classes and distribution of
    samples selected by AL
  """
  n_batches = int(np.ceil(len(y_s) * 1.0 / batch_size))
  counts = get_class_counts(y_s, y_s)
  true_dist = counts / (len(y_s) * 1.0)
  entropy = []
  for b in range(n_batches):
    sample = y_s[b * batch_size:(b + 1) * batch_size]
    counts = get_class_counts(y_s, sample)
    sample_dist = counts / (1.0 * len(sample))
    entropy.append(scipy.stats.entropy(true_dist, sample_dist))
  return entropy

def h5py_fancy_indexing(data, indices):
    '''
    To spd up h5py clumsy fancy indexing method
    WARNING: MUST BE SORTED INDICES!!
    '''
    print("Fancy indexing {} indices...".format(len(indices)))
    ts = time.time()
    max_batch = 5000
    starting_index = 0
    tmp_ret = None
    while starting_index < len(data):
        chunk_ts = time.time()
        
        
        # idx = [i for i in indices if ( (i >= starting_index) and (i < (starting_index + max_batch)) )]
        # print("filtered in {} sec".format(time.time()-ts))
        idx = np.asarray(indices)
        idx = idx[idx >= starting_index]
        idx = idx[idx < (starting_index + max_batch)]
        idx = idx - starting_index

        if len(idx) == 0:
            starting_index += max_batch
            continue
        tmp_data = data[starting_index:(starting_index+max_batch)]
        if tmp_ret is None:
            tmp_ret = tmp_data[idx]
        else:
            tmp_ret = np.concatenate([tmp_ret, tmp_data[idx]])
        
        print("Indexing {}~{} in {} sec".format(str(starting_index), str(starting_index+max_batch), time.time()-chunk_ts))
        starting_index += max_batch

    print("Selected {} samples in {} sec".format(tmp_ret.shape[0], time.time()-ts))



    return tmp_ret


# ''' Still too slow... try for loop '''
# def h5py_fancy_indexing(data, indices):
#     '''
#     To spd up h5py clumsy fancy indexing method
#     '''
#     print("Fancy indexing {} indices...".format(len(indices)))
#     ts = time.time()
#     tmp_ret = []
#     count = 0
#     sorted_indices = sorted(indices)
#     for idx in sorted_indices:
#         chunk_ts = time.time()
#         tmp_data = data[idx]

#         tmp_ret.append(tmp_data)
    
#         print("Indexing no. {} ({}) in {} sec".format(count, idx,  time.time()-chunk_ts))
#         count += 1
#     print("Selected {} samples in {} sec".format(len(tmp_ret), time.time()-ts))

#     sort_t = time.time()
#     ret = []
#     for i in range(len(indices)):
#         query_t = time.time()
#         ret[i] = tmp_ret[sorted_indices.index(indices[i])]
#         print("Queried index {} in {} sec".format(i, time.time()-query_t))
#     print("reordered in {} sec".format(time.time()-sort_t))
#     return ret

# ''' Still too slow... try for loop with multi-threading... '''
# def h5py_fancy_indexing(data, indices):
#     '''
#     To spd up h5py clumsy fancy indexing method
#     '''
#     print("Fancy indexing {} indices...".format(len(indices)))
#     ts = time.time()
#     tmp_ret = []
#     count = 0
#     sorted_indices = sorted(indices)
#     for idx in sorted_indices:
#         chunk_ts = time.time()
#         tmp_data = data[idx]

#         tmp_ret.append(tmp_data)
    
#         print("Indexing no. {} ({}) in {} sec".format(count, idx,  time.time()-chunk_ts))
#         count += 1
#     print("Selected {} samples in {} sec".format(len(tmp_ret), time.time()-ts))

#     sort_t = time.time()
#     ret = []
#     for i in range(len(indices)):
#         query_t = time.time()
#         ret[i] = tmp_ret[sorted_indices.index(indices[i])]
#         print("Queried index {} in {} sec".format(i, time.time()-query_t))
#     print("reordered in {} sec".format(time.time()-sort_t))
#     return ret


def get_train_val_test_splits(X, y, max_points, seed, confusion, seed_batch,
                              split=(2./3, 1./6, 1./6)):
  """Return training, validation, and test splits for X and y.

  Args:
    X: features
    y: targets
    max_points: # of points to use when creating splits.
    seed: seed for shuffling.
    confusion: labeling noise to introduce.  0.1 means randomize 10% of labels.
    seed_batch: # of initial datapoints to ensure sufficient class membership.
    split: percent splits for train, val, and test.
  Returns:
    indices: shuffled indices to recreate splits given original input data X.
    y_noise: y with noise injected, needed to reproduce results outside of
      run_experiments using original data.
  """
  np.random.seed(seed)
  # X_copy = copy.copy(X)
  y_copy = copy.copy(y)

  # Introduce labeling noise
  y_noise = flip_label(y_copy, confusion)

  indices = np.arange(len(y))

  if max_points is None:
    max_points = len(y_noise)
  else:
    max_points = min(len(y_noise), max_points)
  train_split = int(max_points * split[0])
  val_split = train_split + int(max_points * split[1])
  print("Seed {}, train {}".format(seed_batch, train_split))
  assert seed_batch <= train_split

  # Do this to make sure that the initial batch has examples from all classes
  min_shuffle = 3
  max_shuffle = 10
  n_shuffle = 0
  y_tmp = y_noise

  # Need at least 4 obs of each class for 2 fold CV to work in grid search step
  while (any(get_class_counts(y_tmp, y_tmp[0:seed_batch]) < 4)
         or n_shuffle < min_shuffle):
    # print("Shuffling Indexes...")
    np.random.shuffle(indices)
    y_tmp = y_noise[indices]
    n_shuffle += 1
    if n_shuffle > max_shuffle:
      break


  # X_train = np.array()
  # print("Index X_train")
  # ts = time.time()
  # n=5000
  # X_train = X_copy[:n]
  # print("Chunky slicing of {} takes {} sec".format(X_train.shape[0], time.time()-ts))
  
  # for i in indices[0:train_split]:
  #     X_train = X_copy[i]
  #     print("loaded {}".format(i)) # takes about 1 sec per sample

  '''
  a little hack here, if X_train is too large, just return h5 dataset, instead of the np array
  the h5 dataset could have the wrong length, which is the length including the val and test... for now
  '''
  
  train_t = time.time()
  # if train_split > 200000:
  X_train = X
  y_train = y_noise
  # else:
  #     X_train = X[sorted(indices[0:train_split])]
  #     y_train = y_noise[sorted(indices[0:train_split])]
  # X_train = h5py_fancy_indexing(X_copy, indices)
  print("Index X_train in {} sec".format(time.time()-train_t))

  val_t = time.time()
  X_val = X[sorted(indices[train_split:val_split])]
  # X_val = h5py_fancy_indexing(X, sorted(indices[train_split:val_split]))
  print("Index X_val in {} sec".format(time.time()-val_t))
  
  test_t = time.time()
  X_test = X[sorted(indices[val_split:max_points])]
  # X_test = h5py_fancy_indexing(X, sorted(indices[val_split:max_points]))
  print("Index X_test in {} sec".format(time.time()-test_t))
  
  y_val = y_noise[sorted(indices[train_split:val_split])]
  print("Index y_val")
  
  y_test = y_noise[sorted(indices[val_split:max_points])]
  print("Index y_test")

  # for testing purpose only
  
  # val_t = time.time()
  # X_val = X[train_split:val_split]
  # print("Index X_val {} in {} sec".format(X_val.shape[0], time.time()-train_t))
  
  # test_t = time.time()
  # X_test = X[val_split:max_points]
  # print("Index X_test {} in {} sec".format(X_test.shape[0], time.time()-test_t))
  
  # y_val = y_noise[train_split:val_split]
  # print("Index y_val {}".format(y_val.shape[0]))
  
  # y_test = y_noise[val_split:max_points]
  # print("Index y_test {}".format(y_test.shape[0]))

  # Make sure that we have enough observations of each class for 2-fold cv
  # assert all(get_class_counts(y_noise, y_train[0:seed_batch]) >= 4)
  # Make sure that returned shuffled indices are correct
  # assert all(y_noise[indices[0:max_points]] == np.concatenate((y_train, y_val, y_test), axis=0))
  return (indices[0:max_points], X_train, y_train,
          X_val, y_val, X_test, y_test, y_noise)



def eval_margin_acc(X_train, y_train, min, min_margin, model, in_selected, already_selected=[], depth=10):
    step = (1.0 - min) / depth
    # hist = np.histogram(min_margin, range=[min, 1])
    hist = []
    acc = []
    for m in np.arange(min, 1., step):
      n = m + step
      if in_selected:
        ind = [x for x in range(len(min_margin)) if m < min_margin[x] <= n and (x in already_selected)]
      else:
        ind = [x for x in range(len(min_margin)) if m < min_margin[x] <= n and (x not in already_selected)]
      if len(ind) == 0:
        acc.append(0)
        hist.append(0)
        continue
      data_x = X_train[sorted(ind)]
      # data_x = h5py_fancy_indexing(X_train, sorted(ind))
      data_y = y_train[sorted(ind)]
      acc.append(model.score(data_x, data_y))
      hist.append(len(ind))

    return hist, acc


def eval_margin_acc_logscale(X_train, y_train, min_margin, model, in_selected, already_selected=[], depth=20):
  # print("eval margin accuracy (log scale)")

  step = 0.1
  acc = []
  hist = []
  depth_count = 0
  for m in np.logspace(1, depth, base=step, num=depth):
    lower = 1. - m
    upper = 1. - m * step
    depth_count += 1
    if depth_count == depth:
      upper = 1.0

    if in_selected:
      ind = [x for x in range(len(min_margin)) if (lower < min_margin[x] <= upper) and (x in already_selected)]
    else:
      ind = [x for x in range(len(min_margin)) if (lower < min_margin[x] <= upper) and (x not in already_selected)]
    if len(ind) == 0:
      acc.append(0)
      hist.append(0)
      continue
    data_x = X_train[sorted(ind)]
    # data_x = h5py_fancy_indexing(X_train, sorted(ind))
    data_y = y_train[sorted(ind)]
    acc.append(model.score(data_x, data_y))
    hist.append(len(ind))

  return hist, acc

from tqdm import tqdm

def list_subtract(a,b):
    # a = a.tolist()
    # b = b.tolist()
    if TEST:
        return a
    for x in tqdm(b):
        try:
            a.remove(x)
        except ValueError:
            pass
    return a


'''
New acc hist sweep function, 
- dynamically load the data and evaluate
- remove the selection overhead of sweeping different percentage ratios
'''
def acc_hist_sweep(X_train, y_train, metrics_out, model, already_selected=[], intervals=[], fixed_amount=False, image_path_mode=False, al_metric='margin', num_classes=None, minibatch=None, class_names=None):
    
    print("Acc Hist Sweep...")
    ts = time.time()
    hist = []
    acc = []
    # print(min_margin)
    # print(len(min_margin))
    """todo: assuming metrics in increasing order, no compatible with entorpy """
    if al_metric in ['entropy', 'kcenter']:
        rank_ind_desc = np.argsort(metrics_out)
    else:
        metrics_out = -metrics_out
        rank_ind_desc = np.argsort(metrics_out)
        metrics_out = -metrics_out

    # rank_ind_desc = [i for i in rank_ind_desc if i not in already_selected]
    rank_ind_desc = list_subtract(rank_ind_desc.tolist(), already_selected)
    print("index filtered: {} sec".format(time.time()-ts))

    # Default intervals [0.05:0.05:1]
    if len(intervals) == 0:
        intervals = list(range(5, 101, 5)) # was intervals = list(range(5, 101, 5))
        intervals = list(np.asarray(intervals) / 100.0)

    total_count = len(rank_ind_desc)
    if fixed_amount:
      total_count = len(metrics_out)

    # get the data chunk by chunk and evaluate on the fly... 
    # use the percentage as the chunk size for convenience
    chunk_size = int(0.05 * total_count)

    count_evaled = 0
    for ratio in intervals:
        count = int(ratio * total_count)

        if count > len(rank_ind_desc) or count == 0:
            hist.append(-1)
            acc.append(-1)
            continue
        

        inds = rank_ind_desc[count_evaled:count]
        data_x = X_train[sorted(inds)]
        # data_y = y_train[inds]
        # data_x = h5py_fancy_indexing(X_train, sorted(inds))
        data_y = y_train[sorted(inds)]
        hist.append(count)
        print("Scoring {} samples".format(len(data_x)))
        sts = time.time()

        if image_path_mode:
            data_x_ds = image_dataset_loader.image_dataset_from_image_paths(data_x, data_y, num_classes,
                                                                               image_size=image_dataset_loader.image_size,
                                                                               batch_size=minibatch,
                                                                               label_mode="categorical", normalization=True)
            # data_x_ds = image_dataset_loader.image_dataset_iterator_from_image_paths(data_x, data_y, num_classes,
            #                                                                 image_size=image_dataset_loader.image_size,
            #                                                                 batch_size=minibatch,
            #                                                                 label_mode="raw",
            #                                                                 normalization=True, class_names=class_names)
            chunk_acc = model.score(data_x_ds, data_y)
        else:
            chunk_acc = model.score(data_x, data_y)

        acc_evaled = 0
        if len(acc) != 0:
            acc_evaled = acc[-1]
        interval_acc = ( acc_evaled * count_evaled + chunk_acc * (count-count_evaled) ) / float(count)        
        print("Chunk acc: {}({}), evaled acc: {}({}), Interval Acc: {}({})".format(chunk_acc, str(count-count_evaled), acc_evaled, count_evaled, interval_acc, count ))
        acc.append(interval_acc)
        print("{} sec".format(time.time()-sts))

        count_evaled = count

    print("Acc Hist Done: {} sec".format(time.time()-ts))
    return hist, acc, len(rank_ind_desc)

# def acc_hist_sweep(X_train, y_train, min_margin, model, already_selected=[], intervals=[], fixed_amount=False):
    
#     print("Acc Hist Sweep...")
#     ts = time.time()
#     hist = []
#     acc = []
#     min_margin = -min_margin
#     rank_ind_desc = np.argsort(min_margin)
#     min_margin = -min_margin

#     # rank_ind_desc = [i for i in rank_ind_desc if i not in already_selected]
#     rank_ind_desc = list_subtract(rank_ind_desc.tolist(), already_selected)
#     print("index filtered: {} sec".format(time.time()-ts))

#     # count_ratio = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1]
#     # for i in range(len(count_ratio)):
#     # for i in range(5, 100, 5):

#     # Default intervals [0.05:0.05:1]
#     if len(intervals) == 0:
#         intervals = list(range(5, 101, 5)) # was intervals = list(range(5, 101, 5))
#         intervals = list(np.asarray(intervals) / 100.0)

#     total_count = len(rank_ind_desc)
#     if fixed_amount:
#       total_count = len(min_margin)

#     for ratio in intervals:
#         count = int(ratio * total_count)

#         if count > len(rank_ind_desc) or count == 0:
#             hist.append(0)
#             acc.append(0)
#             continue
#         inds = rank_ind_desc[0:count]
#         # data_x = X_train[inds]
#         # data_y = y_train[inds]
#         data_x = h5py_fancy_indexing(X_train, inds)
#         data_y = y_train[inds]
#         hist.append(count)
#         print("Scoring {} samples".format(data_x.shape[0]))
#         sts = time.time()
#         acc.append(model.score(data_x, data_y))
#         print("{} sec".format(time.time()-sts))

#     print("Acc Hist Done: {} sec".format(time.time()-ts))
#     return hist, acc, len(rank_ind_desc)




def get_remaining_correct_predictions_at_accruacy_threshold(X_train, y_train, acc_thresh, min_margin, model,
                                                              already_selected=[], last_metric_thresh=-1):


    step = 100
    high_conf_ind = []
    correct_prediction_ind = []
    min_margin = -min_margin
    rank_ind_desc = np.argsort(min_margin)
    min_margin = -min_margin

    rank_ind_desc = [i for i in rank_ind_desc if i not in already_selected]

    print("eval model correct predictions over the remaining {} samples".format(len(rank_ind_desc)))

    margin_thresh = 1.

    count = 1000 # TODO: change to dataset agnostic value, e.g. 1%
    if last_metric_thresh != -1:
        remaining_margin = min_margin[rank_ind_desc]
        count = sum(remaining_margin > last_metric_thresh)


    acc = 1.0
    while count < len(rank_ind_desc):
        high_conf_ind = rank_ind_desc[0:count]
        data_x = X_train[sorted(high_conf_ind)]
        # data_y = y_train[high_conf_ind]
        # data_x = h5py_fancy_indexing(X_train, sorted(high_conf_ind))
        data_y = y_train[sorted(high_conf_ind)]
        acc = model.score(data_x, data_y)

        print("Scoring at top {}: Metric={}, Acc={}".format(len(high_conf_ind), min_margin[high_conf_ind[-1]], acc))

        if acc >= acc_thresh:
            correct_prediction_ind = high_conf_ind
            count += step
        else:
            # roll back the last bucket
            high_conf_ind = correct_prediction_ind
            count = len(high_conf_ind)
            if count == 0:
                # nothing gets trained yet
                acc = 1.0
                break
            margin_thresh = min_margin[high_conf_ind[-1]]
            break

    return correct_prediction_ind, acc, margin_thresh


# def get_remaining_correct_predictions_at_accruacy_threshold(X_train, y_train, acc_thresh, min_margin, model,
#                                                             already_selected=[], depth=20):
#   # print("eval model correct predictions over the remaining {} samples".format(len(min_margin)))
#   step = 10
#   high_conf_ind = []
#   correct_prediction_ind = []
#   min_margin = -min_margin
#   rank_ind_desc = np.argsort(min_margin)
#   min_margin = -min_margin
#
#   margin_thresh = 1.
#
#   for m in np.logspace(1, depth, base=step, num=depth) / 1e20:
#     # from high acc to low acc
#     lower = 1. - m
#     upper = 1.
#
#     ind = [x for x in range(len(min_margin)) if (lower < min_margin[x] <= upper) and (x not in already_selected)]
#
#     if len(ind) == 0:
#       continue
#
#     high_conf_ind = ind
#     # print(high_conf_ind)
#     data_x = X_train[high_conf_ind]
#     data_y = y_train[high_conf_ind]
#     acc = model.score(data_x, data_y)
#
#     print("Scoring at top {}: {}".format(len(high_conf_ind), acc))
#
#     if acc >= acc_thresh:
#       correct_prediction_ind = ind
#     else:
#       # retrace back the last bucket
#       high_conf_ind = correct_prediction_ind
#       count = len(high_conf_ind)
#       if count == 0:
#         # nothing gets trained yet
#         break
#       # find the right index to start
#       x = 0
#       for i in range(count):
#         while rank_ind_desc[x] in already_selected:
#           x += 1
#         x += 1
#       # include each one in desc order, and eval until threshold break
#       acc = 1.0
#       while acc >= acc_thresh and x < len(min_margin):
#         margin_thresh = min_margin[high_conf_ind[-1]]
#         correct_prediction_ind = high_conf_ind
#
#         y = x + max(int(x * (acc - acc_thresh)), 10)
#         next_batch_ind = [rank_ind_desc[i] for i in range(x, y) if rank_ind_desc[i] not in already_selected]
#         x = y
#         high_conf_ind += next_batch_ind
#         data_x = X_train[high_conf_ind]
#         data_y = y_train[high_conf_ind]
#         acc = model.score(data_x, data_y)
#         print("Scoring at top {}: {}".format(len(high_conf_ind), acc))
#
#       break
#
#   return correct_prediction_ind, acc, margin_thresh

label_cost_per_image = 0.003
GPU_cost_per_sec_Azure = 1.084 / 3600

def log_cost(FLAGS, acc, acc_thresh, train_size, total_training_time, batch_training_time, total_selection_time, total_profiling_time, n_train, t_start, n_nextbatchtotal=0, predicted_cost_nextbatch=0):
    # log the cost
    CorrPredRatio_thisBatch = 1.
    for i in range(len(acc)):
        if acc[i] < acc_thresh:
            CorrPredRatio_thisBatch = i * 0.05
            break

    
    
    noGPUs = len(FLAGS.gpu.split(','))
    GPU_cost_per_sec = GPU_cost_per_sec_Azure * noGPUs

    total_predicted = CorrPredRatio_thisBatch * train_size
    total_labeled = train_size - total_predicted
    extra_labeled = total_labeled - n_train
    total_label_cost = label_cost_per_image * total_labeled
    total_training_cost = GPU_cost_per_sec * total_training_time
    total_selection_cost = GPU_cost_per_sec * total_selection_time
    total_profiling_cost = GPU_cost_per_sec * total_profiling_time
    total_cost = total_label_cost + total_training_cost + total_selection_cost + total_profiling_cost
    total_time = time.time()-t_start

    f = open(FLAGS.save_dir + "/GPU{}_cost.txt".format(FLAGS.gpu), "a")
    f.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
        n_train, # len(selected_inds),
        n_nextbatchtotal, predicted_cost_nextbatch,
        total_time, batch_training_time, 
        total_training_time, total_training_cost,
        total_selection_time, total_selection_cost,
        total_profiling_time, total_profiling_cost,
        CorrPredRatio_thisBatch, total_predicted, 
        extra_labeled, total_labeled, total_label_cost,
        total_cost))
    f.close()  
    return total_cost

def load_data(FLAGS):
    data_t = time.time()
    print("Loading Data...")
    ALLOWLIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')
    image_path_mode = False
    num_classes = None
    class_names = None
    if FLAGS.dataset == 'image_path_dataset':
        image_paths, labels, class_names = dataset_utils.index_directory(
            FLAGS.dataset_dir,
            labels='inferred',
            formats=ALLOWLIST_FORMATS,
            class_names=None,
            shuffle=True,
            seed=None,
            follow_links=False)
        X = np.asarray(image_paths)
        y = np.asarray(labels)
        image_path_mode = True
        num_classes = len(class_names)
        print("Collecting {} data points\n{}...".format(len(image_paths), image_paths[0]))
        print("Added {} labelse\n{}({})...".format(len(labels), labels[0], class_names[labels[0]]))

    else:
        X, y = get_mldata(FLAGS.data_dir, FLAGS.dataset)
    print("Data loaded: {} sec".format(time.time() - data_t))

    return X, y, image_path_mode, num_classes, class_names

from sampling_methods.constants import AL_MAPPING

def init_samplers(FLAGS, sampler, labeling_sampler, X_train, y_train, X_val, y_val, X_test, y_test, seed, image_path_mode=False, num_classes=None,class_names=None):
    print("Initialize Samplers...")
    uniform_sampler = None
    train_sampler = None
    machine_labeling_sampler = None
    X_train_ds = None
    X_val_ds = None
    X_test_ds = None
    if image_path_mode:
        X_train_path = X_train
        X_val_path = X_val
        X_test_path = X_test
        # X_train_ds = image_dataset_loader.image_dataset_iterator_from_image_paths(X_train_path, y_train, num_classes,
        #                                                                           image_size=image_dataset_loader.image_size,
        #                                                                           batch_size=FLAGS.minibatch,
        #                                                                           label_mode="raw", normalization=True,
        #                                                                           class_names=class_names)
        X_train_ds = image_dataset_loader.image_dataset_from_image_paths(X_train_path, y_train, num_classes,
                                                                         image_size=image_dataset_loader.image_size,
                                                                         batch_size=FLAGS.minibatch,
                                                                         label_mode="categorical", normalization=True)
        """validation data doesnot support generator type"""
        # X_val_ds = image_dataset_loader.image_dataset_iterator_from_image_paths(X_val_path, y_val, num_classes,
        #                                                                image_size=image_dataset_loader.image_size,
        #                                                                batch_size=FLAGS.minibatch,
        #                                                                label_mode="raw", normalization=True, class_names=class_names)
        X_val_ds = image_dataset_loader.image_dataset_from_image_paths(X_val_path, y_val, num_classes,
                                                                       image_size=image_dataset_loader.image_size,
                                                                       batch_size=FLAGS.minibatch,
                                                                       label_mode="categorical", normalization=True)
        # X_test_ds = image_dataset_loader.image_dataset_iterator_from_image_paths(X_test_path, y_test, num_classes,
        #                                                                 batch_size=FLAGS.minibatch,
        #                                                                 label_mode="raw", normalization=True, class_names=class_names)
        X_test_ds = image_dataset_loader.image_dataset_from_image_paths(X_test_path, y_test, num_classes,
                                                                        batch_size=FLAGS.minibatch,
                                                                        label_mode="categorical", normalization=True)




        uniform_sampler = AL_MAPPING["uniform"](X_train_ds, y_train, seed, total_size=len(X_train_path))
        train_sampler = sampler(X_train_ds, y_train, seed, total_size=len(X_train_path))
        machine_labeling_sampler = labeling_sampler(X_train_ds, y_train, seed, total_size=len(X_train_path))
    else:
        uniform_sampler = AL_MAPPING["uniform"](X_train, y_train, seed, total_size=len(X_train))
        train_sampler = sampler(X_train, y_train, seed, total_size=len(X_train))
        machine_labeling_sampler = labeling_sampler(X_train, y_train, seed, total_size=len(X_train))

    return uniform_sampler, train_sampler, machine_labeling_sampler, X_train_ds, X_val_ds, X_test_ds