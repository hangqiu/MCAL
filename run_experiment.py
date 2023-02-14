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

"""Run active learner on classification tasks.

Supported datasets include mnist, letter, cifar10, newsgroup20, rcv1,
wikipedia attack, and select classification datasets from mldata.
See utils/create_data.py for all available datasets.

For binary classification, mnist_4_9 indicates mnist filtered down to just 4 and
9.
By default uses logistic regression but can also train using kernel SVM.
2 fold cv is used to tune regularization parameter over a exponential grid.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time
from time import gmtime
from time import strftime

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from absl import app
from absl import flags
from tensorflow.io import gfile

from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping
from utils import utils

from utils import image_dataset_loader


import gc
import h5py

flags.DEFINE_string("dataset", "letter", "Dataset name")
flags.DEFINE_string("dataset_dir", "letter", "Dataset Directory")
flags.DEFINE_string("sampling_method", "margin",
                    ("Name of sampling method to use, can be any defined in "
                     "AL_MAPPING in sampling_methods.constants"))
flags.DEFINE_string("labeling_sampling_method", "margin",
                    ("Name of machine labeling sampling method to use, can be any defined in "
                     "AL_MAPPING in sampling_methods.constants"))
flags.DEFINE_string(
    "warmstart_size", "0.02",
    ("Can be float or integer.  Float indicates percentage of training data "
     "to use in the initial warmstart model")
)
flags.DEFINE_string(
    "batch_size", "0.02",
    ("Can be float or integer.  Float indicates batch size as a percentage "
     "of training data size.")
)
flags.DEFINE_integer("trials", 1,
                     "Number of curves to create using different seeds")
flags.DEFINE_integer("seed", 1, "Seed to use for rng and random state")
# TODO(lisha): add feature noise to simulate data outliers
flags.DEFINE_string("confusions", "0.", "Percentage of labels to randomize")
flags.DEFINE_string("active_sampling_percentage", "1.0",
                    "Mixture weights on active sampling.")
flags.DEFINE_string(
    "score_model", "logistic",
    "Method to use to calculate accuracy.")
flags.DEFINE_string(
    "select_model", "None",
    "Method to use for selecting points.")
flags.DEFINE_string("normalize_data", "False", "Whether to normalize the data.")
flags.DEFINE_string("standardize_data", "True",
                    "Whether to standardize the data.")
flags.DEFINE_string("save_dir", "./results/",
                    "Where to save outputs")
flags.DEFINE_string("data_dir", "/tmp/data",
                    "Directory with predownloaded and saved datasets.")
flags.DEFINE_string("max_dataset_size", "100000000000",
                    ("maximum number of datapoints to include in data "
                     "zero indicates no limit"))
flags.DEFINE_float("train_horizon", "1.0",
                   "how far to extend learning curve as a percent of train")
flags.DEFINE_string("do_save", "True",
                    "whether to save log and results")
flags.DEFINE_string("gpu", "0", "gpu id")
flags.DEFINE_integer("cell", -1, "how many decimation, -1 means not a growing model")
flags.DEFINE_integer("layer", 1, "whether to save log and results")
flags.DEFINE_integer("kernel", 32, "whether to save log and results")
flags.DEFINE_integer("augmentation", 0, "whether to augment data, not for mnist")
flags.DEFINE_float("accthresh", "0.95", "accuracy requirement")
flags.DEFINE_integer("minibatch", 32, "mini batch sizes")
flags.DEFINE_string("test", "False",
                    "Whether to test with minimal execution overhead")
flags.DEFINE_string("profile", "False",
                    "Whether to profile al metrics")
FLAGS = flags.FLAGS


get_wrapper_AL_mapping()


def generate_one_curve(X,
                       y,
                       sampler,
                       labeling_sampler,
                       score_model,
                       seed,
                       warmstart_size,
                       batch_size,
                       select_model=None,
                       confusion=0.,
                       active_p=1.0,
                       max_points=None,
                       standardize_data=False,
                       norm_data=False,
                       train_horizon=0.5,
                       image_path_mode=False,
                       num_classes=None,
                       class_names=None
                       ):
  """Creates one learning curve for both active and passive learning.

  Will calculate accuracy on validation set as the number of training data
  points increases for both PL and AL.
  Caveats: training method used is sensitive to sorting of the data so we
    resort all intermediate datasets

  Args:
    X: training data
    y: training labels
    sampler: sampling class from sampling_methods, assumes reference
      passed in and sampler not yet instantiated.
    score_model: model used to score the samplers.  Expects fit and predict
      methods to be implemented.
    seed: seed used for data shuffle and other sources of randomness in sampler
      or model training
    warmstart_size: float or int.  float indicates percentage of train data
      to use for initial model
    batch_size: float or int.  float indicates batch size as a percent of
      training data
    select_model: defaults to None, in which case the score model will be
      used to select new datapoints to label.  Model must implement fit, predict
      and depending on AL method may also need decision_function.
    confusion: percentage of labels of one class to flip to the other
    active_p: percent of batch to allocate to active learning
    max_points: limit dataset size for preliminary
    standardize_data: wheter to standardize the data to 0 mean unit variance
    norm_data: whether to normalize the data.  Default is False for logistic
      regression.
    train_horizon: how long to draw the curve for.  Percent of training data.

  Returns:
    results: dictionary of results for all samplers
    sampler_states: dictionary of sampler objects for debugging
  """
  # TODO(lishal): add option to find best hyperparameter setting first on
  # full dataset and fix the hyperparameter for the rest of the routine
  # This will save computation and also lead to more stable behavior for the
  # test accuracy

  # TODO(lishal): remove mixture parameter and have the mixture be specified as
  # a mixture of samplers strategy
  def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
                   **kwargs):
    if utils.TEST and not utils.TEST_SAMPLER:
        total = sampler.total_size
        assert total is not None
        new_batch = np.linspace(0,total-1,total,dtype=int).tolist()
        for x in already_selected:
            try:
                new_batch.remove(x)
            except ValueError:
                pass
        return new_batch[0:N], np.zeros(sampler.total_size)
    n_active = int(mixture * N)
    # n_passive = N - n_active
    kwargs["N"] = n_active
    kwargs["already_selected"] = already_selected
    batch_AL, metrics = sampler.select_batch(**kwargs)
    # already_selected = already_selected + batch_AL
    # kwargs["N"] = n_passive
    # kwargs["already_selected"] = already_selected
    # batch_PL, _ = uniform_sampler.select_batch(**kwargs)
    # return batch_AL + batch_PL, metrics
    return batch_AL, metrics

  def profile_samplers(FLAGS, X, Y, seed, uniform_sampler, active_p, n_sample, selected_inds, total_size, **select_batch_inputs):

      kcenter_sampler = get_AL_sampler('kcenter')(X, Y, seed, total_size=total_size)
      margin_sampler = get_AL_sampler('margin')(X, Y, seed, total_size=total_size)
      entropy_sampler = get_AL_sampler('entropy')(X, Y, seed, total_size=total_size)
      least_confidence_sampler = get_AL_sampler('least_confidence')(X, Y, seed, total_size=total_size)

      kcenter_selected_index, kcenter_metrics = select_batch(kcenter_sampler, uniform_sampler, active_p,
                                                             n_sample,
                                                             selected_inds, **select_batch_inputs)

      margin_selected_index, margin_metrics = select_batch(margin_sampler, uniform_sampler, active_p,
                                                             n_sample,
                                                             selected_inds, **select_batch_inputs)

      entropy_selected_index, entropy_metrics = select_batch(entropy_sampler, uniform_sampler, active_p,
                                                             n_sample,
                                                             selected_inds, **select_batch_inputs)

      least_confidence_selected_index, least_confidence_metrics = select_batch(least_confidence_sampler, uniform_sampler, active_p,
                                                             n_sample,
                                                             selected_inds, **select_batch_inputs)


      indices = [kcenter_selected_index, margin_selected_index, entropy_selected_index, least_confidence_selected_index]
      metrics = [kcenter_metrics, margin_metrics, entropy_metrics, least_confidence_metrics]

      indices = np.array(indices)
      metrics = np.array(metrics)

      np.savetxt(FLAGS.save_dir + "/GPU{}_metric_profile_{}.txt".format(FLAGS.gpu, len(selected_inds)), metrics)
      np.savetxt(FLAGS.save_dir + "/GPU{}_metric_profile_selected_indices_{}.txt".format(FLAGS.gpu, len(selected_inds)), indices)

      # f = open(FLAGS.save_dir + "/GPU{}_metric_profile.txt".format(FLAGS.gpu), "a")
      # f.write(str(metrics))
      # f.close()
      #
      # f = open(FLAGS.save_dir + "/GPU{}_metric_profile_selected_indices.txt".format(FLAGS.gpu), "a")
      # f.write(str(indices))
      # f.close()

  np.random.seed(seed)
  # data_splits = [9./10, 1./20, 1./20]
  # data_splits = [98./100, 1./100, 1./100]
  data_splits = utils.data_splits

  # 2/3 of data for training
  if max_points is None:
    max_points = len(y)
  # train_size = int(min(max_points, len(y)) * data_splits[0])
  # to be compatible with large dataset, trainval split now use the entire dataset as train.
  train_size = int(min(max_points, len(y)))
  if batch_size <= 1:
    batch_size = int(batch_size * train_size)
  else:
    batch_size = int(batch_size)
  if warmstart_size <= 1:
    # Set seed batch to provide enough samples to get at least 4 per class
    # TODO(lishal): switch to sklearn stratified sampler
    seed_batch = int(warmstart_size * train_size)
  else:
    seed_batch = int(warmstart_size)
  seed_batch = max(seed_batch, 6 * len(np.unique(y)))

  indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise = (
      utils.get_train_val_test_splits(X,y,max_points,seed,confusion,
                                      seed_batch, split=data_splits))

  # # Preprocess data
  # if norm_data:
  #   print("Normalizing data")
  #   X_train = normalize(X_train)
  #   X_val = normalize(X_val)
  #   X_test = normalize(X_test)
  # if standardize_data:
  #   print("Standardizing data")
  #   scaler = StandardScaler().fit(X_train)
  #   X_train = scaler.transform(X_train)
  #   X_val = scaler.transform(X_val)
  #   X_test = scaler.transform(X_test)
  # print("active percentage: " + str(active_p) + " warmstart batch: " +
  #       str(seed_batch) + " batch size: " + str(batch_size) + " confusion: " +
  #       str(confusion) + " seed: " + str(seed))

  # Initialize samplers
  uniform_sampler, train_sampler, machine_labeling_sampler, X_train_ds, X_val_ds, X_test_ds = utils.init_samplers(FLAGS,
                                                                                                                  sampler,
                                                                                                                  labeling_sampler,
                                                                                                                  X_train,
                                                                                                                  y_train,
                                                                                                                  X_val,
                                                                                                                  y_val,
                                                                                                                  X_test,
                                                                                                                  y_test,
                                                                                                                  seed,
                                                                                                                  image_path_mode=image_path_mode,
                                                                                                                  num_classes=num_classes,
                                                                                                                  class_names=class_names)


  results = {}
  data_sizes = []
  accuracy = []
  # cuz using the entire h5 as train set, need to randomize seedbatch index
  all_inds = np.arange(train_size)
  np.random.shuffle(all_inds)
  np.random.shuffle(all_inds)
  np.random.shuffle(all_inds)
  
  selected_inds = all_inds[0:seed_batch]
  selected_inds = selected_inds.tolist()
  selected_inds = sorted(selected_inds)
  # selected_inds = list(range(seed_batch))

  # If select model is None, use score_model
  same_score_select = False
  if select_model is None:
    select_model = score_model
    same_score_select = True

  n_batches = int(np.ceil((train_horizon * train_size - seed_batch) *
                          1.0 / batch_size)) + 1

  x_data = []
  CorrPredRatio = []
  sample_intervals = list(range(5, 101, 5))
  sample_intervals = list(np.asarray(sample_intervals) / 100.0)
  acc_thresh=FLAGS.accthresh
  total_size = X_train.shape[0]

  t_start = time.time()
  total_training_time = 0
  total_selection_time = 0
  total_profiling_time = 0

  for b in range(n_batches):
      n_train = seed_batch + min(train_size - seed_batch, b * batch_size)
      print("Training model on " + str(n_train) + " datapoints")

      x_data.append(n_train)
      assert n_train == len(selected_inds)
      data_sizes.append(n_train)

      # Sort active_ind so that the end results matches that of uniform sampling
      partial_X = X_train[sorted(selected_inds)]
      # partial_y = y_train[sorted(selected_inds)]
      # partial_X = utils.h5py_fancy_indexing(X_train, sorted(selected_inds))
      partial_y = y_train[sorted(selected_inds)]
      # save it to disk to avoid memory issue when loading
      # h5name = "partial_X.h5"
      # h5f = h5py.File(h5name, 'w')
      # h5f.create_dataset('data', data=partial_X)
      # h5f.close()
      # h5f = h5py.File(h5name,'r')
      # partial_X = h5f['data']

      if score_model.model is None and FLAGS.cell != -1:
          _gpus = FLAGS.gpu.split(',')
          _gpus = [int(i) for i in _gpus]
          if image_path_mode:
              input_shape = image_dataset_loader.inputshape
          else:
              input_shape = X_train.shape[1:]
          print("InputShape {}".format(input_shape))
          score_model.build_model(input_shape, y_train, FLAGS.cell, FLAGS.layer, FLAGS.kernel, _gpus, image_path_mode=image_path_mode)

      print("Started Training")
      # t_start_batch = time.time()
      # score_model.epochs = 1
      # history = score_model.fit(partial_X, partial_y, X_val, y_val)
      # t_end_train = time.time()
      # print("Epoch time: {}".format(t_end_train-t_start))
      # t_start = time.time()
      # score_model.epochs = 2
      # history = score_model.fit(partial_X, partial_y, X_val, y_val)
      # t_end_train = time.time()
      # print("Epoch time: {}".format(t_end_train - t_start))

      partial_X_ds = None
      if image_path_mode:
          partial_X_ds = image_dataset_loader.image_dataset_from_image_paths(partial_X, partial_y, num_classes, image_size=image_dataset_loader.image_size,
                                                                             batch_size=FLAGS.minibatch,
                                                                             label_mode="categorical", normalization=True)
          # partial_X_ds = image_dataset_loader.image_dataset_iterator_from_image_paths(partial_X, partial_y, num_classes,
          #                                                                    image_size=image_dataset_loader.image_size,
          #                                                                    batch_size=FLAGS.minibatch,
          #                                                                    label_mode="raw", normalization=True, class_names=class_names)

          t_start_batch = time.time()
          history = score_model.fit(partial_X_ds, partial_y, X_val_ds, y_val)
          t_end_batch = time.time()
      else:
          t_start_batch = time.time()
          history = score_model.fit(partial_X, partial_y, X_val, y_val)
          t_end_batch = time.time()

      batch_training_time = t_end_batch - t_start_batch
      total_training_time += batch_training_time
      total_time = t_end_batch - t_start

      if not utils.TEST:
          f = open(FLAGS.save_dir + "/GPU{}_fit_acc_history.txt".format(FLAGS.gpu), "a")
          print(history.history.keys())
          f.write(str(n_train) + " " + str(history.history["accuracy"]) + "\n")
          f.write(str(n_train) + " " + str(history.history["val_accuracy"]) + "\n")
          f.close()
          f = open(FLAGS.save_dir + "/GPU{}_fit_loss_history.txt".format(FLAGS.gpu), "a")
          f.write(str(n_train) + " " + str(history.history["loss"]) + "\n")
          f.write(str(n_train) + " " + str(history.history["val_loss"]) + "\n")
          f.close()

      # if not same_score_select:
      #   select_model.fit(partial_X, partial_y)

      print("Scoring...")
      train_acc = 0
      test_acc = 0
      if not utils.TEST:
          if image_path_mode:
              train_acc = score_model.score(partial_X_ds, partial_y)
              test_acc = score_model.score(X_test_ds, y_test)
          else:
              train_acc = score_model.score(partial_X, partial_y)
              test_acc = score_model.score(X_test, y_test)
      accuracy.append(test_acc)
      print("Sampler: %s, Training Accuracy: %.2f%%, Test Accuracy: %.2f%%" % (train_sampler.name, train_acc*100, accuracy[-1]*100))
      f = open(FLAGS.save_dir + "/GPU{}_acc.txt".format(FLAGS.gpu), "a")
      f.write(str(n_train) + " " + str(train_acc) + " " + str(test_acc) + "\n")
      f.close()

      # # recycle the temporary h5
      # h5f.close()
      # os.remove(h5name)


      n_sample = min(batch_size, train_size - len(selected_inds))
      select_batch_inputs = {
          "model": select_model,
          "labeled": dict(zip(selected_inds, y_train[selected_inds])),
          "eval_acc": accuracy[-1],
          "X_test": X_val,
          "y_test": y_val,
          "y": y_train
      }

      select_t = time.time()
      print("Calculating AL Metric for {} sample".format(train_size))
      new_batch, _ = select_batch(train_sampler, uniform_sampler, active_p, n_sample,
                                  selected_inds, **select_batch_inputs)
      _, metrics_out = select_batch(machine_labeling_sampler, uniform_sampler, active_p, n_sample,
                                    selected_inds, **select_batch_inputs)

      print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))

      """Profling differences"""

      total_size = len(X_train)
      if FLAGS.profile == "True":
          if image_path_mode:
              profile_samplers(FLAGS, X_train_ds, y_train, seed, uniform_sampler, active_p, n_sample, selected_inds, total_size,
                               **select_batch_inputs)
          else:
              profile_samplers(FLAGS, X_train, y_train, seed, uniform_sampler, active_p, n_sample, selected_inds, total_size,
                               **select_batch_inputs)

      batch_selection_time = time.time() - select_t
      print("AL metric extracted:{} sec".format(batch_selection_time))
      total_selection_time += batch_selection_time

      # # use fixed amount from the remaining to fit for error curve
      # profile_t = time.time()
      # hist, acc, remaining = utils.acc_hist_sweep(X_train, y_train, metrics_out, score_model, selected_inds, fixed_amount=True)
      # batch_profiling_time = time.time() - profile_t
      # total_profiling_time += batch_profiling_time
      # f = open(FLAGS.save_dir + "/GPU{}_acc_hist_remaining_fixed_amount.txt".format(FLAGS.gpu), "a")
      # # f.write("{} {} {}\n".format(n_train, remaining, hist))
      # f.write("{} {} {}\n".format(n_train, remaining, acc))
      # f.close()

      # use fixed test set fit for error curve
      profile_t = time.time()
      hist, acc, remaining = utils.acc_hist_sweep(X_train, y_train, metrics_out, score_model, selected_inds,
                                                  fixed_amount=True, image_path_mode=image_path_mode, al_metric=FLAGS.sampling_method,
                                                  num_classes=num_classes, minibatch=FLAGS.minibatch, class_names=class_names)
      batch_profiling_time = time.time() - profile_t
      total_profiling_time += batch_profiling_time
      f = open(FLAGS.save_dir + "/GPU{}_acc_hist_remaining_fixed_amount.txt".format(FLAGS.gpu), "a")
      # f.write("{} {} {}\n".format(n_train, remaining, hist))
      f.write("{} {} {}\n".format(n_train, remaining, acc))
      f.close()

      CorrPredRatio.append(acc)

      # # use ratio from the remaining to log cost, and termination point
      # hist, acc, remaining = utils.acc_hist_sweep(X_train,y_train, metrics_out, score_model, selected_inds)
      # f = open(FLAGS.save_dir + "/GPU{}_acc_hist_remaining.txt".format(FLAGS.gpu), "a")
      # # f.write("{} {} {}\n".format(n_train, remaining, hist))
      # f.write("{} {} {}\n".format(n_train, remaining, acc))
      # f.close()

      utils.log_cost(FLAGS, acc, acc_thresh, total_size,
                     total_training_time, batch_training_time, total_selection_time, total_profiling_time, n_train, t_start)

      CorrPredRatio_thisBatch = 1.
      for i in range(len(acc)):
          if acc[i] < acc_thresh:
              CorrPredRatio_thisBatch = i * 0.05
              break

      # # total_predicted = CorrPredRatio_thisBatch * remaining
      # total_predicted = CorrPredRatio_thisBatch * total_size
      # total_labeled = total_size - total_predicted
      # extra_labeled = total_labeled - n_train
      # total_label_cost = 0.001 * total_labeled
      # total_training_cost = 1.084 / 3600 * total_training_time
      # total_selection_cost = 1.084 / 3600 * total_selection_time
      # total_profiling_cost = 1.084 / 3600 * total_profiling_time
      # total_cost = total_label_cost + total_training_cost + total_selection_cost + total_profiling_cost

      # f = open(FLAGS.save_dir + "/GPU{}_cost.txt".format(FLAGS.gpu), "a")
      # f.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
      #     n_train,
      #     total_time, batch_training_time,
      #     total_training_time, total_training_cost,
      #     total_selection_time, total_selection_cost,
      #     total_profiling_time, total_profiling_cost,
      #     CorrPredRatio_thisBatch, total_predicted,
      #     extra_labeled, total_labeled, total_label_cost,
      #     total_cost))
      # f.close()

      if len(x_data) > 1:
          index, amp, cov = fit_CorrPred_Ratio_Curve(CorrPredRatio, x_data, sample_intervals)
          TrainingSizeAtAccThresh = calculate_TrainingSize_at_accThresh(index, amp, cov, acc_thresh=acc_thresh)
          TrainingSizeAtAccThresh = np.asarray(TrainingSizeAtAccThresh)
          # TotalLabelSize = TrainingSizeAtAccThresh + (total_size - TrainingSizeAtAccThresh) * (
          #         1 - np.asarray(sample_intervals))
          TotalLabelSize = TrainingSizeAtAccThresh + (total_size) * (
                  1 - np.asarray(sample_intervals))

          print("TotalLabelSize")
          print(TotalLabelSize)
          print("Training Size")
          print(TrainingSizeAtAccThresh)
          f = open(FLAGS.save_dir + "/GPU{}_MinCost_Estimates.txt".format(FLAGS.gpu), "a")
          f.write("TotalLabelSize {}\n".format(TotalLabelSize))
          f.write("TrainingSize {}\n".format(TrainingSizeAtAccThresh))
          f.close()

      # if CorrPredRatio_thisBatch == 1.0:
      #     # just for profileing purpose only
      #     break

      if (remaining - CorrPredRatio_thisBatch * total_size) < total_size * 0.05:
          # just for profileing purpose only
          # regarded as can predict everything!
          print("Breaking: Remaining {}, Ratio {}*0.05 = {}".format(remaining, CorrPredRatio_thisBatch, total_size * 0.05))
          break

      selected_inds.extend(new_batch)
      selected_inds = sorted(selected_inds)
      print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
      assert len(new_batch) == n_sample
      assert len(list(set(selected_inds))) == len(selected_inds)

  # Check that the returned indice are correct and will allow mapping to
  # training set from original data
  # assert all(y_noise[indices[selected_inds]] == y_train[selected_inds])
  results["accuracy"] = accuracy
  results["selected_inds"] = selected_inds
  results["data_sizes"] = data_sizes
  results["indices"] = indices
  results["noisy_targets"] = y_noise
  return results, train_sampler


def calculate_TrainingSize_at_accThresh(index, amp, cov, acc_thresh):
    TrainingSizeAtAccThresh = []
    for i in range(len(index)):
        # print("Predicting for {}-th line: \n\tindex {}, amp {}".format(i,index[i], amp[i]))
        trainingSize = -1
        if index[i] > 0:
          # print(index[i])
          trainingSize = 10**((np.log10(1-acc_thresh)-np.log10(index[i]))/amp[i])
          
        TrainingSizeAtAccThresh.append(trainingSize)
    # print(TrainingSizeAtAccThresh)
    return TrainingSizeAtAccThresh


from scipy.optimize import curve_fit

def powerlaw_fitting(xdata,ydata):

    def powerlaw(x,index, amp):
        return index* (x**amp)

    param, covariance = curve_fit(powerlaw, xdata, ydata)
    print("fitting {}, param {}".format(ydata, param))

    # print(param)
    return param[0], param[1], covariance


def fit_CorrPred_Ratio_Curve(CorrPredRatioSamples, x_data, sample_intervals):

    """

    :param CorrPredRatioSamples: a list of correct prediction ratio list,
    each of which includes the corrpred ratio at each percentage
    :return:
    """
    index = []
    amp = []
    cov = []
    # ampErr = []
    f = open(FLAGS.save_dir + "/GPU{}_powerlaw_fitting.txt".format(FLAGS.gpu), "a")
    f.write("X-Data {}\n".format(x_data))
    for i in range(len(CorrPredRatioSamples[0])):
        y_data = []
        for j in range(len(CorrPredRatioSamples)):
            y_data.append(1-CorrPredRatioSamples[j][i])
        try:
            _index, _amp, _cov = powerlaw_fitting(x_data, y_data)
        except Exception as e:
            _index, _amp, _cov = -1, -1, -1
        f.write("Ramining Ratio {} fitting {} param {} {}\n".format(sample_intervals[i], y_data, _index, _amp))
        index.append(_index)
        amp.append(_amp)
        cov.append(_cov)
            # ampErr.append(_ampErr)
    f.close()
    return index, amp, cov

def main(argv):
  del argv


  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
  batches = [float(t) for t in FLAGS.batch_size.split(",")]

  WarmFlag = False
  if FLAGS.warmstart_size == "SAME":
    WarmFlag = True

  if not gfile.exists(FLAGS.save_dir):
    gfile.mkdir(FLAGS.save_dir)

  if FLAGS.test == "True":
      utils.TEST = True

  for batch in batches:
      if WarmFlag:
        FLAGS.warmstart_size = batch
      FLAGS.save_dir = "./results/{}_AUG{}_ACC{}_T{}_L{}_{}_C{}_L{}_K{}_B{}_WB{}_MINIB{}".format(FLAGS.dataset, FLAGS.augmentation,
                                                                               FLAGS.accthresh,
                                                                               FLAGS.sampling_method,
                                                                               FLAGS.labeling_sampling_method,
                                                                               FLAGS.score_model,
                                                                               FLAGS.cell, FLAGS.layer, FLAGS.kernel,
                                                                               batch, float(FLAGS.warmstart_size), float(FLAGS.minibatch))
      
      
      if not gfile.exists(FLAGS.save_dir):
        try:
          gfile.mkdir(FLAGS.save_dir)
        except:
          print(('WARNING: error creating save directory, '
                 'directory most likely already created.'))

      save_dir = os.path.join(
          FLAGS.save_dir,
          FLAGS.dataset + "_" + FLAGS.sampling_method)
      do_save = FLAGS.do_save == "True"

      

      if do_save:
        if not gfile.exists(save_dir):
          try:
            gfile.mkdir(save_dir)
          except:
            print(('WARNING: error creating save directory, '
                   'directory most likely already created.'))
        # Set up logging
        filename = os.path.join(
            save_dir, "log-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + ".txt")
        sys.stdout = utils.Logger(filename)

      confusions = [float(t) for t in FLAGS.confusions.split(" ")]
      mixtures = [float(t) for t in FLAGS.active_sampling_percentage.split(" ")]

      all_results = {}
      max_dataset_size = None if FLAGS.max_dataset_size == "0" else int(
          FLAGS.max_dataset_size)
      normalize_data = FLAGS.normalize_data == "True"
      standardize_data = FLAGS.standardize_data == "True"
      
      X, y, image_path_mode, num_classes, class_names = utils.load_data(FLAGS)
      starting_seed = FLAGS.seed

      for c in confusions:
        for m in mixtures:
          for seed in range(starting_seed, starting_seed + FLAGS.trials):
            train_sampler = get_AL_sampler(FLAGS.sampling_method)
            labeling_sampler = get_AL_sampler(FLAGS.labeling_sampling_method)
            t_start = time.time()
            if utils.TEST:
                epochs = 1
            elif FLAGS.dataset == 'image_path_dataset':
                epochs = 25
            else:
                epochs = 200
            score_model = utils.get_model(FLAGS.score_model, seed, FLAGS.augmentation, batch=FLAGS.minibatch,epochs=epochs)
            if (FLAGS.select_model == "None" or
                FLAGS.select_model == FLAGS.score_model):
              select_model = None
            else:
              select_model = utils.get_model(FLAGS.select_model, seed, FLAGS.augmentation, batch=FLAGS.minibatch, epochs=epochs)
            t_end_model = time.time()
            print("Model Prepared: {}".format(t_end_model-t_start))
            results, sampler_state = generate_one_curve(
                X, y, train_sampler, labeling_sampler, score_model, seed, float(FLAGS.warmstart_size),
                batch, select_model, c, m, max_dataset_size,
                standardize_data, normalize_data, FLAGS.train_horizon, image_path_mode=image_path_mode, num_classes=num_classes, class_names=class_names)
            key = (FLAGS.dataset, FLAGS.sampling_method, FLAGS.score_model,
                   FLAGS.select_model, m, float(FLAGS.warmstart_size), batch,
                   c, standardize_data, normalize_data, seed)
            sampler_output = sampler_state.to_dict()
            results["sampler_output"] = sampler_output
            all_results[key] = results
      fields = [
          "dataset", "sampler", "score_model", "select_model",
          "active percentage", "warmstart size", "batch size", "confusion",
          "standardize", "normalize", "seed"
      ]
      all_results["tuple_keys"] = fields

      if do_save:
        filename = ("results_score_" + FLAGS.score_model +
                    "_select_" + FLAGS.select_model +
                    "_norm_" + str(normalize_data) +
                    "_stand_" + str(standardize_data))
        existing_files = gfile.glob(os.path.join(save_dir, filename + "*.pkl"))
        filename = os.path.join(save_dir,
                                filename + "_" + str(1000+len(existing_files))[1:] + ".pkl")
        pickle.dump(all_results, gfile.GFile(filename, "w"))
        sys.stdout.flush_file()


if __name__ == "__main__":
  app.run(main)
