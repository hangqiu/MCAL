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

from optimal_labeling_utils import *

from utils import image_dataset_loader
from tensorflow.python.keras.preprocessing import dataset_utils

import gc


flags.DEFINE_string("dataset", "letter", "Dataset name")
flags.DEFINE_string("dataset_dir", "letter", "Dataset Directory")
flags.DEFINE_string("sampling_method", "margin",
                    ("Name of training sampling method to use, can be any defined in "
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
flags.DEFINE_string("max_dataset_size", "10000000000",
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
FLAGS = flags.FLAGS


get_wrapper_AL_mapping()

def generate_one_curve(FLAGS,
                       X,
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
                       mixture=0,
                       max_points=None,
                       standardize_data=False,
                       norm_data=False,
                       train_horizon=0.5,
                       image_path_mode=False,
                       num_classes=None,
                       class_names=None):

  
    seed_batch, total_size, _, X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_data(X,y,seed,warmstart_size,batch_size,confusion,active_p, max_points,standardize_data,norm_data)
    # Initialize samplers
    # uniform_sampler = AL_MAPPING["uniform"](X_train, y_train, seed)
    # train_sampler = sampler(X_train, y_train, seed)
    # # test_sampler = sampler(X_test, y_test, seed)
    # machine_labeling_sampler = labeling_sampler(X_train, y_train, seed)

    # Initialize samplers
    uniform_sampler, train_sampler, machine_labeling_sampler, X_train_ds, X_val_ds, X_test_ds = utils.init_samplers(
        FLAGS,
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

    if score_model.model is None and FLAGS.cell != -1:
        _gpus = FLAGS.gpu.split(',')
        _gpus = [int(i) for i in _gpus]
        if image_path_mode:
            input_shape = image_dataset_loader.inputshape
        else:
            input_shape = X_train.shape[1:]
        print("InputShape {}".format(input_shape))
        score_model.build_model(input_shape, y_train, FLAGS.cell, FLAGS.layer, FLAGS.kernel, _gpus, image_path_mode=image_path_mode)
        # score_model.build_model(X_train, y_train, FLAGS.cell, FLAGS.layer, FLAGS.kernel, _gpus)
    select_model = score_model

    sample_intervals = list(range(5,101,5))
    sample_intervals = list(np.asarray(sample_intervals)/100.0)
    total_training_time = 0
    total_selection_time = 0
    total_profiling_time = 0
    t_start = time.time()

    selected_inds = []
    x_data, AccOfRemaining_FixAmount, accuracy, batch_training_time, batch_selection_time, batch_profiling_time, selected_inds, AccOfFixedTestSet = train_for_CorrPred_Curve(
        X_train, y_train, X_test, y_test, X_val, y_val, 
        seed_batch, selected_inds, score_model, sample_intervals, train_sampler, machine_labeling_sampler, uniform_sampler,
        active_p, mixture,FLAGS, sample_points=[0.5, 0.5], image_path_mode=image_path_mode, num_classes=num_classes, X_val_ds=X_val_ds, X_test_ds=X_test_ds, class_names=class_names)
    total_training_time += batch_training_time         
    total_selection_time += batch_selection_time
    total_profiling_time += batch_profiling_time


    seed_cost = utils.log_cost(FLAGS, AccOfRemaining_FixAmount[-1], FLAGS.accthresh, 
        total_size, total_training_time, batch_training_time, total_selection_time, total_profiling_time, len(selected_inds), t_start)    

    _x_data = []
    _AccOfRemaining_FixAmount = []
    _AccOfDynamicTestset = []
    _AccOfFixedTestset = []
    
    # estimate the training cost using seed_batch
    total_training_time_per_image = 0.0
    total_images_trained = 0.0
    # for t in range(len(total_training_time)):
    #     total_training_time_per_image += total_training_time[t]
    #     total_images_trained += x_data[t]
    total_training_time_per_image = total_training_time / (seed_batch * 1.5)
    # total_training_time_per_image /= total_images_trained

    noGPUs = len(FLAGS.gpu.split(','))
    GPU_cost_per_sec = utils.GPU_cost_per_sec_Azure * noGPUs
    total_training_cost_per_image = total_training_time_per_image * GPU_cost_per_sec


    # keep a record of previous predictions
    Predicted_Total_Batch_Sizes = []
    Predicted_Total_MinCosts = []
    Last_Predicted_Total_Batch_Size = -1
    Last_Predicted_Total_MinCost = -1   
    

    CostDiffThresh = 0.10
    full_label_cost = utils.label_cost_per_image * total_size
    while True:
        MOREPOINTS = False

        NextBatchTotal, NextTotalMinCost = evaluate_mincost_batchsize(FLAGS, x_data, AccOfRemaining_FixAmount,
                                                                      sample_intervals, FLAGS.accthresh, total_size,
                                                                      total_training_time, total_selection_time,
                                                                      total_profiling_time,
                                                                      total_training_cost_per_image, fix="Amount")
        Predicted_Total_Batch_Sizes.append(NextBatchTotal)
        Predicted_Total_MinCosts.append(NextTotalMinCost)
        NextBatchSize = NextBatchTotal - len(selected_inds)  


        f = open(FLAGS.save_dir + "/GPU{}_log.txt".format(FLAGS.gpu), "a")

        if NextBatchTotal > 1 and NextBatchSize < 0:
            # not jumping if the jump is less than 1%
            print("Already passed the optimal point")
            f.write("Next total {}, next batch {}. Already passed the optimal point\n".format(NextBatchTotal, NextBatchSize))
            f.close()
            break   

        # calculate the cost change percentage, not an absolute value, if cost goes up, end immediately
        CostChange = abs((Last_Predicted_Total_MinCost-NextTotalMinCost) / float(Last_Predicted_Total_MinCost))
        f.write("Cost Change {}; Last Predicted Cost {}\n".format(CostChange, Last_Predicted_Total_MinCost)) 
        f.write("Last Cost Prediction: {} ({}), Current Cost Prediction {} ({})\n".format(Last_Predicted_Total_MinCost, Last_Predicted_Total_Batch_Size, 
                NextTotalMinCost, NextBatchTotal))

        if CostChange > CostDiffThresh:
            # print("Predicted point is not stable within 5%, getting one more sample point") 
            f.write("Predicted point is not stable within {}, getting one more sample point\n".format(CostDiffThresh)) 
            MOREPOINTS = True

        if NextBatchTotal < 1 or NextBatchTotal > total_size or NextTotalMinCost > full_label_cost:
            print("Batch size too small, couldn't get monotonously decreaseing error curve to fit.")
            f.write("Batch size too small, couldn't get monotonously decreaseing error curve to fit.\n")
            print("OR: Predicted size larger than total size. Suggesting all samples should be human labeled")
            print("Getting one more sample point, using double the current batch size")
            MOREPOINTS = True        

        # smaller jump for active learning benefit
        jumps = 1
        if MOREPOINTS:
            NextBatchSize = len(selected_inds) 
            f.write("Doubling down\n")
            next_training_cost = len(selected_inds) * 2.0 * total_training_cost_per_image
            # if NextBatchSize > CostDiffThresh * total_size: # if additional sample points increased to 10%, break
            if next_training_cost * 3 / 2.0 > utils.label_cost_per_image * total_size * CostDiffThresh: 
                # if total training cost has exceeded 10%
                # print("additional points too large, stopping")
                f.write("training cost ${} has exceeded {} of the total label cost ${}, stopping\n".format(next_training_cost * 3 / 2.0, CostDiffThresh, utils.label_cost_per_image * total_size))
                f.close()
                break            
        else:
            # Finish probing, now decide how many to jump
            # estimate by training cost
            mean_training_cost = (NextBatchTotal + len(selected_inds)) / 2 * total_training_cost_per_image
            while mean_training_cost * jumps < CostDiffThresh * NextTotalMinCost:
                if jumps > 3:
                    break
                if NextBatchSize / jumps < 0.05 * total_size:
                    break
                jumps += 1
                
                
            f.write("Mean training cost is {}, current cost estimate is {}, within {} of cost, allowing {} jumps, with a min jump size of {}\n".format(mean_training_cost, NextTotalMinCost, CostDiffThresh, jumps, 0.01*total_size))
            f.write("jumping {} times, from {} to {}\n".format(jumps, len(selected_inds), NextBatchTotal))


        f.close()
        for i in range(jumps):
            _x_data, _AccOfRemaining_FixAmount, _accuracy, _batch_training_time, _batch_selection_time, _batch_profiling_time, selected_inds, _AccOfFixedTestset = train_for_CorrPred_Curve(
                X_train, y_train, X_test, y_test, X_val, y_val, 
                NextBatchSize / jumps, # current jump size
                selected_inds, score_model, sample_intervals, train_sampler, machine_labeling_sampler, uniform_sampler,
                active_p, mixture, FLAGS,sample_points=[1],image_path_mode=image_path_mode, num_classes=num_classes, X_val_ds=X_val_ds, X_test_ds=X_test_ds, class_names=class_names)
            
            total_training_time += _batch_training_time         
            total_selection_time += _batch_selection_time
            total_profiling_time += _batch_profiling_time 

            x_data += _x_data
            AccOfRemaining_FixAmount += _AccOfRemaining_FixAmount
            # AccOfDynamicTestset += _AccOfDynamicTestset
            AccOfFixedTestSet += _AccOfFixedTestset  

            seed_cost = utils.log_cost(FLAGS, AccOfRemaining_FixAmount[-1], FLAGS.accthresh, 
                total_size, total_training_time, batch_training_time, total_selection_time, total_profiling_time, 
                len(selected_inds), t_start, NextBatchTotal, NextTotalMinCost)
        
        Last_Predicted_Total_Batch_Size = NextBatchTotal
        Last_Predicted_Total_MinCost = NextTotalMinCost


def main(argv):
  del argv

  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
  # batches = [float(t) for t in FLAGS.batch_size.split(",")]
  batch = float(FLAGS.batch_size)
  warmbatches = [float(t) for t in FLAGS.warmstart_size.split(",")]
  if not gfile.exists(FLAGS.save_dir):
    gfile.mkdir(FLAGS.save_dir)

  for warmbatch in warmbatches:
    FLAGS.save_dir = "./results/{}_AUG{}_ACC{}_T{}_L{}_{}_C{}_L{}_K{}_B{}_WB{}_MINIB{}_EXPOOPT".format(FLAGS.dataset, FLAGS.augmentation,
                                                                               FLAGS.accthresh,
                                                                               FLAGS.sampling_method,
                                                                               FLAGS.labeling_sampling_method,
                                                                               FLAGS.score_model,
                                                                               FLAGS.cell, FLAGS.layer, FLAGS.kernel,
                                                                               batch, warmbatch, float(FLAGS.minibatch))

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

          if utils.TEST:
              epochs = 1
          elif FLAGS.dataset=='image_path_dataset':
              epochs = 25
          else:
              epochs = 200
          t_start = time.time()
          score_model = utils.get_model(FLAGS.score_model, seed, FLAGS.augmentation, batch=FLAGS.minibatch, epochs=epochs)
          if (FLAGS.select_model == "None" or
              FLAGS.select_model == FLAGS.score_model):
            select_model = None
          else:
            select_model = utils.get_model(FLAGS.select_model, seed, FLAGS.augmentation, batch=FLAGS.minibatch, epochs=epochs)
          t_end_model = time.time()
          print("Model Prepared: {}".format(t_end_model-t_start))
          generate_one_curve(
              FLAGS, X, y, train_sampler, labeling_sampler, score_model, seed, warmbatch,
              batch, select_model, c, 1.0, m, max_dataset_size,
              standardize_data, normalize_data, FLAGS.train_horizon, image_path_mode=image_path_mode, num_classes=num_classes, class_names=class_names)
          # key = (FLAGS.dataset, FLAGS.sampling_method, FLAGS.score_model,
          #       FLAGS.select_model, m, warmbatch, batch,
          #       c, standardize_data, normalize_data, seed)
          # sampler_output = sampler_state.to_dict()
          # results["sampler_output"] = sampler_output
          # all_results[key] = results
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
