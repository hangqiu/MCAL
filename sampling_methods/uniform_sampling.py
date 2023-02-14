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

"""Margin based AL method.

Samples in batches based on margin scores.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sampling_methods.sampling_def import SamplingMethod

import io
import time
from utils import utils


class UniformSampling(SamplingMethod):
  def __init__(self, X, y, seed, total_size=None):
    self.X = X
    self.y = y
    self.name = 'uniform'
    self.total_size = total_size

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with smallest margin/highest uncertainty.

    For binary classification, can just take the absolute distance to decision
    boundary for each point.
    For multiclass classification, must consider the margin between distance for
    top two most likely classes.

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """

    print("Uniform Sampling: ...")
    inf_t = time.time()
    # try:
    #   distances = model.decision_function(self.X)
    # except:
    #   distances = model.predict_proba(self.X)
    # if len(distances.shape) < 2:
    #   min_margin = abs(distances)
    # else:
    #   sort_distances = np.sort(distances, 1)[:, -2:]
    #   min_margin = sort_distances[:, 1] - sort_distances[:, 0]
    #
    #
    # rank_ind = np.argsort(min_margin)
    # # rank_ind = [i for i in rank_ind if i not in already_selected]
    # rank_ind = list_subtract(rank_ind.tolist(), already_selected)
    #
    # # let's try using those most confident ones and try to redirect the rest
    # # min_margin = -min_margin
    # # rank_ind = np.argsort(min_margin)
    # # rank_ind = [i for i in rank_ind if i not in already_selected]
    # # min_margin = -min_margin
    #
    # N = min(N, len(rank_ind))
    # if N!=0:
    #   print("Evaluated AL metric 0-{}: [{},{}]".format(N, min_margin[rank_ind[0]], min_margin[rank_ind[N-1]]))
    # else:
    #   print("Nothing left to select next batch")
    #
    # active_samples = rank_ind[0:N]

    # # return AL metric, but uniform sampling
    # sample = [i for i in range(self.X.shape[0]) if i not in already_selected]
    print("Total Sample:{}".format(self.total_size))

    sample = utils.list_subtract(list(range(self.total_size)), already_selected)
    active_samples = sample[0:N]
    
    print("Finished: {} sec".format(time.time()-inf_t))
    return active_samples, np.zeros(self.total_size)

  def record_margin_acc(self, min, GPU, min_margin, model, already_selected=[]):
    step = (1.0 - min) / 10.
    hist = np.histogram(min_margin, range=[min, 1])
    f = open("GPU{}_margin_{}-1.txt".format(GPU, min), "a")
    f.write(np.array2string(hist[0]) + "\n")
    f.close()

    acc = []
    for m in np.arange(min, 1., step):
      n = m + step
      ind = [x for x in range(len(min_margin)) if m < min_margin[x] <= n]
      data_x = self.X[ind]
      data_y = self.y[ind]
      acc.append(model.score(data_x, data_y))

    f = open("GPU{}_margin_acc_{}-1.txt".format(GPU, min), "a")
    f.write(str(acc))
    f.write("\n")
    f.close()

  def eval_margin_acc_logscale(self, min_margin, model, already_selected=[], depth=20):
    # print("eval margin accuracy (log scale)")

    step = 0.1
    acc = []
    hist = []
    depth_count = 0
    for m in np.logspace(1, depth, base=step, num=depth):
      lower = 1. - m
      upper = 1. - m*step
      depth_count+=1
      if depth_count == depth:
        upper = 1.0

      ind = [x for x in range(len(min_margin)) if (lower < min_margin[x] <= upper) and (x not in already_selected)]
      if len(ind)==0:
        acc.append(0)
        hist.append(0)
        continue
      data_x = self.X[ind]
      data_y = self.y[ind]
      acc.append(model.score(data_x, data_y))
      hist.append(len(ind))

    return hist, acc


