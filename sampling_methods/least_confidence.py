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

from utils import utils
from sampling_methods.sampling_def import SamplingMethod

import io
import time


class LeastConfidence(SamplingMethod):
  def __init__(self, X, y, seed, total_size=None):
    self.X = X
    self.y = y
    self.name = 'least_confidence'
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

    print("Model Inference...")
    inf_t = time.time()
    try:
      distances = model.decision_function(self.X)
    except:
      distances = model.predict_proba(self.X)
    print("Finished: {} sec".format(time.time()-inf_t))

    inf_t = time.time()
    confidence = distances.max(axis=1)
    print("Index calculated: {} sec".format(time.time()-inf_t))

    # # for testing purpose only
    # min_margin = np.random.uniform(size=self.X.shape[0])

    
    inf_t = time.time()
    rank_ind = confidence.argsort(axis=0)
    print("Index sorted: {} sec".format(time.time()-inf_t))
    inf_t = time.time()
    # rank_ind = [i for i in rank_ind if i not in already_selected]
    rank_ind = utils.list_subtract(rank_ind.tolist(), already_selected)
    print("Index filtered: {} sec".format(time.time()-inf_t))
    # let's try using those most confident ones and try to redirect the rest
    # min_margin = -min_margin
    # rank_ind = np.argsort(min_margin)
    # rank_ind = [i for i in rank_ind if i not in already_selected]
    # min_margin = -min_margin

    N = min(N, len(rank_ind))
    if N!=0:
      print("Evaluated least confidence metric 0-{}: [{},{}]".format(N, confidence[rank_ind[0]], confidence[rank_ind[N-1]]))
    else:
      print("Nothing left to select next batch")

    active_samples = rank_ind[0:N]

    # # return AL metric, but uniform sampling
    # sample = [i for i in range(self.X.shape[0]) if i not in already_selected]
    # active_samples = sample[0:N]

    return active_samples, confidence




