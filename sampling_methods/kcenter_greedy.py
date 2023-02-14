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

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from sampling_methods.sampling_def import SamplingMethod
from tqdm import tqdm
from utils import utils

class kCenterGreedy(SamplingMethod):

  def __init__(self, X, y, seed, metric='euclidean', total_size=None):
    self.X = X
    self.y = y
    # self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = None
    self.metric = metric
    self.min_distances = None
    self.n_obs = total_size
    self.already_selected = []
    self.total_size = total_size

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
        self.min_distances = None
    if only_new:
        cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
        # Update min_distances for all examples given new cluster center.
        min_dist = None
        x = self.features[cluster_centers]
        if utils.TEST:
            min_dist = np.zeros([self.total_size, 1]).reshape(-1, 1)
        else:
            if len(x) == 1:
                ############################ original version
                dist = pairwise_distances(self.features, x, metric=self.metric)
                min_dist = np.min(dist, axis=1).reshape(-1,1)
            else:
                ############################ working with large sets, using working memory
                print("pairwise distance from {} elements to {} centers".format(len(self.features), len(x)))
                dist_chunk = pairwise_distances_chunked(self.features, x, metric=self.metric, working_memory=10000)
                while True:
                  dist = next(dist_chunk, None)
                  if dist is None:
                    break
                  # print("Getting a chunk of mindist with shape {}".format(dist.shape))
                  reduced_min_dist = np.min(dist, axis=1).reshape(-1, 1)
                  if min_dist is None:
                    min_dist = reduced_min_dist.tolist()
                  else:
                    # print("{} Min dist updated".format(sum(self.min_distances!=reduced_min_dist)))
                    # min_dist = np.concatenate([min_dist, reduced_min_dist], axis=0)
                    # print("Min Dist Shape {}".format(min_dist.shape))
                    min_dist = min_dist + reduced_min_dist.tolist()
                    # print("Min Dist length {}".format(len(min_dist)))
                min_dist = np.array(min_dist).reshape(-1, 1)
                # print("Min Dist Final Shape {}".format(min_dist.shape))

        if self.min_distances is None:
            self.min_distances = min_dist
        else:
            self.min_distances = np.minimum(self.min_distances, min_dist)

  def select_batch_(self, model, already_selected, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    # try:
    # Assumes that the transform function takes in original data and not
    # flattened data.
    print('Getting transformed features...')
    # self.features = model.transform(self.X)
    # temporary test
    self.features = np.zeros([self.total_size, 1280])
    print("Feature shape {}".format(self.features.shape))
    print('Calculating distances...')
    self.update_distances(already_selected, only_new=False, reset_dist=True)
    # except:
    #   print('Using flat_X as features.')
    #   self.update_distances(already_selected, only_new=True, reset_dist=False)

    new_batch = []
    if utils.TEST:
        new_batch = np.linspace(1, self.total_size, self.total_size)
    else:
        print("Update distance for newly selected centers...")
        for _ in tqdm(range(N)):
          if self.already_selected is None:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(self.n_obs))
          else:
            ind = np.argmax(self.min_distances)
          # New examples should not be in already selected since those points
          # should have min_distance of zero to a cluster center.
          assert ind not in already_selected

          self.update_distances([ind], only_new=True, reset_dist=False)
          new_batch.append(ind)
    print('Maximum Min distance from cluster centers is {}, {}'.format(max(self.min_distances), min(self.min_distances)))
    print(self.min_distances.shape)
    min_dist_list = self.min_distances.reshape((-1,))
    print(min_dist_list.shape)

    self.already_selected = already_selected

    return new_batch, min_dist_list

