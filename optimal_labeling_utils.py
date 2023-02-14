import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from absl import app
from absl import flags
from tensorflow.io import gfile

from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping
from utils import utils, image_dataset_loader

import time
import math
import scipy
from scipy.optimize import curve_fit

truncated = False

def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
                 **kwargs):
    if utils.TEST:
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

def select_AL_batch(n_sample, score_model, train_size, sampler, uniform_sampler, mixture, selected_inds,
                    y_train, X_val, y_val, active_p):
    n_sample = min(n_sample, train_size - len(selected_inds))

    select_batch_inputs = {
        "model": score_model,
        "labeled": dict(zip(selected_inds, y_train[selected_inds])),
        "eval_acc": 0,
        "X_test": X_val,
        "y_test": y_val,
        "y": y_train
    }
    new_batch, metrics_out = select_batch(sampler, uniform_sampler, active_p, n_sample,
                                        selected_inds, **select_batch_inputs)
    print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
    assert n_sample == len(new_batch)

    return new_batch, metrics_out

def select_random_batch(n_sample, score_model, train_size, uniform_sampler, selected_inds,
                        y_train, X_val, y_val, active_p):  
    n_sample = min(n_sample, train_size - len(selected_inds))
    select_batch_inputs = {
        "model": score_model,
        "labeled": dict(zip(selected_inds, y_train[selected_inds])),
        "eval_acc": 0,
        "X_test": X_val,
        "y_test": y_val,
        "y": y_train
    }  
    select_batch_inputs["N"] = n_sample
    select_batch_inputs["already_selected"] = selected_inds
    batch_PL, metrics = uniform_sampler.select_batch(**select_batch_inputs)
    return batch_PL, metrics

def preprocess_data(X,y,seed,warmstart_size,batch_size,confusion=0.,active_p=1.0, max_points=None,standardize_data=False,norm_data=False):
    np.random.seed(seed)
    # data_splits = [2./3, 1./6, 1./6]
    # data_splits = [9./10, 1./20, 1./20]
    data_splits = utils.data_splits

    # 2/3 of data for training
    if max_points is None:
        max_points = len(y)
    train_size = int(min(max_points, len(y)) * data_splits[0])
    if batch_size < 1:
        batch_size = int(batch_size * train_size)
    else:
        batch_size = int(batch_size)
    if warmstart_size < 1:
        seed_batch = int(warmstart_size * train_size)
    else:
        seed_batch = int(warmstart_size)
    seed_batch = max(seed_batch, 6 * len(np.unique(y)))  # ensure 6 sample per class

    indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise = (
        utils.get_train_val_test_splits(X,y,max_points,seed,confusion,
                                        seed_batch, split=data_splits))

    # Preprocess data
    if norm_data:
        print("Normalizing data")
        X_train = normalize(X_train)
        X_val = normalize(X_val)
        X_test = normalize(X_test)
    if standardize_data:
        print("Standardizing data")
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    print("active percentage: " + str(active_p) + " warmstart batch: " +
            str(seed_batch) + " batch size: " + str(batch_size) + " confusion: " +
            str(confusion) + " seed: " + str(seed))
    
    return seed_batch, train_size, indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise

def log_fit_history(FLAGS, n_train, history):
    f = open(FLAGS.save_dir + "/GPU{}_fit_acc_history_optimized_labeling.txt".format(FLAGS.gpu), "a")
    f.write(str(n_train) + " " + str(history.history["accuracy"]) + "\n")
    f.write(str(n_train) + " " + str(history.history["val_accuracy"]) + "\n")
    f.close()
    f = open(FLAGS.save_dir + "/GPU{}_fit_loss_history_optimized_labeling.txt".format(FLAGS.gpu), "a")
    f.write(str(n_train) + " " + str(history.history["loss"]) + "\n")
    f.write(str(n_train) + " " + str(history.history["val_loss"]) + "\n")
    f.close()

def log_acc(FLAGS, n_train, train_acc, test_acc):
    f = open(FLAGS.save_dir + "/GPU{}_acc_optimized_labeling.txt".format(FLAGS.gpu), "a")
    f.write(str(n_train) + " " + str(train_acc) + " " + str(test_acc) + "\n")
    f.close()

def RemainingAccHistSweep(FLAGS, metrics_out, n_train, score_model, selected_inds, X_train, y_train, intervals, image_path_mode=None, num_classes=None, class_names=None):
    

    # hist, acc, remaining = utils.acc_hist_sweep(X_train, y_train, metrics_out, score_model, selected_inds, intervals=intervals)
    # f = open(FLAGS.save_dir + "/GPU{}_acc_hist_remaining_optimized_labeling.txt".format(FLAGS.gpu), "a")
    # # f.write("{} {} {}\n".format(n_train, remaining, hist))
    # f.write("{} {} {}\n".format(n_train, remaining, acc))
    # f.close()

    # hist, acc, remaining = utils.acc_hist_sweep(X_train, y_train, metrics_out, score_model, intervals=intervals)
    # f = open(FLAGS.save_dir + "/GPU{}_acc_hist_optimized_labeling.txt".format(FLAGS.gpu), "a")
    # f.write("{} {} {}\n".format(n_train, remaining, hist))
    # f.write("{} {} {}\n".format(n_train, remaining, acc))
    # f.close()

    hist, acc, remaining = utils.acc_hist_sweep(X_train, y_train, metrics_out, score_model, selected_inds,
                                                fixed_amount=True, image_path_mode=image_path_mode,
                                                al_metric=FLAGS.sampling_method,
                                                num_classes=num_classes, minibatch=FLAGS.minibatch, class_names=class_names)
    f = open(FLAGS.save_dir + "/GPU{}_acc_hist_remaining_fixed_amount_optimized_labeling.txt".format(FLAGS.gpu), "a")
    f.write("{} {} {}\n".format(n_train, remaining, acc))
    f.close()

    return acc, hist, remaining

# def DynamicTestset_AccHistSweep(FLAGS, TestInds, n_train, X_train, y_train, metrics_out, score_model):
#     hist, acc, remaining = utils.acc_hist_sweep(X_train[TestInds], y_train[TestInds], metrics_out[TestInds], score_model,[])
#     f = open(FLAGS.save_dir + "/GPU{}_acc_hist_dynamic_testset_optimized_labeling.txt".format(FLAGS.gpu), "a")
#     # f.write("{} {} {}\n".format(n_train, remaining, hist))
#     f.write("{} {} {}\n".format(n_train, remaining, acc))
#     f.close()
#
#     return acc, hist, remaining
    

def train_for_CorrPred_Curve(X_train, y_train, X_test, y_test, X_val, y_val, this_batch_size, selected_inds, score_model,
                             intervals,
                             train_sampler, machine_labeling_sampler, uniform_sampler,
                             active_p, mixture,
                             FLAGS,
                             sample_points=[0.5, 0.5],
                             image_path_mode=None, num_classes=None, X_val_ds=None, X_test_ds=None, class_names=None):

    test_accuracy = []

    AccOfRemainingFixedAmount = []
    AccOfDynamicTestset = []
    AccOfFixedTestset = []
    x_data = []

    training_time = 0
    train_size = X_train.shape[0]
    for sample_ratio in sample_points:

        n_sample = int(sample_ratio * this_batch_size)
        


        new_batch_inds, _ = select_AL_batch(n_sample, score_model, train_size, train_sampler, uniform_sampler, mixture, selected_inds,
                                            y_train, X_val, y_val, active_p)
        selected_inds.extend(new_batch_inds)

        n_train = len(selected_inds)
        x_data.append(n_train)

        # Sort active_ind so that the end results matches that of uniform sampling
        partial_X = X_train[sorted(selected_inds)]
        # partial_y = y_train[sorted(selected_inds)]
        # partial_X = utils.h5py_fancy_indexing(X_train, sorted(selected_inds))
        partial_y = y_train[sorted(selected_inds)]
        

        print("Started training model on " + str(n_train) + " datapoints")
        partial_X_ds = None
        if image_path_mode:
            partial_X_ds = image_dataset_loader.image_dataset_from_image_paths(partial_X, partial_y, num_classes,
                                                                               image_size=image_dataset_loader.image_size,
                                                                               batch_size=FLAGS.minibatch,
                                                                               label_mode="categorical",
                                                                               normalization=True)
            # partial_X_ds = image_dataset_loader.image_dataset_iterator_from_image_paths(partial_X, partial_y, num_classes,
            #                                                                    image_size=image_dataset_loader.image_size,
            #                                                                    batch_size=FLAGS.minibatch,
            #                                                                    label_mode="raw",
            #                                                                    normalization=True, class_names=class_names)
            t_start_batch = time.time()
            history = score_model.fit(partial_X_ds, partial_y, X_val_ds, y_val)
            t_end_batch = time.time()
        else:
            t_start_batch = time.time()
            history = score_model.fit(partial_X, partial_y, X_val, y_val)
            t_end_batch = time.time()
        training_time += (t_end_batch - t_start_batch)

        log_fit_history(FLAGS, n_train, history)

        if image_path_mode:
            train_acc = score_model.score(partial_X_ds, partial_y)
            test_acc = score_model.score(X_test_ds, y_test)
        else:
            train_acc = score_model.score(partial_X, partial_y)
            test_acc = score_model.score(X_test, y_test)

        test_accuracy.append(test_acc)
        log_acc(FLAGS, n_train, train_acc, test_acc)
        print("Training Accuracy: %.2f%%, Test Accuracy: %.2f%%" % (train_acc * 100, test_accuracy[-1] * 100))


        select_t = time.time()
        _, metrics_out = select_AL_batch(n_train, score_model, train_size, machine_labeling_sampler, uniform_sampler, mixture, selected_inds, y_train, X_val, y_val, active_p)
        batch_selection_time = time.time() - select_t

        profile_t = time.time()
        acc, _, _ = RemainingAccHistSweep(FLAGS, metrics_out, n_train, score_model, selected_inds, X_train, y_train, intervals, image_path_mode=image_path_mode, num_classes=num_classes, class_names=class_names)
        AccOfRemainingFixedAmount.append(acc)
        batch_profiling_time = time.time() - profile_t


        # n_test = 2000
        # TestInds, _ = select_random_batch(n_test, score_model, train_size, uniform_sampler, selected_inds, y_train, X_val, y_val, active_p)

        # acc, _, _ = DynamicTestset_AccHistSweep(FLAGS, TestInds, n_train, X_train, y_train, metrics_out, score_model)
        # AccOfDynamicTestset.append(acc)
        
        # TODO: use the fixed test set to get acc curve
        # _, testset_metrics_out = select_AL_batch(n_train, score_model, train_size, test_sampler, uniform_sampler, mixture, [], y_train, X_val, y_val, active_p)
        # acc, _, _ = RemainingAccHistSweep(FLAGS, testset_metrics_out, n_train, score_model, [], X_test, y_test, intervals)
        AccOfFixedTestset.append(acc)

    return x_data, AccOfRemainingFixedAmount, test_accuracy, training_time, batch_selection_time, batch_profiling_time, selected_inds, AccOfFixedTestset

def calculate_TrainingSize_at_accThresh(index, amp, cov, acc_thresh):
    TrainingSizeAtAccThresh = []
    for i in range(len(index)):
        # print("Predicting for {}-th line: \n\tindex {}, amp {}".format(i,index[i], amp[i]))
        if truncated:
            trainingSize = solve_truncated_powerlaw(index, amp, 1-acc_thresh)
        else:
            trainingSize = 10**((np.log10(1-acc_thresh)-np.log10(index[i]))/amp[i])
        TrainingSizeAtAccThresh.append(trainingSize)
    # print(TrainingSizeAtAccThresh)
    return TrainingSizeAtAccThresh

def powerlaw(x,index, amp):
    return index* (x**amp)

# def truncated_powerlaw(x, index, amp, K=10000):
#     return index * (x**amp) * math.exp(-x/K)

def ln_of_truncated_powerlaw(x, index, amp):
    return np.log(index) + amp * np.log(x) - x/10000


def solve_truncated_powerlaw(index, amp, error_thresh):
    def F(x):
        return np.exp(ln_of_truncated_powerlaw(x, index, amp)) - error_thresh
    training_size = scipy.optimize.broyden1(F)
    return training_size
# import gekko
# def solve_truncated_powerlaw(index, amp, error_thresh):
#     m = gekko()
#     training_size = m.Var(1)
#     m.Equation([truncated_powerlaw(training_size, index, amp)=error_thresh])
#     m.solve(disp=False)
#     return training_size.value

def powerlaw_fitting(xdata,ydata):
    param, covariance = curve_fit(powerlaw, xdata, ydata)
    # param, covariance = curve_fit(truncated_powerlaw, xdata, ydata)
    print("fitting {}, param {}".format(ydata, param))

    # print(param)
    return param[0], param[1], covariance


def truncated_power_law_fitting(xdata,ydata):
    param, covariance = curve_fit(ln_of_truncated_powerlaw, xdata, np.log(ydata))
    return param[0], param[1], covariance


def fit_ErrorRate_Curve(FLAGS, CorrPredRatioSamples, x_data, sample_intervals):

    """

    :param CorrPredRatioSamples: a list of correct prediction ratio list,
    each of which includes the corrpred ratio at each percentage
    :return:
    """
    index = []
    amp = []
    cov = []
    # ampErr = []
    f = open(FLAGS.save_dir + "/GPU{}_powerlaw_fitting_optimized_labeling.txt".format(FLAGS.gpu), "a")
    f.write("X-Data {}\n".format(x_data))
    for i in range(len(CorrPredRatioSamples[0])):
        y_data = []
        for j in range(len(CorrPredRatioSamples)):
            y_data.append(1-CorrPredRatioSamples[j][i])
        if 2 in y_data:
            _index, _amp, _cov = 0, 0, 0
        else:
            try:
                if truncated:
                    _index, _amp, _cov = truncated_power_law_fitting(x_data, y_data)
                else:
                    _index, _amp, _cov = powerlaw_fitting(x_data, y_data)
            except Exception as e:
                _index, _amp, _cov = 0, 0, 0
        f.write("Ramining Ratio {} fitting {} param {} {}\n".format(sample_intervals[i], y_data, _index, _amp))
        index.append(_index)
        amp.append(_amp)
        cov.append(_cov)
            # ampErr.append(_ampErr)
    f.close()
    return index, amp, cov


def evaluate_mincost_batchsize(FLAGS, x_data, AccOfRemaining_vec, sample_intervals, acc_thresh, total_size, 
                                total_training_time, total_selection_time, total_profiling_time,
                                total_training_cost_per_image,
                                fix="Amount"):
    # total_x_data = x_data + _x_data
    # Total_AccOfRemaining_vec = AccOfRemaining_vec + _AccOfRemaining_vec
    total_x_data = x_data
    Total_AccOfRemaining_vec = AccOfRemaining_vec
    
    index, amp, cov = fit_ErrorRate_Curve(FLAGS, Total_AccOfRemaining_vec, total_x_data, sample_intervals)
    TrainingSizeAtAccThresh = calculate_TrainingSize_at_accThresh(index, amp, cov, acc_thresh=acc_thresh)
    TrainingSizeAtAccThresh = np.asarray(TrainingSizeAtAccThresh)

    TotalLabelSize = TrainingSizeAtAccThresh + total_size * (1 - np.asarray(sample_intervals))
    ext = "_FixAmount"
    if fix=="Ratio":
        TotalLabelSize = TrainingSizeAtAccThresh + (total_size - TrainingSizeAtAccThresh) * (1 - np.asarray(sample_intervals))
        ext = "_FixRatio"
    
    # total satyam cost estimate
    noGPUs = len(FLAGS.gpu.split(','))
    GPU_cost_per_sec = utils.GPU_cost_per_sec_Azure * noGPUs

    total_label_cost = utils.label_cost_per_image * TotalLabelSize
    total_training_cost = GPU_cost_per_sec * total_training_time
    total_selection_cost = GPU_cost_per_sec * total_selection_time
    total_profiling_cost = GPU_cost_per_sec * total_profiling_time
    total_cost = total_label_cost + total_training_cost + total_selection_cost + total_profiling_cost

    # extra_training_cost = total_training_cost_per_image * TrainingSizeAtAccThresh
    extra_training_cost = total_training_cost_per_image * TrainingSizeAtAccThresh * 8
    # total_predicted_cost = extra_training_cost + total_cost
    total_predicted_cost = extra_training_cost + total_label_cost + total_training_cost*8 + total_selection_cost*8 + total_profiling_cost

    # choosing minimal total label size for now, assuming one-time training cost is negligible
    print("TotalLabelSize")
    print(TotalLabelSize)
    print("Training Size")
    print(TrainingSizeAtAccThresh)
    f = open(FLAGS.save_dir + "/GPU{}_MinCost_Estimates_optimized_labeling{}.txt".format(FLAGS.gpu, ext), "a")
    f.write("TotalLabelSize {}\n".format(TotalLabelSize))
    f.write("TrainingSize {}\n".format(TrainingSizeAtAccThresh))
    f.write("TotalCost {}\n".format(total_cost))
    f.write("TotalPredictedCost {}\n".format(total_predicted_cost))

    # min_cost_index = np.argmin(TotalLabelSize) 
    min_cost_index = -1
    min_label_size = 9999999999
    min_label_cost = 9999999999
    for i in range(len(TotalLabelSize)):
      if Total_AccOfRemaining_vec[-1][i] == 0.0:
        break
      # if TotalLabelSize[i] !=0 and TotalLabelSize[i] < min_label_size:
      #   min_label_size = TotalLabelSize[i]
      #   min_cost_index = i
      if total_predicted_cost[i] != 0 and total_predicted_cost[i] < min_label_cost:
        min_label_cost = total_predicted_cost[i]
        min_cost_index = i

    # TODO: labeling cost >> training cost for now, change to total cost min, if needed
    NextBatchSize = 0
    MinCost = 9999999999
    if min_cost_index != -1:
        NextBatchSize = int(TrainingSizeAtAccThresh[min_cost_index])
        # MinCost = total_cost[min_cost_index]
        MinCost = total_predicted_cost[min_cost_index]

    print("Predicted optimal training size: {}".format(NextBatchSize))

    f.write("NextBatchTotalSize {}\n".format(NextBatchSize))
    f.close()

    return NextBatchSize, MinCost



