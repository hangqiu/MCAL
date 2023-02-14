import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas
import math
import numpy as np

res_path = "ActiveLabelingResults/results_VM1/"
# res_path = "ActiveLabelingResults/results_VM2/"
# res_path = "ActiveLabelingResults/results_VM3/"
# res_path = "./"
dataset = 'fashion'
dataset = 'cifar10'
# dataset = 'cifar100'
model = 'CNN18'
model_name = "plain_grow_C3_L3_K16"
model = 'Res18'
model_name = "resnet_grow_C3_L3_K16"
# model = 'Res50'
# model_name = "resnet_grow_C3_L9_K16"
# model_name = "resnet_grow_C3_L9_K64"
# config = "cifar10_keras_AUG1_ACC0.95_margin_plain_grow_C3_L3_K16_B1000.0_WB1000.0_MINIB256.0/"
# config = "cifar10_keras_AUG1_ACC0.95_margin_resnet_grow_C3_L3_K16_B1000.0_WB1000.0_MINIB256.0/"
# config = "cifar10_keras_AUG1_ACC0.95_margin_resnet_grow_C3_L9_K16_B1000.0_WB1000.0_MINIB256.0/"
# config = "cifar100_keras_AUG1_ACC0.95_margin_resnet_grow_C3_L9_K64_B1000.0_WB1000.0_MINIB256.0/"
config = "{}_keras_AUG1_ACC0.95_margin_{}_B1000.0_WB1000.0_MINIB256.0/".format(dataset, model_name)
# config = "{}_keras_AUG1_ACC0.95_margin_{}_B500.0_WB500.0_MINIB256.0/".format(dataset, model_name)
file = "GPU0,1,2,3_powerlaw_fitting.txt"


def analyze_fitting_result(file):
    f = open(file,'r')
    training_size = []
    param_alpha = dict()
    param_beta = dict()
    error = dict()
    for line in f.readlines():
        for s in '[,]':
            # print(s)
            line=line.replace(s,'')
        # print(line)
        line=line.strip()
        words = line.split(' ')

        if words[0] == "X-Data":
            if len(training_size) == 0:
                training_size.append(int(words[-2]))
            training_size.append(int(words[-1]))
        else:
            ratio = words[2]
            if not ratio in param_alpha:
                param_alpha[ratio] = []
                param_beta[ratio] = []
                error[ratio] = []
            param_alpha[ratio].append(float(words[-2]))
            param_beta[ratio].append(float(words[-1]))
            if len(training_size) == 2:
                error[ratio].append(float(words[-5]))
            error[ratio].append(float(words[-4]))

    # print(training_size)
    # print(param_alpha)
    # print(param_beta)
    # print(error['0.1'])
    param = dict()
    for k in param_alpha.keys():
        param[k]=[param_alpha[k], param_beta[k]]
    return training_size, param, error


def plot_param_alpha_beta(training_size, param, y_label, legend):
    print(training_size)
    print(param)
    l = plt.plot(training_size, param)
    plt.ylabel(y_label)
    plt.xlabel("Training Size")
    # print(legend)
    # plt.legend(l, legend)
    # plt.ylim(top=1)
    plt.show()

def powerlaw(x,index, amp):
    return index* (x**amp)

def ln_of_truncated_powerlaw(x, index, amp, K):
    return np.log(index) + amp * np.log(x) - x*K

def power_law_fitting(xdata,ydata):
    param, covariance = curve_fit(powerlaw, xdata, ydata)
    return param[0], param[1], covariance

def truncated_power_law_fitting(xdata,ydata):
    param, covariance = curve_fit(ln_of_truncated_powerlaw, xdata, np.log(ydata))
    return param, covariance

def error_fitting_powerlaw(training_size, data_series):
    a = []
    b = []
    c = []

    for i in range(len(training_size)-1):
        try:
            print("fitting {}, and {}".format(training_size[:i+2], data_series[:i+2]))
            alpha, beta, cov = power_law_fitting(training_size[:i+2], data_series[:i+2])
        except:
            print("not fitting")
            alpha, beta, cov = 0, 0, 0
        a.append(alpha)
        b.append(beta)
        c.append(cov)

    return a, b, c

def latest_K_point_fitting(training_size, data_series, K=3):
    a = []
    b = []
    c = []

    for i in range(len(training_size)-K):
        # print(training_size[i:i+K])
        # print(data_series[i:i+K])
        try:
            alpha, beta, cov = power_law_fitting(training_size[i:i+K], data_series[i:i+K])
        except:
            alpha, beta, cov = 0,0,0
        a.append(alpha)
        b.append(beta)
        c.append(cov)

    return a, b, c

def error_fitting_truncated_powerlaw(training_size, data_series):
    a = []
    b = []
    k = []
    c = []

    for i in range(len(training_size)-1):
        try:
            # print("fitting {}, and {}".format(training_size[:i+2], data_series[:i+2]))
            param, cov = truncated_power_law_fitting(training_size[:i+2], data_series[:i+2])
            alpha, beta, K = param[0], param[1], param[2]
        except:
            # print("not fitting")
            alpha, beta, K, cov = 0, 0, 0, 0
        a.append(alpha)
        b.append(beta)
        k.append(K)
        c.append(cov)

    param = [a,b,k]
    return param, c


def calculate_prediction_error(training_size, error, param_alpha, param_beta, lastK=3):
    # print(training_size, error, param_alpha, param_beta)
    prediction_error_dict = dict()

    for ind_i, i in enumerate(training_size):
        prediction_error_dict[i] = []
        alpha = param_alpha[ind_i]
        beta = param_beta[ind_i]
        for ind_j, j in enumerate(training_size):
            if ind_j < len(training_size) - lastK:
                continue
            predicted_error = alpha * (j ** beta)
            error_diff = error[ind_j] - predicted_error
            prediction_error_dict[i].append(error_diff)
        # print("At size {}, for size {}, predicted error {}, actual error {}".format(i,j,predicted_error, error[ind_j]))

    df = pandas.DataFrame(prediction_error_dict)
    return prediction_error_dict, df

def plot_fitting_result(training_size, error_data, param, param_trunc, sample_point, gamma):
    fitted_error = []
    fitted_error_trunc = []
    for i in training_size:
        fitted_error_trunc.append(np.exp(ln_of_truncated_powerlaw(i, param_trunc[0][sample_point], param_trunc[1][sample_point], param_trunc[2][sample_point])))
        fitted_error.append(powerlaw(i, param[0][sample_point], param[1][sample_point]))
    plt.plot(training_size, error_data, 'o')
    plt.plot(training_size, fitted_error, '-')
    plt.plot(training_size, fitted_error_trunc, '--')
    plt.legend(["error_profile", "power_law", "truncated_power_law"])
    plt.title("Fitting using the first {} points".format(sample_point))
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig("{}_fitting_on_{}%.png".format(sample_point, float(gamma) * 100))
    plt.close()
    # print(training_size)
    # print(error_data)
    # print(fitted_error)
    # print(fitted_error_trunc)

def plot_fitting_result_multipoint(training_size, error_data, param, param_trunc, gamma):
    file_name = "Fitting_on_{}%_{}_{}.txt".format(float(gamma) * 100, dataset, model)
    f = open(file_name,'w')
    f.write("Training Size\n")
    f.write(str(training_size))
    f.write("\nError_Profile\n")
    f.write(str(error_data))

    # plt.figure(figsize=[1.6,1.2])
    fig, ax = plt.subplots(figsize=[4.4,3.3])
    plt.rcParams.update({'font.size': 10})
    # ax.set_yticks([0.01, 0.05, 0.1, 0.2])
    # ax.set_xticks([2000, 4000, 8000, 16000])
    plt.plot(training_size, error_data, 'o')
    legend = ["Error Profile"]
    # for sample_point in [3, 5, 7, 9, len(training_size)-2]:
    for sample_point in [3, 5, 7, 9]:
    # for sample_point in [3, 4, 5]:
        fitted_error = []
        fitted_error_trunc = []
        for i in training_size:
            fitted_error_trunc.append(np.exp(ln_of_truncated_powerlaw(i, param_trunc[0][sample_point], param_trunc[1][sample_point], param_trunc[2][sample_point])))
            fitted_error.append(powerlaw(i, param[0][sample_point], param[1][sample_point]))
        plt.plot(training_size, fitted_error, '-')
        plt.plot(training_size, fitted_error_trunc, '--')
        legend.append("{} points".format(sample_point))
        legend.append("{} points (truncated)".format(sample_point))
        f.write("\npower_law_{}_points\n".format(sample_point))
        f.write(str(fitted_error))
        f.write("\ntruncated_power_law_{}_points\n".format(sample_point))
        f.write(str(fitted_error_trunc))
    # plt.legend(["error_profile", "power_law", "truncated_power_law"])
    plt.legend(legend)
    # plt.title("Error Fitting on {} using {}".format(dataset, model))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Training Size')
    plt.ylabel('Error')
    # plt.yticks(np.array([0.01,0.05,0.1,0.2,0.5]))
    # plt.xticks(np.array([2000, 4000, 8000, 16000]))
    plt.xticks([])
    plt.yticks([])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.minorticks_off()
    # ax.set_yticks([0.001, 0.005, 0.01]) #plain
    ax.set_yticks([0.01, 0.05, 0.1, 0.2]) #cifar10
    # ax.set_yticks([0.2, 0.4, 0.6]) #cifao100
    ax.set_xticks([2000, 4000, 8000, 16000])
    # ax.set_xticks([2000, 4000, 8000])
    figure_title = "Fitting_on_{}_{}_{}.png".format(float(gamma), dataset, model)
    print(figure_title)
    plt.tight_layout()
    # plt.show()
    plt.savefig(figure_title)
    plt.close()
    f.close()


training_size, param, error = analyze_fitting_result(res_path+config+file)
gamma = '0.5'
# K=5
# a,b,c = latest_K_point_fitting(training_size,error[gamma], K)
# print(len(a), len(b))
"""truncated powerlaw fitting"""
# sample_point = 5
# a_t, b_t, c_t = error_fitting_powerlaw(training_size, error[gamma])
param_trunc, c_t = error_fitting_truncated_powerlaw(training_size, error[gamma])
# print(len(a_t), len(b_t))
# print(a_t)
# for sample_point in [3,5,7,9, len(training_size)-2]:
#     plot_fitting_result(training_size[1:],error[gamma][1:],param[gamma], param_trunc, sample_point, gamma)
plot_fitting_result_multipoint(training_size[1:],error[gamma][1:],param[gamma], param_trunc, gamma)

# for sample_point in [5,7,9, len(training_size)-2]:
#     plot_fitting_result(training_size[1:],error[gamma][1:],a[sample_point-K], b[sample_point-K], a_t[sample_point], b_t[sample_point], sample_point, gamma)

"""raw"""
# plot_param_alpha_beta(training_size[1:], param_alpha[gamma], "alpha", "Gamma = "+gamma)
# plot_param_alpha_beta(training_size[1:], param_beta[gamma], "beta", "Gamma = "+gamma)
"""latest K point fitting"""
# plot_param_alpha_beta(training_size[K:], a, "alpha", "Gamma = "+gamma)
# plot_param_alpha_beta(training_size[K:], b, "beta", "Gamma = "+gamma)
"""prediction error"""
# _, df = calculate_prediction_error(training_size[1:],error[gamma],param_alpha[gamma],param_beta[gamma],lastK=3)
# df.describe().to_excel("allfor3.xlsx")
# _, df = calculate_prediction_error(training_size[K:],error[gamma],a,b, lastK=3)
# df.describe().to_excel("latest3for3.xlsx")
