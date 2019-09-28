import json
import numpy as np
import os
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def to_train(filename):
    checkpoints = os.listdir("checkpoints/")
    if filename in checkpoints:
        return False
    else:
        return True


def save_dict(filename, dict):

    with open(filename, 'w') as fp:
        json.dump(dict, fp)


def calc_selective_risk(model, regression, calibrated_coverage=None):
    prediction, pred = model.predict()
    if calibrated_coverage is None:
        threshold = 0.5
    else:
        threshold = np.percentile(prediction[:, -1], 100 - 100 * calibrated_coverage)
    covered_idx = prediction[:, -1] > threshold

    coverage = np.mean(covered_idx)
    y_hat = np.argmax(prediction[:, :-1], 1)
    if regression:
        loss = np.sum(np.mean((prediction[covered_idx, :-1] - model.y_test[covered_idx, :-1]) ** 2, -1)) / np.sum(
            covered_idx)
    else:
        loss = np.sum(y_hat[covered_idx] != np.argmax(model.y_test[covered_idx, :], 1)) / np.sum(covered_idx)
    return loss, coverage


def train_profile(model_name, model_cls, coverages, model_baseline=None, regression=False, alpha=0.5):
    results = {}
    for coverage_rate in coverages:
        print("running {}_{}.h5".format(model_name, coverage_rate))
        model = model_cls(train=to_train("{}_{}.h5".format(model_name, coverage_rate)),
                          filename="{}_{}.h5".format(model_name, coverage_rate),
                          coverage=coverage_rate,
                          alpha=alpha)

        loss, coverage = calc_selective_risk(model, regression)

        results[coverage] = {"lambda": coverage_rate, "selective_risk": loss}
        if model_baseline is not None:
            if regression:
                results[coverage]["baseline_risk"] = (model_baseline.selective_risk_at_coverage(coverage))

            else:

                results[coverage]["baseline_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
            results[coverage]["percentage"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_risk"]

        save_dict("results/{}.json".format(model_name), results)


def post_calibration(model_name, model_cls, lamda, calibrated_coverage=None, model_baseline=None, regression=False):
    results = {}
    print("calibrating {}_{}.h5".format(model_name, lamda))
    model = model_cls(train=to_train("{}_{}.h5".format(model_name, lamda)),
                      filename="{}_{}.h5".format(model_name, lamda), coverage=lamda)
    loss, coverage = calc_selective_risk(model, regression, calibrated_coverage)

    results[coverage]={"lambda":lamda, "selective_risk":loss}
    if model_baseline is not None:
        if regression:
            results[coverage]["baseline_risk"] = (model_baseline.selective_risk_at_coverage(coverage))

        else:
            results[coverage]["baseline_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage))
            results[coverage]["mc_risk"] = (1 - model_baseline.selective_risk_at_coverage(coverage, mc=True))

        results[coverage]["percentage"] = 1 - results[coverage]["selective_risk"] / results[coverage]["baseline_risk"]

    return results


def my_generator(func, x_train, y_train, batch_size, k=10):
    while True:
        res = func(x_train, y_train, batch_size
                   ).next()
        yield [res[0], [res[1], res[1][:, :-1]]]

def my_dual_generator(func, x_train, y_train, batch_size):
    while True:
        res = func(x_train, y_train, batch_size
                   ).next()
        yield [res[0], [res[1], res[1][:, :-1], res[0]]]
