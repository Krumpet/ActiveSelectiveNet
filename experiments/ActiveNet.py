import argparse

from models.catdog_vgg_selectivenet import CatsvsDogVgg as CatsvsDogSelective
from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective
from models.svhn_vgg_selectivenet import SvhnVgg as SVHNSelective
from sklearn.utils import shuffle
from selectivnet_utils import *
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import tensorflow as tf
np.random.seed(1234)
tf.set_random_seed(1234)

NUMBER_OF_RUNS = 5
INCREMENT = 5000
INITIAL_TRAINING_SIZE = 10000

MODELS = {"cifar_10": cifar10Selective, "catsdogs": CatsvsDogSelective, "SVHN": SVHNSelective}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar_10')

parser.add_argument('--model_name', type=str, default='test')
parser.add_argument('--baseline', type=str, default='none')
parser.add_argument('--alpha', type=float, default=0.5)

args = parser.parse_args()

model_cls = MODELS[args.dataset]
model_name = args.model_name
baseline_name = args.baseline

from sklearn.model_selection import train_test_split

# init net for data usage, this net will not be trained or tested
net = cifar10Selective()
# get all data and merge between the test and train
original_x_train, original_y_train = net.x_train, net.y_train
original_x_test, original_y_test = net.x_test, net.y_test
x_total = np.copy(original_x_train)
y_total = np.copy(original_y_train)
_, normalize_x_test = net.normalize(original_x_train, original_x_test)

# run several times to get more accurate result
for i in range(NUMBER_OF_RUNS):
    # shuffle each time we run test
    x_total, y_total = shuffle(x_total, y_total)

    # init the train indices and size
    cur_train_size = INITIAL_TRAINING_SIZE
    cur_train_indexes = np.arange(cur_train_size)
    all_indexes = np.arange(x_total.shape[0])

    # starting loop of the increment of train data by confidence level
    while cur_train_size + INCREMENT <= x_total.shape[0]:
        # tuning coverage to the amount of increment train size
        coverage = 1 - INCREMENT / (x_total.shape[0] - cur_train_size)

        # set file name
        file_name = "round_{}_trainSize_{}.h5".format(i, cur_train_size)
        # create the current net for this round
        cur_net = cifar10Selective(coverage=coverage, filename=file_name)

        # set the train and test examples
        x_train, y_train = x_total[cur_train_indexes], y_total[cur_train_indexes]
        cur_test_indexes = all_indexes[np.logical_not(np.isin(all_indexes, cur_train_indexes))]
        x_test, y_test = x_total[cur_test_indexes], y_total[cur_test_indexes]

        x_train, x_test = cur_net.normalize(x_train, x_test)

        cur_net.set_train_and_test(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test)

        # train net with desired train set
        cur_net.train(cur_net.model)

        # check performance on separate test
        scores = cur_net.model.evaluate(normalize_x_test, [original_y_test, original_y_test[:, :-1]], 128)
        metrics_names = cur_net.model.metrics_names
        sio.savemat('test_result/test_result_{}.mat'.format(file_name[:-3]), {'metrics_names': metrics_names,
                                                                                       'scores': scores})

        # predict on the rest of the examples
        y_pred, __ = cur_net.predict()

        # get increment size of examples indices with lowest confidence and add them to the train indices
        y_confidence = y_pred[:, -1]
        cur_train_size += INCREMENT
        sorted_ind = np.argsort(y_confidence)
        index_add_to_train = cur_test_indexes[sorted_ind[:INCREMENT]]
        cur_train_indexes = np.concatenate((cur_train_indexes, index_add_to_train))
