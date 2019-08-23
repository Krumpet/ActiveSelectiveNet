import argparse
from sklearn.utils import shuffle

from models.cifar10_vgg_dual_selectivenet import cifar10vgg as cifar10Selective


import numpy as np

NUMBER_OF_RUNS = 5
INCREMENT = 5000
INITIAL_TRAINING_SIZE = 10000
INITIAL_ENCODER_WEIGHT = 0.8
FINAL_ENCODER_WEIGHT = 0.2
# how much to shift weights in favor of selectiveNet confidence with each step
CONF_WEIGHT_STEP = (INITIAL_ENCODER_WEIGHT - FINAL_ENCODER_WEIGHT) / NUMBER_OF_RUNS


def mse(exp_batch: np.ndarray, pred_batch: np.ndarray):
    """
    calculates mean square error between each element in the numpy arrays.
    e.g. shape is (10,32,32,3) so 10 images of 32x32(x3 channels)
    :param exp_batch:
    :param pred_batch:
    :return:
    """
    mse = ((exp_batch - pred_batch) ** 2).mean(axis=tuple(np.arange(pred_batch.ndim)[1:]))
    return mse


def normalize_as_confidence(mse: np.ndarray):
    return 1 - (mse / mse.max())


def get_dual_confidence_by_step(y_confidence, x_train, autoencoded_images, step):
    autoencoder_confidence = normalize_as_confidence(mse(x_train, autoencoded_images))
    weight_shift = CONF_WEIGHT_STEP * step
    conf_weight, encoder_weight = 1 - INITIAL_ENCODER_WEIGHT + weight_shift, INITIAL_ENCODER_WEIGHT - weight_shift
    return conf_weight * y_confidence + encoder_weight + autoencoder_confidence


MODELS = {"cifar_10": cifar10Selective}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar_10')

parser.add_argument('--model_name', type=str, default='test')
parser.add_argument('--baseline', type=str, default='none')
parser.add_argument('--alpha', type=float, default=0.5)

args = parser.parse_args()

model_cls = MODELS[args.dataset]
model_name = args.model_name
baseline_name = args.baseline

# init net for data usage, this net will not be trained or tested
net = cifar10Selective()
# get all data and merge between the test and train
original_x_train, original_y_train = net.x_train, net.y_train
original_x_test, original_y_test = net.x_test, net.y_test
x_total = np.concatenate((original_x_train, original_x_test), axis=0)
y_total = np.concatenate((original_y_train, original_y_test), axis=0)

# run with shallow or deep autoencoder:
for encoder in ["deep"]:
    # run several times to get more accurate result
    for i in range(NUMBER_OF_RUNS):
        # shuffle each time we run test
        x_total, y_total = shuffle(x_total, y_total)

        # init the train indices and size
        cur_train_size = INITIAL_TRAINING_SIZE
        cur_train_indexes = np.arange(cur_train_size)
        all_indexes = np.arange(x_total.shape[0])

        # starting loop of the increment of train data by confidence level
        while cur_train_size <= x_total.shape[0] - original_x_test.shape[0]:
            # tuning coverage to the amount of increment train size
            coverage = 1 - INCREMENT / (x_total.shape[0] - cur_train_size)

            # set file name
            file_name = "round_{}_trainSize_{}_{}.h5".format(i, cur_train_size, encoder)
            baseline_file_name = "baseline_round_{}_trainSize_{}_{}.h5".format(i, cur_train_size, encoder)

            # create the current net for this round
            cur_net = cifar10Selective(coverage=coverage, filename=file_name, autoencoder=encoder)

            x_train, y_train = x_total[cur_train_indexes], y_total[cur_train_indexes]
            cur_test_indexes = all_indexes[np.logical_not(np.isin(all_indexes, cur_train_indexes))]
            x_test, y_test = x_total[cur_test_indexes], y_total[cur_test_indexes]

            x_train, x_test = cur_net.normalize(x_train, x_test)

            cur_net.set_train_and_test(train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test)

            # train net with desired train set
            cur_net.train(cur_net.model)

            # predict on the rest of the examples
            y_pred, __, encoder_res = cur_net.predict()

            cur_train_size += INCREMENT
            # get increment size of examples indices with lowest confidence and add them to the train indices
            y_confidence = y_pred[:, -1]
            dual_confidence = get_dual_confidence_by_step(y_confidence, cur_net.x_test, encoder_res, i)
            sorted_ind = np.argsort(dual_confidence)
            index_add_to_train = cur_test_indexes[sorted_ind[:INCREMENT]]
            cur_train_indexes = np.concatenate((cur_train_indexes, index_add_to_train))
