import argparse

from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective
from sklearn.utils import shuffle
from selectivnet_utils import *
import scipy.io as sio
import matplotlib.pyplot as plt

NUMBER_OF_RUNS = 5
INCREMENT = 5000
INITIAL_TRAINING_SIZE = 10000

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
    while cur_train_size <= x_total.shape[0]:
        # tuning coverage to the amount of increment train size
        coverage = 1 - INCREMENT / (x_total.shape[0] - cur_train_size + 1e-7)

        # set file name
        baseline_file_name = "baseline_round_{}_trainSize_{}.h5".format(i, cur_train_size)

        # create the current net for this round
        baseline_net = cifar10Selective(filename=baseline_file_name, baseline=True)

        # set the train and test examples
        x_train_baseline, y_train_baseline = x_total[:cur_train_size], y_total[:cur_train_size]
        x_test_baseline, y_test_baseline = x_total[cur_train_size:], y_total[cur_train_size:]

        x_train_baseline, x_test_baseline = baseline_net.normalize(x_train_baseline, x_test_baseline)
        baseline_net.set_train_and_test(x_train_baseline, y_train_baseline, x_test_baseline, y_test_baseline)

        # train net with desired train set
        baseline_net.train(baseline_net.model)

        scores = baseline_net.model.evaluate(normalize_x_test, [original_y_test, original_y_test[:, :-1]], 128)
        metrics_names = baseline_net.model.metrics_names
        sio.savemat('test_result/test_result_{}.mat'.format(baseline_file_name[:-3]), {'metrics_names': metrics_names,
                                                                           'scores': scores})

        # get increment size of examples indices with lowest confidence and add them to the train indices
        cur_train_size += INCREMENT
