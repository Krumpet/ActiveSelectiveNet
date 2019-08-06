from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective
from sklearn.model_selection import train_test_split

initial_training_size = 10000

# init net
net = cifar10Selective()
net._load_data()

# create model
model = net.build_model()

# take 10000 samples for initial training
# TODO: shuffle
original_x_train, original_y_train = net.x_train, net.y_train
x_train, y_train = original_x_train[:initial_training_size], original_x_train[:initial_training_size]

# TODO: data is normalized by all of the data points rather than just this subset, is this ok?
net.x_train, net.y_train = x_train, y_train

net.train(model)

# TODO: save model

# predict on the rest of the training set
net.predict(x_train[initial_training_size:])
# TODO: get softmax layer output, estimate certainty/confidence, sort and take top e.g. 5000 for the next epoch