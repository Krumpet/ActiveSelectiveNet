# need to have graphviz installed (standalone installer), then set this path to the correct one. Might also need to
# add it to the conda environment

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from keras.utils.vis_utils import plot_model
from models.cifar10_vgg_selectivenet import cifar10vgg as cifar10Selective
from models.cifar10_vgg_dual_selectivenet import cifar10vgg as cifar10SelectiveDual

selective = cifar10Selective().model
dual_shallow = cifar10SelectiveDual("shallow").model
dual_deep = cifar10SelectiveDual("deep").model

plot_model(selective, to_file='SelectiveNet.png', show_shapes=True, show_layer_names=True)
plot_model(dual_shallow, to_file='SelectiveNetDualShallow.png', show_shapes=True, show_layer_names=True)
plot_model(dual_deep, to_file='SelectiveNetDualDeep.png', show_shapes=True, show_layer_names=True)