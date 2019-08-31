from __future__ import print_function

import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras import backend as K
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.datasets import cifar10
from keras.engine.topology import Layer
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio

from models.autoencoders import autoencoder_dict
from selectivnet_utils import *


class cifar10vgg:
    def __init__(self, autoencoder, train=True, filename="weightsvgg.h5", coverage=0.875, alpha=0.5, baseline=False):
        self.lamda = coverage
        self.alpha = alpha
        self.mc_dropout_rate = K.variable(value=0)
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.autoencoder_type = autoencoder
        self._load_data()

        self.x_shape = self.x_train.shape[1:]
        self.filename = filename

        self.model = self.build_model()
        # if baseline:
        #     self.alpha = 0
        #
        # if train:
        #     self.model = self.train(self.model)
        # else:
        #     self.model.load_weights("checkpoints/{}".format(self.filename))

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        weight_decay = self.weight_decay
        basic_dropout_rate = 0.3
        input = Input(shape=self.x_shape)

        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(input)
        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate)(curr)

        curr = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.1)(curr)

        curr = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = MaxPooling2D(pool_size=(2, 2))(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)

        curr = Flatten()(curr)
        curr = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(curr)

        curr = Activation('relu')(curr)
        curr = BatchNormalization()(curr)
        curr = Dropout(basic_dropout_rate + 0.2)(curr)
        curr = Lambda(lambda x: K.dropout(x, level=self.mc_dropout_rate))(curr)

        # classification head (f)
        curr1 = Dense(self.num_classes, activation='softmax')(curr)

        curr2 = Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(curr)
        curr2 = Activation('relu')(curr2)
        curr2 = BatchNormalization()(curr2)
        # this normalization is identical to initialization of batchnorm gamma to 1/10
        curr2 = Lambda(lambda x: x / 10)(curr2)
        curr2 = Dense(1, activation='sigmoid')(curr2)

        # autoencoder head (ae)
        autoencoder_output_layer = autoencoder_dict[self.autoencoder_type](input).out

        # selection head (g)
        selective_output = Concatenate(axis=1, name="selective_head")([curr1, curr2])

        # auxiliary head (h)
        auxiliary_output = Dense(self.num_classes, activation='softmax', name="classification_head")(curr)

        # output is ((f,g), h, ae), output shape is [(,11), (,10), (, 32, 32, 3)]
        model = Model(inputs=input, outputs=[selective_output, auxiliary_output, autoencoder_output_layer])

        self.input = input
        self.model_embeding = Model(inputs=input, outputs=curr)
        return model

    def normalize(self, X_train: np.ndarray, X_test: np.ndarray):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the training set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def predict(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model.predict(x, batch_size)

    # def predict_reconstruction_loss_confidence(self, x=None, batch_size=128):
    #     if self.autoencoder_type is None:
    #         return None
    #
    #     if x is None:
    #         x = self.x_test
    #     predictions = self.model_autoencoder.predict(x, batch_size)
    #     return normalize_as_confidence(mse(x, predictions))

    def predict_embedding(self, x=None, batch_size=128):
        if x is None:
            x = self.x_test
        return self.model_embeding.predict(x, batch_size)

    def mc_dropout(self, batch_size=1000, dropout=0.5, iter=100):
        K.set_value(self.mc_dropout_rate, dropout)
        repititions = []
        for i in range(iter):
            _, pred = self.model.predict(self.x_test, batch_size)
            repititions.append(pred)
        K.set_value(self.mc_dropout_rate, 0)

        repititions = np.array(repititions)
        mc = np.var(repititions, 0)
        mc = np.mean(mc, -1)
        return -mc

    def selective_risk_at_coverage(self, coverage, mc=False):
        _, pred = self.predict()

        if mc:
            sr = np.max(pred, 1)
        else:
            sr = self.mc_dropout()
        sr_sorted = np.sort(sr)
        threshold = sr_sorted[pred.shape[0] - int(coverage * pred.shape[0])]
        covered_idx = sr > threshold
        selective_acc = np.mean(np.argmax(pred[covered_idx], 1) == np.argmax(self.y_test[covered_idx], 1))
        return selective_acc

    def _load_data(self):

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test_label) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train, self.x_test = self.normalize(x_train, x_test)

        self.y_train = keras.utils.to_categorical(y_train, self.num_classes + 1)
        self.y_test = keras.utils.to_categorical(y_test_label, self.num_classes + 1)

    def set_train_and_test(self, train_x, train_y, test_x, test_y):
        self.y_train = train_y
        self.y_test = test_y
        self.x_train, self.x_test = self.normalize(train_x, test_x)

    def train(self, model):
        c = self.lamda
        lamda = 32

        def selective_loss(y_true, y_pred):
            loss = K.categorical_crossentropy(
                K.repeat_elements(y_pred[:, -1:], self.num_classes, axis=1) * y_true[:, :-1],
                y_pred[:, :-1]) + lamda * K.maximum(-K.mean(y_pred[:, -1]) + c, 0) ** 2
            return loss

        def selective_acc(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            temp1 = K.sum(
                (g) * K.cast(K.equal(K.argmax(y_true[:, :-1], axis=-1), K.argmax(y_pred[:, :-1], axis=-1)), K.floatx()))
            temp1 = temp1 / K.sum(g)
            return K.cast(temp1, K.floatx())

        def coverage(y_true, y_pred):
            g = K.cast(K.greater(y_pred[:, -1], 0.5), K.floatx())
            return K.mean(g)

        # training parameters
        batch_size = 128
        # TODO: set back to 300
        maxepoches = 2
        learning_rate = 0.1

        lr_decay = 1e-6

        lr_drop = 25

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))

        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        # data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(self.x_train)

        # optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)

        losses = {
            "selective_head": selective_loss,
            "classification_head": 'categorical_crossentropy',
            "decoder_output": 'mean_squared_error'
        }

        metrics = {
            "selective_head": [selective_acc, 'accuracy', coverage],
            "classification_head": [selective_acc, 'accuracy', coverage],
            "decoder_output": 'mean_squared_error'  # accuracy is the same as loss for tracking progress
        }

        model.compile(loss=losses, loss_weights=[self.alpha * 2 / 3, (1 - self.alpha) * 2 / 3, 1 / 3],
                      optimizer=sgd, metrics=metrics)

        historytemp = model.fit_generator(my_dual_generator(datagen.flow, self.x_train, self.y_train,
                                                            batch_size=batch_size),
                                          steps_per_epoch=self.x_train.shape[0] // batch_size,
                                          epochs=maxepoches, callbacks=[reduce_lr],
                                          validation_data=(
                                              self.x_test, [self.y_test, self.y_test[:, :-1], self.x_test]))
        sio.savemat('result_autoencoder_{}.mat'.format(self.filename[:-3]),
                    {'classification_loss_training': historytemp.history['classification_head_acc'],
                     'classification_loss_val': historytemp.history['val_classification_head_acc']})
        # fig, ax = plt.subplots(nrows=1, ncols=1)
        # ax.plot(historytemp.history['classification_head_acc'])
        # ax.plot(historytemp.history['val_classification_head_acc'])
        # ax.title('model accuracy')
        # ax.ylabel('accuracy')
        # ax.xlabel('epoch')
        # ax.legend(['train', 'test'], loc='upper left')
        # fig.savefig("checkpoints/{}_acc_graph.png".format(self.filename[:-3]))

        with open("checkpoints_autoencoder/{}_history.pkl".format(self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        model.save_weights("checkpoints_autoencoder/{}".format(self.filename))

        return model
