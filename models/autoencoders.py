from keras.layers import Conv2D, BatchNormalization, UpSampling2D, Flatten, Dense, Reshape, Conv2DTranspose
from keras import backend as K, Input, Model
from keras import Sequential
from keras_preprocessing.image import ImageDataGenerator
import scipy.io as sio
import numpy as np


#
#
# def shallow_autoencoder(input_layer = Input(shape=(32,32,3))):
#     """
#     Taken from https://github.com/rtflynn/Cifar-Autoencoder
#     optimizer='adam', metrics=['accuracy'], loss='mean_squared_error'
#     :param input_layer: keras tenson
#     :return: the output layer
#     """
#     curr = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3))(input_layer)
#     curr = BatchNormalization()(curr)  # 32x32x32
#     curr = Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(curr)  # 16x16x32
#     curr = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(curr)  # 16x16x32
#     curr = BatchNormalization()(curr)  # 16x16x32
#     curr = UpSampling2D()(curr)
#     curr = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(curr)  # 32x32x32
#     curr = BatchNormalization()(curr)
#     curr = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid')(curr)  # 32x32x3
#     return curr
#
#
# def deep_autoencoder(input_layer = Input(shape=(32,32,3))):
#     """
#     Taken from https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
#     loss='mse', optimizer='adam'
#     :param input_layer: a keras tensor
#     :return: the output layer
#     """
#     kernel_size = 3
#     latent_dim = 256
# # encoder/decoder number of CNN layers and filters per layer
#     layer_filters = [64, 128, 256]
#     curr = input_layer
#     for filters in layer_filters:
#         curr = Conv2D(filters=filters,
#                    kernel_size=kernel_size,
#                    strides=2,
#                    activation='relu',
#                    padding='same')(curr)
#
#     # shape info needed to build decoder model so we don't do hand computation
#     # the input to the decoder's first Conv2DTranspose will have this shape
#     # shape is (4, 4, 256) which is processed by the decoder back to (32, 32, 3)
#     shape = K.int_shape(curr)
#
#     # generate a latent vector
#     curr = Flatten()(curr)
#     curr = Dense(latent_dim, name='latent_vector')(curr)
#
#     curr = Dense(shape[1] * shape[2] * shape[3])(curr)
#     curr = Reshape((shape[1], shape[2], shape[3]))(curr)
#
#     # stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
#     for filters in layer_filters[::-1]:
#         curr = Conv2DTranspose(filters=filters,
#                             kernel_size=kernel_size,
#                             strides=2,
#                             activation='relu',
#                             padding='same')(curr)
#
#     # output layer
#     outputs = Conv2DTranspose(filters=3,
#                               kernel_size=kernel_size,
#                               activation='sigmoid',
#                               padding='same',
#                               name='decoder_output')(curr)
#
#     return outputs

def normalize(X_train: np.ndarray, X_test: np.ndarray):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


class shallow_autoencoder:
    def __init__(self, input_layer):
        # model = Sequential()
        # if input_layer is not None:
        #     model.add(input_layer)
        #     model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))
        # else:
        #     model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
        curr = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(input_layer)
        curr = BatchNormalization()(curr)
        curr = Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(curr)  # 16x16x32
        curr = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(curr)  # 16x16x32
        curr = BatchNormalization()(curr)  # 16x16x32
        curr = UpSampling2D()(curr)
        curr = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(curr)  # 32x32x32
        curr = BatchNormalization()(curr)
        curr = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid', name='decoder_output')(
            curr)  # 32x32x3
        self.out = curr
        #
        # model.add(BatchNormalization())  # 32x32x32
        # model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))  # 16x16x32
        # model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))  # 16x16x32
        # model.add(BatchNormalization())  # 16x16x32
        # model.add(UpSampling2D())
        # model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))  # 32x32x32
        # model.add(BatchNormalization())
        # model.add(Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid', name='decoder_output'))  # 32x32x3
        #
        # model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
        # self.model = model
        # model.summary()


class deep_autoencoder:
    def __init__(self, input_layer):

        input_shape = (32, 32, 3)
        kernel_size = 3
        latent_dim = 256
        # encoder/decoder number of CNN layers and filters per layer
        layer_filters = [64, 128, 256]

        # build the autoencoder model
        # first build the encoder model

        x = input_layer
        # stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
        for filters in layer_filters:
            x = Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=2,
                       activation='relu',
                       padding='same')(x)

        # shape info needed to build decoder model so we don't do hand computation
        # the input to the decoder's first Conv2DTranspose will have this shape
        # shape is (4, 4, 256) which is processed by the decoder back to (32, 32, 3)
        shape = K.int_shape(x)

        # generate a latent vector
        x = Flatten()(x)
        x = Dense(latent_dim, name='latent_vector')(x)

        # instantiate encoder model
        # encoder = Model(inputs, latent, name='encoder')
        # encoder.summary()

        # build the decoder model
        # latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
        x = Dense(shape[1] * shape[2] * shape[3])(x)
        x = Reshape((shape[1], shape[2], shape[3]))(x)

        # stack of Conv2DTranspose(256)-Conv2DTranspose(128)-Conv2DTranspose(64)
        for filters in layer_filters[::-1]:
            x = Conv2DTranspose(filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                activation='relu',
                                padding='same')(x)

        outputs = Conv2DTranspose(filters=3,
                                  kernel_size=kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        # instantiate decoder model
        # decoder = Model(latent_inputs, outputs, name='decoder')
        # decoder.summary()

        # autoencoder = encoder + decoder
        # instantiate autoencoder model
        # autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        # autoencoder.summary()
        #
        # # Mean Square Error (MSE) loss function, Adam optimizer
        # autoencoder.compile(loss='mse', optimizer='adam')
        # self.model = autoencoder
        self.out = outputs


autoencoder_dict = {"shallow": shallow_autoencoder, "deep": deep_autoencoder}
