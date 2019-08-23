from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, UpSampling2D, Flatten, Dense, Reshape, Conv2DTranspose


class shallow_autoencoder:
    """
    Taken from https://github.com/rtflynn/Cifar-Autoencoder
    optimizer='adam', metrics=['accuracy'], loss='mean_squared_error'
    """

    def __init__(self, input_layer):
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


class deep_autoencoder:
    """
    Taken from https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
    loss='mse', optimizer='adam'
     """

    def __init__(self, input_layer):

        kernel_size = 3
        latent_dim = 256
        # encoder/decoder number of CNN layers and filters per layer
        layer_filters = [64, 128, 256]

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

        self.out = outputs


autoencoder_dict = {"shallow": shallow_autoencoder, "deep": deep_autoencoder}
