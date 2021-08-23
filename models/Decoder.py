from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Conv2DTranspose
from tensorflow.keras.regularizers import L2
from tensorflow import Variable, constant, complex64, expand_dims, reshape, math, squeeze
from math import pi


class Decoder(Layer):

    def __init__(self, image_shape=(512, 512)):
        super(Decoder, self).__init__()
        self.latent_dim = int((image_shape[0]*image_shape[1])**0.5)
        self.filter_size = 11
        self.image_shape = image_shape
        # self.conv_options = [
        #     {"filters": 16, "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": 32, "strides": (2, 2), "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": 32, "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": 32, "strides": (2, 2), "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": self.pool, "kernel_size": 3, "activation": "relu", "padding": "same"},
        # ]
        # self.normalize = LayerNormalization(axis=-1)
        self.dense = Dense(self.latent_dim, kernel_regularizer=L2(20.0), activation="relu")
        self.dense_out = Dense(self.image_shape[0]*self.image_shape[1], activation="sigmoid")
        # self.conv_layers = [Conv2DTranspose(**options, kernel_regularizer=L2(100.0)) for options in self.conv_options]
        self.f_weights = Variable(constant([-2 * pi / self.filter_size * i for i in range(self.filter_size)], dtype=complex64))

    def call(self, inputs, training=True, mask=None):
        x = expand_dims(inputs, axis=-1)
        x = math.real(x * math.exp(-1j * self.f_weights))
        x = reshape(x, (x.shape[0], -1))
        x = self.dense(x)
        x = self.dense_out(x)
        x = reshape(x, (x.shape[0], *self.image_shape))
        return squeeze(x)

