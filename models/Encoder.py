from tensorflow.keras.layers import Layer, MaxPooling2D, Flatten, Conv2D, Dense
from tensorflow.keras.regularizers import L2
from tensorflow import Variable, constant, complex64, reshape, reduce_sum, exp, zeros_like, complex as tf_complex
from math import pi


class Encoder(Layer):

    def __init__(self, image_shape=(512, 512)):
        super(Encoder, self).__init__()
        self.latent_dim = int((image_shape[0]*image_shape[1])**0.5)
        self.filter_size = 11
        # self.conv_options = [
        #     {"filters": 2, "strides": (2, 2), "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": 4, "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": 8, "strides": (2, 2), "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": 8, "strides": (2, 2), "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": 8, "strides": (2, 2), "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     # {"filters": 8, "strides": (2, 1), "kernel_size": 3, "activation": "relu", "padding": "same"},
        #     {"filters": self.filter_size, "kernel_size": 3, "activation": "relu", "padding": "same"},
        # ]
        # self.conv_layers = [Conv2D(**options, kernel_regularizer=L2(100.0)) for options in self.conv_options]
        self.dense_test = [Flatten(), Dense(64, activation="relu"), Dense(self.filter_size*self.latent_dim, activation="relu")]
        self.f_weights = Variable(constant([2*pi/self.filter_size * i for i in range(self.filter_size)], dtype=complex64))
        self.pool = MaxPooling2D((2, 2))
        self.dense = Dense(self.latent_dim, kernel_regularizer=L2(100))

    def call(self, inputs, training=True, mask=None):
        x = inputs
        for d in self.dense_test:
            x = d(x)
        x = reshape(x, (x.shape[0], self.latent_dim, self.filter_size))
        x = reduce_sum(exp(1j * self.f_weights) * tf_complex(x, zeros_like(x)), axis=-1)
        x = reshape(x, (x.shape[0], -1))
        return x
