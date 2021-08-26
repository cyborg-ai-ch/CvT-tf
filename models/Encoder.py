from tensorflow.keras.layers import Layer, MaxPooling2D, Flatten, Conv2D, Dense, Embedding
from tensorflow.keras.regularizers import L2
from tensorflow import Variable, constant, transpose, expand_dims, complex64, reshape, reduce_sum, exp, zeros_like, complex as tf_complex
from math import pi
from typing import List


class Encoder(Layer):

    def __init__(self, image_shape=(512, 512)):
        super(Encoder, self).__init__()
        self.embedding = []
        for i in range(1, 7):
            self.embedding.extend(self._convolutional_unit(2**i, 2**(i+1)))
        self.pos_embedding = Embedding(input_dim=128, output_dim=64)

    def call(self, inputs, training=True, mask=None):
        if inputs.shape.ndims < 4:
            inputs = expand_dims(inputs, axis=-1)
        x = inputs
        for layer in self.embedding:
            x = layer(x)
        return transpose(x, [0, 3, 1, 2]) + reshape(self.pos_embedding(constant(range(128))), (-1, 8, 8))

    def _convolutional_unit(self, filters1: int, filters2: int) -> List[Layer]:
        c1 = Conv2D(filters1, kernel_size=3, activation="relu", padding="same")
        pool = MaxPooling2D((2, 2), strides=2, padding="same")
        c2 = Conv2D(filters2, kernel_size=3, activation="relu", padding="same")
        return [c1, pool, c2]

