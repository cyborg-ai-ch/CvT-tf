from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from numpy import save, load, zeros
from typing import Union
import tensorflow as tf


def save_weights(model: Union[Model, Layer], name="weights"):
    weights = model.weights
    save("weights/{}.npy".format(name), [weight.numpy() for weight in weights], allow_pickle=True)


def load_weights(model: Union[Model, Layer], name="weights", input_shape=(1, 512, 512)):
    weights = load("weights/{}.npy".format(name), allow_pickle=True)
    model(zeros(input_shape))
    for index, weight in enumerate(weights):
        tf.compat.v1.assign(model.weights[index], weight)
