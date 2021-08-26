from numpy import stack, ndarray
from numpy.random import randint
from tensorflow.keras.datasets import cifar100
from tensorflow.python import data
from typing import Tuple, Iterable


class DataLoader:

    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = cifar100.load_data()

    def load_images(self, batch_size=125, test=False) -> data.Dataset:
        x = data.Dataset.from_tensor_slices(self.train_x)
        y = data.Dataset.from_tensor_slices(self.train_y)
        return data.Dataset.zip((x, y)).batch(batch_size)

    def get_random_test_image(self) -> Tuple[ndarray, ndarray]:
        index = randint(0, len(self.test_x))
        return self.test_x[index], self.test_y[index]

    def get_random_test_images(self, number_of_images) -> Iterable[Tuple[ndarray, ndarray]]:
        for i in range(number_of_images):
            yield self.get_random_test_image()
