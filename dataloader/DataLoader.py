from numpy import arange
from numpy.random import randint, shuffle, seed as np_seed
from tensorflow.keras.datasets import cifar100
from os import urandom
from struct import unpack
from tensorflow.python import data
from tensorflow import constant, float32, Tensor
from .Augmentor import ImageAugmentor
from typing import Tuple, Iterable, Generator, List, Union


class DataLoader:

    def __init__(self):
        np_seed(1)

    def get_random_test_images(self,
                               number_of_images,
                               split="test",
                               seed: Union[None, int] = 1) -> Iterable[Tuple[Tensor]]:
        raise NotImplementedError

    def batch_generator(self, batch_size=128, split="train") -> Generator[Iterable[Tuple[Tensor]], None, None]:
        raise NotImplementedError

    def validation_set(self, size=128) -> Tuple[Tensor]:
        raise NotImplementedError

    @property
    def image_size(self) -> List[int]:
        return [0, 0, 0]

    @property
    def num_classes(self) -> int:
        return 0


class DataLoaderCifar(DataLoader):

    def __init__(self, image_size: Union[None, List[int]] = None):
        super().__init__()
        (self.x, self.y), (self.x_test, self.y_test) = cifar100.load_data()
        self._image_size = image_size or [32, 32, 3]
        self._validation_set = None
        self.augmentor = ImageAugmentor(image_size=self._image_size)

    def load_images(self, split="train"):
        if split == "train":
            shuffled_indices = arange(0, len(self.y))
            shuffle(shuffled_indices)
            x = data.Dataset.from_tensor_slices(self.x[shuffled_indices]/255.0)
            y = data.Dataset.from_tensor_slices(self.y[shuffled_indices])
        else:
            x = data.Dataset.from_tensor_slices(self.x_test/255.0)
            y = data.Dataset.from_tensor_slices(self.y_test)
        return data.Dataset.zip((x, y))

    def get_random_test_image(self, split="test") -> Tuple[Tensor]:
        [x, y] = [self.x_test, self.y_test] if split == "test" else [self.x, self.y]
        x = self.augmentor.resize(x).numpy()
        index = randint(0, len(x))
        return x[index]/constant(255.0), y[index]

    def get_random_test_images(self, number_of_images, split="test", seed=None) -> Iterable[Tuple[Tensor]]:
        [x, y] = [self.x_test, self.y_test] if split == "test" else [self.x, self.y]
        seed = seed or abs(unpack("i", urandom(4))[0]//2) - 1
        np_seed(seed)
        indices = randint(0, len(x), number_of_images)
        x = x[indices]
        y = y[indices]
        x = self.augmentor.resize(x)
        return x/255.0, y

    def batch_generator(self, batch_size=128, split="train") -> Generator[Iterable[Tuple[Tensor]], None, None]:
        ds = self.load_images(split=split)
        ds = ds.batch(batch_size)
        for batch in ds:
            x, y = batch
            x_augmented = self.augmentor(x*1)
            yield x_augmented, y

    def validation_set(self, size=128, random=True) -> Iterable[Tuple[Tensor]]:
        if random:
            return self.get_random_test_images(size, split="test")
        elif self._validation_set is None or len(self._validation_set[0]) != size:
            x = self.augmentor.resize(constant(self.x_test[:size]/255.0, dtype=float32))
            y = self.y_test[:size]
            self._validation_set = (x, y)
        return self._validation_set

    @property
    def image_size(self) -> List[int]:
        return self._image_size

    @property
    def num_classes(self):
        return 100

