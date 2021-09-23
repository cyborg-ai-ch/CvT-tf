from numpy import stack, ndarray, asarray, arange
from numpy.random import randint, shuffle, seed as np_seed
from tensorflow.keras.datasets import cifar100
from os import urandom
from struct import unpack
from tensorflow_datasets import load, ImageFolder
from tensorflow_datasets.image_classification import imagenet2012_multilabel
from tensorflow.python import data
from tensorflow import one_hot, constant, float32
from itertools import cycle
from PIL import Image
from .Augmentor import ImageAugmentor
from typing import Tuple, Iterable, Generator, List, Union, Callable


class DataLoader:

    def __init__(self):
        np_seed(1)

    def load_images(self):
        raise NotImplementedError

    def get_random_test_images(self,
                               number_of_images,
                               split="test",
                               seed: Union[None, int] = 1) -> Iterable[Tuple[ndarray]]:
        raise NotImplementedError

    def batch_generator(self, batch_size=128, split="train") -> Generator[Iterable[Tuple[ndarray]], None, None]:
        raise NotImplementedError

    def validation_set(self, size=128):
        raise NotImplementedError

    @property
    def image_size(self) -> List[int]:
        return [0, 0, 0]

    @property
    def num_classes(self):
        return 0


class DataLoaderImageNet(DataLoader):

    def __init__(self, image_size=(72, 72, 3)):
        super().__init__()
        self.builder = ImageFolder("~/data/imagenet/ILSVRC/Data/CLS-LOC/")
        self._image_size = list(image_size)
        self.augmentor = ImageAugmentor(image_size=self._image_size)
        self._set = {}
        self._set_gen = {}

    def get_random_test_images(self, number_of_images, split="test", seed=1) -> Iterable[Tuple[ndarray]]:
        return self._set_next(split, number_of_images, self.augmentor.resize)

    def batch_generator(self, batch_size=128, split="train") -> Generator[Iterable[Tuple[ndarray]], None, None]:
        if split not in self._set or split not in self._set_gen:
            self._set_init(split, batch_size, self.augmentor)
        return self._set_gen[split]

    def validation_set(self, size=128):
        return self._set_next("val", size, self.augmentor.resize)

    def _set_next(self, split: str, batch_size: int, augmentor: Callable):
        if split not in self._set or split not in self._set_gen:
            self._set_init(split, batch_size, augmentor)
        return self._set_gen[split].__next__()

    def _set_init(self, split: str, batch_size: int, augmentor: Callable):
        self._set[split] = self.builder.as_dataset(split="split",
                                                   shuffle_files=True,
                                                   batch_size=batch_size,
                                                   as_supervised=True)
        self._set_gen[split] = cycle(self._set_generator(self._set[split], augmentor))

    @staticmethod
    def _set_generator(batched_set: data.Dataset, augmentor: Callable):
        for x, y in batched_set:
            yield augmentor(x / 255.0), y

    @property
    def image_size(self) -> List[int]:
        return [224, 224, 3]

    @property
    def num_classes(self):
        return 1000


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

    def get_random_test_image(self, split="test") -> Tuple[ndarray]:
        [x, y] = [self.x_test, self.y_test] if split == "test" else [self.x, self.y]
        x = self.augmentor.resize(x).numpy()
        index = randint(0, len(x))
        return x[index]/255.0, y[index]

    def get_random_test_images(self, number_of_images, split="test", seed=None) -> Iterable[Tuple[ndarray]]:
        [x, y] = [self.x_test, self.y_test] if split == "test" else [self.x, self.y]
        seed = seed or abs(unpack("i", urandom(4))[0]//2) - 1
        np_seed(seed)
        indices = randint(0, len(x), number_of_images)
        x = x[indices]
        y = y[indices]
        x = self.augmentor.resize(x)
        return x/255.0, y

    def batch_generator(self, batch_size=128, split="train") -> Generator[Iterable[Tuple[ndarray]], None, None]:
        ds = self.load_images(split=split)
        ds = ds.batch(batch_size)
        test_augmentation = False
        first_batch = False

        if test_augmentation:
            import matplotlib.pyplot as plt
            fig: plt.Figure = plt.figure()
            ax_real: plt.Axes = fig.add_subplot(121)
            ax_real.set_title("Real Image")
            ax_augmented: plt.Axes = fig.add_subplot(122)
            ax_augmented.set_title("Augmented Image")

        for batch in ds:
            x, y = batch
            x_augmented = self.augmentor(x*1)

            if first_batch and test_augmentation:
                first_batch = False
                ax_real.imshow(x.numpy()[0])
                ax_augmented.imshow(x_augmented.numpy()[0])
                fig.show()
                fig.canvas.draw()
                done = False
                while not done:
                    try:
                        fig.canvas.flush_events()
                    except Exception:
                        done = True

            yield x_augmented, y

    def validation_set(self, size=128, random=True):
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

