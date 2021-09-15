from numpy import stack, ndarray, asarray, arange
from numpy.random import randint, shuffle
from tensorflow.keras.datasets import cifar100
from tensorflow_datasets import load
from tensorflow.python import data
from PIL import Image
from typing import Tuple, Iterable, Generator, List


class DataLoader:

    def load_images(self):
        raise NotImplementedError

    def get_random_test_images(self, number_of_images) -> Iterable[Tuple[ndarray]]:
        raise NotImplementedError

    def batch_generator(self, batch_size=128, split="train") -> Generator[Iterable[Tuple[ndarray]], None, None]:
        raise NotImplementedError

    @property
    def image_size(self) -> List[int]:
        return [0, 0, 0]

    @property
    def num_classes(self):
        return 0


class DataLoaderImageNet(DataLoader):

    def __init__(self):
        self.image_net = {}
        self.image_net["train"], image_net_info = load("ImagenetV2", with_info=True)
        self.image_net["test"], image_net_info = load("ImagenetV2", split="test", with_info=True)

    def load_images(self, batch_size=125) -> data.Dataset:
        x, y = self._load_images(split="train")
        x = data.Dataset.from_tensor_slices(x)
        y = data.Dataset.from_tensor_slices(y)
        return data.Dataset.zip((x, y)).batch(batch_size)

    def get_random_test_image(self) -> Tuple[ndarray, ndarray]:
        test_x, test_y = self._load_images(split="test")
        index = randint(0, len(test_x))
        return test_x[index], test_y[index]

    def _load_images(self, split="train"):
        return self._numpyfy(self.image_net[split], 1000)

    @staticmethod
    def _numpyfy(dataset, max_size=-1):
        x = []
        y = []
        for element in dataset.take(max_size):
            i = element["image"].numpy()
            i = Image.fromarray(i)
            i = i.resize((224, 224))
            x.append(asarray(i))
            y.append(element["label"].numpy())
        x = stack(x)
        y = stack(y)
        return x / 255., y

    def get_random_test_images(self, number_of_images) -> Iterable[Tuple[ndarray]]:
        for i in range(number_of_images):
            yield self.get_random_test_image()

    def batch_generator(self, batch_size=128, split="train") -> Generator[Iterable[Tuple[ndarray]], None, None]:
        size = self.image_net[split].cardinality().numpy()

        def generator(ds):
            index = 0
            while index < size:
                ds.skip(index)
                index += batch_size
                if index >= batch_size:
                    ds.shuffle(2*batch_size)
                yield ds.take(batch_size)

        for window in generator(self.image_net[split]):
            yield self._numpyfy(window)

    @property
    def image_size(self) -> List[int]:
        return [224, 224, 3]

    @property
    def num_classes(self):
        return 1000


class DataLoaderCifar(DataLoader):

    def __init__(self):
        (self.x, self.y), (self.x_test, self.y_test) = cifar100.load_data()

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
        index = randint(0, len(x))
        return x[index]/255.0, y[index]

    def get_random_test_images(self, number_of_images, split="test") -> Iterable[Tuple[ndarray]]:
        return [self.get_random_test_image(split=split) for i in range(number_of_images)]

    def batch_generator(self, batch_size=128, split="train") -> Generator[Iterable[Tuple[ndarray]], None, None]:
        ds = self.load_images(split=split)
        ds = ds.batch(batch_size)
        for batch in ds:
            yield batch

    @property
    def image_size(self) -> List[int]:
        return [32, 32, 3]

    @property
    def num_classes(self):
        return 100

