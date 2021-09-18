from numpy import stack, ndarray, asarray, arange, squeeze
from numpy.random import randint, shuffle, seed as np_seed
from tensorflow.keras.datasets import cifar100
from tensorflow_datasets import load
from tensorflow.python import data
from tensorflow import one_hot, constant, float32
from PIL import Image
from .Augmentor import ImageAugmentor
from typing import Tuple, Iterable, Generator, List, Union


class DataLoader:

    def __init__(self):
        np_seed(1)

    def load_images(self):
        raise NotImplementedError

    def get_random_test_images(self, number_of_images, split="test", seed=1) -> Iterable[Tuple[ndarray]]:
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

    def __init__(self):
        super().__init__()
        self.image_net = {}
        self.image_net["train"], image_net_info = load("ImagenetV2", with_info=True)
        self.image_net["test"], image_net_info = load("ImagenetV2", split="test", with_info=True)
        self._validation_set = None

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

    def get_random_test_images(self, number_of_images, split="test") -> Iterable[Tuple[ndarray]]:
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

    def validation_set(self, size=128):
        if self._validation_set is None or len(self.validation_set[0]) != size:
            test_x, test_y = self._load_images(split="test")
            self._validation_set = test_x[:size], test_y[:size]
        return self._validation_set

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

    def get_random_test_images(self, number_of_images, split="test", seed=1) -> Iterable[Tuple[ndarray]]:
        [x, y] = [self.x_test, self.y_test] if split == "test" else [self.x, self.y]
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

    def validation_set(self, size=128):
        if self._validation_set is None or len(self._validation_set[0]) != size:
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

