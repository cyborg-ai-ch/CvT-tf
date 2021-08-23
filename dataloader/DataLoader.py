from os.path import isdir, join as pjoin
from os import listdir
from numpy import stack, ndarray
from numpy.random import randint
from .ImageLoader import load_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataLoader:

    def __init__(self, data_path: str):
        self.data_path = data_path
        if isdir(data_path):
            self._images = [pjoin(data_path, "images", file_name) for file_name in listdir(pjoin(data_path, "images"))
                            if file_name.endswith("png")]
        else:
            self._images = []
        self._test_images = self._images[:len(self._images)//5]
        self._train_images = self._images[len(self._test_images):]

    def _load_images(self, batch_size=125, test=False) -> ndarray:
        image_names = self._train_images if not test else self._test_images
        image_list = [load_image(image_name, (512, 512)) for image_name in image_names]
        return stack(image_list)

    def load_images(self, batch_size=125, test=False):
        data_gen = ImageDataGenerator(rescale=1./255., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        return data_gen.flow_from_directory(self.data_path, (512, 512), color_mode="grayscale", batch_size=batch_size, class_mode=None)

    def get_random_test_image(self):
        index = randint(0, len(self._test_images))
        return load_image(self._test_images[index], (512, 512))

    def get_random_test_images(self, number_of_images):
        for i in range(number_of_images):
            yield self.get_random_test_image()
