from tensorflow import image
from tensorflow import Tensor, where, random, constant, expand_dims
from tensorflow.python import data
from tensorflow.keras.layers import Layer
from tensorflow_addons.image import transform
from numpy import array, stack, ndarray
from math import sin, cos, pi
from random import uniform, seed as python_seed
from typing import Union


class ImageAugmentor(Layer):

    def __init__(self, image_size=(32, 32, 3),
                 brightness=0.2,
                 lower_contrast=0.5,
                 upper_contrast=2.0,
                 lower_saturation=0.75,
                 upper_saturation=1.25,
                 hue_max_delta=0.1,
                 max_rotation=pi/10.0,
                 max_stretch=1.3,
                 salt_and_pepper=1.0/18.0,
                 random_seed=1):
        super(ImageAugmentor, self).__init__()
        self.seed = random_seed
        self.image_size = image_size
        self.brightness = brightness
        self.lower_contrast = lower_contrast
        self.upper_contrast = upper_contrast
        self.lower_saturation = lower_saturation
        self.upper_saturation = upper_saturation
        self.hue_max_delta = hue_max_delta
        self.max_rotation = max_rotation
        self.max_stretch = max_stretch
        self._uniform = random.uniform
        self.salt_and_pepper_ratio = salt_and_pepper
        random.set_seed(self.seed)
        python_seed(self.seed)

    def augement(self, inputs: Union[Tensor, data.Dataset]) -> Union[data.Dataset, Tensor]:
        batch_size, _, _, _ = inputs.shape
        self.seed += 1
        random.set_seed(self.seed)
        python_seed(self.seed)
        images = image.resize(inputs, size=self.image_size[:-1])
        images = image.random_brightness(images, self.brightness, seed=self.seed)
        images = image.random_contrast(images, self.lower_contrast, self.upper_contrast, seed=self.seed)
        images = image.random_saturation(images, self.lower_saturation, self.upper_saturation, seed=self.seed)
        images = image.random_hue(images, self.hue_max_delta, seed=self.seed)
        images = image.random_flip_left_right(images, seed=self.seed)
        transformations = self.generate_transformations(batch_size)
        images = transform(images, transformations, interpolation="bilinear", fill_mode="reflect")
        images = self.salt_and_pepper(images, self.salt_and_pepper_ratio, self.salt_and_pepper_ratio)
        return images

    def _generate_transformation(self):
        rotation_angle = uniform(-self.max_rotation, self.max_rotation)
        rotation = array([[cos(rotation_angle), -sin(rotation_angle)],
                          [sin(rotation_angle), cos(rotation_angle)]])
        stretch = array([[uniform(1./self.max_stretch, self.max_stretch), 0.0],
                         [0.0, uniform(1./self.max_stretch, self.max_stretch)]])
        transformation = stretch@rotation
        return array([transformation[0, 0], transformation[0, 1], 0.0,
                      transformation[1, 0], transformation[1, 1], 0.0,
                      0.0, 0.0])

    def salt_and_pepper(self, images: Tensor, ratio_salt: float, ratio_pepper):
        b, h, w, c = images.shape
        mask = expand_dims((ratio_salt + self._uniform((b, h, w))), axis=-1) + constant([0.0, 0.0, 0.0])
        images = where(mask > 1.0, 1.0, images)
        mask = expand_dims((ratio_pepper + self._uniform((b, h, w))), axis=-1) + constant([0.0, 0.0, 0.0])
        images = where(mask > 1.0, 0.0, images)
        return images

    def test(self, images: Union[Tensor, ndarray]):
        import matplotlib.pyplot as plt
        image_iterator = images.as_numpy_iterator() if isinstance(images, Tensor) else images
        augmented_images = self.augement(images*1.0)
        for image_augmented, image in zip(augmented_images.numpy(), image_iterator):
            fig: plt.Figure = plt.figure()
            ax1: plt.Axes = fig.add_subplot(121)
            ax2: plt.Axes = fig.add_subplot(122)
            ax1.imshow(image_augmented/255.)
            ax2.imshow(image)
            plt.show()

    def resize(self, inputs):
        return image.resize(inputs, size=self.image_size[:-1])

    def generate_transformations(self, number_of_transformations=1):
        return stack([self._generate_transformation() for i in range(number_of_transformations)])

    def call(self, inputs, training=False, mask=None):
        return self.augement(inputs)
