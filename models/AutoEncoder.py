from tensorflow import GradientTape, math, squeeze
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from .Encoder import Encoder
from .Decoder import Decoder


class AutoEncoder(Model):

    def __init__(self, image_shape=(512, 512)):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(image_shape=image_shape)
        self.decoder = Decoder(image_shape=image_shape)
        self.optimizer = Adam(1e-2)

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)

    def call(self, inputs, training=False, mask=None):
        return self.decode(self.encode(inputs))

    def _loss(self, inputs):
        return math.reduce_variance(squeeze(inputs) - squeeze(self(inputs)))

    def train_step(self, data):
        with GradientTape() as tape:
            loss = self._loss(data)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}

