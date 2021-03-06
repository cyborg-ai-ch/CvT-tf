from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization, Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow import Variable, squeeze, reshape, reduce_mean, GradientTape, one_hot, math, reduce_max
from numpy import isnan
from tensorflow_addons.optimizers import AdamW

from .cvtBlocks import VisionTransformerStage


class ConvolutionalVisionTransformer(Model):

    def __init__(self,
                 num_classes=100,
                 spec=None,
                 learning_rate=5e-4,
                 learning_rate_schedule=None):
        """
        :param num_classes: The Number of outputs for the classifier Head.
        :param spec: The configuration as in config/config.py
        :param learning_rate: Start learning rate.
        :param learning_rate_schedule: The learning rate decay or None
        """
        super(ConvolutionalVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_stages = spec['NUM_STAGES']
        self.stages = []
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'drop_patch_rate': spec['DROP_PATCH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'] if i == self.num_stages - 1 else False,
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformerStage(
                act_layer="gelu",
                norm_layer=LayerNormalization,
                **kwargs
            )
            self.stages.append(stage)

        self.norm = LayerNormalization(axis=-1)
        self.cls_token = spec['CLS_TOKEN']

        # Classifier head
        self.head = Dense(num_classes, kernel_initializer=TruncatedNormal(stddev=0.02), activation="softmax")

        self.step = Variable(0.0, trainable=False)
        lr = learning_rate * learning_rate_schedule(self.step) if learning_rate_schedule is not None else learning_rate
        wd = lambda: learning_rate / 2.0 * learning_rate_schedule(self.step) \
            if learning_rate_schedule is not None else learning_rate / 2.0
        self._cvt_optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        self.num_classes = num_classes

    def _call_features(self, x, training=False, mask=None):
        cls_token = None
        for i in range(self.num_stages):
            x, cls_tokens = self.stages[i](x, training=training, mask=mask)

        if cls_token is not None:
            # cls_token.shape == 'b c'
            x = self.norm(cls_token)
            x = squeeze(x)
        else:
            # x.shape == 'b (h w) c'
            x = self.norm(x)
            x = reshape(x, (x.shape[0], -1, x.shape[-1]))
            # 'b (h w) c' -> 'b c'
            x = reduce_max(x, axis=1)
        return x

    def call(self, x, training=False, mask=None):
        """

        :param x: A Tensor containing Images.
        :param training: True if the model is to be trained. (mainly activates drop out layers.)
        :param mask: Has no effect (just to match the parent function call).
        :return:
        """
        x = self._call_features(x, training=training, mask=mask)
        x = self.head(x)
        return x

    def from_config(self, config, custom_objects=None):
        """
        to be implemented.
        :param config: the serialized configuration of the model, gained by calling get_config.
        :param custom_objects: always None ?
        :return: a ConvolutionalVisionTransformer model instance.
        """
        raise NotImplementedError

    def get_config(self):
        """
        to be implemented.
        :return: the serialization of the model configuration.
        """
        raise NotImplementedError

    def train_step(self, data, validation_data=None):
        """
        Tain a single step over a Batch
        :param data: the batch consisting of a tuple (images, labels).
        :param validation_data: A validation set or None.
        :return: A Dictionary containing the loss, and the val_loss if validation_data is not None.
        """
        x, y = data

        with GradientTape() as tape:
            x = self(x, training=True)
            loss = self.cvt_loss(x, one_hot(squeeze(y), self.num_classes))

        if not isnan(loss.numpy()):
            grad = tape.gradient(loss, self.trainable_weights)
            self._cvt_optimizer.apply_gradients(zip(grad, self.trainable_weights))

        self.step.assign_add(1.0)

        if validation_data is not None:
            x_val, y_val = validation_data
            x_val = self(x_val, training=False)
            val_loss = self.cvt_loss(x_val, one_hot(squeeze(y_val), self.num_classes))
            return {"loss": float(loss.numpy()), "val_loss": float(val_loss.numpy())}

        return {"loss": float(loss.numpy())}

    @staticmethod
    def cvt_loss(y, y_true):
        """
        Crossentropy
        :param y: calculated value
        :param y_true: one hot embedded label value
        :return: the loss.
        """
        return reduce_mean(-y_true*math.log(y))
