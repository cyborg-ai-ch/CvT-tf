# from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization, Dense
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay, CosineDecay
from tensorflow.keras.optimizers import Adam
from tensorflow import Variable, squeeze, reshape, reduce_mean, GradientTape, one_hot, math, reduce_max
from numpy import isnan
from .cvtBlocks import VisionTransformerStage


class ConvolutionalVisionTransformer(Model):

    def __init__(self,
                 num_classes=1000,
                 act_layer="gelu",
                 norm_layer=LayerNormalization,
                 spec=None,
                 learning_rate=5e-4):
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
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            self.stages.append(stage)

        self.norm = norm_layer(axis=-1)
        self.cls_token = spec['CLS_TOKEN']

        # Classifier head
        self.head = Dense(num_classes, kernel_initializer=TruncatedNormal(stddev=0.02), activation="softmax")

        self.step = Variable(0.0, trainable=False)
        # schedule = PiecewiseConstantDecay([50, 150, 360], [1.0, 5e-2, 1e-3, 1e-4])
        learning_rate = CosineDecay(learning_rate, 5000, learning_rate/5.0)
        # lr = lambda: learning_rate * schedule(self.step)
        # wd = lambda: learning_rate / 2.0 * schedule(self.step)
        # self._cvt_optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        self._cvt_optimizer = Adam(learning_rate)
        self.num_classes = num_classes

    def call_features(self, x, training=False, mask=None):
        cls_token = None
        for i in range(self.num_stages):
            x, cls_tokens = self.stages[i](x, training=training, mask=mask)

        if cls_token is not None:
            x = self.norm(cls_token)
            x = squeeze(x)
        else:
            # 'b c h w -> b (h w) c'
            x = self.norm(x)
            x = reshape(x, (x.shape[0], -1, x.shape[-1]))
            # 'b (h w) c' -> 'b c'
            x = reduce_max(x, axis=1)
        return x

    def call(self, x, training=False, mask=None):
        x = self.call_features(x, training=training, mask=mask)
        x = self.head(x)
        return x

    def from_config(self, config, custom_objects=None):
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    def train_step(self, data):
        x, y = data
        with GradientTape() as tape:
            x = self(x, training=True)
            loss = self.cvt_loss(x, one_hot(squeeze(y), self.num_classes))
        if not isnan(loss.numpy()):
            grad = tape.gradient(loss, self.trainable_weights)
            self._cvt_optimizer.apply_gradients(zip(grad, self.trainable_weights))
        self.step.assign_add(1.0)
        return {"loss": loss.numpy()}

    @staticmethod
    def cvt_loss(y, y_true):
        """
        Crossentropy
        :param y: calculated value
        :param y_true: one hot embedded label value
        :return: the loss.
        """
        return reduce_mean(-y_true*math.log(y))
