from tensorflow.keras.layers import Layer, Conv2D, MultiHeadAttention, LayerNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import reshape, transpose, ones


class MLP(Layer):

    def __init__(self, in_features, out_features=None, hidden_features=None, activation="gelu", dropout_rate=0.0):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden = Dense(hidden_features, activation=activation)
        self.out = Dense(out_features)
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, mask=None, training=True):
        x = self.hidden(inputs)
        x = self.out(x)
        if training:
            return self.dropout(x)
        return x


class ConvAttention(Layer):

    def __init__(self, input_dimension, output_dimension=None, number_of_heads=4, channels=3):
        super(ConvAttention, self).__init__()
        self.output_dimension = output_dimension or input_dimension
        self.input_dimension = input_dimension
        self.conv_projection_k = Conv2D(channels, kernel_size=3, padding="same", strides=1)
        self.conv_projection_v = Conv2D(channels, kernel_size=3, padding="same", strides=1)
        self.conv_projection_q = Conv2D(channels, kernel_size=3, padding="same", strides=1)
        b, h, w, c = input_dimension
        self.attention = MultiHeadAttention(number_of_heads, key_dim=(h*w, channels))

    def call(self, inputs, mask=None, training=True):
        q, k, v = inputs
        q = self.conv_projection_q(q)
        v = self.conv_projection_v(v)
        k = self.conv_projection_k(k)
        b, h, w, c = v.shape
        q = reshape(transpose(q, [0, 3, 1, 2]), (b, c, h*w))
        k = reshape(transpose(k, [0, 3, 1, 2]), (b, c, h*w))
        v = reshape(transpose(v, [0, 3, 1, 2]), (b, c, h*w))
        out = self.attention(q, v, key=k)
        out = transpose(reshape(out, (b, c, h, w)), [0, 2, 3, 1])
        return out


class TransformerBlock(Layer):

    def __init__(self, input_dim, output_dim, number_of_heads):
        super(TransformerBlock, self).__init__()
        b, h, w, c = output_dim
        self.attention = ConvAttention(input_dim, output_dim, number_of_heads=number_of_heads, channels=c)
        self.mlp = MLP(in_features=h*w*c, out_features=h*w*c, hidden_features=h*w*c)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, mask=None, training=True):
        key, value, query = inputs
        x = self.attention(query, value, key=key, attention_mask=mask, training=training, mask=mask)
        x = self.norm1(x + value)
        x = self.mlp(x, training=training)
        x = self.norm2(x + value)
        return x


class ConvEmbed(Layer):

    def __init__(self, channels, norm=True):
        super(ConvEmbed, self).__init__()
        self.convs = [
            Conv2D(channels, kernel_size=3, strides=2, padding="same"),
            Conv2D(channels, kernel_size=3, strides=2, padding="same"),
            Conv2D(channels, kernel_size=3, padding="same")]
        self.norm = LayerNormalization(epsilon=1e-6) if norm else None

    def call(self, inputs, mask=None, training=True):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        if self.norm is not None:
            x = self.norm(x)  # TODO: reshape, along which axis??
        return x


class VisionTransformerStage(Layer):

    def __init__(self, depth, image_dim=(32, 32, 3)):
        super(VisionTransformerStage, self).__init__()
        h, w, c = image_dim
        self.embedding = ConvEmbed(c)
        h = h // 4
        w = w // 4
        self.blocks = [
            TransformerBlock((None, h, w, c), (None, h, w, c), 4)
            for _ in range(depth)
        ]

    @staticmethod
    def _init_head(shape):
        return ones(shape) * 0.2

    def call(self, inputs, mask=None, training=True):
        x = inputs
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, x, x)
        return x

    def get_config(self):
        # stub
        raise NotImplementedError

    def from_config(self, config, custom_objects=None):
        # stub
        raise NotImplementedError


class VisionTransformer(Model):

    def __init__(self, num_classes=100, image_dim=(32, 32, 3)):
        super(VisionTransformer, self).__init__()
        self.optimizer = Adam(1e-3)
        self.stages = [VisionTransformerStage(depth=1, image_dim=image_dim),
                       VisionTransformerStage(depth=1, image_dim=(image_dim[0]//4, image_dim[1]//4, 3)),
                       VisionTransformerStage(depth=1, image_dim=(image_dim[0]//16, image_dim[2]//16, 3))]
        self.head = Dense(num_classes, activation="sigmoid", kernel_initializer=self._init_head)

    def call(self, inputs, mask=None, training=False):
        x = inputs
        batch_size = x.shape[0]
        for stage in self.stages:
            x = stage(x)
        x = reshape(x, (batch_size, -1))
        return self.head(x)

    def loss(self, x, y):


    def train_step(self, data):
        x, y = data

