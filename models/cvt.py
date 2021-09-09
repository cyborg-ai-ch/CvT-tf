from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, Dense, Dropout, \
    BatchNormalization, AveragePooling2D, MultiHeadAttention
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow import reshape, Tensor, transpose, constant, split, concat, einsum, divide, Variable, linspace, \
    squeeze, reduce_mean, GradientTape, one_hot, optimizers
from tensorflow import random
from tensorflow import math
from numpy import linspace as np_linspace, isnan
from typing import List, Union, Iterable
import tensorflow_addons as tfa


class QuickGELU(Layer):

    def call(self, x: Tensor, mask=None, training=True) -> Tensor:
        return x * math.sigmoid(constant(1.702) * x)  # TODO: 1.702 is basically useless, remove??


class DropPath(Layer):

    def __init__(self, drop_probability=0.0):
        super(DropPath, self).__init__()
        assert 0.0 < drop_probability < 1.0
        self.drop_probability = drop_probability

    def call(self, x: Tensor, mask=None, training=False):
        if not self.drop_probability > 0.0 or not training:
            return x
        else:
            keep_probability = 1 - self.drop_probability
            shape = x.shape
            random_tensor = random.uniform(shape) + keep_probability
            random_tensor = random_tensor // 1.0
            return divide(x, keep_probability) * random_tensor


class Mlp(Layer):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer="gelu",
                 drop=0.0):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, activation=act_layer)
        self.fc2 = Dense(out_features)  # TODO: no activation function, should there be one??
        self.drop = Dropout(drop)

    def call(self, x, mask=None, training=True):
        x = self.fc1(x)
        if training:
            x = self.drop(x)
        x = self.fc2(x)
        if training:
            x = self.drop(x)
        return x


class Rearrange(Layer):

    def __init__(self, permutation: List[int], new_shape: List[Union[int, str]] = None):
        super(Rearrange, self).__init__()
        self.permutation = permutation
        self.new_shape = new_shape

    def call(self, x: Tensor, mask=None, training=True) -> Tensor:
        x = self.operation(x, self.permutation, self.new_shape)
        return x

    @staticmethod
    def operation(x: Tensor, permutation: List[int], new_shape: List[Union[int, str]] = None) -> Tensor:
        shape = x.shape
        x = transpose(x, permutation)
        if new_shape is not None:
            x = reshape(x, Rearrange.parse_shape(shape, new_shape))
        return x

    @staticmethod
    def parse_shape(old_shape: List[int], new_shape: List[Union[int, str]]) -> List[int]:
        _new_shape = []
        for p in new_shape:
            if isinstance(p, str):
                _new_shape.append(old_shape[int(p)])
            elif isinstance(p, Iterable):
                s = 1
                for i in p:
                    s *= old_shape[int(i)] if isinstance(i, str) else i
                _new_shape.append(s)
            else:
                _new_shape.append(p)
        return _new_shape


class Attention(Layer):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(dim_in, dim_out, kernel_size, stride_q, method)
        self.conv_proj_k = self._build_projection(dim_in, dim_out, kernel_size, stride_kv, method)
        self.conv_proj_v = self._build_projection(dim_in, dim_out, kernel_size, stride_kv, method)

        # self.proj_q = Dense(dim_out, use_bias=qkv_bias)
        # self.proj_k = Dense(dim_out, use_bias=qkv_bias)
        # self.proj_v = Dense(dim_out, use_bias=qkv_bias)

        # self.attn_drop = Dropout(attn_drop)
        # self.proj = Dense(dim_out)
        self.proj_drop = Dropout(proj_drop)
        self.attention = MultiHeadAttention(self.num_heads, dim_out)


    @staticmethod
    def _build_projection(dim_in,
                          dim_out,
                          kernel_size,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = Sequential([
                Conv2D(dim_out, kernel_size, padding="same", strides=stride, use_bias=False, groups=dim_in),
                BatchNormalization(),
                #  'b h w c -> b (h w) c '
                Rearrange([0, 1, 2, 3], ["0", ["1", "2"], "3"])
            ])
        elif method == 'avg':
            proj = Sequential([
                AveragePooling2D(pool_size=kernel_size, padding="same", strides=stride),
                #  'b h w c -> b (h w) c'
                Rearrange([0, 1, 2, 3], ["0", ["1", "2"], "3"])
            ])
        else:
            proj = Rearrange([0, 1, 2, 3], ["0", ["1", "2"], "3"])
        return proj

    def call_conv(self, x, h, w):
        cls_token = None
        if self.with_cls_token:
            cls_token, x = split(x, [1, h*w], axis=1)
        x = reshape(x, (x.shape[0], h, w, x.shape[2]))

        q = self.conv_proj_q(x)
        k = self.conv_proj_k(x)
        v = self.conv_proj_v(x)

        if cls_token is not None:
            q = concat((cls_token, q), axis=1)
            k = concat((cls_token, k), axis=1)
            v = concat((cls_token, v), axis=1)

        return q, k, v

    def call(self, inputs, mask=None, training=True, h=1, w=1):
        x = inputs
        q, k, v = self.call_conv(x, h, w)
        x = self.attention(q, v, key=k)
        if training:
            x = self.proj_drop(x)
        return x


class Block(Layer):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer="gelu",
                 norm_layer=LayerNormalization,
                 with_cls_token=False,
                 method=""):
        super().__init__()

        self.with_cls_token = with_cls_token

        self.norm1 = norm_layer(axis=-1)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else None
        self.norm2 = norm_layer(axis=-1)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def call(self, x, training=True, mask=None, h=1, w=1):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h=h, w=w, training=training, mask=mask)

        if training and self.drop_path is not None:
            x = res + self.drop_path(attn, training=training, mask=mask)
            x = x + self.drop_path(self.mlp(self.norm2(x), training=training, mask=mask), training=training, mask=mask)
        else:
            x = res + attn
            x = x + self.mlp(self.norm2(x), training=training, mask=mask)

        return x


class ConvEmbed(Layer):
    """
    The ConvEmbed Class embeds the input image to embed_dim number of filters.
    The number of patches is represented by (image height)*(image width) // stride**2
    """
    def __init__(self,
                 patch_size=7,
                 embed_dim=64,
                 stride=4,
                 padding="same",
                 norm_layer=None):
        super().__init__()
        self.patch_size = patch_size if not isinstance(patch_size, int) else (patch_size, patch_size)
        self.proj = Conv2D(
            embed_dim,   # number of filters/channels
            patch_size,  # kernel size == patch size
            strides=stride,
            padding=padding
        )
        self.norm = norm_layer(axis=-1) if norm_layer else None

    def call(self, x, training=True, mask=None):
        x = self.proj(x)
        if self.norm:
            # normalize along the last axis if present
            x = self.norm(x)
        return x


class VisionTransformerStage(Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer="gelu",
                 norm_layer=LayerNormalization,
                 method="",
                 with_cls_token=False,
                 **kwargs):
        super(VisionTransformerStage, self).__init__()
        self.embed_dim = embed_dim

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            stride=patch_stride,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        if with_cls_token:
            self.cls_token = Variable(random.truncated_normal((1, 1, embed_dim), stddev=0.02))
        else:
            self.cls_token = None

        self.pos_drop = Dropout(drop_rate)
        dpr = [x for x in np_linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = []
        for j in range(depth):
            self.blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    with_cls_token=with_cls_token,
                    method=method
                )
            )

    def call(self, x, mask=None, training=True):
        x = self.patch_embed(x)
        b, h, w, c = x.shape

        # 'b h w c-> b (h w) c'
        x = reshape(x, (b, h*w, c))

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = concat((cls_tokens, x), dim=1)

        if training:
            x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x, h=h, w=w, training=training, mask=mask)

        if self.cls_token is not None:
            cls_tokens, x = split(x, [1, h*w], 1)
        # 'b (h w) c -> b h w c'
        x = reshape(x, (b, h, w, x.shape[-1]))

        return x, cls_tokens


class ConvolutionalVisionTransformer(Model):

    def __init__(self,
                 num_classes=1000,
                 act_layer="gelu",
                 norm_layer=LayerNormalization,
                 spec=None,
                 learning_rate=0.02):
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
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
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

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(axis=-1)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = Dense(num_classes, kernel_initializer=TruncatedNormal(stddev=0.02), activation="softmax")
        # self._cvt_optimizer = Adam(learning_rate)
        step = Variable(0, trainable=False)
        schedule = optimizers.schedules.PiecewiseConstantDecay(
            [500, 1200], [2e-2, 1e-3, 1e-4])
        # lr and wd can be a function or a tensor
        lr = 1e-1 * schedule(step)
        wd = lambda: 5e-2 * schedule(step)
        self._cvt_optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
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
            # x = Rearrange.operation(x, [0, 2, 3, 1], ["0", ["2", "3"], "1"])
            x = self.norm(x)
            x = reshape(x, (x.shape[0], -1, x.shape[-1]))
            x = reduce_mean(x, axis=1)
        return x

    def call(self, x, training=False, mask=None):
        x = self.call_features(x, training=training, mask=mask)
        x = self.head(x)
        return x

    def from_config(cls, config, custom_objects=None):
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
        # return self._cvt_loss(y, y_true)
