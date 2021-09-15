from tensorflow.keras.layers import Layer, Dense, Conv2D, Dropout, MultiHeadAttention, AveragePooling2D, \
    BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow import Tensor, divide, concat, random, split, reshape, transpose, float32
from typing import List, Union, Iterable


class Rearrange(Layer):
    '''
     @Brief: Permutate and reshape a tensor.
    inputs:
    permutation: a list containing the permutated indices of the output eg. for the permutation [1, 2, 3] -> [2, 1, 3] th

    '''

    def __init__(self, permutation: Union[List[int], None], new_shape: List[Union[int, str]] = None):
        super(Rearrange, self).__init__()
        self.permutation = permutation
        self.new_shape = new_shape

    def call(self, x: Tensor, mask=None, training=True) -> Tensor:
        x = self.operation(x, self.permutation, self.new_shape)
        return x

    @staticmethod
    def operation(x: Tensor, permutation: Union[List[int], None], new_shape: List[Union[int, str]] = None) -> Tensor:
        shape = x.shape
        if permutation is not None:
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


class DropPatch(Layer):

    def __init__(self, drop_probability=0.0):
        super(DropPatch, self).__init__()
        assert 0.0 < drop_probability < 1.0
        self.drop_probability = drop_probability

    def call(self, x: Tensor, mask=None, training=False):
        if not self.drop_probability > 0.0 or not training:
            return x
        else:
            keep_probability = 1 - self.drop_probability
            b, hw, c = x.shape
            random_tensor = random.uniform((b, hw, 1), minval=0.0, maxval=1.0, dtype=float32) + keep_probability
            random_tensor = random_tensor // 1.0
            return divide(x, keep_probability) * random_tensor


class ConvEmbed(Layer):
    """
    The ConvEmbed Class embeds the input image to embed_dim number of filters.
    The number of patches is represented by (image height)*(image width) // stride**2
    """
    def __init__(self,
                 patch_padding="same",
                 patch_size=7,
                 embed_dim=64,
                 stride=4,
                 norm_layer=None):
        super().__init__()
        self.patch_size = patch_size if not isinstance(patch_size, int) else (patch_size, patch_size)
        self.proj = Conv2D(
            embed_dim,   # number of filters/channels
            patch_size,  # kernel size == patch size
            strides=stride,
            padding=patch_padding
        )
        self.norm = norm_layer(axis=-1) if norm_layer else None

    def call(self, x, training=True, mask=None):
        x = self.proj(x)
        if self.norm:
            # normalize along the last axis if self.norm is set
            x = self.norm(x)
        return x


class Attention(Layer):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 proj_drop=0.,
                 kernel_size=3,
                 stride_kv=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True,
                 stride_q=1,
                 with_cls_token=False):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(dim_in, kernel_size, stride_q, padding_q)
        self.conv_proj_k = self._build_projection(dim_in, kernel_size, stride_kv, padding_kv)
        self.conv_proj_v = self._build_projection(dim_in, kernel_size, stride_kv, padding_kv)

        self.attention = MultiHeadAttention(self.num_heads, dim_out, use_bias=attention_bias)
        self.proj_drop = Dropout(proj_drop)

    @staticmethod
    def _build_projection(filters, kernel_size, stride, padding):
        proj = Sequential([
            Conv2D(filters, kernel_size, padding=padding, strides=stride, use_bias=False),
            BatchNormalization(),
            #  'b h w c -> b (h w) c '
            Rearrange(None, ["0", ["1", "2"], "3"])
        ])
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


class Mlp(Layer):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer="relu",
                 drop=0.0):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, activation=act_layer)
        self.fc2 = Dense(out_features)
        self.drop1 = Dropout(drop)
        self.drop2 = Dropout(drop)

    def call(self, x, mask=None, training=True):
        x = self.fc1(x)
        if training:
            x = self.drop1(x)
        x = self.fc2(x)
        if training:
            x = self.drop2(x)
        return x


