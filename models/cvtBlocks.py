from tensorflow.keras.layers import Layer, LayerNormalization, Dropout
from tensorflow import reshape, concat, zeros, split, Variable, float32
from numpy import linspace as np_linspace
from .cvtLayers import Attention, DropPath, Mlp, ConvEmbed


class Block(Layer):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 drop_path=0.,
                 act_layer="gelu",
                 norm_layer=LayerNormalization,
                 with_cls_token=False):
        super().__init__()

        self.with_cls_token = with_cls_token

        self.norm1 = norm_layer(axis=-1)
        self.attn = Attention(dim_in, dim_out, num_heads,
                              proj_drop=drop, attention_bias=qkv_bias, with_cls_token=with_cls_token)

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
            x = self.drop_path(res + attn, training=training, mask=mask)
            x = x + self.mlp(self.norm2(x), training=training, mask=mask)
        else:
            x = res + attn
            x = x + self.mlp(self.norm2(x), training=training, mask=mask)

        return x


class VisionTransformerStage(Layer):

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer="gelu",
                 norm_layer=LayerNormalization,
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
            self.cls_token = True
        else:
            self.cls_token = False

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
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    with_cls_token=with_cls_token,
                )
            )

        if with_cls_token:
            self.cls = Variable(initial_value=[1.0 for i in range(embed_dim)], dtype=float32, shape=(embed_dim, ))

    def call(self, x, mask=None, training=True):
        x = self.patch_embed(x)
        b, h, w, c = x.shape
        x = reshape(x, (b, h*w, c))

        cls_tokens = None
        if self.cls_token:
            cls_tokens = reshape(zeros((b, 1)) + self.cls, (b, 1, c))
            x = concat((cls_tokens, x), axis=1)

        if training:
            x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x, h=h, w=w, training=training, mask=mask)

        if self.cls_token:
            cls_tokens, x = split(x, [1, h*w], 1)
        # 'b (h w) c -> b h w c'
        x = reshape(x, (b, h, w, x.shape[-1]))

        return x, cls_tokens
