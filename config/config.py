
SPEC = {
    #  MODEL OPTIONS
    "NUM_STAGES": 3,  # Number of Vision Transformer Stages, note that the length of
                      # EMBEDDING, STAGE and ATTENTION OPTIONS must be equal to NUM_STAGES.

    "CLS_TOKEN": True,  # True if the model should use the Class token output for the classification, otherwise a
                        # max pooling layer (over all transformer block output patches) is used.


    #  EMBEDDING OPTIONS
    "PATCH_SIZE": [7, 3, 3],  # Kernel size of the embedding convolution.

    "PATCH_STRIDE": [4, 2, 2],  # Stride of the embedding convolution.
                                # (rule of thumb stride == kernel_size // 2 + kernel_size%2)

    "PATCH_PADDING": ["valid", "same", "same"],  # Padding of the embedding convolution.
                                                 # 'same' generates a padding,
                                                 # whereas 'valid' decreases the height and width.

    "DIM_EMBED": [32, 64, 128],  # the dimension of the embedded image patches.


    # STAGE OPTIONS
    "DEPTH": [1, 2, 5],  # Number of 'Transformer blocks' per Stage.

    "DROP_RATE": [0.01, 0.0, 0.0],  # Probability to drop a 'pixel'.

    "DROP_PATCH_RATE": [0.01, 0.0, 0.05],  # Probability to drop a patch.

    # ATTENTION OPTIONS
    "NUM_HEADS": [1, 4, 6],  # Number of heads of the multi head attention.

    "QKV_BIAS": [True, True, True],  # True if the attention should use a bias.

    "PADDING_KV": ["same", "same", "same"],  # padding of the k and v convolutional projections.

    "STRIDE_KV": [2, 2, 2],  # stride of the kv convolutional projection.

    "PADDING_Q": ["same", "same", "same"],  # padding of the q convolutional projection.

    "STRIDE_Q": [1, 1, 1],  # stride of the q convolutional projection.

    "MLP_RATIO": [4.0, 4.0, 4.0],  # ratio between hidden units and input units in the Multi Layer Perceptron.
                                   # e.g. ratio == 4 means there are four times more hidden units than input units.
}

