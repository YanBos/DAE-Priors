import tensorflow as tf
import utils.utils as utils


class ResNetBlock(tf.keras.layers.Layer):
    """
    Basic ResNet Block as described in the paper by He et. al.
    """

    def __init__(self, out_act=True, batch_norm=True, plain=False):
        """
        Constructor of the ResNetBlock.
        :param out_act: bool if relu after addition
        :param batch_norm: bool if batch normalization should be used
        :return:
        """
        super(ResNetBlock, self).__init__()

        self.plain=plain
        self.conv_0, self.conv_1 = Conv(act=False), Conv(act=False)
        if batch_norm:
            self.bn_or_id_0, self.bn_or_id_1 = tf.keras.layers.BatchNormalization(), tf.keras.layers.BatchNormalization()
        else:
            self.bn_or_id_0, self.bn_or_id_1 = tf.keras.layers.Lambda(lambda x:x), tf.keras.layers.Lambda(lambda x:x)
        self.act = tf.keras.layers.ReLU()
        self.act_out = tf.keras.layers.ReLU()
        self.out_act = out_act

    def get_config(self):
        """
        Updating the layer configuration
        :return: The configuration
        """

        return {'plain': self.plain,
                'conv_0': self.conv_0,
                'conv_1': self.conv_0,
                'bn_or_id_0': self.bn_or_id_0,
                'bn_or_id_1': self.bn_or_id_1,
                'act': self.act,
                'out_act': self.out_act}

    def call(self, x, **kwargs):
        """
        Forward pass.
        :param x:
        :param kwargs:
        :return:
        """
        out = self.conv_0(x)
        out = self.bn_or_id_0(out)
        out = self.act(out)
        out = self.conv_1(out)
        out = self.bn_or_id_1(out)
        if not self.plain:
            out = out + x
        if self.out_act:
            out = self.act_out(out)

        return out


class Conv(tf.keras.layers.Layer):
    """
    Just a wrapper for a normal convolution operation.
    """

    def __init__(self, act=True, filters=utils.FILTERS):
        """
        Constructor for convolutional layer with arguments coming from a utils File.
        :param act:
        """

        super(Conv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=utils.KS,
                                           strides=1,
                                           padding='same',
                                           activation=tf.keras.layers.Activation(utils.ACTIVATION) if act else None)

    def get_config(self):
        """
        Updating the layer configuration
        :return: The configuration
        """

        return {'conv': self.conv}

    def call(self, x, **kwargs):
        """
        Forward pass
        :param x:
        :param kwargs:
        :return: the output of the layer
        """

        out = self.conv(x)
        return out


class KGL(tf.keras.layers.Layer):
    """
    Kernel Generating layer as described in Tang et al.
    TODO: Get reason for numerical instability
    """

    def __init__(self):
        """
        Constructor for the kernel generating layer.
        TODO: Remove hard coded arguments constructing the kernel generating layers.
        """

        super(KGL, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=9,kernel_size=utils.KS, strides=8, padding='same')
        self.ds = tf.keras.layers.Dense(utils.FILTERS**2)

        self.conv_kernels = lambda x: tf.keras.layers.Reshape((utils.KS, utils.KS, utils.FILTERS, 1))(self.conv(x))
        self.dense = lambda x: tf.keras.layers.Reshape((1, 1, utils.FILTERS, utils.FILTERS))(self.ds(x))

    def get_config(self):
        """
        Updating the layer configuration
        :return: The configuration
        """

        return {'conv_kernels': self.conv_kernels,
                'dense': self.dense,
                'conv': self.conv,
                'ds': self.dense}

    def call(self, x, **kwargs):
        """
        Forward pass.
        :param x:
        :param kwargs:
        :return:
        """
        if not utils.RSZ:
            patches = tf.image.extract_patches(images=x,
                                               sizes=[1, utils.PSZ, utils.PSZ, 1],
                                               strides=[1, utils.PSZ, utils.PSZ, 1],
                                               rates=[1, 1, 1, 1],padding='VALID')
            patches = tf.reshape(patches,(patches.shape[1]*patches.shape[2], x.shape[0], utils.PSZ, utils.PSZ, utils.FILTERS))

            W_G1 = tf.map_fn(lambda inp: tf.clip_by_norm(self.conv_kernels(inp), 1, axes=[1,2]), patches)
            W_G2 = tf.map_fn(lambda inp: tf.clip_by_norm(self.dense(tf.keras.layers.GlobalAveragePooling2D()(inp)), 1, axes=[1,2,3]), patches)
        else:
            x = tf.reshape(x, (utils.FILTERS, x.shape[0], x.shape[1], x.shape[2]))
            x = tf.map_fn(lambda inp: tf.image.resize(tf.expand_dims(inp, axis=3), (utils.PSZ, utils.PSZ)), x)
            x = tf.reshape(x, (x.shape[1], utils.PSZ, utils.PSZ, utils.FILTERS))

            W_G2 = tf.clip_by_norm(self.dense(tf.keras.layers.GlobalAveragePooling2D()(x)), 1, axes=[1, 2, 3])
            W_G1 = tf.clip_by_norm(self.conv_kernels(x), 1, axes=[1, 2])

        return (W_G1, W_G2)


class Guided_Conv(tf.keras.layers.Layer):
    """
    Guided convolutional layer as described in Tang et al.
    """

    def __init__(self):
        """
        Constructor of the guided convolutional module
        """

        super(Guided_Conv, self).__init__()
        self.kgl = KGL()

    def depthwise_conv(self, inp):
        """
        A wrapper for the depthwise convolution that is applied on a sample basis
        :param inp:
        :return: tuple the same size as inp
        """

        conv = tf.nn.depthwise_conv2d(tf.expand_dims(inp[1], axis=0),
                                      filter=inp[0],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
        return tf.squeeze(conv), inp[0]

    def crosschannel_conv(self, inp):
        """
        A wrapper for the crosschannel convolution that is applied on a sample basis
        :param inp:
        :return: tuple the same size as inp
        """

        conv = tf.nn.conv2d(tf.expand_dims(inp[1], axis=0),
                            filters=inp[0],
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        return tf.squeeze(conv), inp[0]

    def get_config(self):
        """
        Updating the layer configuration
        :return: The configuration
        """

        return {'kgl': self.kgl}

    def call(self, x, **kwargs):
        """
        Forward pass
        :param x:
        :param kwargs:
        :return:
        """

        guidance = x[0]
        depth = x[1]

        if not utils.RSZ:

            depth_patches = tf.image.extract_patches(images=depth,
                                               sizes=[1, utils.PSZ, utils.PSZ, 1],
                                               strides=[1, utils.PSZ, utils.PSZ, 1],
                                               rates=[1, 1, 1, 1], padding='VALID')
            depth_patches = tf.reshape(depth_patches,
                                 (depth_patches.shape[1] * depth_patches.shape[2], guidance.shape[0], utils.PSZ, utils.PSZ, utils.FILTERS))

            W_G1, W_G2 = self.kgl(guidance)

            dc, _ = tf.map_fn(lambda inp:
                tf.map_fn(self.depthwise_conv, (inp[0], inp[1]))
            , (W_G1, depth_patches))
            cc, _ = tf.map_fn(lambda inp:
                tf.map_fn(self.crosschannel_conv,(inp[0], inp[1]))
            , (W_G2, dc))
        else:
            W_G1, W_G2 = self.kgl(guidance)
            dc, _ = tf.map_fn(self.depthwise_conv, (W_G1, depth))
            cc, _ = tf.map_fn(self.crosschannel_conv,(W_G2, dc))

        return tf.reshape(cc, (guidance.shape[0], guidance.shape[1], guidance.shape[2], utils.FILTERS)) if not utils.RSZ else cc