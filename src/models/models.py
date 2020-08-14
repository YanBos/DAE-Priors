import libs.layers as layers
import tensorflow as tf
import utils.utils as utils

class GuidanceNet():
    """
    GuideNet as described in Tang et al.
    """

    def __init__(self, sz=utils.NSZ, batch_norm=True, plain=False):
        """
        Constructor for the GuideNet
        :param sz: integer depth of the neural network
        """

        self.sz = sz
        self.batch_norm = batch_norm
        self.plain = plain
        self.model = self.model_guidance()

    def decoder_disparity(self, last, disparity_encoder_outputs):
        """
        Decoder for the disparity.
        :param last: last layer of the guidance encoder
        :param disparity_encoder_outputs: outputs of the disparity encoder
        :return: The end of the network
        """
        dec_depth = last
        for i in range(self.sz):
            dec_depth = layers.Conv()(dec_depth + disparity_encoder_outputs[self.sz - 1 - i])

        output = layers.Conv(act=False, filters=1)(dec_depth)

        return output

    def decoder_guidance(self, last, geo):
        """
        Guidance decoder
        :param last: last layer of the guidance encoder
        :param geo: The outputs of the guidance encoder
        :return: The decoder outputs (inputs for the guided module)
        """

        dec = layers.Conv()(last)
        decoder_outputs = []
        for i in reversed(range(self.sz - 1)):
            addition = geo[i] + dec
            decoder_outputs.append(addition)
            if i != 0:
                dec = layers.Conv()(addition)

        return decoder_outputs

    def encoder_disparity(self, disparity, guidance_decoder_outputs):
        """
        Disparity encoder
        :param disparity: disparity image
        :param guidance_decoder_outputs: The outputs of the guidance decoder (inputs for the guided module)
        :return:
        """

        conv_depth_1 = layers.Conv()(disparity)
        disparity_encoder_outputs = []
        before_depth = conv_depth_1
        for i in range(self.sz):
            before_depth = layers.ResNetBlock(plain=self.plain, batch_norm=self.batch_norm)(before_depth)
            disparity_encoder_outputs.append(before_depth)
            if i != self.sz - 1:
                before_depth = layers.Guided_Conv()((before_depth, guidance_decoder_outputs[self.sz - 2 - i]))
        return disparity_encoder_outputs

    def encoder_guidance(self, guidance):
        """
        Guidance encoder
        :param guidance: guidance input
        :return: Guidance encoder output features
        """

        conv = layers.Conv()(guidance)
        guidance_encoder_outputs = []
        before = conv
        for i in range(self.sz):
            before = layers.ResNetBlock(plain=self.plain, batch_norm=self.batch_norm)(before)
            if i != (self.sz - 1):
                guidance_encoder_outputs.append(before)

        last = before
        return guidance_encoder_outputs, last

    def model_guidance(self):
        """
        Model definition for the guidance net from the paper of Tang et al.
        :param sz:
        :return:
        """

        # definition of the inputs
        inp_disp = tf.keras.Input(shape=(utils.PSZ, utils.PSZ, 1), batch_size=utils.BSZ)
        inp_guidance = tf.keras.Input(shape=(utils.PSZ, utils.PSZ, 3), batch_size=utils.BSZ)

        # Guidance encoder
        geo, last_encoder = self.encoder_guidance(inp_guidance)

        # Guidance decoder
        gdo = self.decoder_guidance(last_encoder, geo)

        # Disparity encoder
        deo = self.encoder_disparity(inp_disp, gdo)

        # Disparity decoder
        output = self.decoder_disparity(last_encoder, deo)

        return tf.keras.Model([inp_disp, inp_guidance], output)


class PlainNet():
    """
    PlainNet as described in Bigdeli and Zwicker.
    """

    def __init__(self, sz=utils.NSZ, batch_norm=True):
        """
        Constructor for the GuideNet
        :param sz: integer depth of the neural network
        """

        self.sz = sz
        self.batch_norm = batch_norm
        self.model = self.model_plain()

    def model_plain(self):
        """
        Model definition for the guidance net from the paper of Tang et al.
        :param sz:
        :return:
        """

        # definition of the inputs
        inp_disp = tf.keras.Input(shape=(utils.PSZ, utils.PSZ, 1), batch_size=utils.BSZ)

        layer_inp = layers.Conv()(inp_disp)

        for i in range(self.sz - 1):
           layer_inp = layers.ResNetBlock(plain=True, batch_norm=self.batch_norm)(layer_inp)

        output = layers.Conv(act=False, filters=1)(layer_inp)

        return tf.keras.Model(inp_disp, output)


class ResNet():
    """
    PlainNet as described in Bigdeli and Zwicker.
    """

    def __init__(self, sz=utils.NSZ, batch_norm=True, plain=True, act_after_addition=False):
        """
        Constructor for the GuideNet
        :param sz: integer depth of the neural network
        """

        self.sz = sz
        self.batch_norm = batch_norm
        self.plain = plain
        self.act = tf.keras.layers.ReLU() if act_after_addition else lambda x:x
        self.model = self.model_resnet()

    def model_resnet(self):
        """
        ResNet with residual skip connections.
        :param sz:
        :return:
        """

        # definition of the input
        inp_disp = tf.keras.Input(shape=(utils.PSZ, utils.PSZ, 1), batch_size=utils.BSZ)

        layer_inp = layers.Conv()(inp_disp)
        encoder_outputs = [layer_inp]
        for i in range(self.sz - 1):
            layer_inp = layers.ResNetBlock(plain=self.plain, batch_norm=self.batch_norm)(layer_inp)
            encoder_outputs.append(layer_inp)

        layer_inp = layers.ResNetBlock(plain=self.plain, batch_norm=self.batch_norm)(layer_inp)

        for i in range(self.sz - 1):
            if i <= self.sz - 2:
                layer_inp = self.act(encoder_outputs[self.sz - 1 - i] + layer_inp)
            layer_inp = layers.ResNetBlock(plain=self.plain, batch_norm=self.batch_norm)(layer_inp)
        output = layers.Conv(act=False, filters=1)(self.act(layer_inp + encoder_outputs[0])) + inp_disp

        return tf.keras.Model(inp_disp, output)
