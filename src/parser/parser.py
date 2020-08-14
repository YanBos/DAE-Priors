import utils.utils as utils
import argparse
import os
import tensorflow as tf

opt = None

def update_utils():
    global opt

    if opt is None:
        return
    utils.NSZ = opt.NSZ
    utils.BSZ = opt.BSZ
    utils.FILTERS = opt.FILTERS
    utils.KS = opt.KS
    utils.SD = opt.SD
    utils.ACTIVATION = opt.ACTIVATION
    utils.PSZ = opt.PSZ
    utils.EPOCHS = opt.EPOCHS
    utils.LR = opt.LR
    utils.LNL = opt.LNL
    utils.UNL = opt.UNL
    utils.RSZ = opt.RSZ
    utils.OFFSET += opt.OFFSET
    if utils.RSZ:
        utils.OFFSET += "_RSZ"
    utils.SAVE = utils.SD + utils.OFFSET + "_NSZ-" + str(utils.NSZ) + "_PSZ-" + str(utils.PSZ) + "_BSZ-" + str(utils.BSZ) + "_FILTERS-" + str(utils.FILTERS) + "_NL-" + str(utils.LNL) + "-" + str(utils.UNL) + "_ACT-" + utils.ACTIVATION
    utils.TBP = utils.TBP + utils.OFFSET + "_NSZ-" + str(utils.NSZ) + "_PSZ-" + str(utils.PSZ) + "_BSZ-" + str(utils.BSZ) + "_FILTERS-" + str(utils.FILTERS) + "_NL-" + str(utils.LNL) + "-" + str(utils.UNL) + "_ACT-" + utils.ACTIVATION
    utils.TB = tf.keras.callbacks.TensorBoard(log_dir=utils.TBP, profile_batch=0)

parser = argparse.ArgumentParser()
parser.add_argument('--OFFSET', type=str, default=utils.OFFSET)
parser.add_argument('--RSZ', type=int, default=utils.RSZ)
parser.add_argument('--NSZ', type=int, default=utils.NSZ)
parser.add_argument('--BSZ', type=int, default=utils.BSZ)
parser.add_argument('--FILTERS', type=int, default=utils.FILTERS)
parser.add_argument('--KS', type=int, default=utils.KS)
parser.add_argument('--SD', type=str, default=utils.SD)
parser.add_argument('--ACTIVATION', type=str, default=utils.ACTIVATION)
parser.add_argument('--PSZ', type=int, default=utils.PSZ)
parser.add_argument('--EPOCHS', type=int, default=utils.EPOCHS)
parser.add_argument('--LR', type=float, default=utils.LR)
parser.add_argument('--LNL', type=float, default=utils.LNL)
parser.add_argument('--UNL', type=float, default=utils.UNL)

