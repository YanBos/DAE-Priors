#!/usr/bin/python
import tensorflow as tf
import libs.dataset as dataset
from utils import utils as utils
from callbacks import callbacks_guidance as cb
import models.models as models
import os
import parser.parser as parser

# update the util variables before training.
parser.opt = parser.parser.parse_args()
parser.update_utils()
net = models.GuidanceNet(utils.NSZ, batch_norm=True, plain=True).model
ds, ds_cb, vds = dataset.prepare_dataset(gd=True)
adam = tf.keras.optimizers.Adam(learning_rate=utils.LR)
net.compile(optimizer=adam, loss=tf.keras.losses.MSE)
cbs = [] if not os.path.exists('/mnt/tf-logs') else [utils.ES, utils.TB, cb.Image(ds_cb)]
net.fit(ds,
        epochs=utils.EPOCHS,
        steps_per_epoch=utils.SPE, validation_data=vds, validation_steps=utils.BVS, callbacks=cbs)

try:
    tf.keras.models.save_model(net, utils.SAVE)
except AssertionError as e:
    print('Couldn\'t save the model.')
    print(str(e))

try:
    net.save_weights(utils.SAVE + "/weights.h5")
except Exception as e:
    print('Couldn\'t save weights.')
    print(str(e))
