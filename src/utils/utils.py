import tensorflow as tf
import os.path as os
import math

TDS = ["/mnt/data/lightfield/blender.tfrecords", "/mnt/data/lightfield/benchmark.tfrecords" if os.exists("/mnt/data/lightfield/benchmark.tfrecords") else "/Users/yannickbosch/Desktop/full_data/benchmark.tfrecords", "/mnt/data/lightfield/povray.tfrecords" if os.exists("/mnt/data/lightfield/povray.tfrecords") else "/Users/yannickbosch/Desktop/full_data/povray.tfrecords"]
SD ="/mnt/data/models/FINAL_V2" if os.exists("/mnt/data/models/FINAL") else "GI"
TBP = "/mnt/tf-logs/FINAL_V2"
BSZ = 8

# Dataset information
NB_DS_RNG = 4.031761861577326
NB_DS_NE = 49

DS_RNG = 4.234977262230762
DS_NE = 1245

CDS = 1245
CDSS = math.floor(1245*0.1)
BVS = math.floor(CDSS/BSZ)
SPE = math.floor(8*CDS/BSZ)

# parameters for the network
EPOCHS = int(1200000/SPE) if os.exists("/mnt/data/lightfield") else 1
FILTERS = 64
NSZ = 10
KS = 3
OFFSET="/"

# Convolution
ACTIVATION = 'relu'
PADDING = 'same'

LR = 1e-4
PSZ = 64

RSZ = 1

LNL = 0.1
UNL = 0.1

# save string
SAVE = ""

# dataset dict
FEATURES = {'width': tf.io.FixedLenFeature((), tf.int64),
            'height': tf.io.FixedLenFeature((), tf.int64),
            "disp": tf.io.FixedLenFeature((), tf.string),
            "guidance": tf.io.FixedLenFeature((), tf.string)}

# callbacks
ES = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100, restore_best_weights=True)
TB = None
FP = ""

